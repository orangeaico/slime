import math
from collections.abc import Sequence

import torch


def _resolve_active_mask(
    active_mask: Sequence[bool] | None,
    *,
    length: int,
) -> list[bool]:
    if active_mask is None:
        return [True] * length

    resolved_mask = [bool(v) for v in active_mask]
    if len(resolved_mask) != length:
        raise ValueError(f"active_mask length {len(resolved_mask)} does not match expected length {length}.")
    return resolved_mask


def get_active_sample_mask_from_loss_masks(loss_masks: Sequence[Sequence[int]]) -> list[bool]:
    return [sum(loss_mask) > 0 for loss_mask in loss_masks]


def get_prompt_group_indices(group_indices: Sequence[int | None], n_samples_per_prompt: int) -> list[int]:
    resolved_group_indices = []
    for sample_idx, group_index in enumerate(group_indices):
        if group_index is None:
            group_index = sample_idx // n_samples_per_prompt
        resolved_group_indices.append(int(group_index))
    return resolved_group_indices


def get_prompt_group_mean_centered_rewards(
    raw_rewards: Sequence[float],
    group_indices: Sequence[int],
    active_mask: Sequence[bool] | None = None,
) -> torch.Tensor:
    rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    centered_rewards = torch.zeros_like(rewards)
    resolved_active_mask = _resolve_active_mask(active_mask, length=len(raw_rewards))

    rewards_by_group: dict[int, list[tuple[int, float]]] = {}
    for idx, (reward, group_index, is_active) in enumerate(zip(raw_rewards, group_indices, resolved_active_mask, strict=True)):
        if not is_active:
            continue
        rewards_by_group.setdefault(group_index, []).append((idx, reward))

    for entries in rewards_by_group.values():
        group_reward_tensor = torch.tensor([reward for _, reward in entries], dtype=rewards.dtype)
        group_mean = group_reward_tensor.mean()
        for (sample_idx, _), centered_reward in zip(entries, group_reward_tensor - group_mean, strict=True):
            centered_rewards[sample_idx] = centered_reward

    return centered_rewards


def get_batch_normalized_prompt_rewards(
    raw_rewards: Sequence[float],
    group_indices: Sequence[int],
    eps: float = 1e-6,
    active_mask: Sequence[bool] | None = None,
) -> list[float]:
    centered_rewards = get_prompt_group_mean_centered_rewards(raw_rewards, group_indices, active_mask=active_mask)
    resolved_active_mask = _resolve_active_mask(active_mask, length=len(raw_rewards))
    active_centered_rewards = centered_rewards[torch.tensor(resolved_active_mask, dtype=torch.bool)]
    if active_centered_rewards.numel() > 1:
        batch_std = active_centered_rewards.std()
    else:
        batch_std = torch.zeros((), dtype=centered_rewards.dtype)
    return (centered_rewards / (batch_std + eps)).tolist()


def get_prompt_loss_token_weights(
    loss_masks: Sequence[Sequence[int]],
    group_indices: Sequence[int],
) -> tuple[list[float], int]:
    prompt_token_counts: dict[int, int] = {}
    for loss_mask, group_index in zip(loss_masks, group_indices, strict=True):
        prompt_token_counts[group_index] = prompt_token_counts.get(group_index, 0) + int(sum(loss_mask))

    weights = []
    for group_index in group_indices:
        prompt_token_count = prompt_token_counts[group_index]
        weights.append(0.0 if prompt_token_count <= 0 else 1.0 / prompt_token_count)

    num_prompt_groups = sum(prompt_token_count > 0 for prompt_token_count in prompt_token_counts.values())
    return weights, num_prompt_groups


def get_required_prompt_group_multiple(
    dp_size: int,
    n_samples_per_prompt: int,
    require_prompt_group_alignment: bool,
) -> int:
    if not require_prompt_group_alignment:
        return dp_size
    return math.lcm(dp_size, n_samples_per_prompt)


def should_force_per_token_loss(loss_type: str, prompt_level_loss_aggregation: bool) -> bool:
    return loss_type in {"cispo_loss", "dispo_loss"} and not prompt_level_loss_aggregation


def compute_dapo_style_length_penalty(
    response_length: int,
    *,
    max_response_len: int,
    cache_len: int,
) -> float:
    if cache_len <= 0:
        raise ValueError("cache_len must be positive for dapo_style length penalty.")

    penalty = (max_response_len - response_length) / cache_len - 1.0
    return float(max(-1.0, min(0.0, penalty)))


def apply_length_penalty(
    raw_reward: float,
    response_length: int,
    *,
    length_penalty_type: str,
    max_response_len: int,
    cache_len: int | None,
) -> float:
    if length_penalty_type != "dapo_style":
        return float(raw_reward)

    if cache_len is None:
        raise ValueError("cache_len must be provided when dapo_style length penalty is enabled.")

    if raw_reward != 1.0:
        return float(raw_reward)

    return float(
        raw_reward
        + compute_dapo_style_length_penalty(
            response_length,
            max_response_len=max_response_len,
            cache_len=cache_len,
        )
    )


def update_step_pass_rate_window(
    window: Sequence[float],
    step_pass_rate: float,
    *,
    window_steps: int,
) -> list[float]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive.")

    updated_window = [*window, float(step_pass_rate)]
    if len(updated_window) > window_steps:
        updated_window = updated_window[-window_steps:]
    return updated_window


def should_discard_prompt(
    step_pass_rates: Sequence[float],
    *,
    threshold: float,
    window_steps: int,
) -> bool:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive.")

    return len(step_pass_rates) == window_steps and all(rate >= threshold for rate in step_pass_rates)


def normalize_rewards_for_training(
    args,
    raw_rewards: Sequence[float],
    group_indices: Sequence[int],
    active_mask: Sequence[bool] | None = None,
) -> list[float]:
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        centered_rewards = get_prompt_group_mean_centered_rewards(raw_rewards, group_indices, active_mask=active_mask)
        resolved_active_mask = _resolve_active_mask(active_mask, length=len(raw_rewards))

        if args.batch_level_normalization:
            return get_batch_normalized_prompt_rewards(raw_rewards, group_indices, active_mask=resolved_active_mask)

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            normalized_rewards = centered_rewards.clone()
            grouped_indices: dict[int, list[int]] = {}
            for sample_idx, (group_index, is_active) in enumerate(zip(group_indices, resolved_active_mask, strict=True)):
                if not is_active:
                    continue
                grouped_indices.setdefault(group_index, []).append(sample_idx)
            for group_sample_indices in grouped_indices.values():
                group_rewards = centered_rewards[group_sample_indices]
                group_std = group_rewards.std() if len(group_sample_indices) > 1 else torch.zeros((), dtype=group_rewards.dtype)
                normalized_rewards[group_sample_indices] = group_rewards / (group_std + 1e-6)
            return normalized_rewards.tolist()

        return centered_rewards.tolist()

    resolved_rewards = list(raw_rewards)
    if active_mask is None:
        return resolved_rewards

    return [reward if is_active else 0.0 for reward, is_active in zip(resolved_rewards, active_mask, strict=True)]


def get_train_metric_normalizers(
    keys: Sequence[str],
    *,
    num_samples: int,
    num_tokens: int,
    num_prompt_groups: int,
    prompt_level_loss_aggregation: bool,
    calculate_per_token_loss: bool,
) -> list[int]:
    """Return per-metric denominators for Megatron train logging.

    Most train metrics are reduced as a sum of per-sample means and should
    therefore be normalized by the number of samples. When prompt-level loss
    aggregation is enabled, only the prompt-aggregated objectives (`loss` and
    `pg_loss`) should switch to prompt-group normalization; auxiliary metrics
    like `ratio` and `ppo_kl` remain sample-based.
    """
    if calculate_per_token_loss and not prompt_level_loss_aggregation:
        return [num_tokens] * len(keys)

    if not prompt_level_loss_aggregation:
        return [num_samples] * len(keys)

    prompt_aggregated_keys = {"loss", "pg_loss"}
    return [num_prompt_groups if key in prompt_aggregated_keys else num_samples for key in keys]


def validate_scalerl_args(args) -> None:
    length_penalty_type = getattr(args, "length_penalty_type", "none")
    length_penalty_cache_len = getattr(args, "length_penalty_cache_len", None)
    apf_threshold = getattr(args, "adaptive_prompt_filter_threshold", None)
    apf_window_steps = getattr(args, "adaptive_prompt_filter_window_steps", None)
    apf_drop_prob = getattr(args, "adaptive_prompt_filter_drop_prob", 1.0)

    if length_penalty_type == "dapo_style":
        if length_penalty_cache_len is None or length_penalty_cache_len <= 0:
            raise ValueError("--length-penalty-cache-len must be positive when --length-penalty-type dapo_style.")
    elif length_penalty_cache_len is not None and length_penalty_cache_len <= 0:
        raise ValueError("--length-penalty-cache-len must be positive when provided.")

    if (apf_threshold is None) != (apf_window_steps is None):
        raise ValueError(
            "--adaptive-prompt-filter-threshold and --adaptive-prompt-filter-window-steps must be set together."
        )
    if apf_threshold is not None and not (0.0 <= apf_threshold <= 1.0):
        raise ValueError("--adaptive-prompt-filter-threshold must be between 0 and 1.")
    if apf_window_steps is not None and apf_window_steps <= 0:
        raise ValueError("--adaptive-prompt-filter-window-steps must be positive.")
    if not (0.0 <= apf_drop_prob <= 1.0):
        raise ValueError("--adaptive-prompt-filter-drop-prob must be between 0 and 1.")

    if args.batch_level_normalization:
        if args.advantage_estimator not in ["grpo", "gspo"]:
            raise ValueError("--batch-level-normalization is supported only for --advantage-estimator grpo or gspo.")
        if args.normalize_advantages:
            raise ValueError("--batch-level-normalization cannot be combined with --normalize-advantages.")
        if not args.rewards_normalization:
            raise ValueError("--batch-level-normalization cannot be combined with --disable-rewards-normalization.")

    if args.prompt_level_loss_aggregation:
        if args.train_backend != "megatron":
            raise ValueError("--prompt-level-loss-aggregation is currently supported only on the Megatron backend.")
        if args.custom_pg_loss_reducer_function_path is not None:
            raise ValueError(
                "--prompt-level-loss-aggregation cannot be combined with --custom-pg-loss-reducer-function-path."
            )
        if args.loss_type not in ["policy_loss", "cispo_loss", "dispo_loss"]:
            raise ValueError(
                "--prompt-level-loss-aggregation is supported only for policy-style losses "
                "(policy_loss, cispo_loss, dispo_loss)."
            )

    if args.batch_level_normalization or args.prompt_level_loss_aggregation:
        if args.global_batch_size % args.n_samples_per_prompt != 0:
            raise ValueError(
                "When --batch-level-normalization or --prompt-level-loss-aggregation is enabled, "
                "--global-batch-size must be divisible by --n-samples-per-prompt."
            )
