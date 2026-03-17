import math
from collections.abc import Sequence

import torch


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
) -> torch.Tensor:
    rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    centered_rewards = torch.empty_like(rewards)

    rewards_by_group: dict[int, list[tuple[int, float]]] = {}
    for idx, (reward, group_index) in enumerate(zip(raw_rewards, group_indices, strict=True)):
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
) -> list[float]:
    centered_rewards = get_prompt_group_mean_centered_rewards(raw_rewards, group_indices)
    if centered_rewards.numel() > 1:
        batch_std = centered_rewards.std()
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

    return weights, len(prompt_token_counts)


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


def validate_scalerl_args(args) -> None:
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
