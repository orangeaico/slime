from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from slime.utils.scalerl_utils import (
    apply_length_penalty,
    get_prompt_group_indices,
    normalize_rewards_for_training,
)
from slime.utils.types import Sample

PROMPT_ID_METADATA_KEY = "scalerl_prompt_id"
APF_METADATA_KEY = "scalerl_adaptive_prompt_filter"
APF_STEP_PASS_RATES_KEY = "prompt_step_pass_rates"
SCALERL_ROLLOUT_METRICS_METADATA_KEY = "scalerl_rollout_metrics"

__all__ = [
    "PROMPT_ID_METADATA_KEY",
    "APF_METADATA_KEY",
    "APF_STEP_PASS_RATES_KEY",
    "SCALERL_ROLLOUT_METRICS_METADATA_KEY",
    "flatten_samples",
    "get_scalar_reward",
    "get_scalerl_prompt_id",
    "get_shaped_rewards",
    "compute_scalerl_metrics_from_samples",
    "attach_scalerl_rollout_metrics",
    "post_process_rewards_with_dapo_style",
    "update_step_window_adaptive_prompt_filter",
]


def flatten_samples(samples: Sequence[Sample] | Sequence[Sequence[Sample]]) -> list[Sample]:
    if not samples:
        return []
    first = samples[0]
    if isinstance(first, Sample):
        return list(samples)  # type: ignore[arg-type]
    return [sample for group in samples for sample in group]  # type: ignore[list-item]


def get_scalar_reward(args, sample: Sample) -> float:
    return float(sample.get_reward_value(args))


def get_scalerl_prompt_id(sample: Sample) -> int:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    prompt_id = metadata.get(PROMPT_ID_METADATA_KEY)
    if prompt_id is None:
        raise ValueError(f"Missing {PROMPT_ID_METADATA_KEY} in sample metadata.")
    return int(prompt_id)


def get_shaped_rewards(args, samples: Sequence[Sample]) -> tuple[list[float], list[float]]:
    raw_rewards = [get_scalar_reward(args, sample) for sample in samples]
    shaped_rewards = [
        apply_length_penalty(
            raw_reward,
            sample.response_length,
            length_penalty_type=getattr(args, "length_penalty_type", "none"),
            max_response_len=args.rollout_max_response_len,
            cache_len=getattr(args, "length_penalty_cache_len", None),
        )
        for raw_reward, sample in zip(raw_rewards, samples, strict=True)
    ]
    return raw_rewards, shaped_rewards


def attach_scalerl_rollout_metrics(samples: Sequence[Sample], metrics: dict[str, float]) -> None:
    for sample in samples:
        metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
        metadata[SCALERL_ROLLOUT_METRICS_METADATA_KEY] = dict(metrics)
        sample.metadata = metadata


def _resolve_scalerl_data_source(data_source_getter: Callable[..., Any]):
    data_source = getattr(data_source_getter, "__self__", None)
    if data_source is None:
        raise TypeError("rollout_all_samples_process_path expects the bound data source get_samples method.")
    if not hasattr(data_source, "record_step_pass_rates"):
        raise TypeError("Data source must define record_step_pass_rates for adaptive prompt filtering.")
    return data_source


def compute_scalerl_metrics_from_samples(args, samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {}

    metrics: dict[str, float] = {}

    if getattr(args, "length_penalty_type", "none") == "dapo_style":
        raw_rewards, shaped_rewards = get_shaped_rewards(args, samples)
        penalty_deltas = [shaped_reward - raw_reward for raw_reward, shaped_reward in zip(raw_rewards, shaped_rewards, strict=True)]

        metrics["scalerl/penalty_delta_mean"] = sum(penalty_deltas) / len(penalty_deltas)
        metrics["scalerl/penalized_sample_frac"] = sum(delta != 0.0 for delta in penalty_deltas) / len(penalty_deltas)

    rollout_metrics = None
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if SCALERL_ROLLOUT_METRICS_METADATA_KEY in metadata:
            rollout_metrics = metadata[SCALERL_ROLLOUT_METRICS_METADATA_KEY]
            break

    if isinstance(rollout_metrics, dict):
        metrics |= {f"scalerl/apf/{key}": value for key, value in rollout_metrics.items()}

    return metrics


def post_process_rewards_with_dapo_style(args, samples: list[Sample] | list[list[Sample]], **kwargs):
    flat_samples = flatten_samples(samples)
    raw_rewards, shaped_rewards = get_shaped_rewards(args, flat_samples)

    for sample, raw_reward in zip(flat_samples, raw_rewards, strict=True):
        metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
        metadata["raw_reward"] = raw_reward
        sample.metadata = metadata

    group_indices = get_prompt_group_indices([sample.group_index for sample in flat_samples], args.n_samples_per_prompt)
    normalized_rewards = normalize_rewards_for_training(args, shaped_rewards, group_indices)
    return raw_rewards, normalized_rewards


def update_step_window_adaptive_prompt_filter(args, all_samples, data_source, **kwargs):
    if getattr(args, "adaptive_prompt_filter_threshold", None) is None:
        return

    data_source_instance = _resolve_scalerl_data_source(data_source)

    prompt_correct_and_total: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for sample in flatten_samples(all_samples):
        prompt_id = get_scalerl_prompt_id(sample)
        prompt_correct_and_total[prompt_id][0] += int(get_scalar_reward(args, sample) == 1.0)
        prompt_correct_and_total[prompt_id][1] += 1

    step_pass_rates = {
        prompt_id: correct / total
        for prompt_id, (correct, total) in prompt_correct_and_total.items()
        if total > 0
    }
    rollout_metrics = data_source_instance.record_step_pass_rates(step_pass_rates)
    attach_scalerl_rollout_metrics(flatten_samples(all_samples), rollout_metrics)
