from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slime.utils.types import Sample


def _numeric_stats(values: Sequence[int], *, prefix: str) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{prefix}/min": min(values),
        f"{prefix}/max": max(values),
        f"{prefix}/mean": sum(values) / len(values),
    }


def parse_policy_version(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def get_sample_policy_versions(sample: Sample) -> list[int]:
    return [version for version in (parse_policy_version(item) for item in sample.weight_versions) if version is not None]


def get_sample_policy_version_summary(sample: Sample) -> dict[str, int] | None:
    versions = get_sample_policy_versions(sample)
    if not versions:
        return None

    min_version = min(versions)
    max_version = max(versions)
    return {
        "policy_version_min": min_version,
        "policy_version_max": max_version,
        "policy_version_last": versions[-1],
        "policy_version_span": max_version - min_version,
        "policy_version_count": len(versions),
        "policy_version_mixed": int(len(set(versions)) > 1),
    }


def build_policy_version_train_data(samples: Sequence[Sample]) -> dict[str, list[int]]:
    summaries = [get_sample_policy_version_summary(sample) for sample in samples]
    if not summaries or any(summary is None for summary in summaries):
        return {}

    field_names = [
        "policy_version_min",
        "policy_version_max",
        "policy_version_last",
        "policy_version_span",
        "policy_version_count",
        "policy_version_mixed",
    ]
    return {
        field_name: [summary[field_name] for summary in summaries if summary is not None]
        for field_name in field_names
    }


def compute_policy_version_metrics_from_samples(samples: Sequence[Sample]) -> dict[str, float]:
    summaries = [get_sample_policy_version_summary(sample) for sample in samples]
    summaries = [summary for summary in summaries if summary is not None]
    if not summaries:
        return {}

    return {
        "policy_version/mixed_frac": sum(summary["policy_version_mixed"] for summary in summaries) / len(summaries)
    }


def compute_policy_lag_metrics_from_rollout_data(
    rollout_data: dict[str, Any],
    current_policy_version: int | None,
) -> dict[str, float]:
    if current_policy_version is None:
        return {}

    metrics: dict[str, float] = {}

    last_versions = rollout_data.get("policy_version_last")
    if last_versions:
        parsed_last_versions = [version for version in (parse_policy_version(item) for item in last_versions) if version is not None]
        if parsed_last_versions:
            last_lags = [max(current_policy_version - version, 0) for version in parsed_last_versions]
            metrics["policy_lag/mean"] = sum(last_lags) / len(last_lags)

    oldest_versions = rollout_data.get("policy_version_min")
    if oldest_versions:
        parsed_oldest_versions = [
            version for version in (parse_policy_version(item) for item in oldest_versions) if version is not None
        ]
        if parsed_oldest_versions:
            oldest_lags = [max(current_policy_version - version, 0) for version in parsed_oldest_versions]
            metrics["policy_lag/max"] = max(oldest_lags)

    if rollout_data.get("policy_version_mixed"):
        mixed_flags = [int(flag) for flag in rollout_data["policy_version_mixed"]]
        metrics["policy_version/mixed_frac"] = sum(mixed_flags) / len(mixed_flags)

    return metrics
