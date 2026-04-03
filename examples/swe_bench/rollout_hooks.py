from argparse import Namespace
from collections.abc import Sequence

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.scalerl import flatten_samples
from slime.utils.types import Sample

__all__ = [
    "filter_swe_completed_and_submitted",
    "mark_swe_non_submitted_samples_inactive",
    "should_attempt_partial_resume",
    "is_abort_resumable_for_partial_rollout",
    "mask_previous_response_tokens",
]


def _get_exit_status(sample: Sample) -> str:
    info = sample.metadata.get("info") if isinstance(sample.metadata, dict) else None
    return str(info.get("exit_status", "")).strip().lower() if isinstance(info, dict) else ""


def _should_keep_sample(sample: Sample) -> tuple[bool, str | None]:
    if sample.status == Sample.Status.ABORTED:
        return False, "swe_aborted"
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        return False, f"swe_status_{sample.status.value}"
    if "submitted" not in _get_exit_status(sample):
        return False, "swe_not_submitted"
    return True, None


def filter_swe_completed_and_submitted(
    args: Namespace,
    samples: list[Sample],
    **kwargs,
) -> DynamicFilterOutput:
    del args, kwargs

    for sample in samples:
        keep_sample, reason = _should_keep_sample(sample)
        if not keep_sample:
            return DynamicFilterOutput(keep=False, reason=reason)
    return DynamicFilterOutput(keep=True)


def mark_swe_non_submitted_samples_inactive(
    args: Namespace,
    samples: Sequence[Sample] | Sequence[Sequence[Sample]],
    **kwargs,
) -> None:
    del args, kwargs

    for sample in flatten_samples(samples):
        keep_sample, reason = _should_keep_sample(sample)
        sample.remove_sample = not keep_sample

        metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
        if reason is None:
            metadata.pop("swe_rollout_sample_filter_reason", None)
        else:
            metadata["swe_rollout_sample_filter_reason"] = reason
        sample.metadata = metadata


def should_attempt_partial_resume(args: Namespace, sample: Sample) -> bool:
    return (
        bool(getattr(args, "partial_rollout", False))
        and bool(sample.session_id)
        and sample.status == Sample.Status.ABORTED
    )


def is_abort_resumable_for_partial_rollout(
    sample: Sample,
    *,
    exit_status: str,
    total_turn_count: int,
    max_turns: int,
) -> bool:
    return (
        sample.status == Sample.Status.ABORTED
        and int(getattr(sample, "response_length", 0) or 0) > 0
        and "submitted" not in exit_status
        and int(total_turn_count) < int(max_turns)
    )


def mask_previous_response_tokens(loss_mask: list[int], prior_response_length: int) -> list[int]:
    old_token_count = min(max(int(prior_response_length), 0), len(loss_mask))
    if old_token_count > 0:
        loss_mask[:old_token_count] = [0] * old_token_count
    return loss_mask
