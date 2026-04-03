from types import SimpleNamespace

from examples.swe_bench.rollout_hooks import (
    filter_swe_completed_and_submitted,
    is_abort_resumable_for_partial_rollout,
    mark_swe_non_submitted_samples_inactive,
    mask_previous_response_tokens,
    should_attempt_partial_resume,
)
from slime.utils.types import Sample


def _make_args(**overrides):
    base = dict(
        n_samples_per_prompt=1,
        partial_rollout=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_sample(*, status: Sample.Status, exit_status: str, response_length: int = 8) -> Sample:
    sample = Sample(index=0, group_index=0)
    sample.status = status
    sample.response = "x" * response_length
    sample.response_length = response_length
    sample.session_id = "session-1"
    sample.metadata["info"] = {"exit_status": exit_status}
    return sample


def test_filter_swe_completed_and_submitted_keeps_submitted():
    args = _make_args()
    sample = _make_sample(status=Sample.Status.COMPLETED, exit_status="submitted")
    output = filter_swe_completed_and_submitted(args, [sample])
    assert output.keep is True
    assert output.reason is None


def test_filter_swe_completed_and_submitted_drops_not_submitted():
    args = _make_args()
    sample = _make_sample(status=Sample.Status.COMPLETED, exit_status="none")
    output = filter_swe_completed_and_submitted(args, [sample])
    assert output.keep is False
    assert output.reason == "swe_not_submitted"


def test_filter_swe_completed_and_submitted_drops_aborted():
    args = _make_args()
    sample = _make_sample(status=Sample.Status.ABORTED, exit_status="submitted")
    output = filter_swe_completed_and_submitted(args, [sample])
    assert output.keep is False
    assert output.reason == "swe_aborted"


def test_filter_swe_completed_and_submitted_supports_multi_sample_group():
    args = _make_args(n_samples_per_prompt=2)
    first = _make_sample(status=Sample.Status.COMPLETED, exit_status="submitted")
    second = _make_sample(status=Sample.Status.COMPLETED, exit_status="none")
    output = filter_swe_completed_and_submitted(args, [first, second])
    assert output.keep is False
    assert output.reason == "swe_not_submitted"


def test_mark_swe_non_submitted_samples_inactive_marks_only_non_submitted():
    args = _make_args(n_samples_per_prompt=2)
    first = _make_sample(status=Sample.Status.COMPLETED, exit_status="submitted")
    second = _make_sample(status=Sample.Status.COMPLETED, exit_status="none")

    mark_swe_non_submitted_samples_inactive(args, [[first, second]])

    assert first.remove_sample is False
    assert second.remove_sample is True
    assert "swe_rollout_sample_filter_reason" not in first.metadata
    assert second.metadata["swe_rollout_sample_filter_reason"] == "swe_not_submitted"


def test_should_attempt_partial_resume_requires_partial_flag_and_aborted_status():
    sample = _make_sample(status=Sample.Status.ABORTED, exit_status="none")
    assert should_attempt_partial_resume(_make_args(partial_rollout=True), sample) is True

    sample.status = Sample.Status.COMPLETED
    assert should_attempt_partial_resume(_make_args(partial_rollout=True), sample) is False
    assert should_attempt_partial_resume(_make_args(partial_rollout=False), sample) is False


def test_is_abort_resumable_for_partial_rollout():
    sample = _make_sample(status=Sample.Status.ABORTED, exit_status="none", response_length=12)
    assert (
        is_abort_resumable_for_partial_rollout(
            sample,
            exit_status="none",
            total_turn_count=2,
            max_turns=5,
        )
        is True
    )

    sample.status = Sample.Status.COMPLETED
    assert (
        is_abort_resumable_for_partial_rollout(
            sample,
            exit_status="none",
            total_turn_count=2,
            max_turns=5,
        )
        is False
    )


def test_mask_previous_response_tokens_masks_only_prefix():
    loss_mask = [1, 1, 1, 1, 1]
    masked = mask_previous_response_tokens(loss_mask, prior_response_length=2)
    assert masked == [0, 0, 1, 1, 1]
