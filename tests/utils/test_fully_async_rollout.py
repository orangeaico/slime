"""Unit tests for the new filter/post-process logic in generate_rollout_async.

These tests mock the worker queue and heavy SGLang dependencies so they
run without GPUs or a running server.
"""
import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.scalerl import PROMPT_ID_METADATA_KEY
from slime.utils.types import Sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    base = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_group(index, reward=1.0):
    """Return a list[Sample] group (n_samples=1 for simplicity)."""
    s = Sample(index=index, group_index=index)
    s.reward = reward
    s.prompt = "p"
    s.response = "r"
    s.label = "l"
    return [s]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures: patch the global worker so we control what comes out of the queue
# ---------------------------------------------------------------------------

class FakeWorker:
    """A minimal fake that feeds groups from a pre-loaded list."""

    def __init__(self, groups_to_return):
        self._q = queue.Queue()
        self.data_buffer_lock = threading.Lock()
        for gid, g in enumerate(groups_to_return):
            self._q.put((gid, g))

    def get_completed_groups(self):
        items = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def get_queue_size(self):
        return self._q.qsize()

    def get_pipeline_rl_metrics(self):
        return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_rollout_fn_train_output_with_no_filters():
    """Without any filters set, returns RolloutFnTrainOutput wrapping all groups."""
    args = _make_args()
    groups = [_make_group(0), _make_group(1)]
    data_buffer = MagicMock()

    fake_worker = FakeWorker(groups)

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker):
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args, rollout_id=0, data_buffer=data_buffer))

    assert isinstance(result, RolloutFnTrainOutput)
    assert len(result.samples) == 2
    assert result.metrics == {}


def test_dynamic_filter_rejection_drops_group_and_excludes_from_data():
    """Groups rejected by dynamic filter are dropped and not re-queued."""
    # reject group[0] (index=0), accept group[1] (index=1)
    # We need 2 accepted groups total, so add a third group (index=2) for the requeue path.
    # But rollout_batch_size=1 so one accepted group is enough.
    args = _make_args(rollout_batch_size=1)
    rejected_group = _make_group(0)
    accepted_group = _make_group(1)
    data_buffer = MagicMock()

    fake_worker = FakeWorker([rejected_group, accepted_group])

    def fake_filter(args, group):
        # reject the group with index=0
        if group[0].index == 0:
            return DynamicFilterOutput(keep=False, reason="zero_std")
        return DynamicFilterOutput(keep=True)

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_filter):
        args_with_filter = _make_args(
            rollout_batch_size=1,
            dynamic_sampling_filter_path="some.filter.path",
        )
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args_with_filter, rollout_id=0, data_buffer=data_buffer))

    assert isinstance(result, RolloutFnTrainOutput)
    assert len(result.samples) == 1
    assert result.samples[0][0].index == 1  # accepted group

    data_buffer.add_samples.assert_not_called()

    # filter drop metric must be recorded
    assert result.metrics == {"rollout/dynamic_filter/drop_zero_std": 1}


def test_rollout_sample_filter_path_called_on_accepted_data():
    """rollout_sample_filter_path is called with the accepted data after collection."""
    args = _make_args()
    groups = [_make_group(0), _make_group(1)]
    data_buffer = MagicMock()
    fake_worker = FakeWorker(groups)

    sample_filter_calls = []

    def fake_sample_filter(args, data):
        sample_filter_calls.append(list(data))

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_sample_filter):
        args_with_filter = _make_args(rollout_sample_filter_path="some.sample.filter")
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args_with_filter, rollout_id=0, data_buffer=data_buffer))

    assert len(sample_filter_calls) == 1
    assert len(sample_filter_calls[0]) == 2


def test_rollout_all_samples_process_path_called_with_bound_method():
    """rollout_all_samples_process_path receives data_buffer.get_samples (bound method)."""
    args = _make_args()
    groups = [_make_group(0), _make_group(1)]
    data_buffer = MagicMock()
    fake_worker = FakeWorker(groups)

    all_samples_calls = []

    def fake_process(args, all_samples, data_source):
        all_samples_calls.append((all_samples, data_source))

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_process):
        args_with_process = _make_args(rollout_all_samples_process_path="some.process.path")
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args_with_process, rollout_id=0, data_buffer=data_buffer))

    assert len(all_samples_calls) == 1
    all_samples_arg, data_source_arg = all_samples_calls[0]
    assert len(all_samples_arg) == 2
    # Must receive data_buffer.get_samples (bound method), not data_buffer itself
    assert data_source_arg is data_buffer.get_samples


def test_aborted_groups_not_included_in_all_data_or_data():
    """Aborted groups are re-queued and excluded from both data and all_data."""
    aborted_group = _make_group(0)
    aborted_group[0].status = Sample.Status.ABORTED
    accepted_group = _make_group(1)

    data_buffer = MagicMock()
    fake_worker = FakeWorker([aborted_group, accepted_group])

    all_samples_calls = []

    def fake_process(args, all_samples, data_source):
        all_samples_calls.append(all_samples)

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_process):
        args = _make_args(rollout_batch_size=1, rollout_all_samples_process_path="some.process.path")
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args, rollout_id=0, data_buffer=data_buffer))

    assert len(result.samples) == 1
    assert result.samples[0][0].index == 1

    # aborted group must NOT appear in all_samples passed to the process func
    assert len(all_samples_calls[0]) == 1
    assert all_samples_calls[0][0][0].index == 1


def test_metrics_accumulate_across_multiple_filter_rejections():
    """MetricGatherer counts each rejection reason correctly."""
    args = _make_args(rollout_batch_size=1)
    groups = [_make_group(0), _make_group(1), _make_group(2)]
    data_buffer = MagicMock()
    fake_worker = FakeWorker(groups)

    def fake_filter(args, group):
        idx = group[0].index
        if idx == 0:
            return DynamicFilterOutput(keep=False, reason="zero_std_1.0")
        if idx == 1:
            return DynamicFilterOutput(keep=False, reason="zero_std_1.0")
        return DynamicFilterOutput(keep=True)

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_filter):
        args_f = _make_args(rollout_batch_size=1, dynamic_sampling_filter_path="some.filter")
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args_f, rollout_id=0, data_buffer=data_buffer))

    assert result.metrics == {"rollout/dynamic_filter/drop_zero_std_1.0": 2}
    assert len(result.samples) == 1


def test_completed_group_rejected_by_dynamic_filter_is_not_recycled():
    """Completed groups rejected by the filter must not go back into the buffer."""
    args = _make_args(rollout_batch_size=1)
    rejected_group = _make_group(0)
    rejected_group[0].status = Sample.Status.COMPLETED
    accepted_group = _make_group(1)
    accepted_group[0].status = Sample.Status.COMPLETED
    data_buffer = MagicMock()

    fake_worker = FakeWorker([rejected_group, accepted_group])

    def fake_filter(args, group):
        if group[0].index == 0:
            return DynamicFilterOutput(keep=False, reason="zero_std")
        return DynamicFilterOutput(keep=True)

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker), \
         patch("examples.fully_async.fully_async_rollout.load_function", return_value=fake_filter):
        args_f = _make_args(rollout_batch_size=1, dynamic_sampling_filter_path="some.filter")
        from examples.fully_async.fully_async_rollout import generate_rollout_async
        result = _run(generate_rollout_async(args_f, rollout_id=0, data_buffer=data_buffer))

    assert len(result.samples) == 1
    assert result.samples[0][0].index == 1
    data_buffer.add_samples.assert_not_called()


def test_pipeline_rl_metrics_track_step_lead_instead_of_queue_depth():
    from examples.fully_async.fully_async_rollout import _compute_pipeline_rl_metrics

    metrics = _compute_pipeline_rl_metrics(
        trainer_step=5,
        generation_step=3,
        oldest_outstanding_generation_step=2,
    )

    assert metrics == {
        "pipeline_rl/oldest_step_gap": 3,
        "pipeline_rl/step_lead": 2,
    }


def test_pipeline_rl_metrics_are_zero_without_outstanding_generation():
    from examples.fully_async.fully_async_rollout import _compute_pipeline_rl_metrics

    metrics = _compute_pipeline_rl_metrics(
        trainer_step=5,
        generation_step=None,
        oldest_outstanding_generation_step=None,
    )

    assert metrics == {
        "pipeline_rl/oldest_step_gap": 0,
        "pipeline_rl/step_lead": 0,
    }


def test_pipeline_rl_outstanding_group_limit_scales_with_rollout_batch_size():
    from examples.fully_async.fully_async_rollout import _resolve_pipeline_rl_outstanding_group_limit

    args = _make_args(rollout_batch_size=4, pipeline_rl_k=2)

    assert _resolve_pipeline_rl_outstanding_group_limit(args) == 8


def test_generate_rollout_async_accepts_completed_groups_with_generation_step_metadata():
    args = _make_args(rollout_batch_size=1)
    accepted_group = _make_group(1)
    data_buffer = MagicMock()
    fake_worker = FakeWorker([])
    fake_worker.get_completed_groups = MagicMock(return_value=[(0, 7, accepted_group)])

    with patch("examples.fully_async.fully_async_rollout.get_global_worker", return_value=fake_worker):
        from examples.fully_async.fully_async_rollout import generate_rollout_async

        result = _run(generate_rollout_async(args, rollout_id=0, data_buffer=data_buffer))

    assert len(result.samples) == 1
    assert result.samples[0][0].index == 1
