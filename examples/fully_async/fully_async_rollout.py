import asyncio
import atexit
from collections import Counter
import logging
import queue
import threading
import time

import ray

# Import core functions from sglang_rollout directly to avoid code duplication
from slime.ray.pipeline_rl_controller import get_pipeline_rl_oldest_step_gap
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group, generate_rollout as sglang_generate_rollout
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def _compute_pipeline_rl_metrics(
    *,
    trainer_step: int,
    oldest_outstanding_generation_step: int | None,
    waiting_groups: int,
) -> dict[str, float]:
    oldest_step_gap = get_pipeline_rl_oldest_step_gap(
        trainer_step=trainer_step,
        oldest_outstanding_generation_step=oldest_outstanding_generation_step,
    )
    return {
        "pipeline_rl/oldest_group_lag": oldest_step_gap,
        "pipeline_rl/waiting_groups": waiting_groups,
    }


def _reset_sample_for_regeneration(sample: Sample) -> Sample:
    sample.tokens = []
    sample.response = ""
    sample.response_length = 0
    sample.reward = None
    sample.loss_mask = None
    sample.weight_versions = []
    sample.rollout_log_probs = None
    sample.rollout_routed_experts = None
    sample.teacher_log_probs = None
    sample.remove_sample = False
    sample.status = Sample.Status.PENDING
    sample.train_metadata = None
    sample.session_id = None
    sample.non_generation_time = 0.0
    sample.multimodal_train_inputs = None
    sample.spec_info = Sample.SpecInfo()
    sample.prefix_cache_info = Sample.PrefixCacheInfo()
    return sample


def _reset_group_for_regeneration(group: list[Sample]) -> list[Sample]:
    return [_reset_sample_for_regeneration(sample) for sample in group]


def _collect_stale_metrics(
    *,
    stale_drop_groups: int,
) -> dict[str, float]:
    return {
        "pipeline_rl/stale_drop_groups": stale_drop_groups,
    }


def get_global_worker(args, data_buffer):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _reset_generate_state()
            logger.info("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def _reset_generate_state():
    GenerateState._instances.pop(GenerateState, None)


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None
        _reset_generate_state()


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        # Fully async mode intentionally allows the worker to run ahead.
        # Keep the result queue unbounded so task callbacks never block the
        # worker event loop while pushing completed groups.
        self.output_queue = queue.Queue()
        self.worker_thread = None
        self.data_buffer_lock = threading.Lock()
        self.pipeline_state_lock = threading.Lock()
        self.state = GenerateState(args)
        self.pipeline_rl_k = getattr(args, "pipeline_rl_k", None)
        self.pipeline_rl_controller = getattr(args, "pipeline_rl_controller", None)
        self.latest_known_trainer_step = 0
        self.latest_oldest_outstanding_step: int | None = None
        self.latest_newest_outstanding_step: int | None = None
        self._active_step_counts: Counter[int] = Counter()
        self._queued_step_counts: Counter[int] = Counter()
        self._last_reported_pipeline_snapshot: tuple[int | None, int | None] | None = None

    def _sync_trainer_step(self) -> None:
        if self.pipeline_rl_controller is None:
            return
        try:
            self.latest_known_trainer_step = int(ray.get(self.pipeline_rl_controller.get_trainer_step.remote()))
        except Exception as e:
            logger.warning(f"Failed to refresh PipelineRL trainer step: {e}")

    def _recompute_pipeline_snapshot_locked(self) -> tuple[int | None, int | None]:
        outstanding_steps = set(self._active_step_counts) | set(self._queued_step_counts)
        self.latest_oldest_outstanding_step = min(outstanding_steps) if outstanding_steps else None
        self.latest_newest_outstanding_step = max(outstanding_steps) if outstanding_steps else None
        return (
            self.latest_oldest_outstanding_step,
            self.latest_newest_outstanding_step,
        )

    def _get_waiting_group_count_locked(self) -> int:
        return sum(self._queued_step_counts.values())

    def _maybe_report_pipeline_snapshot(self) -> None:
        if self.pipeline_rl_controller is None:
            return
        with self.pipeline_state_lock:
            snapshot = self._recompute_pipeline_snapshot_locked()
        if snapshot == self._last_reported_pipeline_snapshot:
            return
        self._last_reported_pipeline_snapshot = snapshot
        try:
            self.pipeline_rl_controller.report_outstanding.remote(*snapshot)
        except Exception as e:
            logger.warning(f"Failed to report PipelineRL snapshot: {e}")

    def _record_task_started(self, generation_step: int) -> None:
        with self.pipeline_state_lock:
            self._active_step_counts[generation_step] += 1
            self._recompute_pipeline_snapshot_locked()
        self._maybe_report_pipeline_snapshot()

    def _record_task_finished(self, generation_step: int, *, queued_for_training: bool) -> None:
        with self.pipeline_state_lock:
            self._active_step_counts[generation_step] -= 1
            if self._active_step_counts[generation_step] <= 0:
                self._active_step_counts.pop(generation_step, None)
            if queued_for_training:
                self._queued_step_counts[generation_step] += 1
            self._recompute_pipeline_snapshot_locked()
        self._maybe_report_pipeline_snapshot()

    def _record_groups_drained(self, generation_steps: list[int]) -> None:
        if not generation_steps:
            return
        with self.pipeline_state_lock:
            for generation_step in generation_steps:
                self._queued_step_counts[generation_step] -= 1
                if self._queued_step_counts[generation_step] <= 0:
                    self._queued_step_counts.pop(generation_step, None)
            self._recompute_pipeline_snapshot_locked()
        self._maybe_report_pipeline_snapshot()

    def get_pipeline_rl_metrics(self) -> dict[str, float]:
        if self.pipeline_rl_k is None:
            return {}
        self._sync_trainer_step()
        with self.pipeline_state_lock:
            oldest_outstanding_generation_step = self.latest_oldest_outstanding_step
            waiting_groups = self._get_waiting_group_count_locked()
        return _compute_pipeline_rl_metrics(
            trainer_step=self.latest_known_trainer_step,
            oldest_outstanding_generation_step=oldest_outstanding_generation_step,
            waiting_groups=waiting_groups,
        )

    def get_latest_trainer_step(self) -> int:
        self._sync_trainer_step()
        return int(self.latest_known_trainer_step)

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        logger.info("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            logger.warning(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks

                # If active task count hasn't reached limit, try to get new data and start tasks
                self._sync_trainer_step()

                while len(active_tasks) < max_concurrent_tasks and self.running:
                    with self.data_buffer_lock:
                        samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1
                        generation_step = self.latest_known_trainer_step

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback
                        def make_callback(gid, generation_step):
                            def task_done_callback(done_task):
                                try:
                                    result = done_task.result()
                                except Exception as e:
                                    logger.warning(f"Task {gid} failed with exception: {e}")
                                    self._record_task_finished(generation_step, queued_for_training=False)
                                    return
                                self._record_task_finished(generation_step, queued_for_training=True)
                                self.output_queue.put_nowait((gid, generation_step, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id, generation_step))
                        active_tasks.add(task)
                        self._record_task_started(generation_step)
                        break

                self._maybe_report_pipeline_snapshot()

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.exception(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        self._record_groups_drained(
            [generation_step for generation_step, count in self._queued_step_counts.items() for _ in range(count)]
        )
        if self.pipeline_rl_controller is not None:
            try:
                ray.get(self.pipeline_rl_controller.clear_outstanding.remote())
            except Exception as e:
                logger.warning(f"Failed to clear PipelineRL coordinator state: {e}")

        logger.info("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            logger.info("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("Stopped async worker thread")

    def get_completed_groups(self, max_groups: int | None = None) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        drained_steps = []
        while True:
            if max_groups is not None and len(completed) >= max_groups:
                break
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
                drained_steps.append(result[1])
            except queue.Empty:
                break
        self._record_groups_drained(drained_steps)
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker
    """
    assert args.rollout_global_dataset

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    metric_gatherer = MetricGatherer()
    all_data = []  # tracks all non-aborted groups including dynamically filtered ones

    data = []
    completed_groups = {}
    do_print = True
    pipeline_rl_k = getattr(args, "pipeline_rl_k", None)
    stale_drop_groups = 0
    trainer_step_snapshot = worker.get_latest_trainer_step() if pipeline_rl_k is not None else None
    if pipeline_rl_k is not None:
        logger.info(f"PipelineRL trainer_step snapshot for rollout {rollout_id}: {trainer_step_snapshot}")

    logger.info(f"Starting async rollout collection for {target_data_size} groups")
    logger.info(f"Global worker queue size: {worker.get_queue_size()}")

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        remaining_groups_needed = max(target_data_size - len(data), 1)
        completed = worker.get_completed_groups(max_groups=remaining_groups_needed)

        made_progress = False
        for completed_group in completed:
            if len(completed_group) == 3:
                group_id, generation_step, group = completed_group
            else:
                generation_step = None
                group_id, group = completed_group
            completed_groups[group_id] = (generation_step, group)
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            generation_step, group = completed_groups.pop(group_id)
            if pipeline_rl_k is not None and generation_step is not None:
                trainer_step = trainer_step_snapshot
                group_lag = max(0, int(trainer_step) - int(generation_step))
                if group_lag > int(pipeline_rl_k):
                    stale_drop_groups += 1
                    try:
                        reset_group = _reset_group_for_regeneration(group)
                        with worker.data_buffer_lock:
                            data_buffer.add_samples([reset_group])
                        logger.info(
                            "Dropping stale group and requeuing: "
                            f"group_id={group_id}, trainer_step={trainer_step}, "
                            f"generation_step={generation_step}, lag={group_lag}, k={pipeline_rl_k}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to requeue stale group {group_id}: {e}")
                    processed_any = True
                    continue

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    # add back to buffer so it can be retried or handled by buffer policy
                    with worker.data_buffer_lock:
                        data_buffer.add_samples([group])
                    logger.info(f"Returned aborted group {group_id} to data buffer")
                except Exception as e:
                    logger.warning(f"Failed to return aborted group {group_id} to buffer: {e}")
                # don't count as processed for training
                continue

            if do_print:
                logger.info(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                )
                do_print = False

            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                logger.info(f"Dropping group {group[0][0].index if isinstance(group[0], list) else group[0].index} due to {dynamic_filter_output.reason}")
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                processed_any = True
                continue

            data.append(group)
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            logger.warning(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    logger.info(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        logger.info(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
        )

    if getattr(args, "rollout_sample_filter_path", None) is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    all_samples = sorted(all_data, key=lambda group: group[0].index)

    if getattr(args, "rollout_all_samples_process_path", None) is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        # Pass data_buffer.get_samples (bound method) so _resolve_scalerl_data_source
        # can extract the data source instance via __self__.
        with worker.data_buffer_lock:
            process_func(args, all_samples, data_buffer.get_samples)

    data = sorted(data, key=lambda group: group[0].index)
    worker_metrics = worker.get_pipeline_rl_metrics() if hasattr(worker, "get_pipeline_rl_metrics") else {}
    stale_metrics = (
        _collect_stale_metrics(
            stale_drop_groups=stale_drop_groups,
        )
        if pipeline_rl_k is not None
        else {}
    )
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect() | worker_metrics | stale_metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        stop_global_worker()
        try:
            return sglang_generate_rollout(args, rollout_id, data_buffer, evaluation=True)
        finally:
            _reset_generate_state()

    completed_samples = run(generate_rollout_async(args, rollout_id, data_buffer))
    return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
