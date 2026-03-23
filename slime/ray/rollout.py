import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.rollout.scalerl import compute_scalerl_metrics_from_samples
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.scalerl_utils import (
    apply_group_relative_focal_weights_to_rewards,
    get_batch_normalized_prompt_rewards,
    get_prompt_group_indices,
    get_prompt_group_mean_centered_rewards,
    get_prompt_loss_token_weights,
    get_required_prompt_group_multiple,
    normalize_rewards_for_training,
)
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _default_early_stop_status() -> dict[str, Any]:
    return {
        "should_stop_after_training_batch": False,
        "reason": None,
        "max_fresh_prompt_passes": None,
        "active_prompt_count": 0,
        "min_active_prompt_fresh_passes": 0,
        "max_active_prompt_fresh_passes": 0,
        "active_prompt_completion_ratio": 0.0,
    }


def _requires_prompt_group_alignment(args) -> bool:
    return args.batch_level_normalization or args.prompt_level_loss_aggregation


def _get_sample_group_indices(samples: list[Sample], n_samples_per_prompt: int) -> list[int]:
    return get_prompt_group_indices([sample.group_index for sample in samples], n_samples_per_prompt)


def _is_sample_active_for_training(sample: Sample) -> bool:
    if sample.remove_sample:
        return False
    if sample.loss_mask is None:
        return True
    return sum(sample.loss_mask) > 0


def _get_active_samples(samples: list[Sample]) -> list[Sample]:
    return [sample for sample in samples if _is_sample_active_for_training(sample)]


def _get_dp_partitions(
    total_lengths: list[int],
    *,
    dp_size: int,
    balance_data: bool,
    global_batch_size: int,
    preserve_step_boundaries: bool,
) -> list[list[int] | range]:
    if not preserve_step_boundaries:
        if balance_data:
            return get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        return [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    partitions = [[] for _ in range(dp_size)]
    for step_start in range(0, len(total_lengths), global_batch_size):
        step_total_lengths = total_lengths[step_start : step_start + global_batch_size]
        if balance_data:
            step_partitions = get_seqlen_balanced_partitions(step_total_lengths, dp_size, equal_size=True)
        else:
            step_partitions = [range(i, len(step_total_lengths), dp_size) for i in range(dp_size)]

        for rank in range(dp_size):
            partitions[rank].extend(step_start + local_idx for local_idx in step_partitions[rank])

    return partitions


@dataclasses.dataclass
class EngineGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple EngineGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    nodes_per_engine: int
    num_new_engines: int
    role: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0  # global rank of the first engine in this group

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self) -> list:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns a list of Ray ObjectRefs for the init calls.  The caller
        should ``ray.get()`` on them to block until the engines are healthy.
        """
        if self.args.debug_train_only:
            self.num_new_engines = 0
            return []

        num_gpu_per_engine = min(self.args.rollout_num_gpus_per_engine, self.args.num_gpus_per_node)
        total_num_engines = self.args.rollout_num_gpus // num_gpu_per_engine

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group
            base_gpu_id = int(reordered_gpu_ids[global_rank * num_gpu_per_engine])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[global_rank * num_gpu_per_engine],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(self.args, rank=global_rank, worker_type=self.role, base_gpu_id=base_gpu_id)

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return []

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                num_engines=total_num_engines,
                rollout_engines=rollout_engines,
                worker_type=self.role,
            )

        init_handles = [engine.init.remote(**(addr_and_ports[rank])) for rank, engine in rollout_engines]
        return init_handles

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more engine groups.

    Corresponds to one entry in the future ``--sglang-config`` YAML::

        rollout_servers:
          - name: policy
            engine_groups:
              - role: prefill
                ...
              - role: decode
                ...

    Currently only a single EngineGroup is used for non-PD mode;
    PD disaggregation creates separate prefill and decode groups.
    """

    engine_groups: list[EngineGroup]
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def engines(self):
        """All node-0 engines across all groups."""
        return [e for g in self.engine_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.engine_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.engine_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.engine_groups:
            g.num_new_engines = value

    @property
    def nodes_per_engine(self):
        """Nodes per engine. Only valid when all groups share the same value.

        TODO: remove once health_monitor operates per-group.
        """
        values = {g.nodes_per_engine for g in self.engine_groups}
        assert len(values) == 1, f"Heterogeneous nodes_per_engine: {values}"
        return values.pop()

    def recover(self):
        """Recover dead engines across all groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.engine_groups]

        # Start all groups concurrently.
        all_handles = []
        for g in self.engine_groups:
            all_handles.extend(g.start_engines())
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        new_engines_all = []
        for g, dead_indices in zip(self.engine_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (role={g.role})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.args.offload_rollout and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                new_engines_all.extend(new_engines)

        if release_handles:
            ray.get(release_handles)
            ray.get(
                [engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines_all]
            )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.offload())
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.engine_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        init_tracking(args, primary=False)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.server = None
        else:
            init_http_client(args)
            self.server = start_rollout_server(args, pg)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._last_generate_status = _default_early_stop_status()

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for group in self.server.engine_groups:
                monitor = RolloutHealthMonitor(group, args)
                monitor.start()
                self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.engine_groups[0].all_engines and self.server.engine_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.engine_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()

    @property
    def rollout_engines(self):
        if self.server is None:
            return []
        return self.server.engines

    def get_rollout_engines_and_lock(self):
        num_new_engines = self.server.num_new_engines if self.server else 0
        return self.rollout_engines, self.rollout_engine_lock, num_new_engines

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def get_last_generate_status(self):
        return dict(self._last_generate_status)

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)
        self._last_generate_status = _default_early_stop_status()

    def offload(self):
        self.health_monitoring_pause()
        if self.server:
            return self.server.offload()

    def onload(self, tags: list[str] | None = None):
        if self.server:
            return self.server.onload(tags)

    def onload_weights(self):
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def onload_kv(self):
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def recover_rollout_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection."""
        self.health_monitoring_pause()
        srv = self.server
        if self.rollout_id == -1:
            return self.rollout_engines, self.rollout_engine_lock, srv.num_new_engines

        srv.recover()
        return self.rollout_engines, self.rollout_engine_lock, srv.num_new_engines

    def clear_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        if self.server:
            self.server.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _compute_early_stop_status(self) -> dict[str, Any]:
        if hasattr(self.data_source, "get_early_stop_status"):
            return dict(self.data_source.get_early_stop_status())
        return _default_early_stop_status()

    def _get_rollout_data(self, rollout_id):
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
            self._last_generate_status = _default_early_stop_status()
        else:
            rollout_output = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = dict(rollout_output.metrics or {})
            self._last_generate_status = self._compute_early_stop_status()
            metrics |= {
                f"early_stop/{key}": (int(value) if isinstance(value, bool) else value)
                for key, value in self._last_generate_status.items()
                if key != "reason" and value is not None
            }
            if self._last_generate_status["should_stop_after_training_batch"]:
                logger.info(
                    "Early stop condition satisfied after rollout %s: %s",
                    rollout_id,
                    self._last_generate_status,
                )
            data = rollout_output.samples
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if not self.args.disable_rollout_trim_samples and not self.args.debug_rollout_only:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    # TODO: this is a temporary solution, we should directly save dynamic_global_batch_size to rollout data
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    if trim_len == 0:
                        raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                    origin_data_length = len(data)
                    data = data[:trim_len]
                    logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Calculate dynamic global_batch_size to ensure only one training step.

        Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
        This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size
        required_multiple = get_required_prompt_group_multiple(
            dp_size=dp_size,
            n_samples_per_prompt=self.args.n_samples_per_prompt,
            require_prompt_group_alignment=_requires_prompt_group_alignment(self.args),
        )

        # Round down to a multiple of dp_size (and prompt group size when needed)
        dynamic_gbs = (num_samples // required_multiple) * required_multiple

        if dynamic_gbs == 0:
            # Too few samples, use at least the required multiple and let trim validation raise if still insufficient.
            dynamic_gbs = required_multiple
            logger.warning(
                f"num_samples={num_samples} < required_multiple={required_multiple}, "
                f"using required_multiple as global_batch_size"
            )

        # Calculate how many samples will be discarded
        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, required_multiple={required_multiple}, "
                f"num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        group_indices = _get_sample_group_indices(samples, self.args.n_samples_per_prompt)
        active_mask = [not sample.remove_sample for sample in samples]

        if self.custom_reward_post_process_func is not None:
            raw_rewards, normalized_rewards = self.custom_reward_post_process_func(self.args, samples)
        else:
            raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
            active_raw_rewards = [reward for reward, is_active in zip(raw_rewards, active_mask, strict=True) if is_active]
            logger.info(f"[DEBUG] Raw rewards (extracted from samples): {raw_rewards}")
            if active_raw_rewards:
                logger.info(
                    "[DEBUG] Active raw rewards stats: "
                    f"min={min(active_raw_rewards)}, max={max(active_raw_rewards)}, "
                    f"mean={sum(active_raw_rewards)/len(active_raw_rewards)}"
                )
            else:
                logger.info("[DEBUG] No active samples remain for reward normalization.")

            if (
                self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
                and self.args.rewards_normalization
            ):
                centered_rewards = get_prompt_group_mean_centered_rewards(
                    raw_rewards, group_indices, active_mask=active_mask
                )
                logger.info(f"[DEBUG] Mean-centered rewards by group: {centered_rewards}")

                if self.args.batch_level_normalization:
                    normalized_rewards = get_batch_normalized_prompt_rewards(
                        raw_rewards,
                        group_indices,
                        active_mask=active_mask,
                    )
                    logger.info(f"[DEBUG] Rewards after batch-level normalization: {normalized_rewards}")
                else:
                    normalized_rewards = normalize_rewards_for_training(
                        self.args,
                        raw_rewards,
                        group_indices,
                        active_mask=active_mask,
                    )
                    if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                        logger.info(f"[DEBUG] Rewards after prompt-level std division: {normalized_rewards}")
                    else:
                        logger.info(
                            f"[DEBUG] Skipping std normalization "
                            f"(batch_level_normalization={self.args.batch_level_normalization}, "
                            f"grpo_std_normalization={self.args.grpo_std_normalization})"
                        )
            else:
                logger.info(
                    f"[DEBUG] Skipping reward normalization (rewards_normalization={self.args.rewards_normalization})"
                )
                normalized_rewards = normalize_rewards_for_training(
                    self.args,
                    raw_rewards,
                    group_indices,
                    active_mask=active_mask,
                )

        normalized_rewards = [
            reward if is_active else 0.0
            for reward, is_active in zip(normalized_rewards, active_mask, strict=True)
        ]
        normalized_rewards = apply_group_relative_focal_weights_to_rewards(
            normalized_rewards,
            raw_rewards,
            group_indices,
            getattr(self.args, "group_relative_focal_gamma", None),
            active_mask=active_mask,
        )
        logger.info(
            f"[DEBUG] Sum of normalized rewards after focal scaling: {sum(normalized_rewards)}, "
            f"Mean: {sum(normalized_rewards)/len(normalized_rewards)}"
        )
        return raw_rewards, normalized_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
            "group_index": _get_sample_group_indices(samples, self.args.n_samples_per_prompt),
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks
        train_data["active_sample_mask"] = [1 if sum(loss_mask) > 0 else 0 for loss_mask in loss_masks]

        if self.args.prompt_level_loss_aggregation:
            prompt_loss_token_weight, num_prompt_groups = get_prompt_loss_token_weights(loss_masks, train_data["group_index"])
            train_data["prompt_loss_token_weight"] = prompt_loss_token_weight
            train_data["num_prompt_groups"] = num_prompt_groups

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths
        global_batch_size = data.get("dynamic_global_batch_size", getattr(self, "_dynamic_global_batch_size", self.args.global_batch_size))

        partitions = _get_dp_partitions(
            total_lengths,
            dp_size=dp_size,
            balance_data=self.args.balance_data,
            global_batch_size=global_batch_size,
            preserve_step_boundaries=self.args.prompt_level_loss_aggregation,
        )

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "group_index",
                "prompt_loss_token_weight",
                "round_number",
                "sample_indices",
                "active_sample_mask",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
                "num_prompt_groups",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = []
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports.append(
            dict(
                dist_init_addr=addr,
                nccl_port=None,
                host=host,
                port=int(port),
            )
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(*, args, num_engines, rollout_engines, worker_type="regular"):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(
        1, min(args.num_gpus_per_node, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine
    )
    addr_and_ports = [{} for _ in range(num_engines)]

    visited_nodes = set()
    for rank, engine in rollout_engines:
        if rank // num_engines_per_node in visited_nodes:
            continue
        visited_nodes.add(rank // num_engines_per_node)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (rank % num_engines_per_node)

        def get_addr_and_ports(engine):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = 15000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if args.rollout_num_gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = args.rollout_num_gpus_per_engine // args.num_gpus_per_node
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports


def _start_router(args) -> tuple[str, int]:
    """Start sgl router or slime router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user), skip
    launching and return the existing values.
    """
    if args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    router_port = args.sglang_router_port
    if router_port is None:
        router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        assert args.prefill_num_servers is None, "slime router does not support prefill_num_servers."
        from slime.router.router import run_router

        # slime router reads ip/port from args at startup
        args.sglang_router_ip = router_ip
        args.sglang_router_port = router_port
        router_args = args

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = router_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if hasattr(args, "sglang_router_policy") and args.sglang_router_policy:
            router_args.policy = args.sglang_router_policy

        if args.prefill_num_servers is not None:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def start_rollout_server(args, pg) -> RolloutServer:
    """Start a complete rollout server: one router + a set of SGLang engines.

    Combines router startup and engine initialization into a single
    operation. Each RolloutServer represents one model served behind
    a shared router, containing one or more EngineGroups.

    When ``args.prefill_num_servers`` is set, creates separate prefill
    and decode EngineGroups for PD disaggregation.

    Note: init_http_client should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    router_ip, router_port = _start_router(args)
    # Write back for backward compatibility: downstream code (SGLangEngine,
    # rollout functions, examples) still reads args.sglang_router_ip/port.
    # TODO: remove once all consumers read from RolloutServer directly.
    args.sglang_router_ip = router_ip
    args.sglang_router_port = router_port

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
    total_num_engines = args.rollout_num_gpus // num_gpu_per_engine
    nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)

    if args.prefill_num_servers is not None:
        prefill_engine_count = args.prefill_num_servers * args.rollout_num_gpus_per_engine // num_gpu_per_engine
        decode_engine_count = total_num_engines - prefill_engine_count
        assert decode_engine_count > 0, f"No decode engines: total {total_num_engines}, prefill {prefill_engine_count}"

        prefill_group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * prefill_engine_count,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
            role="prefill",
            rank_offset=0,
        )
        decode_group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * decode_engine_count,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
            role="decode",
            rank_offset=prefill_engine_count,
        )

        # Start both groups concurrently — the heavy work (sglang server
        # startup + health-check) runs inside engine.init() on the Ray
        # workers, so overlapping the two groups nearly halves wall time.
        all_handles = prefill_group.start_engines() + decode_group.start_engines()
        ray.get(all_handles)

        engine_groups = [prefill_group, decode_group]
    else:
        group = EngineGroup(
            args=args,
            pg=pg,
            all_engines=[None] * total_num_engines,
            nodes_per_engine=nodes_per_engine,
            num_new_engines=0,
        )
        ray.get(group.start_engines())
        engine_groups = [group]

    return RolloutServer(
        engine_groups=engine_groups,
        router_ip=router_ip,
        router_port=router_port,
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        # print (f"[EVAL DEBUG] Rewards {rewards}")
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        log_dict[f"eval/{key}_avg@k_accuracy"] = rewards.count(1) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    raw_truncated_ratio = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item() if samples else 0.0
    active_samples = _get_active_samples(samples)
    if not active_samples:
        return {
            "response_len/mean": 0.0,
            "response_len/median": 0.0,
            "response_len/max": 0.0,
            "response_len/min": 0.0,
            "repetition_frac": 0.0,
            "truncated_ratio": raw_truncated_ratio,
        }

    response_lengths = [sample.effective_response_length for sample in active_samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= compute_scalerl_metrics_from_samples(args, active_samples)
    log_dict |= _compute_zero_std_metrics(args, active_samples)
    log_dict |= _compute_reward_cat_metrics(args, active_samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in active_samples]).item()
    log_dict["truncated_ratio"] = raw_truncated_ratio
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    active_samples = _get_active_samples(samples)
    non_generation_time = [sample.non_generation_time for sample in active_samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if non_generation_time and max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        if not response_lengths:
            if args.rollout_num_gpus:
                log_dict[f"{key}tokens_per_gpu_per_sec"] = 0.0
            log_dict[f"longest_{key}sample_tokens_per_sec"] = 0.0
            return
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if not non_generation_time or max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in active_samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in active_samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
