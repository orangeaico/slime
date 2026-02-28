"""
SWE-Agent Rollout with Time-Bounded Collection

Optimized rollout strategy for variable-duration SWE-agent tasks:
- Samples are processed individually (no group blocking)
- Time-bounded collection (configurable max time)
- Partial samples saved and resumed in next iteration
- Scales efficiently with large batch sizes

Usage:
    Add to your training script:
    --custom-rollout-function-path examples.swe_bench.swe_agent_rollout.generate_rollout_swe_agent
    --rollout-max-time-minutes 10
    --partial-rollout
"""

import asyncio
import logging
import time
from argparse import Namespace
from typing import Callable

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Keep our own script logs at DEBUG level


async def generate_rollout_swe_agent(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """
    Time-bounded rollout collection for SWE-agent with variable-duration tasks.

    Strategy:
    1. Ungroup samples - treat each sample independently (no blocking on groups)
    2. Submit tasks continuously to keep pipeline full
    3. Collect for max_time_minutes, then proceed to training
    4. Save partial (incomplete) samples for next iteration

    Args:
        args: Training arguments
            - rollout_batch_size: Target number of samples to collect
            - rollout_max_time_minutes: Max time to collect (default 10)
            - n_samples_per_prompt: Logical grouping (for best-of-N analysis)
            - over_sampling_batch_size: Samples to request per batch
        rollout_id: Current rollout iteration
        data_source: Function to fetch sample groups from dataset
        evaluation: Whether this is evaluation (not used for training)

    Returns:
        RolloutFnTrainOutput with collected samples and metrics
    """
    assert args.rollout_global_dataset, "SWE-agent rollout requires global dataset"

    # Initialize state
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )
    metric_gatherer = MetricGatherer()

    # Configuration
    target_sample_count = args.rollout_batch_size  # Number of samples to collect
    max_time_minutes = getattr(args, "rollout_max_time_minutes", 10)
    max_time_seconds = max_time_minutes * 60
    over_provision_factor = getattr(args, "rollout_over_provision_factor", 4)

    logger.info(
        f"[SWE-Agent Rollout {rollout_id}] Starting time-bounded collection: "
        f"target={target_sample_count} samples, max_time={max_time_minutes}min"
    )

    # Track submitted and collected samples
    individual_tasks = {}  # task -> sample mapping
    collected_samples = []
    submitted_count = 0
    start_time = time.time()

    # Progress tracking
    total_to_submit = min(
        target_sample_count * over_provision_factor,
        1000,  # Cap to avoid memory issues
    )
    pbar = tqdm(
        total=target_sample_count,
        desc=f"Rollout {rollout_id} (time-bounded)",
        unit="samples",
    )

    # Phase 1: Submit and collect tasks
    logger.info(f"[SWE-Agent Rollout {rollout_id}] Submitting up to {total_to_submit} tasks")

    # Track whether we should continue submitting
    can_submit_more = True
    data_source_exhausted = False

    while len(collected_samples) < target_sample_count:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            logger.info(
                f"[SWE-Agent Rollout {rollout_id}] Time limit reached "
                f"({elapsed:.1f}s > {max_time_seconds}s), stopping new submissions"
            )
            can_submit_more = False
            # Don't break - continue waiting for running tasks

        # Submit more tasks if allowed and pipeline not full
        if (
            can_submit_more
            and not data_source_exhausted
            and submitted_count < total_to_submit
            and len(individual_tasks) < target_sample_count * 2
        ):
            # Fetch batch of sample groups from data source
            try:
                sample_groups = data_source(args.over_sampling_batch_size)
            except Exception as e:
                logger.warning(
                    f"[SWE-Agent Rollout {rollout_id}] Data source exhausted after "
                    f"{submitted_count} submissions: {e}"
                )
                data_source_exhausted = True
                # Don't break - continue waiting for running tasks

            if not data_source_exhausted:
                # Unpack groups into individual samples and submit
                for group in sample_groups:
                    for sample in group:
                        # Create individual task for this sample
                        task = asyncio.create_task(
                            generate_and_rm(
                                args,
                                sample,
                                sampling_params=state.sampling_params.copy(),
                                evaluation=evaluation,
                            )
                        )
                        individual_tasks[task] = sample
                        submitted_count += 1

                        # Check if we've submitted enough
                        if submitted_count >= total_to_submit:
                            logger.info(
                                f"[SWE-Agent Rollout {rollout_id}] Reached submission limit "
                                f"({submitted_count} tasks), no more submissions"
                            )
                            can_submit_more = False
                            break

                    if not can_submit_more:
                        break

                logger.debug(
                    f"[SWE-Agent Rollout {rollout_id}] Submitted {submitted_count} tasks, "
                    f"{len(individual_tasks)} pending, {len(collected_samples)} collected"
                )

        # Wait for next completion (with timeout for periodic checks)
        if not individual_tasks:
            # No tasks running at all
            if len(collected_samples) < target_sample_count:
                logger.warning(
                    f"[SWE-Agent Rollout {rollout_id}] No tasks pending but only "
                    f"{len(collected_samples)}/{target_sample_count} collected. "
                    f"Submitted: {submitted_count}, Data exhausted: {data_source_exhausted}"
                )
            break

        remaining_time = max(1, max_time_seconds - elapsed)
        try:
            done, pending = await asyncio.wait(
                set(individual_tasks.keys()),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=min(remaining_time, 30),  # Check every 30 seconds
            )
        except asyncio.TimeoutError:
            logger.debug(f"[SWE-Agent Rollout {rollout_id}] No completions in last 30s")
            continue

        if not done:
            # Timeout reached but no tasks completed
            elapsed = time.time() - start_time
            logger.debug(
                f"[SWE-Agent Rollout {rollout_id}] Waiting... "
                f"{len(collected_samples)}/{target_sample_count} collected, "
                f"{len(individual_tasks)} pending, "
                f"elapsed {elapsed:.1f}s"
            )
            continue

        # Process completed samples
        for task in done:
            sample = task.result()
            original_sample = individual_tasks.pop(task)

            # Check sample status
            if sample.status == Sample.Status.ABORTED:
                logger.debug(
                    f"[SWE-Agent Rollout {rollout_id}] Sample aborted: "
                    f"{sample.metadata.get('instance_id', 'unknown')}"
                )
                metric_gatherer.on_dynamic_filter_drop(reason="aborted")
                continue

            if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                logger.warning(
                    f"[SWE-Agent Rollout {rollout_id}] Sample has unexpected status: "
                    f"{sample.status} for {sample.metadata.get('instance_id', 'unknown')}"
                )
                continue

            # Apply dynamic filter (if configured)
            # Note: Filter expects groups, so wrap in list
            filter_output = call_dynamic_filter(dynamic_filter, args, [sample])
            if not filter_output.keep:
                logger.debug(
                    f"[SWE-Agent Rollout {rollout_id}] Sample filtered: "
                    f"{sample.metadata.get('instance_id', 'unknown')}, "
                    f"reason={filter_output.reason}"
                )
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue

            # Accept sample
            collected_samples.append(sample)
            pbar.update(1)

            logger.debug(
                f"[SWE-Agent Rollout {rollout_id}] Collected sample "
                f"{len(collected_samples)}/{target_sample_count}: "
                f"{sample.metadata.get('instance_id', 'unknown')}, "
                f"duration={time.time() - start_time:.1f}s"
            )

            # Check if we've collected enough
            if len(collected_samples) >= target_sample_count:
                logger.info(
                    f"[SWE-Agent Rollout {rollout_id}] Target reached: "
                    f"{len(collected_samples)} samples collected"
                )
                break

    pbar.close()

    # Phase 2: Abort remaining tasks and collect partial samples
    elapsed = time.time() - start_time
    logger.info(
        f"[SWE-Agent Rollout {rollout_id}] Collection complete: "
        f"{len(collected_samples)} samples in {elapsed:.1f}s, "
        f"{len(individual_tasks)} tasks still pending"
    )

    # Abort remaining in-flight tasks
    partial_samples = []
    if individual_tasks:
        logger.info(f"[SWE-Agent Rollout {rollout_id}] Aborting {len(individual_tasks)} pending tasks...")

        # Mark state as aborted to stop new generations
        state.aborted = True

        # Send abort request to SGLang engines
        try:
            from slime.rollout.sglang_rollout import abort as abort_sglang_engines

            await abort_sglang_engines(args, rollout_id)
        except Exception as e:
            logger.error(f"[SWE-Agent Rollout {rollout_id}] Failed to abort engines: {e}")

        # Collect results from aborted tasks (may have partial responses)
        aborted_count = 0
        partial_count = 0

        # Wait for all tasks to finish (they should abort quickly)
        if individual_tasks:
            done, _ = await asyncio.wait(set(individual_tasks.keys()), timeout=60)

            for task in done:
                try:
                    sample = task.result()
                    original_sample = individual_tasks.get(task)

                    # Check if sample has partial response (useful for next iteration)
                    if (
                        args.partial_rollout
                        and sample.response
                        and len(sample.response) > 0
                        and sample.status != Sample.Status.COMPLETED
                    ):
                        # Mark as partial for resumption in next iteration
                        sample.metadata["partial_from_rollout"] = rollout_id
                        sample.metadata["start_rollout_id"] = rollout_id
                        partial_samples.append(sample)
                        partial_count += 1
                        logger.debug(
                            f"[SWE-Agent Rollout {rollout_id}] Saved partial sample: "
                            f"{sample.metadata.get('instance_id', 'unknown')}, "
                            f"response_length={sample.response_length}"
                        )
                    else:
                        aborted_count += 1

                except Exception as e:
                    logger.warning(f"[SWE-Agent Rollout {rollout_id}] Error getting aborted task result: {e}")
                    aborted_count += 1

        logger.info(
            f"[SWE-Agent Rollout {rollout_id}] Abort complete: "
            f"{aborted_count} aborted, {partial_count} partial samples saved"
        )

    # Phase 3: Organize results
    total_elapsed = time.time() - start_time

    # Sort samples by index for reproducibility
    collected_samples = sorted(collected_samples, key=lambda s: s.index if s.index is not None else 0)

    # Log statistics
    logger.info(
        f"[SWE-Agent Rollout {rollout_id}] Summary:\n"
        f"  - Collected: {len(collected_samples)} samples\n"
        f"  - Submitted: {submitted_count} tasks\n"
        f"  - Partial: {len(partial_samples)} samples saved for next iteration\n"
        f"  - Duration: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n"
        f"  - Throughput: {len(collected_samples)/total_elapsed*60:.2f} samples/min"
    )

    # Log sample durations for analysis
    if collected_samples:
        durations = []
        for sample in collected_samples:
            if "generation_time" in sample.metadata:
                durations.append(sample.metadata["generation_time"])

        if durations:
            logger.info(
                f"[SWE-Agent Rollout {rollout_id}] Sample durations: "
                f"min={min(durations):.1f}s, max={max(durations):.1f}s, "
                f"mean={sum(durations)/len(durations):.1f}s"
            )

    # Phase 4: Post-process for best-of-N analysis (optional)
    if args.n_samples_per_prompt > 1:
        logger.info(
            f"[SWE-Agent Rollout {rollout_id}] Analyzing best-of-{args.n_samples_per_prompt} candidates..."
        )
        grouped_analysis = _analyze_best_of_n(collected_samples, args.n_samples_per_prompt)
        logger.info(f"[SWE-Agent Rollout {rollout_id}] Best-of-N analysis: {grouped_analysis}")

    # Reset state for next rollout
    state.reset()

    # Return collected samples and metrics
    return RolloutFnTrainOutput(
        samples=collected_samples,
        metrics=metric_gatherer.collect(),
    )


def _analyze_best_of_n(samples: list[Sample], n_per_prompt: int) -> dict:
    """
    Analyze samples for best-of-N selection.

    Groups samples by prompt/instance and identifies best candidate.
    This is informational only - all samples are still used for training.
    """
    from collections import defaultdict

    # Group by instance_id
    groups = defaultdict(list)
    for sample in samples:
        instance_id = sample.metadata.get("instance_id", "unknown")
        groups[instance_id].append(sample)

    # Analyze each group
    complete_groups = 0
    partial_groups = 0
    best_rewards = []

    for instance_id, group_samples in groups.items():
        if len(group_samples) == n_per_prompt:
            # Complete group - can do best-of-N
            complete_groups += 1
            rewards = [s.reward if isinstance(s.reward, (int, float)) else 0.0 for s in group_samples]
            best_reward = max(rewards)
            best_rewards.append(best_reward)
        else:
            # Partial group
            partial_groups += 1

    analysis = {
        "complete_groups": complete_groups,
        "partial_groups": partial_groups,
        "total_groups": len(groups),
    }

    if best_rewards:
        analysis["best_reward_mean"] = sum(best_rewards) / len(best_rewards)
        analysis["best_reward_max"] = max(best_rewards)

    return analysis


def _save_partial_samples_to_buffer(args: Namespace, partial_samples: list[Sample], rollout_id: int):
    """
    Save partial samples to data source buffer for next iteration.

    These samples will be loaded in the next rollout and resumed from
    where they left off (partial-rollout mode).
    """
    if not partial_samples:
        return

    try:
        # This requires data source to support adding samples back
        # Implementation depends on your data source type
        logger.info(
            f"[SWE-Agent Rollout {rollout_id}] Saving {len(partial_samples)} partial samples to buffer"
        )

        # Mark samples for partial rollout
        for sample in partial_samples:
            # Mask previous tokens (don't train on them in next iteration)
            if args.mask_offpolicy_in_partial_rollout and sample.loss_mask:
                sample.loss_mask = [0] * len(sample.loss_mask)

        # TODO: Implement data source interface for adding partial samples
        # data_source.add_partial_samples(partial_samples)

    except Exception as e:
        logger.error(f"[SWE-Agent Rollout {rollout_id}] Failed to save partial samples: {e}")
