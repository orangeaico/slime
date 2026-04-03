"""Reward function for SWE-bench on-policy distillation.

By default this returns 0.0 task rewards and learns from OPD KL penalty
computed from teacher log-probs.

When `--swe-eval-reward-enable` is set, task rewards are overwritten by
evaluation status directly inside ``reward_func``:
  - resolved: +1.0
  - anything else: -1.0

This follows the pattern from slime/examples/on_policy_distillation/on_policy_distillation.py
and keeps `post_process_rewards` focused on tensor extraction only.
"""

import asyncio

import aiohttp
import torch

from examples.swe_bench.eval.eval_reward_router import evaluate_group_for_reward, sample_eval_key
from slime.utils.types import Sample


def _default_missing_eval_result() -> dict:
    return {
        "resolved": False,
        "status": "missing_eval_result",
        "error": "no eval result",
        "run_id": "",
    }


def _assign_eval_metadata(sample: Sample, *, status: str, reward_value: float, run_id: str, error: str) -> None:
    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    metadata["swe_eval_status"] = status
    metadata["swe_eval_reward"] = reward_value
    metadata["swe_eval_run_id"] = run_id
    metadata["swe_eval_error"] = error
    sample.metadata = metadata


async def _query_teacher_logprobs(args, sample: Sample) -> dict:
    """Query teacher model to get log probabilities for OPD."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _get_eval_reward_value(eval_result: dict) -> float:
    return 1.0 if bool(eval_result.get("resolved", False)) else -1.0


def _apply_reward_and_metadata(
    args,
    sample: Sample,
    reward_payload: dict,
    *,
    eval_result: dict | None,
) -> dict:
    use_eval_reward = bool(getattr(args, "swe_eval_reward_enable", False))
    if use_eval_reward:
        resolved_result = eval_result or _default_missing_eval_result()
        reward_value = _get_eval_reward_value(resolved_result)
        _assign_eval_metadata(
            sample,
            status=str(resolved_result.get("status", "")),
            reward_value=reward_value,
            run_id=str(resolved_result.get("run_id", "")),
            error=str(resolved_result.get("error", "")),
        )
    else:
        reward_value = 0.0
        _assign_eval_metadata(
            sample,
            status="eval_disabled",
            reward_value=reward_value,
            run_id="",
            error="",
        )

    reward_payload["reward"] = reward_value
    return reward_payload


async def reward_func(args, sample, **kwargs):
    """Query teacher model and assign task reward.

    This function sends the student's generated tokens to the teacher model
    to get the teacher's log probabilities, which are used for KL divergence
    computation in on-policy distillation. It also assigns scalar task reward:
    - eval enabled: resolved => +1.0 else -1.0
    - eval disabled: 0.0

    Args:
        args: Training arguments (must have args.rm_url for teacher server)
        sample: Sample or list[Sample] with tokens to evaluate
        **kwargs: Additional arguments

    Returns:
        dict or list[dict]: Teacher model response(s) containing log probabilities
        and scalar reward under ``reward`` key
    """
    if isinstance(sample, list):
        samples = sample
        payloads = await asyncio.gather(*[_query_teacher_logprobs(args, s) for s in samples])

        eval_results_by_key: dict[str, dict] = {}
        if bool(getattr(args, "swe_eval_reward_enable", False)):
            eval_results_by_key = evaluate_group_for_reward(args, samples)

        output_payloads = []
        for i, (s, payload) in enumerate(zip(samples, payloads, strict=False)):
            eval_result = eval_results_by_key.get(sample_eval_key(s, i))
            output_payloads.append(_apply_reward_and_metadata(args, s, payload, eval_result=eval_result))
        return output_payloads

    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    result = await _query_teacher_logprobs(args, sample)
    eval_result = None
    if bool(getattr(args, "swe_eval_reward_enable", False)):
        eval_results_by_key = evaluate_group_for_reward(args, [sample])
        eval_result = eval_results_by_key.get(sample_eval_key(sample, 0))
    return _apply_reward_and_metadata(args, sample, result, eval_result=eval_result)


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs and return scalar rewards for training.

    This function:
    1. Extracts teacher log-probs from reward response (sglang output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards already assigned in `reward_func`
    5. Optimizes sample.reward for saving (truncates logprobs to save space)

    Args:
        args: Training arguments
        samples: List of samples with rewards from reward_func
        **kwargs: Additional arguments

    Returns:
        tuple: (rewards, rewards) - scalar rewards for training and logging
    """
    raw_rewards = [sample.reward for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    def _extract_logprobs_tensor(reward_obj):
        if not isinstance(reward_obj, dict):
            return torch.tensor([], dtype=torch.float32)
        meta_info = reward_obj.get("meta_info", {})
        token_logprobs = meta_info.get("input_token_logprobs", [])
        values = []
        for item in token_logprobs[1:]:
            if isinstance(item, (list, tuple)) and item:
                values.append(item[0])
        return torch.tensor(values, dtype=torch.float32)

    teacher_log_probs = [_extract_logprobs_tensor(reward) for reward in raw_rewards]

    # Trim to match response length (only the generated tokens)
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    # Store teacher log-probs in samples for OPD KL computation
    for sample, t_log_probs, reward in zip(samples, teacher_log_probs, raw_rewards, strict=False):
        sample.teacher_log_probs = t_log_probs

        # Optimize reward dict for saving - keep only a few logprobs for debugging
        # This significantly reduces saved file size
        if isinstance(reward, dict) and "meta_info" in reward:
            original_logprobs = reward["meta_info"].get("input_token_logprobs", [])
            if len(original_logprobs) > 10:
                # Keep only first 5 and last 5 logprobs for inspection
                truncated_logprobs = original_logprobs[:5] + ["... truncated ..."] + original_logprobs[-5:]
                reward["meta_info"]["input_token_logprobs"] = truncated_logprobs
                reward["meta_info"]["_truncated"] = True
                reward["meta_info"]["_original_length"] = len(original_logprobs)

    # Scalar task reward is assigned in reward_func and stored at reward_key.
    scalar_rewards: list[float] = []
    for sample in samples:
        try:
            reward_value = float(sample.get_reward_value(args))
        except Exception:
            reward_obj = sample.reward
            if isinstance(reward_obj, dict):
                reward_value = float(reward_obj.get("reward", 0.0))
            else:
                reward_value = float(reward_obj or 0.0)
        scalar_rewards.append(reward_value)

    return scalar_rewards, scalar_rewards
