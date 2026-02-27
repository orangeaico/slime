"""Reward function for SWE-bench on-policy distillation.

For pure on-policy distillation, we return 0.0 task rewards.
The learning signal comes from the OPD KL penalty computed from teacher log-probs.

This follows the pattern from slime/examples/on_policy_distillation/on_policy_distillation.py
"""

import aiohttp
import torch
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    """Query teacher model to get log probabilities for OPD.

    This function sends the student's generated tokens to the teacher model
    to get the teacher's log probabilities, which are used for KL divergence
    computation in on-policy distillation.

    Args:
        args: Training arguments (must have args.rm_url for teacher server)
        sample: Sample with tokens to evaluate
        **kwargs: Additional arguments

    Returns:
        dict: Teacher model's response containing log probabilities
    """
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Query teacher server for log probabilities on student's tokens
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
            result = await resp.json()
            # Add scalar reward field for metrics computation
            # For pure OPD, task reward is always 0.0
            result["reward"] = 0.0
            return result


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs and return scalar rewards for training.

    This function:
    1. Extracts teacher log-probs from the reward response (sglang output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) for GRPO/PPO
    5. Optimizes sample.reward for saving (truncates logprobs to save space)

    Args:
        args: Training arguments
        samples: List of samples with rewards from reward_func
        **kwargs: Additional arguments

    Returns:
        tuple: (rewards, rewards) - both are lists of 0.0 for pure distillation
    """
    # Access sample.reward directly to get the full dict (not the scalar from get_reward_value)
    raw_rewards = [sample.reward for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32
        )
        for reward in raw_rewards
    ]

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

    # Return scalar task rewards (0.0 for pure distillation)
    # The learning signal comes from the OPD KL penalty
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
