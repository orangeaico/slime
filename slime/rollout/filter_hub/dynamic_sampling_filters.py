import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.scalerl import get_shaped_rewards
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_reward_nonzero_std_with_dapo_style"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_with_dapo_style(args, samples: list[Sample], **kwargs):
    _, shaped_rewards = get_shaped_rewards(args, samples)
    keep = bool(torch.tensor(shaped_rewards, dtype=torch.float32).std() > 0.0)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(shaped_rewards[0], 3)}",
    )
