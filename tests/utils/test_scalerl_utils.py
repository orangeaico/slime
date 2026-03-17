from types import SimpleNamespace

import pytest
import torch

from slime.utils.scalerl_utils import (
    get_batch_normalized_prompt_rewards,
    get_prompt_group_mean_centered_rewards,
    get_prompt_loss_token_weights,
    get_required_prompt_group_multiple,
    should_force_per_token_loss,
    validate_scalerl_args,
)


def test_batch_level_normalization_matches_scalerl_formula():
    raw_rewards = [1.0, 3.0, 10.0, 14.0]
    group_indices = [0, 0, 1, 1]

    centered_rewards = get_prompt_group_mean_centered_rewards(raw_rewards, group_indices)
    normalized_rewards = get_batch_normalized_prompt_rewards(raw_rewards, group_indices)

    expected = centered_rewards / (centered_rewards.std() + 1e-6)
    assert torch.allclose(torch.tensor(normalized_rewards), expected)

    prompt_std_normalized = torch.tensor(
        [
            centered_rewards[0] / (centered_rewards[:2].std() + 1e-6),
            centered_rewards[1] / (centered_rewards[:2].std() + 1e-6),
            centered_rewards[2] / (centered_rewards[2:].std() + 1e-6),
            centered_rewards[3] / (centered_rewards[2:].std() + 1e-6),
        ]
    )
    assert not torch.allclose(torch.tensor(normalized_rewards), prompt_std_normalized)

    token_lengths = [10, 1, 1, 1]
    token_weighted_advs = torch.cat(
        [
            centered_rewards[idx].repeat(token_length)
            for idx, token_length in enumerate(token_lengths)
        ]
    )
    token_whitened = (token_weighted_advs - token_weighted_advs.mean()) / (token_weighted_advs.std() + 1e-6)
    assert not torch.allclose(torch.tensor(normalized_rewards), token_whitened[: len(normalized_rewards)])


def test_prompt_loss_token_weights_match_prompt_average_formula():
    group_indices = [0, 0, 1, 1]
    loss_masks = [
        [1, 1],
        [1, 0],
        [1, 1, 1],
        [1],
    ]
    per_token_losses = [
        torch.tensor([2.0, 4.0]),
        torch.tensor([6.0, 0.0]),
        torch.tensor([1.0, 3.0, 5.0]),
        torch.tensor([7.0]),
    ]

    weights, num_prompt_groups = get_prompt_loss_token_weights(loss_masks, group_indices)
    weighted_sum = sum(
        (token_losses * torch.tensor(loss_mask, dtype=token_losses.dtype)).sum() * weight
        for token_losses, loss_mask, weight in zip(per_token_losses, loss_masks, weights, strict=True)
    )
    prompt_average = weighted_sum / num_prompt_groups

    prompt_0_total = 2.0 + 4.0 + 6.0
    prompt_1_total = 1.0 + 3.0 + 5.0 + 7.0
    expected = 0.5 * (prompt_0_total / 3.0 + prompt_1_total / 4.0)

    assert prompt_average.item() == pytest.approx(expected)
    assert weights == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 4.0])


def test_required_prompt_group_multiple_uses_lcm():
    assert get_required_prompt_group_multiple(6, 4, require_prompt_group_alignment=True) == 12
    assert get_required_prompt_group_multiple(6, 4, require_prompt_group_alignment=False) == 6


def test_should_force_per_token_loss_respects_prompt_aggregation():
    assert should_force_per_token_loss("cispo_loss", prompt_level_loss_aggregation=False) is True
    assert should_force_per_token_loss("dispo_loss", prompt_level_loss_aggregation=False) is True
    assert should_force_per_token_loss("cispo_loss", prompt_level_loss_aggregation=True) is False
    assert should_force_per_token_loss("dispo_loss", prompt_level_loss_aggregation=True) is False
    assert should_force_per_token_loss("policy_loss", prompt_level_loss_aggregation=False) is False


def test_validate_scalerl_args_rejects_invalid_combinations():
    base_args = dict(
        batch_level_normalization=False,
        prompt_level_loss_aggregation=False,
        advantage_estimator="grpo",
        normalize_advantages=False,
        rewards_normalization=True,
        train_backend="megatron",
        custom_pg_loss_reducer_function_path=None,
        loss_type="policy_loss",
        global_batch_size=64,
        n_samples_per_prompt=8,
    )

    with pytest.raises(ValueError, match="normalize-advantages"):
        validate_scalerl_args(
            SimpleNamespace(**(base_args | dict(batch_level_normalization=True, normalize_advantages=True)))
        )

    with pytest.raises(ValueError, match="Megatron backend"):
        validate_scalerl_args(
            SimpleNamespace(**(base_args | dict(prompt_level_loss_aggregation=True, train_backend="fsdp")))
        )

    with pytest.raises(ValueError, match="divisible by --n-samples-per-prompt"):
        validate_scalerl_args(
            SimpleNamespace(**(base_args | dict(prompt_level_loss_aggregation=True, global_batch_size=66)))
        )

    validate_scalerl_args(
        SimpleNamespace(**(base_args | dict(batch_level_normalization=True, prompt_level_loss_aggregation=True)))
    )
