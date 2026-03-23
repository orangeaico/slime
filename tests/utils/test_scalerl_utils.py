from types import SimpleNamespace

import pytest
import torch

from slime.utils.scalerl_utils import (
    apply_group_relative_focal_weights_to_rewards,
    apply_length_penalty,
    compute_dapo_style_length_penalty,
    get_batch_normalized_prompt_rewards,
    get_group_relative_focal_weights,
    get_prompt_group_mean_centered_rewards,
    get_prompt_loss_token_weights,
    get_required_prompt_group_multiple,
    get_train_metric_normalizers,
    normalize_rewards_for_training,
    should_force_per_token_loss,
    should_discard_prompt,
    update_step_pass_rate_window,
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


def test_batch_level_normalization_ignores_inactive_samples():
    raw_rewards = [1.0, 3.0, 10.0, 14.0]
    group_indices = [0, 0, 1, 1]
    active_mask = [True, False, True, True]

    centered_rewards = get_prompt_group_mean_centered_rewards(raw_rewards, group_indices, active_mask=active_mask)
    normalized_rewards = get_batch_normalized_prompt_rewards(raw_rewards, group_indices, active_mask=active_mask)

    assert centered_rewards.tolist() == pytest.approx([0.0, 0.0, -2.0, 2.0])
    assert normalized_rewards[1] == pytest.approx(0.0)

    active_centered = torch.tensor([0.0, -2.0, 2.0])
    expected = torch.tensor([0.0, 0.0, -2.0, 2.0]) / (active_centered.std() + 1e-6)
    assert torch.allclose(torch.tensor(normalized_rewards), expected)


def test_prompt_loss_token_weights_exclude_fully_inactive_groups():
    group_indices = [0, 0, 1, 1]
    loss_masks = [
        [0, 0],
        [0],
        [1, 1, 1],
        [1],
    ]

    weights, num_prompt_groups = get_prompt_loss_token_weights(loss_masks, group_indices)

    assert weights == pytest.approx([0.0, 0.0, 1.0 / 4.0, 1.0 / 4.0])
    assert num_prompt_groups == 1


def test_normalize_rewards_for_training_zeros_inactive_samples_without_reward_normalization():
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        batch_level_normalization=False,
        grpo_std_normalization=False,
    )

    rewards = normalize_rewards_for_training(
        args,
        raw_rewards=[1.0, 2.0, 3.0],
        group_indices=[0, 0, 1],
        active_mask=[True, False, True],
    )

    assert rewards == pytest.approx([1.0, 0.0, 3.0])


def test_group_relative_focal_weights_match_zero_one_groups():
    weights = get_group_relative_focal_weights(
        raw_rewards=[1.0, 0.0, 1.0, 1.0],
        group_indices=[0, 0, 1, 1],
        gamma=1.0,
    )

    assert weights == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_group_relative_focal_weights_match_signed_reward_groups():
    weights = get_group_relative_focal_weights(
        raw_rewards=[1.0, -1.0, 1.0, -1.0],
        group_indices=[0, 0, 1, 1],
        gamma=2.0,
    )

    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_group_relative_focal_weights_cover_all_correct_and_all_incorrect_groups():
    all_correct_weights = get_group_relative_focal_weights(
        raw_rewards=[1.0, 1.0],
        group_indices=[0, 0],
        gamma=1.0,
    )
    all_incorrect_weights = get_group_relative_focal_weights(
        raw_rewards=[0.0, 0.0],
        group_indices=[0, 0],
        gamma=1.0,
    )

    assert all_correct_weights == pytest.approx([0.0, 0.0])
    assert all_incorrect_weights == pytest.approx([1.0, 1.0])


def test_group_relative_focal_weights_ignore_inactive_samples():
    weights = get_group_relative_focal_weights(
        raw_rewards=[1.0, 1.0, -1.0],
        group_indices=[0, 0, 0],
        gamma=1.0,
        active_mask=[True, False, True],
    )

    assert weights == pytest.approx([0.5, 0.5, 0.5])


def test_apply_group_relative_focal_weights_gamma_zero_is_noop():
    scaled_rewards = apply_group_relative_focal_weights_to_rewards(
        normalized_rewards=[2.0, -2.0, 0.0],
        raw_rewards=[1.0, -1.0, 1.0],
        group_indices=[0, 0, 1],
        gamma=0.0,
        active_mask=[True, True, False],
    )

    assert scaled_rewards == pytest.approx([2.0, -2.0, 0.0])


def test_apply_group_relative_focal_weights_none_is_noop():
    scaled_rewards = apply_group_relative_focal_weights_to_rewards(
        normalized_rewards=[0.5, -0.5],
        raw_rewards=[1.0, 0.0],
        group_indices=[0, 0],
        gamma=None,
    )

    assert scaled_rewards == pytest.approx([0.5, -0.5])


def test_apply_group_relative_focal_weights_scales_normalized_rewards():
    scaled_rewards = apply_group_relative_focal_weights_to_rewards(
        normalized_rewards=[2.0, -2.0, 1.0, -1.0],
        raw_rewards=[1.0, 0.0, 1.0, 1.0],
        group_indices=[0, 0, 1, 1],
        gamma=1.0,
    )

    assert scaled_rewards == pytest.approx([1.0, -1.0, 0.0, 0.0])


def test_group_relative_focal_weights_reject_non_binary_rewards():
    with pytest.raises(ValueError, match="binary raw rewards"):
        get_group_relative_focal_weights(
            raw_rewards=[1.0, 0.5],
            group_indices=[0, 0],
            gamma=1.0,
        )


def test_required_prompt_group_multiple_uses_lcm():
    assert get_required_prompt_group_multiple(6, 4, require_prompt_group_alignment=True) == 12
    assert get_required_prompt_group_multiple(6, 4, require_prompt_group_alignment=False) == 6


def test_should_force_per_token_loss_respects_prompt_aggregation():
    assert should_force_per_token_loss("cispo_loss", prompt_level_loss_aggregation=False) is True
    assert should_force_per_token_loss("dispo_loss", prompt_level_loss_aggregation=False) is True
    assert should_force_per_token_loss("cispo_loss", prompt_level_loss_aggregation=True) is False
    assert should_force_per_token_loss("dispo_loss", prompt_level_loss_aggregation=True) is False
    assert should_force_per_token_loss("policy_loss", prompt_level_loss_aggregation=False) is False


def test_compute_dapo_style_length_penalty_matches_piecewise_formula():
    assert compute_dapo_style_length_penalty(7, max_response_len=10, cache_len=2) == pytest.approx(0.0)
    assert compute_dapo_style_length_penalty(9, max_response_len=10, cache_len=2) == pytest.approx(-0.5)
    assert compute_dapo_style_length_penalty(10, max_response_len=10, cache_len=2) == pytest.approx(-1.0)
    assert compute_dapo_style_length_penalty(12, max_response_len=10, cache_len=2) == pytest.approx(-1.0)


def test_apply_length_penalty_only_shapes_correct_rewards():
    assert apply_length_penalty(
        1.0,
        9,
        length_penalty_type="dapo_style",
        max_response_len=10,
        cache_len=2,
    ) == pytest.approx(0.5)
    assert apply_length_penalty(
        -1.0,
        9,
        length_penalty_type="dapo_style",
        max_response_len=10,
        cache_len=2,
    ) == pytest.approx(-1.0)
    assert apply_length_penalty(
        1.0,
        9,
        length_penalty_type="none",
        max_response_len=10,
        cache_len=2,
    ) == pytest.approx(1.0)


def test_update_step_pass_rate_window_keeps_only_recent_steps():
    window = [0.25, 0.5]
    assert update_step_pass_rate_window(window, 0.75, window_steps=2) == pytest.approx([0.5, 0.75])
    assert update_step_pass_rate_window([], 1.0, window_steps=4) == pytest.approx([1.0])


def test_should_discard_prompt_requires_full_window_and_all_steps_above_threshold():
    assert should_discard_prompt([1.0, 1.0, 1.0], threshold=0.9, window_steps=4) is False
    assert should_discard_prompt([1.0, 0.875, 1.0, 1.0], threshold=0.9, window_steps=4) is False
    assert should_discard_prompt([1.0, 1.0, 1.0, 1.0], threshold=0.9, window_steps=4) is True


def test_current_eight_sample_threshold_requires_full_success():
    assert (7 / 8) < 0.9
    assert should_discard_prompt([7 / 8, 1.0, 1.0, 1.0], threshold=0.9, window_steps=4) is False
    assert should_discard_prompt([1.0, 1.0, 1.0, 1.0], threshold=0.9, window_steps=4) is True


def test_train_metric_normalizers_keep_aux_metrics_sample_based_under_prompt_aggregation():
    keys = ["loss", "pg_loss", "entropy_loss", "pg_clipfrac", "ppo_kl", "ratio"]
    counts = get_train_metric_normalizers(
        keys,
        num_samples=4,
        num_tokens=1024,
        num_prompt_groups=8,
        prompt_level_loss_aggregation=True,
        calculate_per_token_loss=False,
    )

    assert counts == [8, 8, 4, 4, 4, 4]


def test_train_metric_normalizers_preserve_ratio_near_one_for_prompt_aggregation_logging():
    keys = ["ratio"]
    counts = get_train_metric_normalizers(
        keys,
        num_samples=4,
        num_tokens=1024,
        num_prompt_groups=8,
        prompt_level_loss_aggregation=True,
        calculate_per_token_loss=False,
    )

    local_ratio_sum = 4.0  # four samples in the local microbatch, each with mean ratio ~= 1
    global_ratio_sum = local_ratio_sum * 8 * 2  # 8 microbatches, 2 DP ranks
    global_count = counts[0] * 8 * 2

    assert global_ratio_sum / global_count == pytest.approx(1.0)


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


def test_validate_scalerl_args_rejects_invalid_length_penalty_and_apf_configs():
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
        length_penalty_type="none",
        length_penalty_cache_len=None,
        adaptive_prompt_filter_threshold=None,
        adaptive_prompt_filter_window_steps=None,
    )

    with pytest.raises(ValueError, match="length-penalty-cache-len"):
        validate_scalerl_args(SimpleNamespace(**(base_args | dict(length_penalty_type="dapo_style"))))

    with pytest.raises(ValueError, match="must be set together"):
        validate_scalerl_args(
            SimpleNamespace(**(base_args | dict(adaptive_prompt_filter_threshold=0.9, adaptive_prompt_filter_window_steps=None)))
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_scalerl_args(
            SimpleNamespace(**(base_args | dict(adaptive_prompt_filter_threshold=1.1, adaptive_prompt_filter_window_steps=4)))
        )


def test_validate_scalerl_args_rejects_negative_group_relative_focal_gamma():
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

    with pytest.raises(ValueError, match="group-relative-focal-gamma"):
        validate_scalerl_args(SimpleNamespace(**(base_args | dict(group_relative_focal_gamma=-0.1))))
