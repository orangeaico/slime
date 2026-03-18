from types import SimpleNamespace

import pytest

import slime.rollout.data_source as rollout_data_source_module
from slime.rollout.data_source import ScaleRLRolloutDataSourceWithBuffer
from slime.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std_with_dapo_style
from slime.rollout.scalerl import (
    APF_METADATA_KEY,
    APF_STEP_PASS_RATES_KEY,
    PROMPT_ID_METADATA_KEY,
    SCALERL_ROLLOUT_METRICS_METADATA_KEY,
    compute_scalerl_metrics_from_samples,
    post_process_rewards_with_dapo_style,
    update_step_window_adaptive_prompt_filter,
)
from slime.utils.types import Sample


class FakeDataset:
    def __init__(self, samples):
        self.origin_samples = samples
        self.samples = list(samples)
        self.epoch_id = -1

    def shuffle(self, new_epoch_id):
        self.epoch_id = new_epoch_id
        self.samples = list(self.origin_samples)

    def __len__(self):
        return len(self.samples)


def _make_args(tmp_path, **overrides):
    base_args = dict(
        rollout_global_dataset=True,
        hf_checkpoint="dummy",
        dump_details=None,
        prompt_data=str(tmp_path / "unused.jsonl"),
        rollout_max_prompt_len=None,
        input_key="prompt",
        multimodal_keys=None,
        label_key="label",
        metadata_key="metadata",
        tool_key="tools",
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=42,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        adaptive_prompt_filter_threshold=0.9,
        adaptive_prompt_filter_window_steps=2,
        reward_key="score",
        length_penalty_type="dapo_style",
        length_penalty_cache_len=2,
        rollout_max_response_len=10,
        advantage_estimator="grpo",
        rewards_normalization=False,
        batch_level_normalization=False,
        grpo_std_normalization=False,
        save=str(tmp_path / "save"),
        load=None,
    )
    base_args.update(overrides)
    return SimpleNamespace(**base_args)


def _make_reward_sample(*, prompt_id: int, group_index: int, reward: float, response_length: int, index: int) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        response_length=response_length,
        reward={"score": reward},
        metadata={PROMPT_ID_METADATA_KEY: prompt_id},
    )


@pytest.fixture
def patched_dataset(monkeypatch):
    dataset = FakeDataset(
        [
            Sample(prompt="p0", label="0", metadata={}),
            Sample(prompt="p1", label="1", metadata={}),
            Sample(prompt="p2", label="2", metadata={}),
        ]
    )
    monkeypatch.setattr(rollout_data_source_module, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_data_source_module, "load_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_data_source_module, "Dataset", lambda *args, **kwargs: dataset)
    return dataset


def test_custom_data_source_assigns_prompt_ids_and_skips_retired_prompts(tmp_path, patched_dataset):
    args = _make_args(tmp_path)
    data_source = ScaleRLRolloutDataSourceWithBuffer(args)

    assert [sample.metadata[PROMPT_ID_METADATA_KEY] for sample in patched_dataset.origin_samples] == [0, 1, 2]

    first_metrics = data_source.record_step_pass_rates({0: 1.0})
    second_metrics = data_source.record_step_pass_rates({0: 1.0})
    assert first_metrics["newly_retired_prompt_count"] == pytest.approx(0.0)
    assert second_metrics["newly_retired_prompt_count"] == pytest.approx(1.0)
    assert second_metrics["retired_prompt_frac"] == pytest.approx(1.0 / 3.0)

    groups = data_source.get_samples(1)
    sampled_prompt_id = groups[0][0].metadata[PROMPT_ID_METADATA_KEY]
    assert sampled_prompt_id != 0
    skipped_metrics = data_source.record_step_pass_rates({})
    assert skipped_metrics["skip_prompt_draw_count"] == pytest.approx(1.0)


def test_custom_data_source_raises_when_no_eligible_prompts_remain(tmp_path, patched_dataset):
    args = _make_args(tmp_path)
    data_source = ScaleRLRolloutDataSourceWithBuffer(args)

    for prompt_id in [0, 1, 2]:
        data_source.record_step_pass_rates({prompt_id: 1.0})
    for prompt_id in [0, 1, 2]:
        data_source.record_step_pass_rates({prompt_id: 1.0})

    with pytest.raises(RuntimeError, match="No eligible fresh prompts remain"):
        data_source.get_samples(1)


def test_update_step_window_apf_aggregates_multiple_groups_per_prompt(tmp_path, patched_dataset):
    args = _make_args(tmp_path)
    data_source = ScaleRLRolloutDataSourceWithBuffer(args)

    all_samples = [
        [
            _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=10, index=0),
            _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=10, index=1),
        ],
        [
            _make_reward_sample(prompt_id=0, group_index=1, reward=1.0, response_length=10, index=2),
            _make_reward_sample(prompt_id=0, group_index=1, reward=-1.0, response_length=10, index=3),
        ],
        [
            _make_reward_sample(prompt_id=1, group_index=2, reward=-1.0, response_length=10, index=4),
            _make_reward_sample(prompt_id=1, group_index=2, reward=-1.0, response_length=10, index=5),
        ],
    ]

    update_step_window_adaptive_prompt_filter(args, all_samples, data_source.get_samples)

    step_pass_rates = data_source.metadata[APF_METADATA_KEY][APF_STEP_PASS_RATES_KEY]
    assert step_pass_rates[0] == pytest.approx([0.75])
    assert step_pass_rates[1] == pytest.approx([0.0])
    rollout_metrics = all_samples[0][0].metadata[SCALERL_ROLLOUT_METRICS_METADATA_KEY]
    assert rollout_metrics["retired_prompt_frac"] == pytest.approx(0.0)
    assert rollout_metrics["newly_retired_prompt_count"] == pytest.approx(0.0)
    assert rollout_metrics["skip_prompt_draw_count"] == pytest.approx(0.0)


def test_apf_state_survives_save_and_load(tmp_path, patched_dataset):
    args = _make_args(tmp_path)
    data_source = ScaleRLRolloutDataSourceWithBuffer(args)
    data_source.record_step_pass_rates({0: 1.0, 1: 0.5})
    data_source.record_step_pass_rates({0: 1.0})
    data_source.save(3)

    reloaded_args = _make_args(tmp_path, load=args.save)
    reloaded_data_source = ScaleRLRolloutDataSourceWithBuffer(reloaded_args)
    reloaded_data_source.load(3)

    assert reloaded_data_source.metadata[APF_METADATA_KEY][APF_STEP_PASS_RATES_KEY] == data_source.metadata[APF_METADATA_KEY][APF_STEP_PASS_RATES_KEY]
    assert reloaded_data_source.get_prompt_step_pass_rates(0) == pytest.approx([1.0, 1.0])
    assert reloaded_data_source.get_prompt_step_pass_rates(1) == pytest.approx([0.5])


def test_dapo_style_dynamic_filter_and_reward_post_process(tmp_path):
    args = _make_args(tmp_path)

    equal_length_group = [
        _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=10, index=0),
        _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=10, index=1),
    ]
    varied_length_group = [
        _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=8, index=2),
        _make_reward_sample(prompt_id=0, group_index=0, reward=1.0, response_length=9, index=3),
    ]

    assert check_reward_nonzero_std_with_dapo_style(args, equal_length_group).keep is False
    assert check_reward_nonzero_std_with_dapo_style(args, varied_length_group).keep is True

    raw_rewards, shaped_rewards = post_process_rewards_with_dapo_style(args, varied_length_group)
    assert raw_rewards == pytest.approx([1.0, 1.0])
    assert shaped_rewards == pytest.approx([1.0, 0.5])
    assert [sample.metadata["raw_reward"] for sample in varied_length_group] == pytest.approx([1.0, 1.0])
    rollout_metrics = compute_scalerl_metrics_from_samples(args, varied_length_group)
    assert rollout_metrics["scalerl/penalty_delta_mean"] == pytest.approx(-0.25)
    assert rollout_metrics["scalerl/penalized_sample_frac"] == pytest.approx(0.5)
