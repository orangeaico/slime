import abc
import copy
import logging
import os
import random
from pathlib import Path

import torch

from slime.rollout.scalerl import (
    APF_METADATA_KEY,
    APF_STEP_PASS_RATES_KEY,
    PROMPT_ID_METADATA_KEY,
    get_scalerl_prompt_id,
)
from slime.utils.data import Dataset
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.scalerl_utils import should_discard_prompt, update_step_pass_rate_window
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

APF_SKIP_DRAW_COUNT_KEY = "skip_prompt_draw_count"
APF_KEEP_DRAW_COUNT_KEY = "kept_retired_prompt_draw_count"
APF_RNG_STATE_KEY = "rng_state"
FRESH_PROMPT_PASS_METADATA_KEY = "fresh_prompt_passes"
FRESH_PROMPT_PASS_STOP_REASON = "all_active_prompts_reached_max_fresh_prompt_passes"


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            self._assign_prompt_ids()
        else:
            self.dataset = None

    def _assign_prompt_ids(self) -> None:
        if self.dataset is None:
            return

        for prompt_id, sample in enumerate(self.dataset.origin_samples):
            metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
            metadata[PROMPT_ID_METADATA_KEY] = prompt_id
            sample.metadata = metadata

    def _fresh_prompt_pass_early_stop_enabled(self) -> bool:
        return (
            self.dataset is not None
            and self.args.rollout_global_dataset
            and getattr(self.args, "max_fresh_prompt_passes", None) is not None
        )

    def _get_fresh_prompt_pass_state(self) -> dict[int, int]:
        return self.metadata.setdefault(FRESH_PROMPT_PASS_METADATA_KEY, {})

    def _record_fresh_prompt_draw(self, prompt_sample: Sample) -> None:
        if not self._fresh_prompt_pass_early_stop_enabled():
            return

        prompt_id = get_scalerl_prompt_id(prompt_sample)
        state = self._get_fresh_prompt_pass_state()
        state[prompt_id] = int(state.get(prompt_id, 0)) + 1

    def get_prompt_fresh_pass_count(self, prompt_id: int) -> int:
        return int(self._get_fresh_prompt_pass_state().get(prompt_id, 0))

    def get_prompt_fresh_pass_counts(self) -> dict[int, int]:
        if self.dataset is None:
            return {}
        return {
            prompt_id: self.get_prompt_fresh_pass_count(prompt_id)
            for prompt_id in (get_scalerl_prompt_id(sample) for sample in self.dataset.origin_samples)
        }

    def get_active_prompt_ids(self) -> list[int]:
        if self.dataset is None:
            return []
        return [get_scalerl_prompt_id(sample) for sample in self.dataset.origin_samples]

    def get_early_stop_status(self) -> dict[str, int | float | bool | str | None]:
        max_passes = getattr(self.args, "max_fresh_prompt_passes", None)
        if not self._fresh_prompt_pass_early_stop_enabled() or max_passes is None:
            return {
                "should_stop_after_training_batch": False,
                "reason": None,
                "active_prompt_count": 0,
                "min_active_prompt_fresh_passes": 0,
                "max_active_prompt_fresh_passes": 0,
                "active_prompt_completion_ratio": 0.0,
            }

        active_prompt_ids = self.get_active_prompt_ids()
        active_prompt_count = len(active_prompt_ids)
        active_prompt_pass_counts = [self.get_prompt_fresh_pass_count(prompt_id) for prompt_id in active_prompt_ids]
        completed_active_prompt_count = sum(pass_count >= max_passes for pass_count in active_prompt_pass_counts)
        should_stop = active_prompt_count == 0 or completed_active_prompt_count == active_prompt_count

        if active_prompt_pass_counts:
            min_active_prompt_fresh_passes = min(active_prompt_pass_counts)
            max_active_prompt_fresh_passes = max(active_prompt_pass_counts)
        else:
            min_active_prompt_fresh_passes = max_passes
            max_active_prompt_fresh_passes = max_passes

        return {
            "should_stop_after_training_batch": should_stop,
            "reason": FRESH_PROMPT_PASS_STOP_REASON if should_stop else None,
            "active_prompt_count": active_prompt_count,
            "min_active_prompt_fresh_passes": min_active_prompt_fresh_passes,
            "max_active_prompt_fresh_passes": max_active_prompt_fresh_passes,
            "active_prompt_completion_ratio": (
                completed_active_prompt_count / active_prompt_count if active_prompt_count > 0 else 1.0
            ),
        }

    def _build_groups_from_prompt_samples(self, prompt_samples: list[Sample]) -> list[list[Sample]]:
        groups = []
        for prompt_sample in prompt_samples:
            self._record_fresh_prompt_draw(prompt_sample)
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        return self._build_groups_from_prompt_samples(prompt_samples)

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

    def __len__(self) -> int:
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


class ScaleRLRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        self._apf_random = random.Random(self.args.rollout_seed)

    def _adaptive_prompt_filter_enabled(self) -> bool:
        return (
            getattr(self.args, "adaptive_prompt_filter_threshold", None) is not None
            and getattr(self.args, "adaptive_prompt_filter_window_steps", None) is not None
        )

    def _get_apf_state(self) -> dict[str, dict[int, list[float]]]:
        state = self.metadata.setdefault(APF_METADATA_KEY, {})
        state.setdefault(APF_STEP_PASS_RATES_KEY, {})
        state.setdefault(APF_SKIP_DRAW_COUNT_KEY, 0)
        state.setdefault(APF_KEEP_DRAW_COUNT_KEY, 0)
        state.setdefault(APF_RNG_STATE_KEY, self._apf_random.getstate())
        return state

    def _get_apf_drop_prob(self) -> float:
        return float(getattr(self.args, "adaptive_prompt_filter_drop_prob", 1.0))

    def _store_apf_rng_state(self) -> None:
        state = self._get_apf_state()
        state[APF_RNG_STATE_KEY] = self._apf_random.getstate()

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if not self._adaptive_prompt_filter_enabled():
            return

        state = self._get_apf_state()
        rng_state = state.get(APF_RNG_STATE_KEY)
        if rng_state is not None:
            self._apf_random.setstate(rng_state)

    def get_prompt_step_pass_rates(self, prompt_id: int) -> list[float]:
        state = self._get_apf_state()
        prompt_step_pass_rates = state[APF_STEP_PASS_RATES_KEY]
        return list(prompt_step_pass_rates.get(prompt_id, []))

    def is_prompt_retired(self, prompt_id: int) -> bool:
        if not self._adaptive_prompt_filter_enabled():
            return False

        return should_discard_prompt(
            self.get_prompt_step_pass_rates(prompt_id),
            threshold=self.args.adaptive_prompt_filter_threshold,
            window_steps=self.args.adaptive_prompt_filter_window_steps,
        )

    def get_active_prompt_ids(self) -> list[int]:
        if self.dataset is None:
            return []
        if not self._adaptive_prompt_filter_enabled():
            return super().get_active_prompt_ids()
        return [
            get_scalerl_prompt_id(sample)
            for sample in self.dataset.origin_samples
            if not self.is_prompt_retired(get_scalerl_prompt_id(sample))
        ]

    def record_step_pass_rates(self, prompt_step_pass_rates: dict[int, float]) -> dict[str, float]:
        state = self._get_apf_state()
        if not self._adaptive_prompt_filter_enabled():
            metrics = {
                "skip_prompt_draw_count": float(state.get(APF_SKIP_DRAW_COUNT_KEY, 0)),
                "kept_retired_prompt_draw_count": float(state.get(APF_KEEP_DRAW_COUNT_KEY, 0)),
            }
            state[APF_SKIP_DRAW_COUNT_KEY] = 0
            state[APF_KEEP_DRAW_COUNT_KEY] = 0
            return metrics

        windows = state[APF_STEP_PASS_RATES_KEY]
        previous_retired = {prompt_id for prompt_id in windows if self.is_prompt_retired(prompt_id)}
        for prompt_id, step_pass_rate in prompt_step_pass_rates.items():
            current_window = windows.get(prompt_id, [])
            windows[prompt_id] = update_step_pass_rate_window(
                current_window,
                step_pass_rate,
                window_steps=self.args.adaptive_prompt_filter_window_steps,
            )

        retired_prompt_ids = {prompt_id for prompt_id in windows if self.is_prompt_retired(prompt_id)}
        total_prompt_count = len(self.dataset.origin_samples) if self.dataset is not None else 0
        skipped_prompt_draw_count = float(state.get(APF_SKIP_DRAW_COUNT_KEY, 0))
        kept_retired_prompt_draw_count = float(state.get(APF_KEEP_DRAW_COUNT_KEY, 0))
        state[APF_SKIP_DRAW_COUNT_KEY] = 0
        state[APF_KEEP_DRAW_COUNT_KEY] = 0

        return {
            "retired_prompt_frac": (len(retired_prompt_ids) / total_prompt_count) if total_prompt_count > 0 else 0.0,
            "newly_retired_prompt_count": float(len(retired_prompt_ids - previous_retired)),
            "skip_prompt_draw_count": skipped_prompt_draw_count,
            "kept_retired_prompt_draw_count": kept_retired_prompt_draw_count,
        }

    def _has_eligible_fresh_prompts(self) -> bool:
        if self.dataset is None or not self._adaptive_prompt_filter_enabled():
            return True
        if any(not self.is_prompt_retired(get_scalerl_prompt_id(sample)) for sample in self.dataset.origin_samples):
            return True
        return len(self.dataset.origin_samples) > 0 and self._get_apf_drop_prob() < 1.0

    def _should_skip_prompt_draw(self, prompt_id: int) -> bool:
        if not self.is_prompt_retired(prompt_id):
            return False

        drop_prob = self._get_apf_drop_prob()
        if drop_prob <= 0.0:
            return False
        if drop_prob >= 1.0:
            return True

        should_skip = self._apf_random.random() < drop_prob
        self._store_apf_rng_state()
        return should_skip

    def _advance_dataset_cursor(self) -> Sample:
        assert self.dataset is not None
        sample = self.dataset.samples[self.sample_offset]
        self.sample_offset += 1
        if self.sample_offset >= len(self.dataset):
            self.sample_offset = 0
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        return sample

    def _get_next_fresh_prompt_sample(self) -> Sample:
        if self.dataset is None:
            return Sample()

        if not self._has_eligible_fresh_prompts():
            raise RuntimeError("No eligible fresh prompts remain after adaptive prompt filtering.")

        while True:
            prompt_sample = self._advance_dataset_cursor()
            prompt_id = get_scalerl_prompt_id(prompt_sample)
            if not self._should_skip_prompt_draw(prompt_id):
                if self.is_prompt_retired(prompt_id):
                    state = self._get_apf_state()
                    state[APF_KEEP_DRAW_COUNT_KEY] = int(state.get(APF_KEEP_DRAW_COUNT_KEY, 0)) + 1
                return prompt_sample
            state = self._get_apf_state()
            state[APF_SKIP_DRAW_COUNT_KEY] = int(state.get(APF_SKIP_DRAW_COUNT_KEY, 0)) + 1

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)
        if num_samples == 0:
            return samples

        prompt_samples = [self._get_next_fresh_prompt_sample() for _ in range(num_samples)]
        samples += self._build_groups_from_prompt_samples(prompt_samples)
        return samples


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
