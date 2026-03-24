from __future__ import annotations

from typing import Any

import ray


def get_pipeline_rl_step_lead(
    *,
    trainer_step: int,
    generation_step: int | None,
) -> int:
    if generation_step is None:
        return 0
    return max(0, int(trainer_step) - int(generation_step))


def get_pipeline_rl_oldest_step_gap(
    *,
    trainer_step: int,
    oldest_outstanding_generation_step: int | None,
) -> int:
    if oldest_outstanding_generation_step is None:
        return 0
    return max(0, int(trainer_step) - int(oldest_outstanding_generation_step))


def is_pipeline_rl_update_allowed(
    *,
    target_trainer_step: int,
    generation_step: int | None,
    max_step_lead: int | None,
) -> bool:
    if max_step_lead is None:
        return True
    # The trainer should track generator progress, not the oldest lingering
    # request that may still be draining after a weight update.
    return (
        get_pipeline_rl_step_lead(
            trainer_step=target_trainer_step,
            generation_step=generation_step,
        )
        < max_step_lead
    )


@ray.remote(num_cpus=0)
class PipelineRLCoordinator:
    def __init__(self):
        self.trainer_step = 0
        self.oldest_outstanding_generation_step: int | None = None
        self.newest_outstanding_generation_step: int | None = None

    def set_trainer_step(self, trainer_step: int) -> None:
        self.trainer_step = int(trainer_step)

    def get_trainer_step(self) -> int:
        return self.trainer_step

    def report_outstanding(
        self,
        oldest_outstanding_generation_step: int | None,
        newest_outstanding_generation_step: int | None,
    ) -> None:
        self.oldest_outstanding_generation_step = oldest_outstanding_generation_step
        self.newest_outstanding_generation_step = newest_outstanding_generation_step

    def clear_outstanding(self) -> None:
        self.oldest_outstanding_generation_step = None
        self.newest_outstanding_generation_step = None

    def get_status(self) -> dict[str, Any]:
        return {
            "trainer_step": self.trainer_step,
            "oldest_outstanding_generation_step": self.oldest_outstanding_generation_step,
            "newest_outstanding_generation_step": self.newest_outstanding_generation_step,
            "generator_step": self.newest_outstanding_generation_step,
            "current_step_lead": get_pipeline_rl_step_lead(
                trainer_step=self.trainer_step,
                generation_step=self.newest_outstanding_generation_step,
            ),
            "current_oldest_step_gap": get_pipeline_rl_oldest_step_gap(
                trainer_step=self.trainer_step,
                oldest_outstanding_generation_step=self.oldest_outstanding_generation_step,
            ),
        }

    def is_update_allowed(self, target_trainer_step: int, max_step_lead: int | None) -> tuple[bool, dict[str, Any]]:
        status = self.get_status()
        return (
            is_pipeline_rl_update_allowed(
                target_trainer_step=target_trainer_step,
                generation_step=status["generator_step"],
                max_step_lead=max_step_lead,
            ),
            status,
        )
