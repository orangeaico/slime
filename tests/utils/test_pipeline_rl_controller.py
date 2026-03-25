from slime.ray.pipeline_rl_controller import (
    get_pipeline_rl_oldest_step_gap,
    get_pipeline_rl_step_lead,
)


def test_pipeline_rl_step_lead_is_zero_without_outstanding_generation():
    assert get_pipeline_rl_step_lead(
        trainer_step=3,
        generation_step=None,
    ) == 0


def test_pipeline_rl_step_lead_uses_generator_progress_step():
    assert get_pipeline_rl_step_lead(
        trainer_step=5,
        generation_step=3,
    ) == 2


def test_pipeline_rl_oldest_step_gap_tracks_stalest_outstanding_generation():
    assert get_pipeline_rl_oldest_step_gap(
        trainer_step=5,
        oldest_outstanding_generation_step=2,
    ) == 3
