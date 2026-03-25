from slime.utils.policy_version_utils import (
    build_policy_version_train_data,
    compute_policy_lag_metrics_from_rollout_data,
    compute_policy_version_metrics_from_samples,
    get_sample_policy_version_summary,
)
from slime.utils.types import Sample


def _make_sample(weight_versions):
    sample = Sample()
    sample.weight_versions = list(weight_versions)
    return sample


def test_get_sample_policy_version_summary_tracks_min_max_last_and_mixed_flag():
    summary = get_sample_policy_version_summary(_make_sample(["3", "5", "5"]))

    assert summary == {
        "policy_version_min": 3,
        "policy_version_max": 5,
        "policy_version_last": 5,
        "policy_version_span": 2,
        "policy_version_count": 3,
        "policy_version_mixed": 1,
    }


def test_build_policy_version_train_data_requires_versions_on_all_samples():
    samples = [_make_sample(["2"]), _make_sample([])]
    assert build_policy_version_train_data(samples) == {}


def test_compute_policy_version_metrics_from_samples_reports_mixed_fraction_only():
    samples = [_make_sample(["4"]), _make_sample(["4", "6"])]

    metrics = compute_policy_version_metrics_from_samples(samples)

    assert metrics == {"policy_version/mixed_frac": 0.5}


def test_compute_policy_version_metrics_from_samples_returns_empty_for_missing_versions():
    assert compute_policy_version_metrics_from_samples([_make_sample([])]) == {}


def test_compute_policy_lag_metrics_from_rollout_data_uses_current_policy_version():
    rollout_data = {
        "policy_version_last": [6, 7],
        "policy_version_min": [5, 7],
        "policy_version_max": [6, 7],
        "policy_version_span": [1, 0],
        "policy_version_mixed": [1, 0],
    }

    metrics = compute_policy_lag_metrics_from_rollout_data(rollout_data, current_policy_version=9)

    assert metrics["policy_lag/max"] == 4
    assert metrics["policy_version/mixed_frac"] == 0.5
