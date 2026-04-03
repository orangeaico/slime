from types import SimpleNamespace

from examples.swe_bench.eval import eval_reward_router as router
from slime.utils.types import Sample


def _make_sample(*, session_id: str) -> Sample:
    sample = Sample(index=0, group_index=0)
    sample.session_id = session_id
    sample.metadata["instance_id"] = "pydata__xarray-1"
    sample.metadata["patch"] = "diff --git a/a b/a\n"
    return sample


def _make_args(**overrides):
    base = {}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_router_uses_eval_runner(monkeypatch):
    args = _make_args()
    s1 = _make_sample(session_id="s1")
    s2 = _make_sample(session_id="s2")
    calls = {"count": 0}

    def fake_eval(args, samples, group_run_id):
        calls["count"] += 1
        return {
            "s1": {"resolved": True, "status": "resolved", "error": "", "run_id": group_run_id},
            "s2": {"resolved": False, "status": "unresolved", "error": "", "run_id": group_run_id},
        }

    monkeypatch.setattr(router, "evaluate_group", fake_eval)
    out = router.evaluate_group_for_reward(args, [s1, s2])

    assert calls["count"] == 1
    assert out["s1"]["resolved"] is True
    assert out["s2"]["resolved"] is False
    assert out["s1"]["status"] == "resolved"
    assert out["s2"]["status"] == "unresolved"


def test_router_returns_router_error_when_eval_runner_fails(monkeypatch):
    args = _make_args()
    s1 = _make_sample(session_id="s1")

    def fake_eval(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(router, "evaluate_group", fake_eval)
    out = router.evaluate_group_for_reward(args, [s1])
    assert out["s1"]["resolved"] is False
    assert out["s1"]["status"] == "router_error"
