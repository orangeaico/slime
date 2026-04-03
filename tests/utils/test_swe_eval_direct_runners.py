import json
from pathlib import Path
from types import SimpleNamespace

from examples.swe_bench.eval.eval_reward_router import evaluate_group
from slime.utils.types import Sample


def _make_sample(*, idx: int, session_id: str, instance_id: str, patch: str) -> Sample:
    sample = Sample(index=idx, group_index=1)
    sample.session_id = session_id
    sample.metadata["instance_id"] = instance_id
    sample.metadata["patch"] = patch
    return sample


def test_eval_direct_runner_alias_roundtrip(tmp_path, monkeypatch):
    source_jsonl = tmp_path / "eval_source.jsonl"
    source_jsonl.write_text(
        json.dumps({"instance_id": "pydata__xarray-1", "image_name": "dummy"}) + "\n",
        encoding="utf-8",
    )

    args = SimpleNamespace(
        swe_eval_jsonl_path=str(source_jsonl),
        swe_eval_logs_root=str(tmp_path / "logs"),
        swe_eval_workers=1,
        swe_eval_pytest_timeout_seconds=1800,
        swe_eval_timeout_seconds=60,
    )

    s1 = _make_sample(
        idx=0,
        session_id="s1",
        instance_id="pydata__xarray-1",
        patch="diff --git a/a b/a\n",
    )
    s2 = _make_sample(
        idx=1,
        session_id="s2",
        instance_id="pydata__xarray-1",
        patch="diff --git a/b b/b\n",
    )

    def fake_run(cmd, check, timeout):
        logs_dir = Path(cmd[cmd.index("--logs-dir") + 1])
        logs_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "resolved_ids": ["pydata__xarray-1__cand0"],
            "unresolved_ids": ["pydata__xarray-1__cand1"],
            "empty_patch_ids": [],
            "error_ids": [],
        }
        (logs_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("examples.swe_bench.eval.eval_reward_router.subprocess.run", fake_run)

    out = evaluate_group(args, [s1, s2], "group_run")

    assert out["s1"]["resolved"] is True
    assert out["s1"]["status"] == "resolved"
    assert out["s2"]["resolved"] is False
    assert out["s2"]["status"] == "unresolved"
