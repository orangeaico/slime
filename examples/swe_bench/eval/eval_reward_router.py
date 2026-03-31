from __future__ import annotations

import copy
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

_EVAL_LOCK = threading.Lock()
_VENDORED_EVAL_RUNNER_PATH = Path(__file__).resolve().parent / "run_patch_checks_docker.sh"


def sample_eval_key(sample: Sample, fallback_idx: int) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None:
        return f"index:{sample.index}"
    return f"fallback:{fallback_idx}"


def _default_result(*, run_id: str, status: str, error: str | None = None) -> dict[str, Any]:
    return {
        "resolved": False,
        "status": status,
        "error": error or "",
        "run_id": run_id,
    }


def _load_jsonl_index(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("instance_id")
            if isinstance(iid, str):
                index[iid] = row
    return index


def _build_group_run_id(samples: list[Sample]) -> str:
    group_marker = samples[0].group_index if samples and samples[0].group_index is not None else "nogroup"
    ts = int(time.time() * 1000)
    nonce = uuid.uuid4().hex[:8]
    return f"swe_eval_g{group_marker}_{ts}_{nonce}"


def evaluate_group(args, samples: list[Sample], group_run_id: str) -> dict[str, dict[str, Any]]:
    """Evaluate one prompt group using the configured SWE patch-check runner."""
    results: dict[str, dict[str, Any]] = {}
    for i, sample in enumerate(samples):
        key = sample_eval_key(sample, i)
        results[key] = _default_result(run_id=group_run_id, status="not_evaluated")

    jsonl_path_raw = getattr(args, "swe_eval_jsonl_path", None)
    if not jsonl_path_raw:
        for value in results.values():
            value.update(status="missing_config", error="swe_eval_jsonl_path is not set")
        return results

    jsonl_path = Path(str(jsonl_path_raw))
    runner_path = _VENDORED_EVAL_RUNNER_PATH
    if not jsonl_path.exists():
        for value in results.values():
            value.update(status="missing_config", error=f"eval jsonl not found: {jsonl_path}")
        return results
    if not runner_path.exists():
        for value in results.values():
            value.update(status="missing_config", error=f"vendored eval runner not found: {runner_path}")
        return results

    try:
        index = _load_jsonl_index(jsonl_path)
    except Exception as e:
        for value in results.values():
            value.update(status="config_error", error=f"failed to parse eval jsonl: {type(e).__name__}: {e}")
        return results

    candidates_to_run: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        key = sample_eval_key(sample, i)
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        instance_id = metadata.get("instance_id")
        patch = metadata.get("patch", "")

        if not isinstance(instance_id, str) or not instance_id:
            results[key].update(status="invalid_sample", error="missing metadata.instance_id")
            continue
        if not isinstance(patch, str) or not patch.strip():
            results[key].update(status="empty_patch", error="empty patch")
            continue
        if instance_id not in index:
            results[key].update(status="unknown_instance", error=f"instance {instance_id!r} not found in eval jsonl")
            continue

        alias_id = f"{instance_id}__cand{i}"
        row = copy.deepcopy(index[instance_id])
        row["instance_id"] = alias_id
        row["bug_dir"] = alias_id
        candidates_to_run.append(
            {
                "sample_key": key,
                "alias_id": alias_id,
                "row": row,
                "patch": patch,
            }
        )

    if not candidates_to_run:
        return results

    eval_logs_root = Path(str(getattr(args, "swe_eval_logs_root", "/tmp/swe_eval_reward_logs")))
    group_logs_dir = eval_logs_root / group_run_id
    group_logs_dir.mkdir(parents=True, exist_ok=True)

    eval_workers = int(getattr(args, "swe_eval_workers", 1) or 1)
    pytest_timeout_seconds = int(getattr(args, "swe_eval_pytest_timeout_seconds", 1800) or 1800)
    eval_timeout_seconds = int(getattr(args, "swe_eval_timeout_seconds", 3600) or 3600)
    router_inputs_dir = group_logs_dir / "_router_inputs"
    router_inputs_dir.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = router_inputs_dir / f"{group_run_id}_group_candidates.jsonl"
    tmp_preds = router_inputs_dir / f"{group_run_id}_group_preds.json"

    with tmp_jsonl.open("w", encoding="utf-8") as fh:
        for candidate in candidates_to_run:
            fh.write(json.dumps(candidate["row"], ensure_ascii=False) + "\n")

    preds_payload = {
        candidate["alias_id"]: {
            "instance_id": candidate["alias_id"],
            "model_patch": candidate["patch"],
            "model_name_or_path": "slime_swe_eval_reward",
        }
        for candidate in candidates_to_run
    }
    tmp_preds.write_text(json.dumps(preds_payload, ensure_ascii=False), encoding="utf-8")

    cmd = [
        "bash",
        str(runner_path),
        "--jsonl",
        str(tmp_jsonl),
        "--preds",
        str(tmp_preds),
        "--logs-dir",
        str(group_logs_dir),
        "--repo-dir",
        "/testbed",
        "--workers",
        str(max(1, eval_workers)),
        "--pytest-timeout-seconds",
        str(max(0, pytest_timeout_seconds)),
    ]

    try:
        subprocess.run(cmd, check=True, timeout=max(1, eval_timeout_seconds))
    except Exception as e:
        err = f"eval runner failed: {type(e).__name__}: {e}"
        for candidate in candidates_to_run:
            results[candidate["sample_key"]].update(status="runner_error", error=err)
        return results

    summary_path = group_logs_dir / "summary.json"
    if not summary_path.exists():
        for candidate in candidates_to_run:
            results[candidate["sample_key"]].update(status="missing_summary", error=f"summary not found: {summary_path}")
        return results

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        err = f"failed to parse summary: {type(e).__name__}: {e}"
        for candidate in candidates_to_run:
            results[candidate["sample_key"]].update(status="parse_error", error=err)
        return results

    resolved_ids = set(summary.get("resolved_ids", []) if isinstance(summary, dict) else [])
    unresolved_ids = set(summary.get("unresolved_ids", []) if isinstance(summary, dict) else [])
    empty_patch_ids = set(summary.get("empty_patch_ids", []) if isinstance(summary, dict) else [])
    error_ids = set(summary.get("error_ids", []) if isinstance(summary, dict) else [])

    for candidate in candidates_to_run:
        sample_key = candidate["sample_key"]
        alias_id = candidate["alias_id"]
        if alias_id in resolved_ids:
            results[sample_key].update(resolved=True, status="resolved", error="")
        elif alias_id in unresolved_ids:
            results[sample_key].update(resolved=False, status="unresolved", error="")
        elif alias_id in empty_patch_ids:
            results[sample_key].update(resolved=False, status="empty_patch", error="")
        elif alias_id in error_ids:
            results[sample_key].update(resolved=False, status="runner_error", error="instance in error_ids")
        else:
            results[sample_key].update(resolved=False, status="not_reported", error="candidate missing from summary")

    return results


def evaluate_group_for_reward(args, samples: list[Sample]) -> dict[str, dict[str, Any]]:
    """Evaluate one prompt group with SWE eval runner and return per-sample statuses."""
    results: dict[str, dict[str, Any]] = {}
    if not samples:
        return results

    for i, sample in enumerate(samples):
        key = sample_eval_key(sample, i)
        results[key] = _default_result(run_id="", status="not_evaluated")

    group_run_id = _build_group_run_id(samples=samples)
    for value in results.values():
        value["run_id"] = group_run_id

    with _EVAL_LOCK:
        try:
            runner_results = evaluate_group(args=args, samples=samples, group_run_id=group_run_id)
        except Exception as e:
            err = f"eval router error: {type(e).__name__}: {e}"
            for value in results.values():
                value.update(status="router_error", error=err, resolved=False)
            return results

    for i, sample in enumerate(samples):
        key = sample_eval_key(sample, i)
        if key in runner_results:
            results[key] = runner_results[key]
        else:
            results[key].update(status="missing_runner_result", error="runner result missing for sample", resolved=False)
    return results
