#!/usr/bin/env python3
"""
Local runner for model patch checks (no SWE-ReX).

For each instance from a JSONL and a predictions JSON file, this script:
  - Locates a local repo checkout (via --repo-dir or --repos-root + slug)
  - Ensures the base commit is present locally (fetches if needed)
  - Checks out the base commit directly in the repo (no worktrees)
  - Applies mirror.patch, then the model patch
  - Reverts modifications to test_*.py files introduced by the model patch
  - Applies test.patch, runs pytest once over FAIL_TO_PASS ∪ PASS_TO_PASS
  - Writes test_output.json and report.json under logs_dir/<bug_dir>

The inner test-run logic mirrors evaluation/run_model_patch_checks.py (inner-run)
but executes entirely on the local machine, similar in spirit to
verification/batch_local_runner.py.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import json
import logging
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from subprocess import CompletedProcess
import importlib


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GIT_APPLY_CMDS: list[list[str]] = [
    ["git", "apply", "--verbose"],
    ["git", "apply", "--verbose", "--reject"],
    ["patch", "--batch", "--fuzz=5", "-p1", "-i"],
]

_CURRENT_LOGGER: contextvars.ContextVar[Optional[logging.Logger]] = contextvars.ContextVar(
    "current_logger", default=None
)


def _get_logger() -> Optional[logging.Logger]:
    return _CURRENT_LOGGER.get()


def _log_command_output(cmd: Sequence[str], result: CompletedProcess) -> None:
    """Write subprocess stdout/stderr to the current log if present."""
    logger = _get_logger() or logging.getLogger(__name__)
    try:
        cmd_rendered = " ".join(shlex.quote(str(c)) for c in cmd)
        logger.info("[subprocess] %s", cmd_rendered)
        if result.stdout:
            logger.info(result.stdout.rstrip("\n"))
        if result.stderr:
            logger.error(result.stderr.rstrip("\n"))
    except Exception:
        pass


def log_info(msg: str) -> None:
    logger = _get_logger() or logging.getLogger(__name__)
    logger.info(msg)


def log_warning(msg: str) -> None:
    logger = _get_logger() or logging.getLogger(__name__)
    logger.warning(msg)


def log_error(msg: str) -> None:
    logger = _get_logger() or logging.getLogger(__name__)
    logger.error(msg)


def log_debug(msg: str) -> None:
    logger = _get_logger() or logging.getLogger(__name__)
    logger.debug(msg)


@contextlib.contextmanager
def instance_log_context(log_path: Path):
    """Configure per-instance logger writing to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"instance.{log_path.stem}.{log_path.parent.name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    token = _CURRENT_LOGGER.set(logger)
    try:
        yield logger
    finally:
        _CURRENT_LOGGER.reset(token)
        for handler in logger.handlers:
            try:
                handler.close()
            except Exception:
                pass
        logger.handlers.clear()


DEFAULT_PYTEST_CMD = "pytest -q --tb=no"


# --------------------------- utilities ---------------------------

def run(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _log_command_output(cmd, result)
    return result


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    kwargs = {"cwd": cwd}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["text"] = True
    result = subprocess.run(args, check=check, **kwargs)
    if capture_output:
        _log_command_output(args, result)
    return result


def _ensure_runtime_deps() -> None:
    """Ensure required packages are available for this script.

    - libcst is needed by generate_patch.py for CST transformations
    - pytest-json-report is preferred for robust test result parsing
    """
    def _have(mod: str) -> bool:
        try:
            importlib.import_module(mod)
            return True
        except Exception:
            return False

    # pytest-json-report (module name: pytest_jsonreport; pip name: pytest-json-report)
    if not _have("pytest_jsonreport"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest-json-report"], check=False)
        except Exception:
            pass

def repo_dir_from_slug(slug: str, repos_root: Path) -> Optional[Path]:
    try:
        _owner, name = slug.split("/", 1)
    except ValueError:
        name = slug
    candidate = repos_root / name
    if candidate.exists() and (candidate / ".git").exists():
        return candidate
    return None


def ensure_commit_available(repo: Path, commit: str) -> bool:
    ok = run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"], None)
    if ok.returncode == 0:
        return True
    fetch = run(["git", "-C", str(repo), "fetch", "--no-tags", "--depth=1", "origin", commit], None)
    if fetch.returncode == 0:
        ok2 = run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"], None)
        if ok2.returncode == 0:
            return True
    fetch_all = run(["git", "-C", str(repo), "fetch", "--all"], None)
    if fetch_all.returncode == 0:
        ok3 = run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"], None)
        return ok3.returncode == 0
    return False


def reset_repo(repo_root: Path, base_commit: str) -> bool:
    try:
        log_info(f"[git] Resetting current branch to base commit {base_commit}")
        run_command(["git", "-C", str(repo_root), "reset", "--hard", base_commit], cwd=repo_root)
        run_command(["git", "-C", str(repo_root), "clean", "-fd"], cwd=repo_root)
        return True
    except subprocess.CalledProcessError as exc:
        log_warning(f"[git] git reset/clean failed for {repo_root}: {exc}")
        return False


def load_jsonl_index(jsonl_path: Path) -> Dict[str, dict]:
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")
    index: Dict[str, dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSONL line {lineno} in {jsonl_path}: {exc}") from exc
            instance_id = data.get("instance_id")
            if not instance_id:
                raise SystemExit(f"Missing instance_id in JSONL line {lineno} ({jsonl_path})")
            index[instance_id] = data
    return index


def load_preds_index(preds_path: Path) -> Dict[str, dict]:
    if not preds_path.is_file():
        raise SystemExit(f"Predictions file not found: {preds_path}")
    with preds_path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse predictions JSON {preds_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Predictions JSON must map instance_id -> payload. Got {type(data).__name__}")
    return data


def _get_bug_dir(entry: dict) -> str:
    # Prefer md_dir name if present (consistent with other tooling), else instance_id
    md_dir = entry.get("md_dir")
    if md_dir:
        try:
            return Path(str(md_dir)).name
        except Exception:
            pass
    return str(entry.get("bug_dir") or entry.get("instance_id"))


# --------------------------- pytest helpers ---------------------------

def normalize_outcome(value: str | None) -> str:
    if not value:
        return "NONE"
    upper = value.upper()
    if upper == "XFAILED":
        return "SKIPPED"
    if upper == "XPASSED":
        return "PASSED"
    if upper == "ERROR":
        return "FAILED"
    return upper


def parse_pytest_json(report_path: Path) -> Dict[str, str]:
    if not report_path.is_file():
        return {}
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError:
        return {}
    tests = payload.get("tests")
    if not isinstance(tests, list):
        return {}
    results: Dict[str, str] = {}
    for entry in tests:
        if not isinstance(entry, dict):
            continue
        nodeid = entry.get("nodeid")
        if not nodeid:
            continue
        outcome = None
        call = entry.get("call")
        if isinstance(call, dict):
            outcome = call.get("outcome")
        if not outcome:
            outcome = entry.get("outcome")
        results[nodeid] = normalize_outcome(outcome)
    return results


def run_pytest_all(
    *,
    nodeids: Sequence[str],
    repo_root: Path,
    base_pytest_cmd: str,
    output_dir: Path,
    timeout_seconds: Optional[int] = None,
) -> Tuple[int, Dict[str, str], List[str], str, bool]:
    """Execute pytest once and return (rc, outcomes, missing_nodeids, log_path, timed_out)."""
    unique_nodeids: List[str] = []
    seen: set[str] = set()
    for nodeid in nodeids:
        if nodeid not in seen:
            seen.add(nodeid)
            unique_nodeids.append(nodeid)
    if not unique_nodeids:
        return 0, {}, [], str(output_dir / "test_output.json"), False

    log_path = output_dir / "test_output.json"
    cmd = shlex.split(base_pytest_cmd)
    cmd.extend(["--json-report", f"--json-report-file={log_path}"])
    cmd.extend(unique_nodeids)

    log_info(f"[pytest] Running {len(unique_nodeids)} tests")
    timeout_value: Optional[int] = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    timed_out = False
    # Capture pytest output to avoid emitting to console
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_value,
        )
        _log_command_output(cmd, result)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        result = CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)
        _log_command_output(cmd, result)
        log_error(f"[pytest] Timed out after {timeout_value}s while running {len(unique_nodeids)} tests.")
        fallback_payload = {
            "error": "timeout",
            "timeout_seconds": timeout_value,
            "returncode": 124,
            "requested_nodeids": unique_nodeids,
        }
        if stdout:
            fallback_payload["stdout_excerpt"] = stdout[-4000:]
        if stderr:
            fallback_payload["stderr_excerpt"] = stderr[-4000:]
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as fh:
                json.dump(fallback_payload, fh, indent=2)
                fh.write("\n")
        except Exception as write_exc:  # noqa: BLE001
            log_warning(f"[pytest] Failed to write timeout payload to {log_path}: {write_exc}")

    outcomes = parse_pytest_json(Path(log_path))
    missing: List[str] = []
    for nodeid in unique_nodeids:
        if nodeid not in outcomes:
            missing.append(nodeid)
            outcomes[nodeid] = "NONE"
    return int(result.returncode), outcomes, missing, str(log_path), timed_out


def summarise_outcomes(outcomes: Mapping[str, str]) -> Dict[str, int]:
    counter = Counter(outcomes.values())
    return dict(sorted(counter.items()))


def revert_test_file_changes(repo_root: Path, base_commit: str) -> None:
    diff_result = run_command(["git", "diff", "--name-only"], cwd=repo_root, capture_output=True)
    changed_files = [line.strip() for line in (diff_result.stdout or "").splitlines() if line.strip()]
    def _is_target_test(path_str: str) -> bool:
        """Return True for test_*.py files located somewhere under a tests/ directory."""
        path = Path(path_str)
        return (
            path.name.startswith("test_")
            and path.suffix == ".py"
            and "tests" in path.parts[:-1]
        )

    test_files = [p for p in changed_files if _is_target_test(p)]
    if not test_files:
        log_info("No test file changes detected before applying test patch.")
        return
    log_info(f"Reverting changes in test files: {', '.join(test_files)}")
    try:
        run_command(
            ["git", "checkout", base_commit, "--", *test_files],
            cwd=repo_root,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log_warning(f"Warning: failed to revert test file changes: {exc}")


class PatchManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._tmpdir = tempfile.TemporaryDirectory()
        self._applied: List[Tuple[str, Path]] = []
        self._counter = 0
        self._errors: Dict[str, str] = {}

    def apply(self, label: str, diff_text: str | None) -> bool:
        contents = diff_text or ""
        if not contents.strip():
            log_info(f"[patch:{label}] No content; skipping.")
            return False
        self._counter += 1
        filename = f"{self._counter:02d}_{label.replace(' ', '_')}.patch"
        patch_path = Path(self._tmpdir.name) / filename
        if not contents.endswith("\n"):
            contents = contents + "\n"
        patch_path.write_text(contents, encoding="utf-8")
        log_info(f"[patch:{label}] Applying {patch_path.name}")

        # Model patch: try multiple apply strategies (matches live/fork harness)
        if label == "model_patch":
            applied = False
            for cmd_template in GIT_APPLY_CMDS:
                cmd = cmd_template + [str(patch_path)]
                result = subprocess.run(cmd, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                _log_command_output(cmd, result)
                if result.returncode == 0:
                    log_info(f"[patch:{label}] Applied with command: {' '.join(cmd_template)}")
                    applied = True
                    break
                err_text = (result.stderr or result.stdout or "").strip()
                if err_text:
                    self._errors[label] = err_text
                log_warning(f"[patch:{label}] Failed with command: {' '.join(cmd_template)} (exit {result.returncode})")

            if not applied:
                try:
                    preview = patch_path.read_text(encoding="utf-8").splitlines()
                    preview_text = "\n".join(preview[:40])
                    log_info(f"[patch:{label}] Patch preview (first 40 lines):\n{preview_text}")
                except Exception as preview_exc:
                    log_warning(f"[patch:{label}] Unable to read patch for debugging: {preview_exc}")
                log_error(f"[patch:{label}] Failed to apply patch after trying all commands.")
                return False

            self._applied.append((label, patch_path))
            return True

        # Mirror/test patch: keep original 3-way apply with pre-check
        try:
            run_command(
                [
                    "git",
                    "apply",
                    "-3",
                    "--whitespace=fix",
                    "--recount",
                    "--check",
                    str(patch_path),
                ],
                cwd=self.repo_root,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            err_text = (getattr(exc, "stdout", None) or getattr(exc, "output", None) or getattr(exc, "stderr", None) or "").strip()
            if err_text:
                self._errors[label] = err_text
            try:
                preview = patch_path.read_text(encoding="utf-8").splitlines()
                preview_text = "\n".join(preview[:40])
                log_info(f"[patch:{label}] Patch preview (first 40 lines):\n{preview_text}")
            except Exception as preview_exc:
                log_warning(f"[patch:{label}] Unable to read patch for debugging: {preview_exc}")
            log_error(f"Failed to apply patch '{label}' (check): {exc}")
            return False

        try:
            run_command(
                [
                    "git",
                    "apply",
                    "-3",
                    "--whitespace=fix",
                    "--recount",
                    str(patch_path),
                ],
                cwd=self.repo_root,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            err_text = (getattr(exc, "stdout", None) or getattr(exc, "output", None) or getattr(exc, "stderr", None) or "").strip()
            if err_text:
                self._errors[label] = err_text
            try:
                preview = patch_path.read_text(encoding="utf-8").splitlines()
                preview_text = "\n".join(preview[:40])
                log_info(f"[patch:{label}] Patch preview (first 40 lines):\n{preview_text}")
            except Exception as preview_exc:
                log_warning(f"[patch:{label}] Unable to read patch for debugging: {preview_exc}")
            log_error(f"Failed to apply patch '{label}': {exc}")
            return False

        self._applied.append((label, patch_path))
        return True

    def get_error(self, label: str) -> Optional[str]:
        return self._errors.get(label)

    def close(self) -> None:
        self._tmpdir.cleanup()


def build_report_payload(
    instance_id: str,
    *,
    patch_is_none: bool,
    patch_exists: bool,
    patch_successfully_applied: bool,
    resolved: bool,
    tests_executed: bool,
    fail_success: Sequence[str],
    fail_failure: Sequence[str],
    pass_success: Sequence[str],
    pass_failure: Sequence[str],
) -> Dict[str, dict]:
    return {
        instance_id: {
            "patch_is_None": bool(patch_is_none),
            "patch_exists": bool(patch_exists),
            "patch_successfully_applied": bool(patch_successfully_applied),
            "resolved": bool(resolved),
            "tests_executed": bool(tests_executed),
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": list(fail_success),
                    "failure": list(fail_failure),
                },
                "PASS_TO_PASS": {
                    "success": list(pass_success),
                    "failure": list(pass_failure),
                },
                "FAIL_TO_FAIL": {"success": [], "failure": []},
                "PASS_TO_FAIL": {"success": [], "failure": []},
            },
        }
    }


def write_report(report_path: Path, payload: Dict[str, dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


# --------------------------- core per-instance runner ---------------------------

def _run_instance_in_repo(
    *,
    repo_dir: Path,
    base_commit: str,
    instance_id: str,
    instance_entry: dict,
    preds_entry: dict,
    pytest_cmd: str,
    pytest_timeout_seconds: int,
    out_dir: Path,
    p2p_same_file_only: bool,
) -> Tuple[int, bool, bool, Path, bool]:
    """Return (exit_code, resolved, tests_executed, report_path, timed_out)."""
    mirror_patch = instance_entry.get("mirror_patch", "")
    test_patch = instance_entry.get("test_patch", "")
    f2p: List[str] = instance_entry.get("FAIL_TO_PASS") or []
    p2p: List[str] = instance_entry.get("PASS_TO_PASS") or []

    if p2p_same_file_only and f2p and p2p:
        f2p_files = {str(nid).split("::", 1)[0] for nid in f2p if isinstance(nid, str)}
        p2p = [nid for nid in p2p if isinstance(nid, str) and nid.split("::", 1)[0] in f2p_files]

    model_patch = preds_entry.get("model_patch") if preds_entry else None
    patch_is_none = model_patch is None
    patch_exists = bool(model_patch and str(model_patch).strip())

    # Prepare output location
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    if not reset_repo(repo_dir, base_commit):
        return 1, False, False, report_path, False

    patch_manager = PatchManager(repo_dir)
    patch_successfully_applied = False
    timed_out = False
    try:
        # Apply patches
        patch_manager.apply("mirror_patch", mirror_patch)
        if patch_exists:
            patch_successfully_applied = patch_manager.apply("model_patch", model_patch)
        else:
            # Attempt anyway to keep the audit trail; apply() will no-op if empty
            patch_manager.apply("model_patch", model_patch)

        # Build combined nodeids
        combined_nodeids: List[str] = []
        seen: set[str] = set()
        for nid in f2p + p2p:
            if nid not in seen:
                seen.add(nid)
                combined_nodeids.append(nid)

        tests_executed = False
        if patch_successfully_applied and combined_nodeids:
            revert_test_file_changes(repo_dir, base_commit)
            patch_manager.apply("test_patch", test_patch)

            rc, outcomes, missing, log_path, timed_out = run_pytest_all(
                nodeids=combined_nodeids,
                repo_root=repo_dir,
                base_pytest_cmd=pytest_cmd,
                output_dir=out_dir,
                timeout_seconds=pytest_timeout_seconds,
            )
            # Consider tests executed only if the JSON report was produced (plugin active and pytest ran)
            tests_executed = Path(log_path).exists() and not timed_out
        else:
            rc, outcomes, missing, log_path = 0, {}, combined_nodeids[:], str(out_dir / "test_output.json")

        # Project outcomes into groups
        def project(group_list: List[str]) -> Tuple[List[str], List[str]]:
            success = []
            failure = []
            for nid in group_list:
                outcome = outcomes.get(nid, "NONE")
                if nid in missing:
                    failure.append(nid)
                elif outcome == "PASSED":
                    success.append(nid)
                else:
                    failure.append(nid)
            return success, failure

        fail_success, fail_failure = project(f2p)
        pass_success, pass_failure = project(p2p)
        resolved = not fail_failure and not pass_failure

        payload = build_report_payload(
            instance_id,
            patch_is_none=patch_is_none,
            patch_exists=patch_exists,
            patch_successfully_applied=patch_successfully_applied,
            resolved=resolved,
            tests_executed=tests_executed,
            fail_success=fail_success,
            fail_failure=fail_failure,
            pass_success=pass_success,
            pass_failure=pass_failure,
        )
        write_report(report_path, payload)
        log_info(f"Wrote report to {report_path}")
        return (0 if resolved else 1), resolved, tests_executed, report_path, timed_out
    finally:
        patch_manager.close()
        reset_repo(repo_dir, base_commit)


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local runner for model patch checks (no SWE-ReX)")
    p.add_argument("--jsonl", required=True, type=Path, help="Path to input JSONL with patches and nodeids")
    p.add_argument("--preds", required=True, type=Path, help="Path to predictions JSON with model_patchs")
    p.add_argument("--instance-id", required=True, help="Process only a single instance id")
    p.add_argument("--repos-root", default=".", help="Directory under which local repos are located")
    p.add_argument("--repo-dir", default=None, help="Explicit single repo path to use for all entries")
    # Match remote script naming: use logs-dir as the base for outputs
    p.add_argument("--logs-dir", default=None, help="Base directory to store per-instance outputs (like remote)")
    p.add_argument("--pytest-cmd", default=DEFAULT_PYTEST_CMD, help="Base pytest command")
    p.add_argument(
        "--pytest-timeout-seconds",
        type=int,
        default=0,
        help="Timeout (seconds) for the pytest invocation (0 = no timeout).",
    )
    p.add_argument("--skip-errors", action="store_true", help="Continue on error and record failure")
    p.add_argument(
        "--p2p-same-file-only",
        action="store_true",
        help="Limit PASS_TO_PASS nodeids to files also present in FAIL_TO_PASS for the instance",
    )
    return p.parse_args()


@dataclass
class InstanceOutcome:
    instance_id: str
    bug_dir: str
    exit_code: int
    resolved: bool
    tests_executed: bool
    report_path: Optional[Path]
    artifacts_dir: Path
    error: Optional[str] = None


def main() -> int:
    args = parse_args()
    _ensure_runtime_deps()
    jsonl_index = load_jsonl_index(args.jsonl)
    preds_index = load_preds_index(args.preds)

    iid = args.instance_id
    if iid not in jsonl_index:
        raise SystemExit(f"Instance {iid!r} not present in {args.jsonl}")
    if iid not in preds_index:
        raise SystemExit(f"Instance {iid!r} not present in {args.preds}")

    # Prepare base logs directory (match remote script semantics)
    script_dir = Path(__file__).resolve().parent
    default_logs = script_dir / "artifacts"
    logs_base = Path(args.logs_dir).expanduser().resolve() if args.logs_dir else default_logs
    logs_base.mkdir(parents=True, exist_ok=True)

    repos_root = Path(args.repos_root).resolve()
    explicit_repo: Optional[Path] = None
    if args.repo_dir:
        explicit_repo = Path(args.repo_dir).resolve()
        if not explicit_repo.exists() or not (explicit_repo / ".git").exists():
            raise SystemExit(f"--repo-dir is not a git repo: {explicit_repo}")

    def process(iid: str) -> InstanceOutcome:
        entry = jsonl_index[iid]
        preds = preds_index.get(iid, {})
        repo_slug = entry.get("repo") or entry.get("repo_name") or ""
        base_commit = entry.get("base_commit")
        bug_dir = _get_bug_dir(entry)
        art_dir = logs_base / bug_dir
        art_dir.mkdir(parents=True, exist_ok=True)
        log_path = art_dir / "run_instance.log"

        with instance_log_context(log_path):
            log_info(f"[instance:{iid}] Starting evaluation for {bug_dir}")

            # Predictions entry present?
            patch_value = preds.get("model_patch") if preds else None
            patch_is_empty = False
            if patch_value is None:
                patch_is_empty = True
            elif isinstance(patch_value, str):
                patch_is_empty = not patch_value.strip()
            if patch_is_empty:
                # Write placeholder artifacts
                log_info(f"[instance:{iid}] Empty model_patch; writing placeholder artifacts.")
                payload = build_report_payload(
                    iid,
                    patch_is_none=patch_value is None,
                    patch_exists=bool(patch_value and str(patch_value).strip()),
                    patch_successfully_applied=False,
                    resolved=False,
                    tests_executed=False,
                    fail_success=[],
                    fail_failure=[],
                    pass_success=[],
                    pass_failure=[],
                )
                write_report(art_dir / "report.json", payload)
                with (art_dir / "test_output.json").open("w", encoding="utf-8") as fh:
                    json.dump({"note": "No tests executed due to empty model_patch.", "instance_id": iid}, fh, indent=2)
                    fh.write("\n")
                return InstanceOutcome(iid, bug_dir, 1, False, False, art_dir / "report.json", art_dir, error="empty_model_patch")

            # Resolve repo
            repo = explicit_repo or repo_dir_from_slug(str(repo_slug), repos_root)
            if not repo or not (repo / ".git").exists():
                log_error(f"[instance:{iid}] Local repo not found for {repo_slug}")
                return InstanceOutcome(iid, bug_dir, 1, False, False, None, art_dir, error=f"local repo not found for {repo_slug}")
            if not base_commit:
                log_error(f"[instance:{iid}] base_commit missing in JSONL")
                return InstanceOutcome(iid, bug_dir, 1, False, False, None, art_dir, error="base_commit missing in JSONL")
            base_commit = str(base_commit)
            if not ensure_commit_available(repo, base_commit):
                log_error(f"[instance:{iid}] base commit not available: {base_commit}")
                return InstanceOutcome(iid, bug_dir, 1, False, False, None, art_dir, error=f"base commit not available: {base_commit}")

            try:
                exit_code, resolved, tests_executed, report_path, timed_out = _run_instance_in_repo(
                    repo_dir=repo,
                    base_commit=base_commit,
                    instance_id=iid,
                    instance_entry=entry,
                    preds_entry=preds,
                    pytest_cmd=args.pytest_cmd,
                    pytest_timeout_seconds=int(args.pytest_timeout_seconds),
                    out_dir=art_dir,
                    p2p_same_file_only=bool(args.p2p_same_file_only),
                )
                error_value = "pytest_timeout" if timed_out else None
                return InstanceOutcome(iid, bug_dir, int(exit_code), bool(resolved), bool(tests_executed), report_path, art_dir, error=error_value)
            except Exception as exc:  # noqa: BLE001
                logger = _get_logger()
                if logger:
                    logger.exception(f"[instance:{iid}] Unhandled exception during processing")
                return InstanceOutcome(iid, bug_dir, 1, False, False, None, art_dir, error=str(exc))

    out = process(iid)
    return 0 if out.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
