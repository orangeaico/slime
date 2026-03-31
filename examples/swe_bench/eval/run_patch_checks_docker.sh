#!/usr/bin/env bash
set -euo pipefail

# Ensure worker containers are stopped on interruption
declare -a CONTAINER_NAMES=()
declare -a WAITER_PIDS=()
PROGRESS_FIFO=""
PROGRESS_PID=""
PROGRESS_FD=""
INTERRUPTED=0

mark_interrupted() {
  INTERRUPTED=1
  if [[ -n "${LOGS_DIR:-}" ]]; then
    touch "$LOGS_DIR/.interrupted" 2>/dev/null || true
  fi
}

is_interrupted() {
  [[ "${INTERRUPTED:-0}" -eq 1 ]] && return 0
  if [[ -n "${LOGS_DIR:-}" && -f "$LOGS_DIR/.interrupted" ]]; then
    return 0
  fi
  return 1
}
cleanup_shards() {
  if [[ "${CLEANUP_SHARDS:-0}" -ne 1 ]]; then
    return 0
  fi
  # Only clean auto-generated shards dir under logs/.image_groups
  if [[ -n "${SHARDS_DIR:-}" && "$SHARDS_DIR" != "/" && "$SHARDS_DIR" == *"/.image_groups/"* ]]; then
    rm -rf "$SHARDS_DIR" 2>/dev/null || true
    # Best-effort prune empty .image_groups parent
    local parent
    parent="$(dirname "$SHARDS_DIR")"
    if [[ -n "$parent" && "$parent" != "/" ]]; then
      rmdir "$parent" 2>/dev/null || true
    fi
  fi
  # Best-effort prune .image_groups under logs dir even if SHARDS_DIR is missing
  if [[ -n "${LOGS_DIR:-}" && "$LOGS_DIR" != "/" ]]; then
    rm -rf "$LOGS_DIR/.image_groups" 2>/dev/null || true
  fi
}

stop_workers() {
  if [[ ${#CONTAINER_NAMES[@]} -gt 0 ]]; then
    echo "Stopping ${#CONTAINER_NAMES[@]} patch-check container(s)..."
    docker rm -f "${CONTAINER_NAMES[@]}" >/dev/null 2>&1 || true
    # Best-effort CPU release even if allocator functions are defined later
    if declare -F release_cpus >/dev/null 2>&1; then
      release_cpus "${CONTAINER_NAMES[@]}"
    fi
  fi
  if declare -F stop_progress >/dev/null 2>&1; then
    stop_progress
  fi
  # Terminate waiter processes quickly
  if [[ ${#WAITER_PIDS[@]} -gt 0 ]]; then
    for pid in "${WAITER_PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
  fi
}
on_sigint() { echo "Interrupted (SIGINT). Cleaning up..."; mark_interrupted; stop_progress_fast; stop_workers; cleanup_shards; exit 130; }
on_sigterm() { echo "Terminated (SIGTERM). Cleaning up..."; mark_interrupted; stop_progress_fast; stop_workers; cleanup_shards; exit 143; }
on_exit() { cleanup_shards; }
trap on_sigint INT
trap on_sigterm TERM
trap on_exit EXIT

# Launch docker containers to run patch checks per instance.
# Each container handles a single instance_id using the Docker image specified
# in the JSONL row (keys: `image_name` / `image` / `docker_image`).
# Use --workers to cap the maximum number of concurrent containers.

usage() {
  cat <<'EOF'
Usage: examples/swe_bench/eval/run_patch_checks_docker.sh \
  --jsonl /host/path/instances.jsonl \
  --preds /host/path/preds.json \
  --logs-dir /host/path/logs \
  [--repos-root /host/path/repos | --repo-dir /container/repo] \
  [--workers N] [--name-prefix PREFIX] [--extra-mount "-v /host:/container"] \
  [--shards-dir DIR] [--default-image IMAGE] [--pytest-cmd CMD] [--pytest-timeout-seconds SECONDS] [--p2p-same-file-only] [--skip-existing|--no-skip-existing]

Notes:
  - JSONL, PREDS, and LOGS_DIR paths must be visible inside the container via mounts.
  - If using --repos-root, mount it to the same absolute path inside containers.
  - Alternatively pass --repo-dir to use a single repo path inside the container.
    Default repo-dir inside the container is /testbed if neither --repo-dir nor --repos-root is provided.
  - The script launches one container per instance_id using the image from each JSONL row.
  - Use --default-image to handle instances missing an image field; without it, such rows are skipped.
  - --shards-dir controls where the per-instance manifest is written on the host.
  - --skip-existing skips instances that already have a well-formed report.json under logs-dir (default).
EOF
}

JSONL=""
PREDS=""
LOGS_DIR=""
REPOS_ROOT=""
REPO_DIR=""
WORKERS=1
EXTRA_MOUNT=""
SHARDS_DIR=""
SHARDS_DIR_USER=0
CLEANUP_SHARDS=0
DEFAULT_IMAGE=""
# Restrict PASS_TO_PASS to tests in the same files as FAIL_TO_PASS
P2P_SAME_FILE_ONLY=0
# Optional: run setup inside containers before invoking the runner (e.g., pip installs)
SETUP_CMD=""
PYTEST_CMD=""
PYTEST_TIMEOUT_SECONDS=0
# Skip instances with existing reports to avoid unnecessary container startup
SKIP_EXISTING=1
# Debug/diagnostics
DEBUG=0
# Keep containers after completion for inspection
KEEP_CONTAINERS=0
NO_SUMMARY=0
# CPU pinning config (shared across concurrent runs/users)
CPU_POOL="${CPU_POOL:-0-23}"  # allowed host cores
CPU_STATE="${CPU_STATE:-/tmp/swe_patch_cpu_state/state.json}"  # shared state file
CPU_LOCK="${CPU_LOCK:-/tmp/swe_patch_cpu_state/state.lock}"
# Wait time (seconds) when acquiring the CPU lock; prevents silent indefinite hangs
CPU_LOCK_WAIT_SECS="${CPU_LOCK_WAIT_SECS:-30}"
# Derive a per-user default container name prefix to avoid cross-user collisions
DEFAULT_USER_FOR_NAME=$(id -un 2>/dev/null || whoami 2>/dev/null || echo user)

# Ensure shared CPU state/lock files exist with permissive perms so multiple users can coordinate
ensure_cpu_state_files() {
  local state="$CPU_STATE"
  local lock="$CPU_LOCK"
  local dir=""
  [[ -n "$state" ]] && dir="$(dirname "$state")"
  if [[ -n "$dir" ]]; then
    mkdir -p "$dir"
    chmod 777 "$dir" 2>/dev/null || true
  fi
  if [[ -n "$state" && ! -e "$state" ]]; then
    : > "$state" 2>/dev/null || true
  fi
  if [[ -n "$lock" && ! -e "$lock" ]]; then
    : > "$lock" 2>/dev/null || true
  fi
  if [[ -n "$state" ]]; then
    chmod 666 "$state" 2>/dev/null || true
  fi
  if [[ -n "$lock" ]]; then
    chmod 666 "$lock" 2>/dev/null || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jsonl) JSONL="$2"; shift 2;;
    --preds) PREDS="$2"; shift 2;;
    --logs-dir) LOGS_DIR="$2"; shift 2;;
    --repos-root) REPOS_ROOT="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --extra-mount) EXTRA_MOUNT="$2"; shift 2;;
    --shards-dir) SHARDS_DIR="$2"; SHARDS_DIR_USER=1; shift 2;;
    --default-image) DEFAULT_IMAGE="$2"; shift 2;;
    --setup-cmd) SETUP_CMD="$2"; shift 2;;
    --pytest-cmd) PYTEST_CMD="$2"; shift 2;;
    --pytest-timeout-seconds) PYTEST_TIMEOUT_SECONDS="$2"; shift 2;;
    --p2p-same-file-only) P2P_SAME_FILE_ONLY=1; shift 1;;
    --skip-existing) SKIP_EXISTING=1; shift 1;;
    --no-skip-existing) SKIP_EXISTING=0; shift 1;;
    --no-summary) NO_SUMMARY=1; shift 1;;
    --debug) DEBUG=1; shift 1;;
    --keep-containers) KEEP_CONTAINERS=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$JSONL" || -z "$PREDS" || -z "$LOGS_DIR" ]]; then
  echo "Missing required args." >&2
  usage; exit 2
fi

# Prepare shared CPU state/lock files (permits cross-user coordination)
ensure_cpu_state_files

# Generate a unique run identifier so container names never collide across runs
# Allow override via RUN_ID env var for reproducibility if desired
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$-$RANDOM}"
echo "Run ID: $RUN_ID"

# Default shards dir: isolate by logs dir + run id to avoid cross-run collisions.
if [[ -z "$SHARDS_DIR" || "$SHARDS_DIR" == "AUTO" ]]; then
  SHARDS_DIR="$LOGS_DIR/.image_groups/$RUN_ID"
  CLEANUP_SHARDS=1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  set -x
  echo "[debug] JSONL=$JSONL PREDS=$PREDS LOGS_DIR=$LOGS_DIR REPOS_ROOT=$REPOS_ROOT REPO_DIR=$REPO_DIR WORKERS=$WORKERS"
fi

# Verify input files exist on host
if [[ ! -f "$JSONL" ]]; then
  echo "JSONL not found on host: $JSONL" >&2; exit 2
fi
if [[ ! -f "$PREDS" ]]; then
  echo "Preds JSON not found on host: $PREDS" >&2; exit 2
fi

# Ensure required paths exist on host
mkdir -p "$LOGS_DIR"
# Clear stale interrupted marker from previous runs
rm -f "$LOGS_DIR/.interrupted" 2>/dev/null || true

# Common mounts: workspace (current repo) and logs dir
# Derive a stable workspace root from the script location so mounting works regardless of caller CWD.
# Prefer mounting repository root so runner path stays stable.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RUNNER_REL_PATH="examples/swe_bench/eval/run_model_patch_checks_local.py"
if [[ -z "$WORKSPACE_ROOT" || ! -d "$WORKSPACE_ROOT" ]]; then
  WORKSPACE_ROOT=$(pwd)
fi
# Fallback to legacy script-local mount layout.
if [[ ! -f "$WORKSPACE_ROOT/$RUNNER_REL_PATH" ]]; then
  WORKSPACE_ROOT=$(cd "$SCRIPT_DIR" && pwd)
  RUNNER_REL_PATH="run_model_patch_checks_local.py"
fi
if [[ "$DEBUG" -eq 1 ]]; then
  echo "[debug] Using WORKSPACE_ROOT=$WORKSPACE_ROOT (script dir: $SCRIPT_DIR, runner: $RUNNER_REL_PATH)"
fi
# Translate container-visible paths (e.g. /root/repo/...) to host paths for docker bind mounts.
REPO_MOUNT_POINT="/root/repo"
REPO_MOUNT_SOURCE="$(awk '$5=="/root/repo"{print $4; exit}' /proc/self/mountinfo 2>/dev/null || true)"
to_host_path() {
  local path="$1"
  if [[ -n "${REPO_MOUNT_SOURCE:-}" && "$path" == "$REPO_MOUNT_POINT"* ]]; then
    echo "${REPO_MOUNT_SOURCE}${path#$REPO_MOUNT_POINT}"
  else
    echo "$path"
  fi
}

WORKSPACE_ROOT_HOST="$(to_host_path "$WORKSPACE_ROOT")"
WORKDIR_MOUNT="-v $WORKSPACE_ROOT_HOST:/workspace -w /workspace"
LOGS_DIR_HOST="$(to_host_path "$LOGS_DIR")"
LOGS_MOUNT="-v $LOGS_DIR_HOST:$LOGS_DIR"

# Sanity check: ensure workspace contains the expected evaluation scripts
RUNNER_SOURCE_PATH="$WORKSPACE_ROOT/$RUNNER_REL_PATH"
if [[ ! -f "$RUNNER_SOURCE_PATH" ]]; then
  echo "Error: Expected script not found at $RUNNER_SOURCE_PATH" >&2
  echo "       Check that WORKSPACE_ROOT is correct or run with --debug to inspect mounts." >&2
  exit 2
fi

# Copy runner into logs dir so candidate containers can execute it from a known-good mount.
RUNNER_SHARED_DIR="$LOGS_DIR/.runner"
RUNNER_SHARED_PATH="$RUNNER_SHARED_DIR/run_model_patch_checks_local.py"
mkdir -p "$RUNNER_SHARED_DIR"
cp -f "$RUNNER_SOURCE_PATH" "$RUNNER_SHARED_PATH"
chmod a+r "$RUNNER_SHARED_PATH" >/dev/null 2>&1 || true
if [[ ! -f "$RUNNER_SHARED_PATH" ]]; then
  echo "Error: Failed to stage runner at $RUNNER_SHARED_PATH" >&2
  exit 2
fi

# Auto-mount parents of JSONL and PREDS to the same absolute paths in the container
JSONL_ABS=$(readlink -f "$JSONL" 2>/dev/null || python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$JSONL")
PREDS_ABS=$(readlink -f "$PREDS" 2>/dev/null || python3 -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$PREDS")
JSONL_HOST_ABS="$(to_host_path "$JSONL_ABS")"
PREDS_HOST_ABS="$(to_host_path "$PREDS_ABS")"
JSONL_DIR=$(dirname "$JSONL_ABS")
PREDS_DIR=$(dirname "$PREDS_ABS")
JSONL_DIR_HOST=$(dirname "$JSONL_HOST_ABS")
PREDS_DIR_HOST=$(dirname "$PREDS_HOST_ABS")
JSONL_MOUNT="-v $JSONL_DIR_HOST:$JSONL_DIR:ro"
PREDS_MOUNT=""
if [[ "$PREDS_DIR" != "$JSONL_DIR" ]]; then
  PREDS_MOUNT="-v $PREDS_DIR_HOST:$PREDS_DIR:ro"
fi

# Repos root mount (optional). Must be identical path inside container for git operations.
REPOS_MOUNT=""
REPOS_ARG=""
if [[ -n "$REPOS_ROOT" ]]; then
  REPOS_ROOT_HOST="$(to_host_path "$REPOS_ROOT")"
  REPOS_MOUNT="-v $REPOS_ROOT_HOST:$REPOS_ROOT"
  REPOS_ARG="--repos-root $REPOS_ROOT"
fi

# Default repo-dir inside the container is /testbed unless repos-root is specified
REPO_DIR_ARG=""
if [[ -z "$REPO_DIR" && -z "$REPOS_ROOT" ]]; then
  REPO_DIR="/testbed"
fi
if [[ -n "$REPO_DIR" ]]; then
  REPO_DIR_ARG="--repo-dir $REPO_DIR"
fi

# Build per-instance manifest from the JSONL file using python3, but only for
# instance_ids present in the predictions file (intersection of JSONL and PREDS).
# Ensure SHARDS_DIR is absolute and, if relative, anchor it under WORKSPACE_ROOT.
if [[ "${SHARDS_DIR#/}" == "$SHARDS_DIR" ]]; then
  SHARDS_DIR="$WORKSPACE_ROOT/$SHARDS_DIR"
fi
mkdir -p "$SHARDS_DIR"

INSTANCES_MANIFEST="$SHARDS_DIR/instances.manifest"
python3 - "$JSONL" "$SHARDS_DIR" "$DEFAULT_IMAGE" "$PREDS_ABS" <<'PY'
import json, sys, os
from pathlib import Path

jsonl_path, out_dir, default_image, preds_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# Load allowed instance IDs from predictions
try:
    with open(preds_path, 'r', encoding='utf-8') as pf:
        preds = json.load(pf)
    if isinstance(preds, dict):
        allowed = set(preds.keys())
    elif isinstance(preds, list):
        allowed = {p.get('instance_id') for p in preds if isinstance(p, dict) and p.get('instance_id')}
    else:
        raise ValueError('Unsupported predictions format')
except Exception as exc:
    print(f"Failed to read predictions at {preds_path}: {exc}", file=sys.stderr)
    sys.exit(2)

manifest_lines = []
missing_image = 0
filtered_out = 0
dupes = 0
seen = set()
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = obj.get('instance_id')
        if not isinstance(iid, str) or iid not in allowed:
            filtered_out += 1
            continue
        if iid in seen:
            dupes += 1
            continue
        img = obj.get('image_name') or obj.get('image') or obj.get('docker_image') or ''
        if not img:
            if default_image:
                img = default_image
            else:
                missing_image += 1
                continue
        def bug_dir_for(entry):
            md_dir = entry.get("md_dir")
            if md_dir:
                try:
                    return Path(str(md_dir)).name
                except Exception:
                    pass
            return str(entry.get("bug_dir") or entry.get("instance_id"))
        seen.add(iid)
        manifest_lines.append((img, iid, bug_dir_for(obj)))

with open(os.path.join(out_dir, 'instances.manifest'), 'w', encoding='utf-8') as mf:
    for img, iid, bug_dir in manifest_lines:
        mf.write(f"{img}\t{iid}\t{bug_dir}\n")

print(
    f"instances={len(manifest_lines)} filtered_out_not_in_preds={filtered_out} "
    f"skipped_no_image={missing_image} duplicates={dupes}"
)
PY

if [[ ! -f "$INSTANCES_MANIFEST" ]]; then
  echo "Failed to produce instances manifest at $INSTANCES_MANIFEST" >&2
  exit 2
fi

# Keep full manifest for summary bookkeeping even if we skip existing reports.
FULL_INSTANCES_MANIFEST="$INSTANCES_MANIFEST"
RUN_INSTANCES_MANIFEST="$INSTANCES_MANIFEST"
SKIPPED_INSTANCES_MANIFEST=""

# Optionally skip instances that already have a well-formed report.json
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  FILTERED_MANIFEST="$SHARDS_DIR/instances.filtered.manifest"
  SKIPPED_INSTANCES_MANIFEST="$SHARDS_DIR/instances.skipped.manifest"
  python3 - "$FULL_INSTANCES_MANIFEST" "$LOGS_DIR" "$FILTERED_MANIFEST" "$SKIPPED_INSTANCES_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
logs_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
skipped_out = Path(sys.argv[4])

skipped = 0
kept = 0
skipped_out.parent.mkdir(parents=True, exist_ok=True)

def has_valid_report(bug_dir: str, iid: str) -> bool:
    report_path = logs_dir / bug_dir / "report.json"
    if not report_path.is_file():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    entry = payload.get(iid)
    return isinstance(entry, dict)

with manifest_path.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out, skipped_out.open("w", encoding="utf-8") as skipped_fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        img, iid = parts[0], parts[1]
        bug_dir = parts[2] if len(parts) > 2 and parts[2] else iid
        if has_valid_report(bug_dir, iid):
            skipped += 1
            skipped_fh.write(f"{img}\t{iid}\t{bug_dir}\n")
            continue
        out.write(f"{img}\t{iid}\t{bug_dir}\n")
        kept += 1

print(f"skip_existing=1 skipped={skipped} remaining={kept}")
PY
  if [[ -f "$FILTERED_MANIFEST" ]]; then
    RUN_INSTANCES_MANIFEST="$FILTERED_MANIFEST"
  fi
fi

echo "Discovered instances (by image):"
awk -F '\t' '{count[$1]++} END {for (img in count) printf("  - %s (%d items)\n", img, count[img])}' \
  "$RUN_INSTANCES_MANIFEST" | sort

# If no instances discovered, exit early with helpful message
INSTANCE_COUNT=$(awk 'END{print NR}' "$RUN_INSTANCES_MANIFEST")
if [[ -z "$INSTANCE_COUNT" || "$INSTANCE_COUNT" -eq 0 ]]; then
  FULL_INSTANCE_COUNT=$(awk 'END{print NR}' "$FULL_INSTANCES_MANIFEST")
  if [[ "$SKIP_EXISTING" -eq 1 && -n "$FULL_INSTANCE_COUNT" && "$FULL_INSTANCE_COUNT" -gt 0 ]]; then
    echo "All instances were skipped because valid report.json files already exist under --logs-dir."
    echo "Nothing to run."
    exit 0
  fi
  echo "No instances were produced."
  echo "Possible causes:"
  echo " - Predictions and JSONL have no overlapping instance_ids"
  echo " - Missing image_name/image/docker_image in JSONL and no --default-image provided"
  echo " - Manifest generation failed; check stderr above"
  exit 2
fi

# Launch containers per instance
declare -A CPU_ALLOCATIONS=()

# Expand a CPU list/range like 0-3,5 into a space-separated list
expand_cpus() {
  local spec="$1"
  python3 - "$spec" <<'PY'
import sys
spec = sys.argv[1]
def expand(part):
    if '-' in part:
        a,b = part.split('-',1)
        return list(range(int(a), int(b)+1))
    return [int(part)]
out=[]
for part in spec.split(','):
    part=part.strip()
    if not part:
        continue
    out.extend(expand(part))
print(" ".join(str(x) for x in out))
PY
}

allocate_cpu() {
  local name="$1"
  local cpu
  cpu=$(flock -x -w "$CPU_LOCK_WAIT_SECS" "$CPU_LOCK" python3 - "$name" "$CPU_POOL" "$CPU_STATE" <<'PY'
import json, os, sys, subprocess

name, pool_spec, state_path = sys.argv[1:4]

def expand(spec):
    out=[]
    for part in spec.split(','):
        part=part.strip()
        if not part:
            continue
        if '-' in part:
            a,b = part.split('-',1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return out

pool = expand(pool_spec)
if not pool:
    print("", end="")
    sys.exit(1)

def running_usage():
    try:
        ids = subprocess.check_output(["docker", "ps", "-q"], text=True).strip().split()
    except Exception:
        return {}
    if not ids:
        return {}
    try:
        lines = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.Name}} {{.HostConfig.CpusetCpus}}", *ids],
            text=True,
        ).splitlines()
    except Exception:
        return {}
    usage={}
    for line in lines:
        parts=line.strip().split(None,1)
        if not parts:
            continue
        cpuspec=parts[1] if len(parts)>1 else ""
        for c in expand(cpuspec):
            usage[c]=usage.get(c,0)+1
    return usage

running = running_usage()
counts = {c: running.get(c, 0) for c in pool}

# load state
alloc={}
if os.path.isfile(state_path):
    try:
        with open(state_path,"r",encoding="utf-8") as f:
            obj=json.load(f)
        if isinstance(obj, dict):
            alloc={k:int(v) for k,v in obj.get("allocations",{}).items() if isinstance(v,int)}
    except Exception:
        alloc={}

# prune stale entries (containers no longer exist)
try:
    existing=set(subprocess.check_output(["docker","ps","-a","--format","{{.Names}}"], text=True).split())
except Exception:
    existing=set()
alloc={k:v for k,v in alloc.items() if not existing or k in existing}

if name in alloc:
    print(alloc[name], end="")
    tmp=state_path+".tmp"
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump({"allocations": alloc}, f)
        os.replace(tmp, state_path)
        os.chmod(state_path, 0o666)
    except Exception:
        pass
    sys.exit(0)

for v in alloc.values():
    if v in counts:
        counts[v]=counts.get(v,0)+1

# choose least used CPU (prefers free)
best=None
best_count=None
for c in sorted(pool):
    cnt=counts.get(c,0)
    if best is None or cnt < best_count:
        best=c
        best_count=cnt

if best is None:
    print("", end="")
    sys.exit(1)

alloc[name]=best
tmp=state_path+".tmp"
try:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    try:
        os.chmod(os.path.dirname(state_path), 0o777)
    except Exception:
        pass
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump({"allocations": alloc}, f)
    os.replace(tmp, state_path)
    os.chmod(state_path, 0o666)
except Exception:
    pass

print(best, end="")
PY
) || {
    echo "Failed to acquire CPU lock within ${CPU_LOCK_WAIT_SECS}s for $name; aborting." >&2
    stop_workers
    exit 1
  }
  echo "$cpu"
}

release_cpus() {
  local names=("$@")
  [[ ${#names[@]} -eq 0 ]] && return
  flock -x -w "$CPU_LOCK_WAIT_SECS" "$CPU_LOCK" python3 - "$CPU_STATE" "${names[@]}" <<'PY'
import json, sys, os
state_path=sys.argv[1]
names=set(sys.argv[2:])
if not os.path.isfile(state_path):
    sys.exit(0)
try:
    with open(state_path,"r",encoding="utf-8") as f:
        obj=json.load(f)
    alloc=obj.get("allocations",{}) if isinstance(obj, dict) else {}
except Exception:
    sys.exit(0)
changed=False
for n in list(alloc.keys()):
    if n in names:
        alloc.pop(n, None)
        changed=True
if not changed:
    sys.exit(0)
tmp=state_path+".tmp"
try:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    try:
        os.chmod(os.path.dirname(state_path), 0o777)
    except Exception:
        pass
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump({"allocations": alloc}, f)
    os.replace(tmp, state_path)
    os.chmod(state_path, 0o666)
except Exception:
    pass
PY
}

start_progress() {
  local total="$1"
  local is_tty=0
  [[ -t 1 ]] && is_tty=1
  if [[ "$is_tty" -ne 1 ]]; then
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  PROGRESS_FIFO="$(mktemp -u "$SHARDS_DIR/progress_fifo.XXXXXX")"
  mkfifo "$PROGRESS_FIFO"
  exec 9<> "$PROGRESS_FIFO"
  PROGRESS_FD=9
  python3 - "$PROGRESS_FIFO" "$total" <<'PY' &
import sys

fifo = sys.argv[1]
try:
    total = int(sys.argv[2])
except Exception:
    total = 0

use_tqdm = True
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    use_tqdm = False

count = 0
resolved = 0
unresolved = 0
error = 0
bar = None
if use_tqdm:
    bar = tqdm(total=total, desc="patch-check", unit="inst", dynamic_ncols=True)
else:
    print(f"Progress: 0/{total}", file=sys.stderr, flush=True)

with open(fifo, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line == "DONE":
            break
        step = 1
        status = ""
        if "|" in line:
            step_part, status = line.split("|", 1)
            try:
                step = int(step_part)
            except Exception:
                step = 1
        else:
            try:
                step = int(line)
            except Exception:
                step = 1
        if status == "R":
            resolved += 1
        elif status == "U":
            unresolved += 1
        elif status == "E":
            error += 1
        count += step
        if bar:
            bar.update(step)
            bar.set_postfix({
                "resolved": resolved,
                "unresolved": unresolved,
                "error": error,
            }, refresh=False)
        else:
            print(f"Progress: {count}/{total}", file=sys.stderr, flush=True)

if bar:
    bar.close()
PY
  PROGRESS_PID=$!
}

progress_tick() {
  if [[ -n "$PROGRESS_PID" ]] && ! kill -0 "$PROGRESS_PID" 2>/dev/null; then
    stop_progress
    return 0
  fi
  local status="${1:-}"
  if [[ -n "$PROGRESS_FD" ]]; then
    if [[ -n "$status" ]]; then
      printf '1|%s\n' "$status" >&"$PROGRESS_FD" 2>/dev/null || true
    else
      printf '1\n' >&"$PROGRESS_FD" 2>/dev/null || true
    fi
  fi
}

stop_progress() {
  if [[ -n "$PROGRESS_FD" ]]; then
    printf 'DONE\n' >&"$PROGRESS_FD" 2>/dev/null || true
    if [[ -n "${PROGRESS_PID:-}" ]]; then
      wait "$PROGRESS_PID" 2>/dev/null || true
    fi
    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    PROGRESS_FD=""
    rm -f "$PROGRESS_FIFO" 2>/dev/null || true
    PROGRESS_FIFO=""
    PROGRESS_PID=""
  fi
}

stop_progress_fast() {
  if [[ -n "$PROGRESS_FD" ]]; then
    printf 'DONE\n' >&"$PROGRESS_FD" 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    PROGRESS_FD=""
  fi
  if [[ -n "${PROGRESS_PID:-}" ]]; then
    kill -TERM "$PROGRESS_PID" 2>/dev/null || true
    kill -KILL "$PROGRESS_PID" 2>/dev/null || true
    PROGRESS_PID=""
  fi
  if [[ -n "${PROGRESS_FIFO:-}" ]]; then
    rm -f "$PROGRESS_FIFO" 2>/dev/null || true
    PROGRESS_FIFO=""
  fi
}

init_summary() {
  python3 - "$INSTANCES_MANIFEST" "$LOGS_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
logs_dir = Path(sys.argv[2])
state_path = logs_dir / ".summary_state.json"
summary_path = logs_dir / "summary.json"

submitted_ids = []
bug_dirs = {}
with manifest_path.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        _, iid = parts[0], parts[1]
        bug_dir = parts[2] if len(parts) > 2 and parts[2] else iid
        if iid not in bug_dirs:
            submitted_ids.append(iid)
            bug_dirs[iid] = bug_dir

state = {
    "submitted_ids": submitted_ids,
    "bug_dirs": bug_dirs,
    "status": {},
}
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

summary = {
    "submitted_instances": len(submitted_ids),
    "completed_instances": 0,
    "resolved_instances": 0,
    "unresolved_instances": 0,
    "empty_patch_instances": 0,
    "error_instances": 0,
    "submitted_ids": submitted_ids,
    "completed_ids": [],
    "resolved_ids": [],
    "unresolved_ids": [],
    "empty_patch_ids": [],
    "error_ids": [],
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
}

update_summary() {
  local iid="$1"
  local lock="$LOGS_DIR/.summary.lock"
  local status
  status=$(flock -x "$lock" python3 - "$LOGS_DIR" "$iid" <<'PY'
import json
import sys
from pathlib import Path

logs_dir = Path(sys.argv[1])
iid = sys.argv[2]
state_path = logs_dir / ".summary_state.json"
summary_path = logs_dir / "summary.json"

if not state_path.is_file():
    # No state; nothing to update
    print("E")
    raise SystemExit(0)

state = json.loads(state_path.read_text(encoding="utf-8"))
submitted_ids = state.get("submitted_ids", [])
bug_dirs = state.get("bug_dirs", {})
status_map = state.get("status", {})

bug_dir = bug_dirs.get(iid, iid)
report_path = logs_dir / bug_dir / "report.json"

patch_exists = True
tests_executed = False
resolved = False

if report_path.is_file():
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        entry = None
        if isinstance(payload, dict):
            entry = payload.get(iid)
            if entry is None and payload:
                entry = next(iter(payload.values()))
        if isinstance(entry, dict):
            patch_exists = bool(entry.get("patch_exists", True))
            tests_executed = bool(entry.get("tests_executed", False))
            resolved = bool(entry.get("resolved", False))
    except Exception:
        pass

status_map[iid] = {
    "patch_exists": patch_exists,
    "tests_executed": tests_executed,
    "resolved": resolved,
}
state["status"] = status_map
state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

completed_ids = []
resolved_ids = []
unresolved_ids = []
empty_patch_ids = []
error_ids = []

for sid in submitted_ids:
    entry = status_map.get(sid)
    if not entry:
        continue
    p_exists = bool(entry.get("patch_exists"))
    t_exec = bool(entry.get("tests_executed"))
    res = bool(entry.get("resolved"))
    if not p_exists:
        empty_patch_ids.append(sid)
        continue
    completed_ids.append(sid)
    if not t_exec:
        error_ids.append(sid)
        continue
    if res:
        resolved_ids.append(sid)
    else:
        unresolved_ids.append(sid)

summary = {
    "submitted_instances": len(submitted_ids),
    "completed_instances": len(completed_ids),
    "resolved_instances": len(resolved_ids),
    "unresolved_instances": len(unresolved_ids),
    "empty_patch_instances": len(empty_patch_ids),
    "error_instances": len(error_ids),
    "submitted_ids": submitted_ids,
    "completed_ids": completed_ids,
    "resolved_ids": resolved_ids,
    "unresolved_ids": unresolved_ids,
    "empty_patch_ids": empty_patch_ids,
    "error_ids": error_ids,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

if not patch_exists:
    print("X")
elif not tests_executed:
    print("E")
elif resolved:
    print("R")
else:
    print("U")
PY
) || status="E"
  echo "$status"
}

wait_for_slot() {
  local max="$1"
  if [[ -z "$max" || "$max" -le 0 ]]; then
    return 0
  fi
  while [[ ${#WAITER_PIDS[@]} -ge "$max" ]]; do
    wait -n || true
    local still_running=()
    for pid in "${WAITER_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still_running+=("$pid")
      fi
    done
    WAITER_PIDS=("${still_running[@]}")
  done
}

if [[ "$NO_SUMMARY" -eq 0 ]]; then
  INSTANCES_MANIFEST="$FULL_INSTANCES_MANIFEST"
  init_summary
  # Pre-populate summary for instances skipped due to existing reports
  if [[ -n "$SKIPPED_INSTANCES_MANIFEST" && -f "$SKIPPED_INSTANCES_MANIFEST" ]]; then
    while IFS=$'\t' read -r _IMG SKIP_IID _SKIP_BUGDIR; do
      if [[ -n "$SKIP_IID" ]]; then
        update_summary "$SKIP_IID" >/dev/null || true
      fi
    done < "$SKIPPED_INSTANCES_MANIFEST"
  fi
fi
start_progress "$INSTANCE_COUNT"
echo "Launching per-instance containers (max parallel: $WORKERS)"
INSTANCE_INDEX=0
while IFS=$'\t' read -r IMAGE IID BUG_DIR; do
  INSTANCE_INDEX=$((INSTANCE_INDEX+1))
  wait_for_slot "$WORKERS"
  IMG_SLUG=$(echo "$IMAGE" | sed 's#[/:@]#_#g')
  IID_SLUG=$(echo "$IID" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')
  if [[ -z "$IID_SLUG" ]]; then
    IID_SLUG="instance-${INSTANCE_INDEX}"
  fi
  IID_SLUG="${IID_SLUG:0:40}"
  IID_HASH=$(printf "%s" "$IID" | sha1sum | cut -c1-10)
  # Use instance-id + run-id + hash to keep names stable and unique.
  NAME="${IID_SLUG}_${RUN_ID}_${IID_HASH}_${DEFAULT_USER_FOR_NAME}"
  if [[ ${#NAME} -gt 120 ]]; then
    suffix="_${RUN_ID}_${IID_HASH}_${DEFAULT_USER_FOR_NAME}"
    max_iid_len=$((120 - ${#suffix}))
    if [[ $max_iid_len -lt 8 ]]; then
      max_iid_len=8
    fi
    IID_SLUG="${IID_SLUG:0:$max_iid_len}"
    NAME="${IID_SLUG}${suffix}"
  fi
  # Keep output clean; container name/instance are available in logs if needed.
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  release_cpus "$NAME"  # clear stale allocation for reused names
  CPU_CHOICE=$(allocate_cpu "$NAME")
  if [[ -z "$CPU_CHOICE" ]]; then
    echo "Failed to allocate CPU for $NAME; aborting." >&2
    stop_workers
    exit 1
  fi
  docker run -d \
    --cpuset-cpus="${CPU_CHOICE}" \
    --name "$NAME" \
    $LOGS_MOUNT \
    $JSONL_MOUNT \
    $PREDS_MOUNT \
    $REPOS_MOUNT \
    $EXTRA_MOUNT \
    -e PYTHONUNBUFFERED=1 \
    -e P2P_SAME_FILE_ONLY="$P2P_SAME_FILE_ONLY" \
    -e "SETUP_CMD=$SETUP_CMD" \
    -e "INSTANCE_ID=$IID" \
    "$IMAGE" \
    bash -lc "chmod -R 777 '$LOGS_DIR' >/dev/null 2>&1 || true; \
      if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi; \
      P2P_FLAG=\"\"; if [ \"\${P2P_SAME_FILE_ONLY:-0}\" = \"1\" ]; then P2P_FLAG=\"--p2p-same-file-only\"; fi; \
      if [ -n \"$SETUP_CMD\" ]; then echo \"[setup] running: $SETUP_CMD\"; eval \"$SETUP_CMD\"; fi; \
      if [ -z \"$PYTEST_CMD\" ]; then \
        if [ -x /testbed/.venv/bin/pytest ]; then DET_PYTEST=\"/testbed/.venv/bin/pytest -q --tb=no\"; else DET_PYTEST=\"pytest -q --tb=no\"; fi; \
      else \
        DET_PYTEST=\"$PYTEST_CMD\"; \
      fi; \
      echo Running: \$PY \"$RUNNER_SHARED_PATH\" --jsonl '$JSONL' --preds '$PREDS' --logs-dir '$LOGS_DIR' --instance-id \"\$INSTANCE_ID\" --skip-errors $REPOS_ARG $REPO_DIR_ARG --pytest-cmd \"\$DET_PYTEST\" --pytest-timeout-seconds '$PYTEST_TIMEOUT_SECONDS' \$P2P_FLAG; \
      \$PY \"$RUNNER_SHARED_PATH\" \
      --jsonl '$JSONL' \
      --preds '$PREDS' \
      --logs-dir '$LOGS_DIR' \
      --instance-id \"\$INSTANCE_ID\" \
      --skip-errors \
      $REPOS_ARG $REPO_DIR_ARG \
      --pytest-cmd \"\$DET_PYTEST\" --pytest-timeout-seconds '$PYTEST_TIMEOUT_SECONDS' \
      \$P2P_FLAG" >/dev/null
  CONTAINER_NAMES+=("$NAME")
  CPU_ALLOCATIONS["$NAME"]="$CPU_CHOICE"
  # Wait per-container and cleanup as soon as it exits
  IID_LOCAL="$IID"
  BUG_DIR_LOCAL="$BUG_DIR"
  IMAGE_LOCAL="$IMAGE"
  (
    EXIT_CODE=$(docker wait "$NAME" 2>/dev/null || true)
    ART_DIR="$LOGS_DIR/$BUG_DIR_LOCAL"
    REPORT_PATH="$ART_DIR/report.json"
    if [[ ! -f "$REPORT_PATH" ]]; then
      if is_interrupted; then
        exit 0
      fi
      mkdir -p "$ART_DIR" || true
      chmod -R u+rwX,g+rwX "$ART_DIR" "$LOGS_DIR" 2>/dev/null || true
      if [[ ! -w "$ART_DIR" ]]; then
        docker run --rm $LOGS_MOUNT "$IMAGE_LOCAL" bash -lc "chmod -R 777 '$LOGS_DIR' || true" >/dev/null 2>&1 || true
      fi
      TMP_LOG=$(mktemp)
      docker logs --tail 200 "$NAME" > "$TMP_LOG" 2>&1 || true
      if ! mv "$TMP_LOG" "$ART_DIR/container.log" 2>/dev/null; then
        chmod -R u+rwX,g+rwX "$ART_DIR" "$LOGS_DIR" 2>/dev/null || true
        mv "$TMP_LOG" "$ART_DIR/container.log" 2>/dev/null || true
      fi
      python3 - "$ART_DIR" "$NAME" "$EXIT_CODE" <<'PY' || {
import json
import sys
from pathlib import Path

art_dir = Path(sys.argv[1])
name = sys.argv[2]
exit_code = sys.argv[3]

payload = {
    "note": "report.json missing after container exit",
    "container_name": name,
    "exit_code": exit_code,
}
out = art_dir / "host_error.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
        chmod -R u+rwX,g+rwX "$ART_DIR" "$LOGS_DIR" 2>/dev/null || true
        python3 - "$ART_DIR" "$NAME" "$EXIT_CODE" <<'PY' || true
import json
import sys
from pathlib import Path

art_dir = Path(sys.argv[1])
name = sys.argv[2]
exit_code = sys.argv[3]

payload = {
    "note": "report.json missing after container exit",
    "container_name": name,
    "exit_code": exit_code,
}
out = art_dir / "host_error.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
      }
    fi
    if [[ "$KEEP_CONTAINERS" -eq 1 ]]; then
      :
    else
      docker rm -f "$NAME" >/dev/null 2>&1 || true
    fi
    release_cpus "$NAME"
    if ! is_interrupted; then
      if [[ "$NO_SUMMARY" -eq 0 ]]; then
        status=$(update_summary "$IID_LOCAL")
        progress_tick "$status"
      else
        progress_tick ""
      fi
    fi
  ) &
  WAITER_PIDS+=("$!")
done < "$RUN_INSTANCES_MANIFEST"

if [[ ${#CONTAINER_NAMES[@]} -eq 0 ]]; then
  echo "No containers were started."
  echo "Check instances manifest at $INSTANCES_MANIFEST and docker run/pull errors above."
  exit 2
fi
echo "Waiting for all containers to finish..."
if [[ ${#WAITER_PIDS[@]} -gt 0 ]]; then
  wait "${WAITER_PIDS[@]}" || true
fi
stop_progress
# Relax permissions inside a container to avoid host write issues
FIRST_IMAGE=$(awk -F '\t' 'NR==1{print $1}' "$INSTANCES_MANIFEST")
if [[ -n "$FIRST_IMAGE" ]]; then
  docker run --rm \
    $LOGS_MOUNT \
    "$FIRST_IMAGE" \
    bash -lc "chmod -R 777 '$LOGS_DIR' || true"
fi

# Proactively remove worker containers after completion to avoid leftovers
if [[ ${#CONTAINER_NAMES[@]} -gt 0 ]]; then
  if [[ "$KEEP_CONTAINERS" -eq 1 ]]; then
    echo "Keeping containers (per --keep-containers)."
  else
    echo "Removing containers..."
    docker rm -f "${CONTAINER_NAMES[@]}" >/dev/null 2>&1 || true
  fi
  # Release CPU allocations regardless of keep/remove
  release_cpus "${CONTAINER_NAMES[@]}"
fi

if [[ "$NO_SUMMARY" -eq 0 ]]; then
echo "Aggregating global summary at: $LOGS_DIR/summary.json"
python3 - "$LOGS_DIR" "$JSONL" "$PREDS" <<'PY'
import os, sys, json
from pathlib import Path

logs_dir = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
preds_path = Path(sys.argv[3])

# Load preds keys
try:
    with preds_path.open('r', encoding='utf-8') as f:
        preds = json.load(f)
    preds_ids = set()
    if isinstance(preds, dict):
        preds_ids = set(preds.keys())
    elif isinstance(preds, list):
        preds_ids = {p.get('instance_id') for p in preds if isinstance(p, dict) and p.get('instance_id')}
except Exception:
    preds_ids = set()

# Build submitted ids from JSONL intersect preds
submitted = []
seen = set()
try:
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = obj.get('instance_id')
            if isinstance(iid, str) and iid in preds_ids and iid not in seen:
                seen.add(iid)
                submitted.append(iid)
except Exception:
    pass

# Scan logs_dir for report.json files and collect outcomes
by_iid = {}
for sub in logs_dir.iterdir() if logs_dir.is_dir() else []:
    if not sub.is_dir():
        continue
    rep = sub / 'report.json'
    if not rep.is_file():
        continue
    try:
        with rep.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload:
            # report has a single instance_id key
            for iid, info in payload.items():
                if isinstance(info, dict):
                    by_iid[iid] = info
                break
    except Exception:
        continue

def unique(seq):
    out = []
    s = set()
    for x in seq:
        if x not in s:
            s.add(x)
            out.append(x)
    return out

submitted_ids = unique(submitted)
resolved_ids = []
unresolved_ids = []
empty_patch_ids = []
error_ids = []
completed_ids = []  # non-empty only

for iid in submitted_ids:
    info = by_iid.get(iid)
    if not isinstance(info, dict):
        continue
    patch_exists = bool(info.get('patch_exists'))
    resolved = bool(info.get('resolved'))
    tests_executed = bool(info.get('tests_executed'))
    if not patch_exists:
        empty_patch_ids.append(iid)
    else:
        completed_ids.append(iid)
        if not tests_executed:
            error_ids.append(iid)
        elif resolved:
            resolved_ids.append(iid)
        else:
            unresolved_ids.append(iid)

completed_non_empty_ids = unique(completed_ids)
aggregate = {
    'submitted_instances': len(submitted_ids),
    'completed_instances': len(completed_non_empty_ids),
    'resolved_instances': len(unique(resolved_ids)),
    'unresolved_instances': len(unique([x for x in unresolved_ids if x not in set(empty_patch_ids) and x not in set(error_ids)])),
    'empty_patch_instances': len(unique(empty_patch_ids)),
    'error_instances': len(unique(error_ids)),
    'submitted_ids': submitted_ids,
    'completed_ids': completed_non_empty_ids,
    'resolved_ids': unique(resolved_ids),
    'unresolved_ids': unique([x for x in unresolved_ids if x not in set(empty_patch_ids) and x not in set(error_ids)]),
    'empty_patch_ids': unique(empty_patch_ids),
    'error_ids': unique(error_ids),
}

out_path = logs_dir / 'summary.json'
tmp_path = out_path.parent / (out_path.name + '.tmp')
tmp_path.write_text(json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')
os.replace(tmp_path, out_path)
print(json.dumps(aggregate, indent=2))
PY
AGG_RC=$?
if [[ $AGG_RC -ne 0 ]]; then
  echo "Local aggregation failed (likely permissions). Retrying inside container..." >&2
  FIRST_IMAGE=$(awk -F '\t' 'NR==1{print $1}' "$INSTANCES_MANIFEST")
  if [[ -z "$FIRST_IMAGE" ]]; then
    echo "No image found to run fallback aggregator." >&2
    exit 1
  fi
  docker run --rm \
    $WORKDIR_MOUNT $LOGS_MOUNT $JSONL_MOUNT $PREDS_MOUNT \
    -e "JSONL=$JSONL" -e "PREDS=$PREDS" -e "LOGS_DIR=$LOGS_DIR" \
    "$FIRST_IMAGE" \
    bash -lc "python3 - <<'PY'
import os, sys, json
from pathlib import Path
logs_dir = Path(os.environ['LOGS_DIR'])
jsonl_path = Path(os.environ['JSONL'])
preds_path = Path(os.environ['PREDS'])
try:
    with preds_path.open('r', encoding='utf-8') as f:
        preds = json.load(f)
    preds_ids = set()
    if isinstance(preds, dict):
        preds_ids = set(preds.keys())
    elif isinstance(preds, list):
        preds_ids = {p.get('instance_id') for p in preds if isinstance(p, dict) and p.get('instance_id')}
except Exception:
    preds_ids = set()
submitted = []
seen = set()
try:
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                obj=json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = obj.get('instance_id')
            if isinstance(iid, str) and iid in preds_ids and iid not in seen:
                seen.add(iid)
                submitted.append(iid)
except Exception:
    pass
by_iid = {}
for sub in logs_dir.iterdir() if logs_dir.is_dir() else []:
    if not sub.is_dir():
        continue
    rep = sub / 'report.json'
    if not rep.is_file():
        continue
    try:
        with rep.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload:
            for iid, info in payload.items():
                if isinstance(info, dict):
                    by_iid[iid] = info
                break
    except Exception:
        continue
def unique(seq):
    out = []
    s = set()
    for x in seq:
        if x not in s:
            s.add(x)
            out.append(x)
    return out
submitted_ids = unique(submitted)
resolved_ids = []
unresolved_ids = []
empty_patch_ids = []
error_ids = []
completed_ids = []
for iid in submitted_ids:
    info = by_iid.get(iid)
    if not isinstance(info, dict):
        continue
    patch_exists = bool(info.get('patch_exists'))
    resolved = bool(info.get('resolved'))
    tests_executed = bool(info.get('tests_executed'))
    if not patch_exists:
        empty_patch_ids.append(iid)
    else:
        completed_ids.append(iid)
        if not tests_executed:
            error_ids.append(iid)
        elif resolved:
            resolved_ids.append(iid)
        else:
            unresolved_ids.append(iid)
completed_non_empty_ids = unique(completed_ids)
aggregate = {
    'submitted_instances': len(submitted_ids),
    'completed_instances': len(completed_non_empty_ids),
    'resolved_instances': len(unique(resolved_ids)),
    'unresolved_instances': len(unique([x for x in unresolved_ids if x not in set(empty_patch_ids) and x not in set(error_ids)])),
    'empty_patch_instances': len(unique(empty_patch_ids)),
    'error_instances': len(unique(error_ids)),
    'submitted_ids': submitted_ids,
    'completed_ids': completed_non_empty_ids,
    'resolved_ids': unique(resolved_ids),
    'unresolved_ids': unique([x for x in unresolved_ids if x not in set(empty_patch_ids) and x not in set(error_ids)]),
    'empty_patch_ids': unique(empty_patch_ids),
    'error_ids': unique(error_ids),
}
out_path = logs_dir / 'summary.json'
tmp_path = out_path.parent / (out_path.name + '.tmp')
tmp_path.write_text(json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')
os.replace(tmp_path, out_path)
print(json.dumps(aggregate, indent=2))
PY
"
fi

echo "Done. Summary written to $LOGS_DIR/summary.json"
fi

# Backup logs into the directory that contains the preds.json file (exclude .image_groups)
if [[ "${NO_COPY_LOGS:-0}" == "1" ]]; then
  echo "Skipping log backup (NO_COPY_LOGS=1)"
else
  echo "Backing up logs to preds directory: $PREDS_DIR"
  mkdir -p "$PREDS_DIR"
  DEST_LOGS_DIR="$PREDS_DIR/$(basename "$LOGS_DIR")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='.image_groups' "$LOGS_DIR"/ "$DEST_LOGS_DIR"/ || true
  else
    cp -r "$LOGS_DIR" "$PREDS_DIR/" || true
    rm -rf "$DEST_LOGS_DIR/.image_groups" 2>/dev/null || true
  fi
fi

# Cleanup auto-generated shards dir (if we created it)
cleanup_shards
