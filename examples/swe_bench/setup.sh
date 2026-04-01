#!/usr/bin/env bash

set -euo pipefail

# Bootstrap SWE-agent runtime dependencies inside the Slime container.
# This script is intentionally idempotent and safe to re-run.
#
# Install modes:
#   safe (default): editable install without dependency resolution to avoid
#                   mutating core Slime/sglang stack versions.
#   full:           full dependency install from swe_livup metadata.
#
# Env vars:
#   SWE_SETUP_INSTALL_MODE=safe|full
#   SWE_SETUP_AUTO_FALLBACK_FULL=0|1

if [[ ! -S /var/run/docker.sock ]]; then
    echo "[swe-setup] ERROR: /var/run/docker.sock is missing."
    echo "[swe-setup] Start the container with: -v /var/run/docker.sock:/var/run/docker.sock"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "[swe-setup] docker CLI not found. Installing docker.io..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y docker.io
else
    echo "[swe-setup] docker CLI already available."
fi

echo "[swe-setup] Verifying docker daemon connectivity..."
if ! docker info >/dev/null 2>&1; then
    echo "[swe-setup] ERROR: docker CLI cannot reach local daemon via /var/run/docker.sock."
    exit 1
fi
echo "[swe-setup] Docker daemon connectivity check passed."

if [[ ! -d /root/swe_livup ]]; then
    echo "[swe-setup] ERROR: /root/swe_livup not found."
    echo "[swe-setup] Mount swe_livup with: -v /home/shramana/evaluation/swe_livup:/root/swe_livup"
    exit 1
fi

git config --global --add safe.directory /root/swe_livup

export SWE_AGENT_CONFIG_ROOT="${SWE_AGENT_CONFIG_ROOT:-/root/swe_livup}"
export SWE_AGENT_CACHE_ROOT="${SWE_AGENT_CACHE_ROOT:-/root/repo/slime/outputs/swe_agent_cache}"
export SWE_AGENT_TRAJECTORY_DIR="${SWE_AGENT_TRAJECTORY_DIR:-/root/repo/slime/outputs/swe_agent_trajectories}"

mkdir -p /home/shared
mkdir -p "${SWE_AGENT_CACHE_ROOT}"
mkdir -p "${SWE_AGENT_TRAJECTORY_DIR}"

install_mode="${SWE_SETUP_INSTALL_MODE:-safe}"
auto_fallback_full="${SWE_SETUP_AUTO_FALLBACK_FULL:-0}"

if [[ "${install_mode}" != "safe" && "${install_mode}" != "full" ]]; then
    echo "[swe-setup] ERROR: invalid SWE_SETUP_INSTALL_MODE='${install_mode}' (expected 'safe' or 'full')."
    exit 1
fi

verify_imports() {
python3 - <<'PY'
import importlib

mods = ("sweagent", "swerex")
for mod in mods:
    importlib.import_module(mod)
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
print("sweagent + swerex import check passed")
PY
}

if [[ "${install_mode}" == "safe" ]]; then
    echo "[swe-setup] Installing swe_livup in editable SAFE mode (no dependency upgrades)..."
    python3 -m pip install -e /root/swe_livup --no-deps
    # Minimal runtime deps required for `from sweagent.agent.agents import DefaultAgent`
    # without broad dependency upgrades that can break sglang/transformers.
    python3 -m pip install \
        "swe-rex>=1.2.0" \
        rich \
        GitPython \
        packaging \
        ruamel.yaml \
        tenacity \
        unidiff \
        simple-parsing \
        rich-argparse \
        fastuuid \
        python-dotenv \
        pydantic_settings \
        ghapi \
        tabulate
    # Keep litellm for sweagent model imports, but do not let pip upgrade the
    # core OpenAI/Transformers stack via transitive dependencies.
    python3 -m pip install --no-deps litellm
    python3 -m pip install "huggingface-hub<1.0" "openai==2.6.1"
else
    echo "[swe-setup] Installing swe_livup in FULL mode (may update shared dependencies)..."
    # --ignore-installed avoids uninstall failures for distro-packaged Python libs
    # (for example blinker from apt in the base image).
    python3 -m pip install --ignore-installed -e /root/swe_livup
fi

echo "[swe-setup] Verifying Python imports..."
if ! verify_imports; then
    if [[ "${install_mode}" == "safe" && "${auto_fallback_full}" == "1" ]]; then
        echo "[swe-setup] SAFE mode import check failed. Retrying with FULL mode..."
        python3 -m pip install --ignore-installed -e /root/swe_livup
        verify_imports
    else
        if [[ "${install_mode}" == "safe" ]]; then
            echo "[swe-setup] SAFE mode import check failed."
            echo "[swe-setup] Re-run with SWE_SETUP_INSTALL_MODE=full if you need full dependency install."
        fi
        exit 1
    fi
fi

echo "[swe-setup] Bootstrap complete."
