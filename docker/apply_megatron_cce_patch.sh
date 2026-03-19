#!/usr/bin/env bash

set -euo pipefail

REFRESH_OVERLAY_FILES=0
TARGET_DIR=/root/Megatron-LM
EXPECTED_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
CCE_COMMIT=b7a02791b234e187b524fb1dba6a812d521b203a
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patch/latest/megatron_cce_rl.patch"
FP32_HELPER="${SCRIPT_DIR}/apply_megatron_fp32_lm_head_patch.sh"
TOUCHED_FILES=(
    megatron/training/arguments.py
    megatron/core/transformer/transformer_config.py
    megatron/core/models/gpt/gpt_model.py
)
NEW_FILES=(
    megatron/core/fusions/cce_loss.py
)

while (($#)); do
    case "$1" in
        --refresh-overlay-files)
            REFRESH_OVERLAY_FILES=1
            shift
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

cce_overlay_present() {
    [[ -f "${TARGET_DIR}/megatron/core/fusions/cce_loss.py" ]] \
        && grep -q -- '--fused-linear-cross-entropy' "${TARGET_DIR}/megatron/training/arguments.py" \
        && grep -q 'use_linear_cross_entropy: bool = False' \
            "${TARGET_DIR}/megatron/core/transformer/transformer_config.py" \
        && grep -q 'from megatron.core.fusions.cce_loss import cce_per_token_loss' \
            "${TARGET_DIR}/megatron/core/models/gpt/gpt_model.py"
}

python - <<'PY' || pip install --no-deps "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@${CCE_COMMIT}"
import cut_cross_entropy  # noqa: F401

print("cut_cross_entropy import ok")
PY

python - <<'PY'
import ast
from pathlib import Path

TARGET_FILES = {
    Path("tl_utils.py"): {
        "tl_softcapping": "return v / softcap",
        "tl_softcapping_grad": "return dv / softcap",
    },
    Path("utils.py"): {
        "softcapping": "return logits / softcap",
    },
}


def locate_target(relative: Path) -> Path:
    import cut_cross_entropy

    module_path = Path(cut_cross_entropy.__file__).resolve()
    target = module_path.parent / relative
    if not target.exists():
        raise SystemExit(f"Unable to locate {relative} under {module_path.parent}")
    return target


def apply_replacements(target_path: Path, replacements: dict[str, str]) -> bool:
    source_text = target_path.read_text()
    tree = ast.parse(source_text)
    lines = source_text.splitlines()
    updated = False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in replacements or not node.body:
            continue
        start = node.body[0].lineno - 1
        end = node.body[-1].end_lineno - 1
        indent = " " * node.body[0].col_offset
        replacement = f"{indent}{replacements[node.name]}"
        if lines[start : end + 1] != [replacement]:
            lines[start : end + 1] = [replacement]
            updated = True

    if updated:
        target_path.write_text("\n".join(lines) + "\n")
    return updated


for relative_path, replacements in TARGET_FILES.items():
    target_file = locate_target(relative_path)
    if apply_replacements(target_file, replacements):
        print(f"Updated {target_file}")
    else:
        print(f"No updates required for {target_file}")
PY

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "Patch file not found: ${PATCH_FILE}" >&2
    exit 1
fi

if [[ ! -f "${FP32_HELPER}" ]]; then
    echo "Required FP32 helper not found: ${FP32_HELPER}" >&2
    exit 1
fi

if [[ ! -d "${TARGET_DIR}/.git" && ! -f "${TARGET_DIR}/.git" ]]; then
    echo "Megatron-LM git checkout not found at ${TARGET_DIR}" >&2
    exit 1
fi

git config --global --add safe.directory "${TARGET_DIR}" >/dev/null 2>&1 || true

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unexpected Megatron-LM commit at ${TARGET_DIR}: ${ACTUAL_COMMIT}" >&2
    echo "Expected commit: ${EXPECTED_COMMIT}" >&2
    exit 1
fi

if git -C "${TARGET_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "CCE RL overlay already applied to ${TARGET_DIR}"
    exit 0
fi

if cce_overlay_present; then
    echo "CCE RL overlay already applied to ${TARGET_DIR}"
    exit 0
fi

"${FP32_HELPER}" "${TARGET_DIR}"

if ! git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    if [[ "${REFRESH_OVERLAY_FILES}" != "1" ]]; then
        echo "Patch does not apply cleanly to ${TARGET_DIR}" >&2
        echo "If this checkout already has an older CCE RL overlay, rerun with --refresh-overlay-files" >&2
        exit 1
    fi

    git -C "${TARGET_DIR}" checkout -- "${TOUCHED_FILES[@]}"
    for path in "${NEW_FILES[@]}"; do
        rm -f "${TARGET_DIR}/${path}"
    done
    "${FP32_HELPER}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}"
git -C "${TARGET_DIR}" apply "${PATCH_FILE}"

echo "Applied CCE RL overlay patch to ${TARGET_DIR}"
