#!/usr/bin/env bash

set -euo pipefail

REFRESH_OVERLAY_FILES=0
TARGET_DIR=/root/Megatron-LM
EXPECTED_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patch/latest/megatron_fp32_lm_head.patch"
TOUCHED_FILES=(
    megatron/training/arguments.py
    megatron/training/yaml_arguments.py
    megatron/core/transformer/transformer_config.py
    megatron/core/models/common/language_module/language_module.py
    megatron/core/models/gpt/gpt_model.py
    megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py
    tests/unit_tests/models/test_gpt_model.py
    tests/unit_tests/inference/model_inference_wrappers/gpt/test_gpt_inference_wrapper.py
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

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "Patch file not found: ${PATCH_FILE}" >&2
    exit 1
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    echo "Megatron-LM git checkout not found at ${TARGET_DIR}" >&2
    exit 1
fi

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unexpected Megatron-LM commit at ${TARGET_DIR}: ${ACTUAL_COMMIT}" >&2
    echo "Expected commit: ${EXPECTED_COMMIT}" >&2
    exit 1
fi

if git -C "${TARGET_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "FP32 LM head overlay already applied to ${TARGET_DIR}"
    exit 0
fi

if ! git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    if [[ "${REFRESH_OVERLAY_FILES}" != "1" ]]; then
        echo "Patch does not apply cleanly to ${TARGET_DIR}" >&2
        echo "If this checkout already has an older FP32 LM head overlay, rerun with --refresh-overlay-files" >&2
        exit 1
    fi

    git -C "${TARGET_DIR}" checkout -- "${TOUCHED_FILES[@]}"
fi

git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}"
git -C "${TARGET_DIR}" apply "${PATCH_FILE}"

echo "Applied FP32 LM head overlay patch to ${TARGET_DIR}"
