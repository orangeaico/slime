#!/bin/bash

####clear before training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"

DEFAULT_MEGATRON_LM_ROOT="/root/Megatron-LM"
if [[ ! -d "$DEFAULT_MEGATRON_LM_ROOT" ]]; then
    DEFAULT_MEGATRON_LM_ROOT="/home/surya/livup/Megatron-LM"
fi
MEGATRON_LM_ROOT="${MEGATRON_LM_ROOT:-$DEFAULT_MEGATRON_LM_ROOT}"

source "${REPO_ROOT}/scripts/models/qwen3-0.6B.sh"

DIGITAL_TWIN_DATA_DIR="${DIGITAL_TWIN_DATA_DIR:-${REPO_ROOT}/examples/digital_twin/data}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${DIGITAL_TWIN_DATA_DIR}/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${DIGITAL_TWIN_DATA_DIR}/eval.jsonl}"

DEFAULT_HF_CHECKPOINT="/home/shared/megatron_dir/hf_models/Qwen3-0.6B"
if [[ ! -e "${DEFAULT_HF_CHECKPOINT}" ]]; then
    DEFAULT_HF_CHECKPOINT="/root/data/hf_models/Qwen3-0.6B"
fi

DEFAULT_REF_LOAD="/home/shared/megatron_dir/mega-models/Qwen3-0.6B"
if [[ ! -e "${DEFAULT_REF_LOAD}" ]]; then
    DEFAULT_REF_LOAD="/root/data/mega-models/Qwen3-0.6B"
fi

DEFAULT_SAVE_DIR="/home/shared/megatron_dir/mega-models/Qwen3-0.6B_digital_twin_grpo_base_non_think"
if [[ ! -e "/home/shared/megatron_dir/mega-models" ]]; then
    DEFAULT_SAVE_DIR="/root/data/mega-models/Qwen3-0.6B_digital_twin_grpo_base_non_think"
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-${DEFAULT_HF_CHECKPOINT}}"
REF_LOAD="${REF_LOAD:-${DEFAULT_REF_LOAD}}"
LOAD_DIR="${LOAD_DIR:-}"
SAVE_DIR="${SAVE_DIR:-${DEFAULT_SAVE_DIR}}"

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-65536}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-57344}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
ROPE_SCALING_FACTOR="${ROPE_SCALING_FACTOR:-2.0}"
ORIGINAL_MAX_POSITION_EMBEDDINGS="${ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"
SGLANG_JSON_MODEL_OVERRIDE_ARGS="${SGLANG_JSON_MODEL_OVERRIDE_ARGS:-$(printf '{"max_position_embeddings":%s,"rope_scaling":{"rope_type":"yarn","factor":%s,"original_max_position_embeddings":%s}}' "${MAX_CONTEXT_LEN}" "${ROPE_SCALING_FACTOR}" "${ORIGINAL_MAX_POSITION_EMBEDDINGS}")}"

echo "Digital-twin GRPO data dir: $DIGITAL_TWIN_DATA_DIR"
echo "RoPE scaling: original_max_position_embeddings=${ORIGINAL_MAX_POSITION_EMBEDDINGS}, factor=${ROPE_SCALING_FACTOR}, effective_max_context=${MAX_CONTEXT_LEN}"

MODEL_ARGS+=(
   --max-position-embeddings "${MAX_CONTEXT_LEN}"
   --use-rope-scaling
   --rotary-scaling-factor "${ROPE_SCALING_FACTOR}"
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_DIR}"
   --no-save-optim
   --no-save-rng
   --save-interval "${SAVE_INTERVAL:-25}"
)
if [[ -n "${LOAD_DIR}" ]]; then
   CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA_PATH}"
   --input-key messages
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle

   --custom-rm-path examples.digital_twin.reward.reward_func
   --reward-key score

   --num-rollout "${NUM_ROLLOUT:-64}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"

   --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}"
   --rollout-top-p "${ROLLOUT_TOP_P:-0.8}"
   --rollout-top-k "${ROLLOUT_TOP_K:-20}"
   --use-rollout-logprobs

   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --balance-data
)

RM_ARGS=(
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-10}"
   --eval-prompt-data twin2k500 "${EVAL_DATA_PATH}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-2}"
   --eval-input-key messages
   --eval-label-key label
   --eval-max-response-len "${MAX_RESPONSE_LEN}"
   --eval-temperature "${EVAL_TEMPERATURE:-0.7}"
   --eval-top-p "${EVAL_TOP_P:-0.8}"
   --eval-top-k "${EVAL_TOP_K:-20}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --loss-type "${LOSS_TYPE:-policy_loss}"
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-digital-twin
   # --wandb-group qwen3-06b-digital-twin
   # --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.8}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
   --sglang-json-model-override-args "${SGLANG_JSON_MODEL_OVERRIDE_ARGS}"
   --partial-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
   --cross-entropy-loss-fusion
   --cross-entropy-fusion-impl te
   --bf16
   --use-distributed-optimizer
   --use-precision-aware-optimizer
   --main-params-dtype fp16
   --main-grads-dtype bf16
   --grad-reduce-in-bf16
   --exp-avg-dtype fp16
   --exp-avg-sq-dtype fp16
   --transformer-impl transformer_engine
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"
  }
}
EOF
)" \
   -- python3 "${REPO_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   --rollout-num-gpus 2 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
