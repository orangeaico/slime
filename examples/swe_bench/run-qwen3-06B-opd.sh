#!/bin/bash

# SWE-bench on-policy distillation training script
# Usage: bash examples/swe_bench/run-qwen3-06B-opd.sh

ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

find_free_port() {
   local start_port="$1"
   local end_port="$2"

   python3 - "$start_port" "$end_port" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
end_port = int(sys.argv[2])

for port in range(start_port, end_port + 1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    raise SystemExit(0)

raise SystemExit(1)
PY
}

# Ensure SWE-agent + SWE-ReX dependencies are installed in this container.
SWE_SETUP_INSTALL_MODE="${SWE_SETUP_INSTALL_MODE:-safe}" \
SWE_SETUP_AUTO_FALLBACK_FULL="${SWE_SETUP_AUTO_FALLBACK_FULL:-0}" \
  bash examples/swe_bench/setup.sh

# Start the teacher model server
TEACHER_IP="127.0.0.1"
TEACHER_PORT=4500
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
    --model-path /root/data/hf_models/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.2 \
    > "$LOG_FILE" 2>&1 &

echo "Starting teacher model server..."

## Wait for the teacher model server to be ready
until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
    echo "Waiting for the teacher model server to start..."
    tail -n 10 "$LOG_FILE"
    sleep 5
done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
sleep 10


export PYTHONBUFFERED=16

# Create SWE-agent required directories
mkdir -p /home/shared
mkdir -p /root/repo/slime/outputs/swe_agent_cache
mkdir -p /root/repo/slime/outputs/swe_agent_trajectories

# Fix git safe.directory issue for swe_livup
git config --global --add safe.directory /root/swe_livup

# Set SWE-agent environment variables
export SWE_AGENT_CONFIG_ROOT=/root/swe_livup
export SWE_AGENT_CACHE_ROOT=/root/repo/slime/outputs/swe_agent_cache
export SWE_AGENT_TRAJECTORY_DIR=/root/repo/slime/outputs/swe_agent_trajectories

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/repo/slime/.env"
source "/root/repo/slime/scripts/models/qwen3-0.6B.sh"

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
if [ -z "${RAY_HEAD_PORT:-}" ]; then
   RAY_HEAD_PORT=$(find_free_port 26000 29999)
fi
if [ -z "${RAY_DASHBOARD_PORT:-}" ]; then
   RAY_DASHBOARD_PORT=$(find_free_port 30000 33999)
fi
if [ "${RAY_DASHBOARD_PORT}" = "${RAY_HEAD_PORT}" ]; then
   RAY_DASHBOARD_PORT=$(find_free_port 34000 37999)
fi
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_swe_${TIMESTAMP}_$$}
echo "Using Ray ports: head=${RAY_HEAD_PORT}, dashboard=${RAY_DASHBOARD_PORT}"

NUM_ROLLOUT=${NUM_ROLLOUT:-50}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
SWE_DOCKER_STARTUP_CONCURRENCY=${SWE_DOCKER_STARTUP_CONCURRENCY:-4}
SWE_DOCKER_STARTUP_TIMEOUT_SECONDS=${SWE_DOCKER_STARTUP_TIMEOUT_SECONDS:-900}
ROLLOUT_SAMPLE_FILTER_PATH=${ROLLOUT_SAMPLE_FILTER_PATH:-examples.swe_bench.rollout_hooks.mark_swe_non_submitted_samples_inactive}
SWE_HARDCODED_RESPONSE_MODE=${SWE_HARDCODED_RESPONSE_MODE:-program}
SWE_HARDCODED_PROGRAM_PATH=${SWE_HARDCODED_PROGRAM_PATH:-/root/repo/slime/examples/swe_bench/hardcoded_programs/hardcoded_program.yaml}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-2048}
SWE_EVAL_REWARD_ENABLE=${SWE_EVAL_REWARD_ENABLE:-0}
SWE_EVAL_LOGS_ROOT=${SWE_EVAL_LOGS_ROOT:-/root/repo/slime/outputs/swe_eval_reward_logs/${TIMESTAMP}}
SWE_EVAL_TIMEOUT_SECONDS=${SWE_EVAL_TIMEOUT_SECONDS:-300}
SWE_EVAL_JSONL_PATH=${SWE_EVAL_JSONL_PATH:-/root/data/swe_mirror/dataset/all_sources_combined_dataset.jsonl}
SWE_EVAL_WORKERS=${SWE_EVAL_WORKERS:-4}
SWE_EVAL_PYTEST_TIMEOUT_SECONDS=${SWE_EVAL_PYTEST_TIMEOUT_SECONDS:-180}


CKPT_ARGS=(
   --hf-checkpoint /root/data/hf_models/Qwen3-0.6B
   --ref-load /root/data/mega-models/Qwen3-0.6B
   # --load /root/data/mega-models/Qwen3-0.6B_swe_bench/
   # --no-load-rng
   # --no-load-optim
   --save /root/data/mega-models/Qwen3-0.6B_swe_bench/
   --no-save-optim
   --no-save-rng
   --save-interval 10
)

CUSTOM_ARGS=(
   # Multi-turn generate function using proper SWE-agent DefaultAgent integration
   --custom-generate-function-path examples.swe_bench.generate_with_sweagent.generate

   # Use fully-async rollout engine for SWE-agent.
   --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async

   # Reward functions for pure distillation
   --custom-rm-path examples.swe_bench.reward.reward_func
   --custom-reward-post-process-path examples.swe_bench.reward.post_process_rewards
   --reward-key reward  # Extract scalar reward from dict for metrics computation
   --group-rm # do rm on a whole group

   # OPD requires rm-url for teacher log-probs
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

ROLLOUT_ARGS=(
   # Prompt data for eval-reward mode uses SWE eval for all samples.
   --prompt-data examples/swe_bench/data/train.jsonl
   --input-key prompt
   # Don't apply chat template - we handle it in generate.py
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"

   # Batch-fill rollout for variable-duration tasks
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"  # Number of samples to collect (increase to 64 for scale)
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --swe-docker-startup-concurrency "${SWE_DOCKER_STARTUP_CONCURRENCY}"  # Parallel startups without overwhelming swe-rex bootstrap
   --swe-docker-startup-timeout-seconds "${SWE_DOCKER_STARTUP_TIMEOUT_SECONDS}"  # Avoid false startup timeout during pipx/swe-rex install
   --partial-rollout  # Enable saving/resuming true aborted samples
   --mask-offpolicy-in-partial-rollout  # Mask old tokens in resumed samples

   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"  # Long context for multi-turn
   --rollout-temperature 0.8

   --global-batch-size "${GLOBAL_BATCH_SIZE}"  # For training (increase to 64 for scale)
   --balance-data
)

if [ -n "${ROLLOUT_SAMPLE_FILTER_PATH}" ]; then
   ROLLOUT_ARGS+=(--rollout-sample-filter-path "${ROLLOUT_SAMPLE_FILTER_PATH}")
fi

if [ "${SWE_HARDCODED_RESPONSE_MODE}" != "none" ]; then
   CUSTOM_ARGS+=(
      --swe-hardcoded-response-mode "${SWE_HARDCODED_RESPONSE_MODE}"
      --swe-hardcoded-program-path "${SWE_HARDCODED_PROGRAM_PATH}"
   )
fi

if [ "${SWE_EVAL_REWARD_ENABLE}" = "1" ]; then
   mkdir -p "${SWE_EVAL_LOGS_ROOT}"
   CUSTOM_ARGS+=(
      --swe-eval-reward-enable
      --swe-eval-logs-root "${SWE_EVAL_LOGS_ROOT}"
      --swe-eval-timeout-seconds "${SWE_EVAL_TIMEOUT_SECONDS}"
      --swe-eval-jsonl-path "${SWE_EVAL_JSONL_PATH}"
      --swe-eval-workers "${SWE_EVAL_WORKERS}"
      --swe-eval-pytest-timeout-seconds "${SWE_EVAL_PYTEST_TIMEOUT_SECONDS}"
   )
fi

EVAL_ARGS=(
   # Evaluation disabled for now
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

   --micro-batch-size 1
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo

   # On-policy distillation settings
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0

   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-swe-bench
   --wandb-group qwen3-06B-opd
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
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

DEBUG_ARGS=(
   # Save rollout trajectories for debugging and analysis
   --save-debug-rollout-data /root/repo/slime/outputs/swe_agent_trajectories/rollout_{rollout_id}.pt
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --port ${RAY_HEAD_PORT} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT} --temp-dir ${RAY_TEMP_DIR}


ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json='{
      "env_vars": {
         "PYTHONPATH": "/root/Megatron-LM/:/root/swe_livup",
         "CUDA_DEVICE_MAX_CONNECTIONS": "1",
         "SWE_AGENT_CONFIG_ROOT": "/root/swe_livup",
         "SWE_AGENT_CACHE_ROOT": "/root/repo/slime/outputs/swe_agent_cache",
         "SWE_AGENT_TRAJECTORY_DIR": "/root/repo/slime/outputs/swe_agent_trajectories"
      }
   }' \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${DEBUG_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
