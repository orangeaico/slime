#!/bin/bash

# SWE-bench on-policy distillation training script
# Usage: bash examples/swe_bench/run-qwen3-06B-opd.sh

ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray

set -ex

# Start the teacher model server
TEACHER_IP="127.0.0.1"
TEACHER_PORT=4500
LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

## Launch the teacher model server in the background
# CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
#     --model-path /root/data/hf_models/Qwen3-1.7B \
#     --host 0.0.0.0 \
#     --port $TEACHER_PORT \
#     --tp 1 \
#     --chunked-prefill-size 4096 \
#     --mem-fraction-static 0.2 \
#     > "$LOG_FILE" 2>&1 &

# echo "Starting teacher model server..."

# ## Wait for the teacher model server to be ready
# until curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null; do
#     echo "Waiting for the teacher model server to start..."
#     tail -n 10 "$LOG_FILE"
#     sleep 5
# done

curl http://$TEACHER_IP:$TEACHER_PORT/get_model_info
echo "Teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT."
# sleep 10


export PYTHONBUFFERED=16

# Create SWE-agent required directories
mkdir -p /home/shared
mkdir -p /tmp/swe_agent_cache
mkdir -p /tmp/swe_agent_trajectories

# Fix git safe.directory issue for swe_livup
git config --global --add safe.directory /root/swe_livup

# Set SWE-agent environment variables
export SWE_AGENT_CONFIG_ROOT=/root/swe_livup
export SWE_AGENT_CACHE_ROOT=/tmp/swe_agent_cache
export SWE_AGENT_TRAJECTORY_DIR=/tmp/swe_agent_trajectories

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/repo/slime/scripts/models/qwen3-1.7B.sh"


CKPT_ARGS=(
   --hf-checkpoint /root/data/himanshu/output/hf_models_converted/qwen3_1.7b/swe_bench_472_trajs_10_epochs
   --ref-load /root/data/mega-models/Qwen3-1.7B
   # --load /root/data/mega-models/Qwen3-1.7B_swe_bench/
   # --no-load-rng
   # --no-load-optim
   --save /root/data/mega-models/Qwen3-1.7B_swe_bench/
   --no-save-optim
   --no-save-rng
   --save-interval 10
)

CUSTOM_ARGS=(
   # Multi-turn generate function with SWE-agent
   --custom-generate-function-path examples.swe_bench.generate.generate

   # Reward functions for pure distillation
   --custom-rm-path examples.swe_bench.reward.reward_func
   --custom-reward-post-process-path examples.swe_bench.reward.post_process_rewards

   # OPD requires rm-url for teacher log-probs
   --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

ROLLOUT_ARGS=(
   --prompt-data examples/swe_bench/data/train_with_images.jsonl
   --input-key prompt
   # Don't apply chat template - we handle it in generate.py
   --rollout-shuffle
   --num-rollout 50
   --rollout-batch-size 1  # REDUCED: 2->1 to avoid simultaneous Docker startups
   --n-samples-per-prompt 1  # REDUCED: 2->1 to avoid resource contention
   --rollout-max-response-len 4096  # Long context for multi-turn
   --rollout-temperature 0.8

   --global-batch-size 1
   --balance-data
)

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
   --max-tokens-per-gpu 2048
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
   #--use-wandb
   # --wandb-project slime-swe-bench
   # --wandb-group qwen3-06B-opd
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   # Partial rollout disabled for now
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
   --save-debug-rollout-data /tmp/swe_agent_trajectories/rollout_{rollout_id}.pt
)




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/:/root/swe_livup",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "SWE_AGENT_CONFIG_ROOT": "/root/swe_livup",
        "SWE_AGENT_CACHE_ROOT": "/tmp/swe_agent_cache",
        "SWE_AGENT_TRAJECTORY_DIR": "/tmp/swe_agent_trajectories"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
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
