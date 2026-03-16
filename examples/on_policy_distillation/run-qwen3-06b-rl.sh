#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

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

LOG_FILE="/tmp/sglang_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/slime/scripts/models/qwen3-0.6B.sh"

MAX_SEQ_LEN=1024

LOSS_TYPE=${LOSS_TYPE:-policy_loss}
CISPO_EPS_CLIP_HIGH=${CISPO_EPS_CLIP_HIGH:-5.0}
DISPO_POS_EPS_CLIP_LOW=${DISPO_POS_EPS_CLIP_LOW:-0.2}
DISPO_POS_EPS_CLIP_HIGH=${DISPO_POS_EPS_CLIP_HIGH:-10}
DISPO_NEG_EPS_CLIP_LOW=${DISPO_NEG_EPS_CLIP_LOW:-1.0}
DISPO_NEG_EPS_CLIP_HIGH=${DISPO_NEG_EPS_CLIP_HIGH:-100}

LOSS_ARGS=(
   --loss-type ${LOSS_TYPE}
)

LOSS_SPECIFIC_ARGS=()
if [ "${LOSS_TYPE}" = "cispo_loss" ]; then
   LOSS_SPECIFIC_ARGS=(
      --eps-clip-high ${CISPO_EPS_CLIP_HIGH}
      --calculate-per-token-loss
   )
elif [ "${LOSS_TYPE}" = "dispo_loss" ]; then
   LOSS_SPECIFIC_ARGS=(
      --dispo-pos-eps-clip-low ${DISPO_POS_EPS_CLIP_LOW}
      --dispo-pos-eps-clip-high ${DISPO_POS_EPS_CLIP_HIGH}
      --dispo-neg-eps-clip-low ${DISPO_NEG_EPS_CLIP_LOW}
      --dispo-neg-eps-clip-high ${DISPO_NEG_EPS_CLIP_HIGH}
      --calculate-per-token-loss
   )
else
   LOSS_SPECIFIC_ARGS=(
      --eps-clip-high 0.28
   )
fi


   
CKPT_ARGS=(
   --hf-checkpoint /root/data/hf_models/Qwen3-0.6B_rl_grpo_non_think_7473_4_steps
   --ref-load /root/data/mega-models/Qwen3-0.6B_rl_grpo_non_think_7473_4_steps
   # --load /root/data/mega-models/Qwen3-0.6B_slime/
   # --no-load-rng
   # --no-load-optim
   --save /root/data/mega-models/Qwen3-0.6B_rl_cispo_non_think_7473
   --no-save-optim
   --no-save-rng
   --save-interval 117
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/datasets/gsm8k/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout 936
   --rollout-batch-size 32
   --num-steps-per-rollout 4
   --over-sampling-batch-size 48
   --n-samples-per-prompt 8
   --rollout-max-response-len $MAX_SEQ_LEN

   --rollout-temperature 0.7
   --rollout-top-p 0.8
   --rollout-top-k 20
   --use-rollout-logprobs

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(   
   --eval-interval 29
   --eval-prompt-data gsm8k /root/data/datasets/gsm8k/test_100.jsonl
   --n-samples-per-eval-prompt 4   

   --eval-input-key prompt
   --eval-label-key label

   --eval-max-response-len $MAX_SEQ_LEN
   --eval-temperature 0.7
   --eval-top-p 0.8
   --eval-top-k 20
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

   --micro-batch-size 4
   # --use-dynamic-batch-size
   --max-tokens-per-gpu $MAX_SEQ_LEN
)

GRPO_ARGS=(
   --advantage-estimator grpo
   ${LOSS_ARGS[@]}
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   ${LOSS_SPECIFIC_ARGS[@]}
   # --disable-rewards-normalization
   # --disable-grpo-std-normalization
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
   # --use-wandb
   # --wandb-project slime-rl
   # --wandb-group qwen3-06B-gsm
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
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




# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   --rollout-num-gpus 2 \
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
   ${RM_ARGS[@]}



####clear after training
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python