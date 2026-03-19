# Launch the same container environment used by the working 0.6B recipes.
docker run --rm --gpus all --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 --network host -v /var/run/docker.sock:/var/run/docker.sock -v /home/surya/livup:/root/repo/ -v /home/shared/megatron_dir:/root/data/ -it slimerl/slime:latest /bin/bash -lc '/root/repo/slime/docker/apply_megatron_fp32_lm_head_patch.sh && exec bash'

# Build the canonical digital-twin dataset.
cd /root/repo/slime
python3 examples/digital_twin/build_dataset.py \
  --digital-twin-root /root/repo/digital-twin-simulation \
  --output-dir examples/digital_twin/data \
  --simulation-config text_simulation/configs/qwen3_0_6b_instruct.yaml \
  --overwrite

# Build the static reward spec used by the custom digital-twin reward.
cd /root/repo/slime
python3 examples/digital_twin/build_reward_spec.py \
  --digital-twin-root /root/repo/digital-twin-simulation \
  --output examples/digital_twin/reward_spec.json

# Run the canonical digital-twin Qwen3-0.6B GRPO recipe with 64k rope scaling.
cd /root/repo/slime
DIGITAL_TWIN_DATA_DIR=/root/repo/slime/examples/digital_twin/data \
HF_CHECKPOINT=/root/data/hf_models/Qwen3-0.6B \
REF_LOAD=/root/data/mega-models/Qwen3-0.6B \
SAVE_DIR=/root/data/mega-models/Qwen3-0.6B_digital_twin_grpo_base_non_think \
MAX_CONTEXT_LEN=65536 \
MAX_PROMPT_LEN=57344 \
MAX_RESPONSE_LEN=8192 \
ROPE_SCALING_FACTOR=2.0 \
bash examples/digital_twin/run-qwen3-06b-grpo.sh
