# Build the canonical digital-twin dataset with the same request shape used by
# the digital-twin pipeline and the eval-aligned label source.
cd /home/surya/livup/slime
python3 examples/digital_twin/build_dataset.py \
  --digital-twin-root /home/surya/livup/digital-twin-simulation \
  --output-dir /tmp/digital_twin_dataset_build \
  --simulation-config text_simulation/configs/qwen3_0_6b_instruct.yaml \
  --overwrite
