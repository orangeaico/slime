#!/usr/bin/env bash
set -euo pipefail

# The input to the script is the timestamp of the current run
TIMESTAMP=$1
MODEL_NAME=Qwen3-0.6B

mkdir -p /root/data/trained-mega-models/$TIMESTAMP/$MODEL_NAME/logs/

CHECKPOINTS_LOCAL_PATH=/root/data/trained-mega-models/$TIMESTAMP/$MODEL_NAME/checkpoints/
CHECKPOINTS_GDRIVE_PATH=gdrive:megatron_dir/trained-mega-models/$TIMESTAMP/$MODEL_NAME/checkpoints/

# Copy the logs as well
echo "Copying the training logs to gdrive"
cp /root/repo/slime/logs.txt /root/data/trained-mega-models/$TIMESTAMP/$MODEL_NAME/logs/
cp /root/repo/slime/examples/on_policy_distillation/run-qwen3-06b-rl.sh /root/data/trained-mega-models/$TIMESTAMP/$MODEL_NAME/logs/
rclone copy /root/data/trained-mega-models/$TIMESTAMP/$MODEL_NAME/logs/ gdrive:megatron_dir/trained-mega-models/$TIMESTAMP/$MODEL_NAME/logs/ --progress 

echo "Copy checkpoints to gdrive"
rclone copy -P --transfers 4 --checkers 16 --drive-chunk-size 32M --buffer-size 32M "$CHECKPOINTS_LOCAL_PATH" "$CHECKPOINTS_GDRIVE_PATH"

echo "All Done!"