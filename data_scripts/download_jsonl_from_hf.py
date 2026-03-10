from datasets import load_dataset
import json

ds = load_dataset("openai/gsm8k", "main")

# train
with open("/home/shared/megatron_dir/datasets/gsm8k/train.jsonl", "w") as f:
    for row in ds["train"]:
        f.write(json.dumps(row) + "\n")

# test
with open("/home/shared/megatron_dir/datasets/gsm8k/test.jsonl", "w") as f:
    for row in ds["test"]:
        f.write(json.dumps(row) + "\n")

