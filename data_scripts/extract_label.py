import json
import re

input_file = "/home/shared/megatron_dir/datasets/gsm8k/train.jsonl"
output_file = "/home/shared/megatron_dir/datasets/gsm8k/train_with_labels.jsonl"

pattern = re.compile(r"####\s*([-+]?\d*.?\d+)")

with open(input_file, "r") as fin, open(output_file, "w") as fout:

    for line in fin:
        # print (line)
        row = json.loads(line)

        answer_text = row.get("answer", "")
        match = pattern.search(answer_text)

        if match:
            label = match.group(1)
            label = label.replace(",", "")  # Remove commas from numbers, e.g., "1,000" -> "1000"
            # convert to int if possible
            if label.isdigit() or (label.startswith("-") and label[1:].isdigit()):
                label = int(label)
            else:
                label = float(label)

            row["label"] = label
        else:
            row["label"] = None

        fout.write(json.dumps(row) + "\n")
