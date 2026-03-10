# Dataset Prep for train and test for basic 1 step RL training. Aim is to create jsonl with label field and clear instruction prompt.

## Step 1
## Download dataset from hf and store in jsonl through download_jsonl_from_hf.py. Update hardcoded paths in this script.

python download_jsonl_from_hf.py

## Step 2
## Extract labels from data and add an explicit label field for every input in the jsonl through extract_label.py. Update hardcoded paths in this script.

python extract_label.py

## Step 3
## Update the prompt field to add prefix and suffix instructions for clarity of output format through the script update_prompts.py. 

python update_prompts.py <input_jsonl>
