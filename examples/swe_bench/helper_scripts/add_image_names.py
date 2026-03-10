#!/usr/bin/env python3
"""Add image_name field to train.jsonl based on instance_id."""

import json
import sys

def instance_id_to_image_name(instance_id: str) -> str:
    """Convert instance_id to Docker image name.

    Example:
        astropy__astropy-12907 -> swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
        django__django-10924 -> swebench/sweb.eval.x86_64.django_1776_django-10924
    """
    # Split on __ to get repo_owner and repo-issue
    parts = instance_id.split("__")
    if len(parts) != 2:
        raise ValueError(f"Invalid instance_id format: {instance_id}")

    repo_owner = parts[0]  # e.g., "astropy"
    repo_issue = parts[1]  # e.g., "astropy-12907"

    # Format: swebench/sweb.eval.x86_64.{repo_owner}_1776_{repo_issue}
    return f"swebench/sweb.eval.x86_64.{repo_owner}_1776_{repo_issue}"


def add_image_names(input_file: str, output_file: str):
    """Add image_name field to each entry in the JSONL file."""

    count = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())

            # Get instance_id and convert to image_name
            instance_id = data['metadata']['instance_id']
            image_name = instance_id_to_image_name(instance_id)

            # Add image_name to metadata
            data['metadata']['image_name'] = image_name

            # Write updated entry
            fout.write(json.dumps(data) + '\n')
            count += 1

            if count <= 5:
                print(f"{instance_id} -> {image_name}")

    print(f"\nProcessed {count} entries")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    input_file = "examples/swe_bench/data/train.jsonl"
    output_file = "examples/swe_bench/data/train_with_images.jsonl"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    add_image_names(input_file, output_file)
