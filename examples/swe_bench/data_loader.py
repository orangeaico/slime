"""Data loader for SWE-bench dataset.

Transforms SWE-bench instances into JSONL format for slime training.
"""

import json
from pathlib import Path


def format_problem_statement(instance: dict) -> str:
    """Format problem statement for the model.

    Args:
        instance: SWE-bench instance

    Returns:
        str: Formatted problem statement
    """
    problem = instance["problem_statement"]
    repo = instance["repo"]
    version = instance.get("version", "unknown")

    return f"""Repository: {repo}
Version: {version}

{problem}

Please analyze the issue and make the necessary code changes to fix it."""


def create_swebench_jsonl(
    output_path: str = "data/train.jsonl",
    split: str = "test",
    max_instances: int = 100,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
):
    """Create JSONL training data from SWE-bench dataset.

    Args:
        output_path: Path to output JSONL file
        split: Dataset split to use ("test" for SWE-bench Lite)
        max_instances: Maximum number of instances to include
        dataset_name: Hugging Face dataset name
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)

    print(f"Processing {min(max_instances, len(ds))} instances...")

    output_file = Path(__file__).parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for idx, instance in enumerate(ds):
            if idx >= max_instances:
                break

            # Create slime-compatible sample
            data = {
                "prompt": format_problem_statement(instance),
                "metadata": {
                    "instance_id": instance["instance_id"],
                    "repo": instance["repo"],
                    "base_commit": instance["base_commit"],
                    "version": instance.get("version", "unknown"),
                    "problem_statement": instance["problem_statement"],
                    "hints_text": instance.get("hints_text", ""),
                    "created_at": instance.get("created_at", ""),
                },
            }

            f.write(json.dumps(data) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} instances...")

    print(f"\nCreated {output_file} with {min(max_instances, len(ds))} instances")
    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create SWE-bench training data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/train.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=100,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name",
    )

    args = parser.parse_args()

    create_swebench_jsonl(
        output_path=args.output,
        max_instances=args.max_instances,
        dataset_name=args.dataset,
    )
