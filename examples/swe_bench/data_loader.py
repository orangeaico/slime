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
    filter_repo: str = None,
):
    """Create JSONL training data from SWE-bench dataset.

    Args:
        output_path: Path to output JSONL file
        split: Dataset split to use ("test" for SWE-bench Lite)
        max_instances: Maximum number of instances to include
        dataset_name: Hugging Face dataset name
        filter_repo: Filter instances by repo name (e.g., "xarray")
    """
    from datasets import load_dataset

    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split=split)

    if filter_repo:
        print(f"Filtering for instances with '{filter_repo}' in instance_id...")
        ds = [instance for instance in ds if filter_repo in instance["instance_id"]]
        print(f"Found {len(ds)} instances matching filter")

    print(f"Processing {min(max_instances, len(ds))} instances...")

    output_file = Path(__file__).parent / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    with open(output_file, "w") as f:
        for idx, instance in enumerate(ds):
            if processed_count >= max_instances:
                break

            instance_id = instance["instance_id"]

            # Extract parts from instance_id
            # Example: "pydata__xarray-7393" -> org="pydata", repo_issue="xarray-7393"
            # Image name template: swebench/sweb.eval.x86_64.{org}_1776_{repo_issue}
            parts = instance_id.split('__')
            if len(parts) == 2:
                org = parts[0]  # e.g., "pydata"
                repo_issue = parts[1]  # e.g., "xarray-7393"
                image_name = f"swebench/sweb.eval.x86_64.{org}_1776_{repo_issue}"
            else:
                # Fallback if format is unexpected
                image_name = f"swebench/sweb.eval.x86_64.unknown_1776_{instance_id}"

            # Create slime-compatible sample
            data = {
                "prompt": format_problem_statement(instance),
                "metadata": {
                    "instance_id": instance_id,
                    "repo": instance["repo"],
                    "base_commit": instance["base_commit"],
                    "version": instance.get("version", "unknown"),
                    "problem_statement": instance["problem_statement"],
                    "hints_text": instance.get("hints_text", ""),
                    "created_at": instance.get("created_at", ""),
                    "image_name": image_name,
                },
            }

            f.write(json.dumps(data) + "\n")
            processed_count += 1

            if processed_count % 10 == 0:
                print(f"Processed {processed_count} instances...")

    print(f"\nCreated {output_file} with {processed_count} instances")
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
        default="princeton-nlp/SWE-bench_Verified",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--filter-repo",
        type=str,
        default="xarray",
        help="Filter instances by repo name in instance_id (e.g., 'xarray')",
    )

    args = parser.parse_args()

    create_swebench_jsonl(
        output_path=args.output,
        max_instances=args.max_instances,
        dataset_name=args.dataset,
        filter_repo=args.filter_repo,
    )
