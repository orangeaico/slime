#!/usr/bin/env python3
"""
Script to add prefix and suffix to prompts in a JSONL file.

This script:
1. Reads a JSONL file
2. For each JSON object, reads the "prompt" field
3. Adds a prefix and suffix to the prompt
4. Saves the output file with "_updated_prompt" suffix

Usage:
    python update_prompts.py input_data.jsonl
    # Output: input_data_updated_prompt.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Prompt prefix and suffix for math problems
PROMPT_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
PROMPT_SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def add_prompt_wrapper(content: str) -> str:
    """Add prefix and suffix to prompt if they don't already exist."""
    if content is None:
        return None

    # Check if prefix already exists
    if not content.strip().startswith("Solve the following math problem"):
        content = PROMPT_PREFIX + content

    # Check if suffix already exists
    if not content.strip().endswith("Remember to put your answer on its own line after \"Answer:\"."):
        content = content + PROMPT_SUFFIX

    return content


def update_prompts_in_jsonl(input_file: str, output_file: str = None) -> None:
    """
    Read a JSONL file, update prompts with prefix and suffix, and write to output file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (default: input_file_updated_prompt.jsonl)
    """
    input_path = Path(input_file)

    # Generate output filename if not provided
    if output_file is None:
        stem = input_path.stem
        suffix = input_path.suffix
        output_file = str(input_path.parent / f"{stem}_updated_prompt{suffix}")

    output_path = Path(output_file)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            processed = 0
            skipped = 0

            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)

                    # Check if prompt field exists
                    if "prompt" in data:
                        original_prompt = data["prompt"]

                        # Handle both string and list formats
                        if isinstance(original_prompt, str):
                            data["prompt"] = add_prompt_wrapper(original_prompt)
                        elif isinstance(original_prompt, list) and len(original_prompt) > 0:
                            # If prompt is a list (chat format), wrap the last user message
                            if isinstance(original_prompt[-1], dict) and original_prompt[-1].get("role") == "user":
                                original_prompt[-1]["content"] = add_prompt_wrapper(original_prompt[-1]["content"])
                            data["prompt"] = original_prompt

                        # Write the modified JSON
                        outfile.write(json.dumps(data) + '\n')
                        processed += 1
                    else:
                        print(f"Warning: Line {line_num} has no 'prompt' field, skipping")
                        # Still write the line as-is
                        outfile.write(line)
                        skipped += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num}: {e}")
                    skipped += 1

        print(f"\nProcessing complete!")
        print(f"  Processed: {processed} lines")
        print(f"  Skipped: {skipped} lines")
        print(f"  Total: {processed + skipped} lines")
        print(f"\nOutput saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add prefix and suffix to prompts in a JSONL file"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: input_file_updated_prompt.jsonl)"
    )

    args = parser.parse_args()

    update_prompts_in_jsonl(args.input_file, args.output)


if __name__ == "__main__":
    main()
