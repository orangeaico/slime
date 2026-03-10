#!/usr/bin/env python3
"""Inspect saved rollout trajectories from SWE-bench training.

By default, shows complete prompts, responses, and patches for full inspection.
Teacher logprobs are automatically truncated in saved files to save space.

Usage:
    python inspect_trajectories.py                        # Show latest rollout (full output)
    python inspect_trajectories.py --rollout-id 0         # Show specific rollout
    python inspect_trajectories.py --sample-idx 0         # Show specific sample
    python inspect_trajectories.py --truncate             # Truncate long outputs for compact view
"""

import argparse
from pathlib import Path

import torch


def find_latest_rollout(trajectory_dir: Path) -> Path | None:
    """Find the most recent rollout file."""
    rollout_files = list(trajectory_dir.glob("rollout_*.pt"))
    if not rollout_files:
        return None
    return max(rollout_files, key=lambda p: p.stat().st_mtime)


def inspect_trajectory(trajectory_path: Path, sample_idx: int | None = None, truncate_output: bool = False):
    """Inspect a saved trajectory file.

    Args:
        trajectory_path: Path to the .pt trajectory file
        sample_idx: Optional specific sample index to show (None = show all)
        truncate_output: If True, truncate long prompts/responses (False by default)
    """
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        return

    print(f"\n{'='*80}")
    print(f"Loading trajectory: {trajectory_path}")
    print(f"{'='*80}\n")

    data = torch.load(trajectory_path)
    rollout_id = data.get("rollout_id", "unknown")
    samples = data["samples"]

    print(f"Rollout ID: {rollout_id}")
    print(f"Number of samples: {len(samples)}")
    print(f"File size: {trajectory_path.stat().st_size / 1024:.2f} KB\n")

    if sample_idx is not None:
        # Show specific sample
        if sample_idx >= len(samples):
            print(f"Error: Sample index {sample_idx} out of range (0-{len(samples)-1})")
            return
        samples_to_show = [samples[sample_idx]]
        start_idx = sample_idx
    else:
        # Show all samples
        samples_to_show = samples
        start_idx = 0

    for idx, sample in enumerate(samples_to_show, start=start_idx):
        print(f"\n{'-'*80}")
        print(f"Sample {idx}")
        print(f"{'-'*80}")

        # Basic info
        print(f"\nInstance ID: {sample.get('metadata', {}).get('instance_id', 'unknown')}")
        print(f"Repository: {sample.get('metadata', {}).get('repo', 'unknown')}")
        print(f"Status: {sample.get('status', 'unknown')}")

        # Prompt - show full by default
        prompt = sample.get("prompt", "")
        if truncate_output and len(prompt) > 1000:
            print(f"\nPrompt (first 1000 chars):\n{prompt[:1000]}...")
            print(f"[Total prompt length: {len(prompt)} chars - use --show-full to see complete prompt]")
        else:
            print(f"\nPrompt ({len(prompt)} chars):\n{prompt}")

        # Response - show full by default
        response = sample.get("response", "")
        if truncate_output and len(response) > 2000:
            print(f"\nResponse (first 2000 chars):\n{response[:2000]}...")
            print(f"[Total response length: {len(response)} chars - use --show-full to see complete response]")
        else:
            print(f"\nResponse ({len(response)} chars):\n{response}")

        # Tokens and loss mask
        tokens = sample.get("tokens", [])
        loss_mask = sample.get("loss_mask", [])
        response_length = sample.get("response_length", 0)

        print(f"\nToken stats:")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Response tokens: {response_length}")
        print(f"  Prompt tokens: {len(tokens) - response_length}")
        if loss_mask:
            trained_tokens = sum(loss_mask)
            print(f"  Trained tokens (loss_mask=1): {trained_tokens}")
            print(f"  Masked tokens (loss_mask=0): {len(loss_mask) - trained_tokens}")

        # Reward
        reward = sample.get("reward", None)
        if reward and isinstance(reward, dict):
            print(f"\nReward: dict with keys {list(reward.keys())}")
            meta_info = reward.get("meta_info", {})
            if "input_token_logprobs" in meta_info:
                logprobs = meta_info["input_token_logprobs"]
                if meta_info.get("_truncated", False):
                    original_length = meta_info.get("_original_length", "unknown")
                    print(f"  Teacher log-probs: {original_length} tokens (truncated to {len(logprobs)} for saving)")
                    print(f"  First 5 logprobs: {logprobs[:5]}")
                    print(f"  Last 5 logprobs: {logprobs[-5:]}")
                else:
                    print(f"  Teacher log-probs available: {len(logprobs)} tokens")
                    if len(logprobs) <= 10:
                        print(f"  Log-probs: {logprobs}")
        else:
            print(f"\nReward: {reward}")

        # Metadata
        metadata = sample.get("metadata", {})
        if "error" in metadata:
            print(f"\nError: {metadata['error']}")
        if "patch" in metadata:
            patch = metadata["patch"]
            if patch:
                print(f"\nPatch generated ({len(patch)} chars):")
                if truncate_output and len(patch) > 1000:
                    print(f"{patch[:1000]}...")
                    print(f"[Truncated - use --show-full to see complete patch]")
                else:
                    print(patch)

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SWE-bench rollout trajectories",
        epilog="By default, shows complete prompts and responses. Use --truncate to shorten long outputs."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("/tmp/swe_agent_trajectories"),
        help="Directory containing trajectory files (default: /tmp/swe_agent_trajectories)",
    )
    parser.add_argument(
        "--rollout-id",
        type=int,
        help="Rollout ID to inspect (default: latest)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        help="Sample index to inspect (default: show all)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate long prompts/responses/patches for compact display",
    )

    args = parser.parse_args()

    # Determine trajectory file to inspect
    if args.rollout_id is not None:
        trajectory_path = args.trajectory_dir / f"rollout_{args.rollout_id}.pt"
    else:
        trajectory_path = find_latest_rollout(args.trajectory_dir)
        if trajectory_path is None:
            print(f"Error: No trajectory files found in {args.trajectory_dir}")
            print(f"\nMake sure training has run at least once with:")
            print(f"  --save-debug-rollout-data /tmp/swe_agent_trajectories/rollout_{{rollout_id}}.pt")
            return

    inspect_trajectory(trajectory_path, args.sample_idx, args.truncate)


if __name__ == "__main__":
    main()
