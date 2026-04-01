#!/usr/bin/env python3
"""
Speculative Decoding Hit Ratio Analysis

This script analyzes the potential hit ratio for n-gram based speculative decoding.
Given multiple rollouts per prompt, it uses the first rollout as a draft and measures
how well the draft can predict tokens in subsequent rollouts.

The n-gram matching scheme (similar to vLLM's approach):
- For each position in the target response, look at the previous n tokens (context)
- Search for this context in the draft response
- If found, use the next k tokens from the draft as speculation candidates
- Count how many of these speculated tokens match the actual target tokens

Usage:
    python speculative_decoding_analysis.py --input all_rollouts.jsonl
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --model /path/to/model
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --k 4
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --context-size 4 --k 5
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-first-match
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Rollout:
    """Represents a single rollout for a prompt."""
    instance_id: int
    rollout_id: int
    user_content: str
    assistant_content: str


@dataclass
class InstanceRollouts:
    """Collection of rollouts for a single instance/prompt."""
    instance_id: int
    rollouts: list[Rollout] = field(default_factory=list)

    def get_draft(self) -> Rollout | None:
        """Get the draft rollout (rollout_id=0)."""
        for r in self.rollouts:
            if r.rollout_id == 0:
                return r
        return None

    def get_verification_rollouts(self) -> list[Rollout]:
        """Get all non-draft rollouts for verification."""
        return [r for r in self.rollouts if r.rollout_id != 0]


@dataclass
class HitRatioResult:
    """Result of hit ratio calculation for a single comparison."""
    total_positions: int  # Total positions where speculation could be attempted
    context_found_positions: int  # Positions where context was found (regardless of speculation tokens)
    matched_positions: int  # Positions where context was found AND tokens were available to speculate
    total_hits: int  # Total number of matching tokens across all speculations
    total_speculated: int  # Total number of tokens speculated
    total_context_length: int  # Sum of longest matching context lengths
    total_candidates: int  # Total number of candidate matches found
    hit_lengths: dict[int, int] = field(default_factory=dict)  # Distribution of hit lengths: length -> count

    @property
    def hit_ratio(self) -> float:
        """Ratio of hits to total speculated tokens."""
        if self.total_speculated == 0:
            return 0.0
        return self.total_hits / self.total_speculated

    @property
    def context_found_rate(self) -> float:
        """Fraction of positions where context was found in draft (regardless of available speculation tokens)."""
        if self.total_positions == 0:
            return 0.0
        return self.context_found_positions / self.total_positions

    @property
    def match_rate(self) -> float:
        """Fraction of positions where context was found AND tokens were available to speculate."""
        if self.total_positions == 0:
            return 0.0
        return self.matched_positions / self.total_positions

    @property
    def avg_context_length(self) -> float:
        """Average longest matching context length per matched position."""
        if self.matched_positions == 0:
            return 0.0
        return self.total_context_length / self.matched_positions

    @property
    def avg_candidates(self) -> float:
        """Average number of candidate matches per matched position."""
        if self.matched_positions == 0:
            return 0.0
        return self.total_candidates / self.matched_positions


# =============================================================================
# Tokenization
# =============================================================================

def load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    Load tokenizer from a model path or HuggingFace model name.

    Args:
        model_path: Path to local model or HuggingFace model identifier

    Returns:
        Loaded AutoTokenizer instance
    """
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    return tokenizer


def tokenize_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    """
    Tokenize text using the model's tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        text: Text to tokenize

    Returns:
        List of token IDs
    """
    return tokenizer.encode(text, add_special_tokens=False)


# =============================================================================
# Data Loading
# =============================================================================

def load_rollouts(input_file: str) -> dict[int, InstanceRollouts]:
    """
    Load rollouts from JSONL file and group by instance_id.

    Expected format per line:
    {"instance_id": int, "rollout_id": int, "messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    instances: dict[int, InstanceRollouts] = {}

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                instance_id = data["instance_id"]
                rollout_id = data["rollout_id"]
                messages = data["messages"]

                # Extract user and assistant content
                user_content = ""
                assistant_content = ""
                for msg in messages:
                    if msg["role"] == "user":
                        user_content = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_content = msg["content"]

                rollout = Rollout(
                    instance_id=instance_id,
                    rollout_id=rollout_id,
                    user_content=user_content,
                    assistant_content=assistant_content,
                )

                if instance_id not in instances:
                    instances[instance_id] = InstanceRollouts(instance_id=instance_id)
                instances[instance_id].rollouts.append(rollout)

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping line {line_num}: {e}")
                continue

    # Sort rollouts within each instance by rollout_id
    for instance in instances.values():
        instance.rollouts.sort(key=lambda r: r.rollout_id)

    logger.info(f"Loaded {len(instances)} instances from {input_file}")
    return instances


# =============================================================================
# N-gram Matching for Speculative Decoding
# =============================================================================

def build_ngram_index(tokens: list[int], n: int) -> dict[tuple[int, ...], list[int]]:
    """
    Build an index mapping n-grams to their positions in the token sequence.

    Args:
        tokens: List of token IDs
        n: Size of n-gram (context window)

    Returns:
        Dictionary mapping n-gram tuples to list of positions where they occur
    """
    index: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        index[ngram].append(i)

    return index


def find_longest_match(
    target_tokens: list[int],
    pos: int,
    draft_tokens: list[int],
    draft_index: dict[tuple[int, ...], list[int]],
    min_context: int,
) -> tuple[int, int, int] | None:
    """
    Find the longest matching context in the draft for a given position.

    Uses the minimum context index to find candidates, then extends the match
    by comparing tokens before the context.

    Args:
        target_tokens: Target token sequence
        pos: Current position in target (where we want to speculate)
        draft_tokens: Draft token sequence
        draft_index: Pre-built n-gram index for minimum context size
        min_context: Minimum context size

    Returns:
        Tuple of (draft_position, context_length, num_candidates) for longest match, or None if no match
    """
    # Get the minimum context from target
    min_ctx = tuple(target_tokens[pos - min_context:pos])

    if min_ctx not in draft_index:
        return None

    candidates = draft_index[min_ctx]
    num_candidates = len(candidates)

    # Find the candidate with the longest matching context
    best_draft_pos = None
    best_context_len = 0

    for draft_pos in candidates:
        # Start with minimum context length
        ctx_len = min_context

        # Extend backwards to find longest match
        # Check how many tokens before the context also match
        target_start = pos - min_context
        draft_start = draft_pos

        while target_start > 0 and draft_start > 0:
            if target_tokens[target_start - 1] == draft_tokens[draft_start - 1]:
                ctx_len += 1
                target_start -= 1
                draft_start -= 1
            else:
                break

        if ctx_len > best_context_len:
            best_context_len = ctx_len
            # Adjust draft_pos to point to start of the extended context
            best_draft_pos = draft_pos - (ctx_len - min_context)

    if best_draft_pos is not None:
        return (best_draft_pos, best_context_len, num_candidates)

    return None


def find_first_match(
    target_tokens: list[int],
    pos: int,
    draft_tokens: list[int],
    draft_index: dict[tuple[int, ...], list[int]],
    min_context: int,
) -> tuple[int, int, int] | None:
    """
    Find the first matching context in the draft for a given position.

    Uses the minimum context index to find the first candidate match (no extension).

    Args:
        target_tokens: Target token sequence
        pos: Current position in target (where we want to speculate)
        draft_tokens: Draft token sequence
        draft_index: Pre-built n-gram index for minimum context size
        min_context: Minimum context size

    Returns:
        Tuple of (draft_position, context_length, num_candidates) for first match, or None if no match
    """
    # Get the minimum context from target
    min_ctx = tuple(target_tokens[pos - min_context:pos])

    if min_ctx not in draft_index:
        return None

    candidates = draft_index[min_ctx]
    num_candidates = len(candidates)

    # Use the first candidate with minimum context length
    draft_pos = candidates[0]
    ctx_len = min_context

    return (draft_pos, ctx_len, num_candidates)


def calculate_hit_ratio(
    draft_tokens: list[int],
    target_tokens: list[int],
    k: int,
    context_size: int = 3,
    use_longest_match: bool = True,
) -> HitRatioResult:
    """
    Calculate hit ratio for speculative decoding.

    For each position in the target (after context_size tokens), we:
    1. Find a matching context (minimum context_size) in the draft
    2. If found, speculate the next k tokens from the draft
    3. Count how many of these match the actual target tokens

    Args:
        draft_tokens: Token IDs from the draft response
        target_tokens: Token IDs from the target response
        k: Number of tokens to speculate
        context_size: Minimum context window size for n-gram matching
        use_longest_match: If True, find longest matching context; if False, use first match

    Returns:
        HitRatioResult with hit statistics
    """
    # Build n-gram index only for minimum context size
    draft_index = build_ngram_index(draft_tokens, context_size)

    total_positions = 0
    context_found_positions = 0
    matched_positions = 0
    total_hits = 0
    total_speculated = 0
    total_context_length = 0
    total_candidates = 0
    hit_lengths: dict[int, int] = defaultdict(int)

    # Select matching function based on configuration
    match_fn = find_longest_match if use_longest_match else find_first_match

    # Iterate through target positions where we can attempt speculation
    for pos in range(context_size, len(target_tokens)):
        total_positions += 1

        # Find matching context
        match = match_fn(
            target_tokens, pos, draft_tokens, draft_index,
            min_context=context_size,
        )

        if match is None:
            continue

        draft_pos, ctx_len, num_candidates = match
        context_found_positions += 1

        # Get speculated tokens from draft (tokens after the matched context)
        spec_start = draft_pos + ctx_len
        spec_end = min(spec_start + k, len(draft_tokens))
        speculated_tokens = draft_tokens[spec_start:spec_end]

        if not speculated_tokens:
            continue

        # Context was found and we have tokens to speculate
        matched_positions += 1
        total_context_length += ctx_len
        total_candidates += num_candidates

        # Get actual tokens from target
        actual_end = min(pos + len(speculated_tokens), len(target_tokens))
        actual_tokens = target_tokens[pos:actual_end]

        # Count matches (consecutive from start)
        hits = 0
        for i, (spec, actual) in enumerate(zip(speculated_tokens, actual_tokens)):
            if spec == actual:
                hits += 1
            else:
                break  # Stop counting at first mismatch

        total_speculated += len(speculated_tokens)
        total_hits += hits
        hit_lengths[hits] += 1

    return HitRatioResult(
        total_positions=total_positions,
        context_found_positions=context_found_positions,
        matched_positions=matched_positions,
        total_hits=total_hits,
        total_speculated=total_speculated,
        total_context_length=total_context_length,
        total_candidates=total_candidates,
        hit_lengths=dict(hit_lengths),
    )


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for the analysis."""
    k: int = 4
    context_size: int = 3
    model_path: str = ""
    use_longest_match: bool = True  # If True, find longest matching context; if False, use first match


@dataclass
class AggregatedResults:
    """Aggregated results across all instances and rollouts."""
    k_value: int
    num_instances: int
    num_comparisons: int
    total_positions: int
    total_context_found_positions: int
    total_matched_positions: int
    total_hits: int
    total_speculated: int
    total_context_length: int
    total_candidates: int = 0
    per_instance_hit_ratios: list[float] = field(default_factory=list)
    hit_lengths: dict[int, int] = field(default_factory=dict)  # Distribution of hit lengths

    @property
    def overall_hit_ratio(self) -> float:
        """Overall hit ratio across all comparisons."""
        if self.total_speculated == 0:
            return 0.0
        return self.total_hits / self.total_speculated

    @property
    def avg_hit_ratio(self) -> float:
        """Average hit ratio per instance."""
        if not self.per_instance_hit_ratios:
            return 0.0
        return sum(self.per_instance_hit_ratios) / len(self.per_instance_hit_ratios)

    @property
    def context_found_rate(self) -> float:
        """Fraction of positions where context was found in draft (regardless of available speculation tokens)."""
        if self.total_positions == 0:
            return 0.0
        return self.total_context_found_positions / self.total_positions

    @property
    def match_rate(self) -> float:
        """Fraction of positions where context was found AND tokens were available to speculate."""
        if self.total_positions == 0:
            return 0.0
        return self.total_matched_positions / self.total_positions

    @property
    def avg_context_length(self) -> float:
        """Average longest matching context length per matched position."""
        if self.total_matched_positions == 0:
            return 0.0
        return self.total_context_length / self.total_matched_positions

    @property
    def avg_candidates(self) -> float:
        """Average number of candidate matches per matched position."""
        if self.total_matched_positions == 0:
            return 0.0
        return self.total_candidates / self.total_matched_positions


def analyze_instance(
    instance: InstanceRollouts,
    tokenizer: AutoTokenizer,
    config: AnalysisConfig,
) -> list[HitRatioResult]:
    """
    Analyze a single instance's rollouts.

    Args:
        instance: Collection of rollouts for this instance
        tokenizer: HuggingFace tokenizer to use
        config: Analysis configuration

    Returns:
        List of HitRatioResults (one per verification rollout)
    """
    draft = instance.get_draft()
    if draft is None:
        logger.warning(f"Instance {instance.instance_id} has no draft rollout")
        return []

    verification_rollouts = instance.get_verification_rollouts()
    if not verification_rollouts:
        logger.warning(f"Instance {instance.instance_id} has no verification rollouts")
        return []

    # Tokenize draft
    draft_tokens = tokenize_text(tokenizer, draft.assistant_content)

    results: list[HitRatioResult] = []

    for rollout in verification_rollouts:
        target_tokens = tokenize_text(tokenizer, rollout.assistant_content)

        result = calculate_hit_ratio(
            draft_tokens=draft_tokens,
            target_tokens=target_tokens,
            k=config.k,
            context_size=config.context_size,
            use_longest_match=config.use_longest_match,
        )
        results.append(result)

    return results


def run_analysis(
    instances: dict[int, InstanceRollouts],
    tokenizer: AutoTokenizer,
    config: AnalysisConfig,
) -> AggregatedResults:
    """
    Run full analysis across all instances.

    Args:
        instances: Dictionary of instance_id to InstanceRollouts
        tokenizer: HuggingFace tokenizer to use
        config: Analysis configuration

    Returns:
        Aggregated results for the specified k value
    """
    # Initialize aggregated results
    aggregated = AggregatedResults(
        k_value=config.k,
        num_instances=0,
        num_comparisons=0,
        total_positions=0,
        total_context_found_positions=0,
        total_matched_positions=0,
        total_hits=0,
        total_speculated=0,
        total_context_length=0,
        total_candidates=0,
    )

    for instance_id, instance in instances.items():
        instance_results = analyze_instance(instance, tokenizer, config)

        if not instance_results:
            continue

        # Aggregate results for this instance
        instance_hits = sum(r.total_hits for r in instance_results)
        instance_speculated = sum(r.total_speculated for r in instance_results)
        instance_positions = sum(r.total_positions for r in instance_results)
        instance_context_found = sum(r.context_found_positions for r in instance_results)
        instance_matched = sum(r.matched_positions for r in instance_results)
        instance_context_len = sum(r.total_context_length for r in instance_results)
        instance_candidates = sum(r.total_candidates for r in instance_results)

        # Aggregate hit_lengths distribution
        for result in instance_results:
            for hit_len, count in result.hit_lengths.items():
                if hit_len not in aggregated.hit_lengths:
                    aggregated.hit_lengths[hit_len] = 0
                aggregated.hit_lengths[hit_len] += count

        if instance_speculated > 0:
            instance_hit_ratio = instance_hits / instance_speculated
            aggregated.per_instance_hit_ratios.append(instance_hit_ratio)

        aggregated.num_instances += 1
        aggregated.num_comparisons += len(instance_results)
        aggregated.total_positions += instance_positions
        aggregated.total_context_found_positions += instance_context_found
        aggregated.total_matched_positions += instance_matched
        aggregated.total_hits += instance_hits
        aggregated.total_speculated += instance_speculated
        aggregated.total_context_length += instance_context_len
        aggregated.total_candidates += instance_candidates

    return aggregated


def print_results(results: AggregatedResults, config: AnalysisConfig) -> None:
    """Print analysis results."""
    print("\n" + "=" * 80)
    print("SPECULATIVE DECODING HIT RATIO ANALYSIS")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Context size (n-gram): {config.context_size}")
    print(f"Speculation length (k): {config.k}")
    match_strategy = "longest match" if config.use_longest_match else "first match"
    print(f"Matching strategy: {match_strategy}")
    print("-" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Hit Ratio (overall):        {results.overall_hit_ratio:.4f}")
    print(f"  Hit Ratio (avg/instance):   {results.avg_hit_ratio:.4f}")
    print(f"  Context found rate:         {results.context_found_rate:.4f}")
    print(f"  Usable context rate:        {results.match_rate:.4f}")
    print(f"  Avg context length:         {results.avg_context_length:.2f}")
    print(f"  Instances analyzed:         {results.num_instances}")
    print(f"  Total comparisons:          {results.num_comparisons}")
    print(f"  Total positions evaluated:  {results.total_positions}")

    print(f"\nHit Length Distribution (for k={config.k}):")
    print(f"  Length | Count | Cumulative %")
    print(f"  " + "-" * 35)

    if results.hit_lengths:
        total_matches = sum(results.hit_lengths.values())
        cumulative_count = 0
        for hit_len in sorted(results.hit_lengths.keys(), reverse=True):
            count = results.hit_lengths[hit_len]
            cumulative_count += count
            cumulative_percentage = (cumulative_count / total_matches * 100) if total_matches > 0 else 0
            print(f"  {hit_len:>6} | {cumulative_count:>5} | {cumulative_percentage:>6.2f}%")
    else:
        print("  No hits recorded")

    print("\n" + "=" * 80)
    print("Metrics explanation:")
    print("  - Hit Ratio: Fraction of speculated tokens that matched (overall)")
    print("  - Avg/Instance: Average hit ratio per instance")
    print("  - Context found rate: Fraction of positions where context was found (regardless of tokens to speculate)")
    print("  - Usable context rate: Fraction of positions where context was found AND tokens were available to speculate")
    print("  - Avg context length: Average longest matching context length per usable match (tokens)")
    print("  - Hit length distribution: Cumulative count of cases where >= N consecutive tokens matched")
    print("    (e.g., 'Length 3' shows how many cases had >= 3 tokens matching)")
    print("")


def save_results(
    results: AggregatedResults,
    config: AnalysisConfig,
    output_file: str,
) -> None:
    """Save results to JSON file."""
    output = {
        "config": {
            "k": config.k,
            "context_size": config.context_size,
            "model_path": config.model_path,
            "use_longest_match": config.use_longest_match,
        },
        "results": {
            "k_value": results.k_value,
            "num_instances": results.num_instances,
            "num_comparisons": results.num_comparisons,
            "total_positions": results.total_positions,
            "total_context_found_positions": results.total_context_found_positions,
            "total_matched_positions": results.total_matched_positions,
            "total_hits": results.total_hits,
            "total_speculated": results.total_speculated,
            "total_context_length": results.total_context_length,
            "total_candidates": results.total_candidates,
            "overall_hit_ratio": results.overall_hit_ratio,
            "avg_hit_ratio": results.avg_hit_ratio,
            "context_found_rate": results.context_found_rate,
            "match_rate": results.match_rate,
            "avg_context_length": results.avg_context_length,
            "avg_candidates": results.avg_candidates,
            "hit_lengths": results.hit_lengths,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze speculative decoding hit ratios using n-gram matching"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with all rollouts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path to model or HuggingFace model name (used for tokenizer, default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Speculation length (number of tokens to speculate) (default: 4)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=5,
        help="Size of context window for n-gram matching (default: 3)",
    )
    parser.add_argument(
        "--use-first-match",
        action="store_true",
        default=False,
        help="If set, use first match instead of longest match (default: use longest match)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results (optional)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Load tokenizer
    tokenizer = load_tokenizer(args.model)

    # Create config
    config = AnalysisConfig(
        k=args.k,
        context_size=args.context_size,
        model_path=args.model,
        use_longest_match=not args.use_first_match,
    )

    logger.info(f"Loading rollouts from {args.input}")
    instances = load_rollouts(args.input)

    if not instances:
        logger.error("No instances loaded")
        return 1

    match_strategy = "longest match" if config.use_longest_match else "first match"
    logger.info(f"Running analysis with k={config.k}, context_size={config.context_size}, strategy={match_strategy}")
    results = run_analysis(instances, tokenizer, config)

    print_results(results, config)

    if args.output:
        save_results(results, config, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
