#!/usr/bin/env python3
"""
Speculative Decoding Hit Ratio Analysis

This script analyzes the potential hit ratio for n-gram based speculative decoding.
Given multiple rollouts per prompt, it can operate in two modes:
1. Single draft mode (default): Uses the first rollout (rollout_id=0) as a draft for all others
2. All rollouts mode (--use-all-rollouts): For each rollout, uses all other rollouts as drafts

The matching scheme:
- By default, uses token-based n-gram matching only
- Optionally (--use-char-fallback), enables hybrid token/character matching:
  - First tries token-based n-gram matching
  - If no token match is found, falls back to character-based prefix matching
  - Character matching uses a configurable threshold (default: 5 characters)
- If found, use the next k tokens from the draft as speculation candidates
- Count how many of these speculated tokens match the actual target tokens

Usage:
    python speculative_decoding_analysis.py --input all_rollouts.jsonl
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --model /path/to/model
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --k 4
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --context-size 4 --k 5
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-first-match
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-all-rollouts
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-char-fallback --char-threshold 10
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
    total_target_tokens: int = 0  # Total tokens to generate (target length - context_size)
    hit_lengths: dict[int, int] = field(default_factory=dict)  # Distribution of hit lengths: length -> count
    total_verification_cost: float = 0.0  # Total cost accounting for per-token verification overhead

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

    @property
    def speedup(self) -> float:
        """
        Speedup compared to non-speculative generation.

        Speedup = (total tokens to generate) / (total generation cost with verification overhead)

        Logic:
        - Without speculation: Need 1 generation step per token = total_target_tokens steps
        - With speculation: Need total_verification_cost steps accounting for verification overhead
        - Speedup shows how many times faster speculative decoding is

        Example: If speedup = 2.0, speculative decoding is 2x faster
        """
        if self.total_verification_cost == 0:
            return 0.0
        return self.total_target_tokens / self.total_verification_cost


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

def decode_tokens_to_text(tokenizer: AutoTokenizer, tokens: list[int]) -> str:
    """Decode tokens to text string."""
    return tokenizer.decode(tokens, skip_special_tokens=True)


def find_longest_char_match(
    target_text: str,
    target_pos_tokens: int,
    target_tokens: list[int],
    draft_text: str,
    draft_tokens: list[int],
    tokenizer: AutoTokenizer,
    min_chars: int,
) -> tuple[int, int] | None:
    """
    Find the longest character-based match in the draft for a given position.

    Args:
        target_text: Target text string
        target_pos_tokens: Current position in target tokens
        target_tokens: Target token sequence
        draft_text: Draft text string
        draft_tokens: Draft token sequence
        tokenizer: Tokenizer for decoding
        min_chars: Minimum character length to consider a match

    Returns:
        Tuple of (draft_token_position, match_length_in_chars) or None if no match
    """
    # Decode target up to current position to get character offset
    target_prefix_text = decode_tokens_to_text(tokenizer, target_tokens[:target_pos_tokens])
    target_char_pos = len(target_prefix_text)

    # Get a reasonable context window in characters (e.g., last 100 chars)
    context_start = max(0, target_char_pos - 100)
    context_text = target_text[context_start:target_char_pos]

    if len(context_text) < min_chars:
        return None

    # Try to find longest matching substring in draft
    best_draft_pos = None
    best_match_len = 0

    # Start with longer substrings and work down to min_chars
    for length in range(len(context_text), min_chars - 1, -1):
        search_str = context_text[-length:]
        pos = draft_text.find(search_str)

        if pos != -1:
            # Found a match, now convert character position back to token position
            # Find the token position in draft that corresponds to this character position
            draft_prefix = draft_text[:pos + length]

            # Binary search or sequential search to find token position
            draft_token_pos = 0
            for i in range(1, len(draft_tokens) + 1):
                decoded = decode_tokens_to_text(tokenizer, draft_tokens[:i])
                if len(decoded) >= len(draft_prefix):
                    draft_token_pos = i
                    break

            best_draft_pos = draft_token_pos
            best_match_len = length
            break

    if best_draft_pos is not None:
        return (best_draft_pos, best_match_len)

    return None


def find_first_char_match(
    target_text: str,
    target_pos_tokens: int,
    target_tokens: list[int],
    draft_text: str,
    draft_tokens: list[int],
    tokenizer: AutoTokenizer,
    min_chars: int,
) -> tuple[int, int] | None:
    """
    Find the first character-based match in the draft for a given position.

    Args:
        target_text: Target text string
        target_pos_tokens: Current position in target tokens
        target_tokens: Target token sequence
        draft_text: Draft text string
        draft_tokens: Draft token sequence
        tokenizer: Tokenizer for decoding
        min_chars: Minimum character length to consider a match

    Returns:
        Tuple of (draft_token_position, match_length_in_chars) or None if no match
    """
    # Decode target up to current position to get character offset
    target_prefix_text = decode_tokens_to_text(tokenizer, target_tokens[:target_pos_tokens])
    target_char_pos = len(target_prefix_text)

    # Get a reasonable context window in characters
    context_start = max(0, target_char_pos - 100)
    context_text = target_text[context_start:target_char_pos]

    if len(context_text) < min_chars:
        return None

    # Try to find first match with at least min_chars
    search_str = context_text[-min_chars:]
    pos = draft_text.find(search_str)

    if pos != -1:
        # Found a match, convert character position to token position
        draft_prefix = draft_text[:pos + min_chars]

        draft_token_pos = 0
        for i in range(1, len(draft_tokens) + 1):
            decoded = decode_tokens_to_text(tokenizer, draft_tokens[:i])
            if len(decoded) >= len(draft_prefix):
                draft_token_pos = i
                break

        return (draft_token_pos, min_chars)

    return None


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


def find_longest_char_match_multi_draft(
    target_text: str,
    target_pos_tokens: int,
    target_tokens: list[int],
    all_draft_texts: list[str],
    all_draft_tokens: list[list[int]],
    tokenizer: AutoTokenizer,
    min_chars: int,
) -> tuple[int, int, int] | None:
    """
    Find the longest character-based match across multiple draft sequences.

    Args:
        target_text: Target text string
        target_pos_tokens: Current position in target tokens
        target_tokens: Target token sequence
        all_draft_texts: List of draft text strings
        all_draft_tokens: List of draft token sequences
        tokenizer: Tokenizer for decoding
        min_chars: Minimum character length to consider a match

    Returns:
        Tuple of (draft_idx, draft_token_position, match_length_in_chars) or None if no match
    """
    best_draft_idx = None
    best_draft_pos = None
    best_match_len = 0

    for draft_idx, (draft_text, draft_tokens) in enumerate(zip(all_draft_texts, all_draft_tokens)):
        char_match = find_longest_char_match(
            target_text, target_pos_tokens, target_tokens,
            draft_text, draft_tokens, tokenizer, min_chars,
        )

        if char_match is not None:
            draft_pos, match_len = char_match
            if match_len > best_match_len:
                best_match_len = match_len
                best_draft_pos = draft_pos
                best_draft_idx = draft_idx

    if best_draft_idx is not None:
        return (best_draft_idx, best_draft_pos, best_match_len)

    return None


def find_first_char_match_multi_draft(
    target_text: str,
    target_pos_tokens: int,
    target_tokens: list[int],
    all_draft_texts: list[str],
    all_draft_tokens: list[list[int]],
    tokenizer: AutoTokenizer,
    min_chars: int,
) -> tuple[int, int, int] | None:
    """
    Find the first character-based match across multiple draft sequences.

    Args:
        target_text: Target text string
        target_pos_tokens: Current position in target tokens
        target_tokens: Target token sequence
        all_draft_texts: List of draft text strings
        all_draft_tokens: List of draft token sequences
        tokenizer: Tokenizer for decoding
        min_chars: Minimum character length to consider a match

    Returns:
        Tuple of (draft_idx, draft_token_position, match_length_in_chars) or None if no match
    """
    for draft_idx, (draft_text, draft_tokens) in enumerate(zip(all_draft_texts, all_draft_tokens)):
        char_match = find_first_char_match(
            target_text, target_pos_tokens, target_tokens,
            draft_text, draft_tokens, tokenizer, min_chars,
        )

        if char_match is not None:
            draft_pos, match_len = char_match
            return (draft_idx, draft_pos, match_len)

    return None


def find_longest_match_multi_draft(
    target_tokens: list[int],
    pos: int,
    all_draft_tokens: list[list[int]],
    all_draft_indices: list[dict[tuple[int, ...], list[int]]],
    min_context: int,
) -> tuple[int, int, int, int] | None:
    """
    Find the longest matching context across multiple draft sequences.

    Args:
        target_tokens: Target token sequence
        pos: Current position in target (where we want to speculate)
        all_draft_tokens: List of draft token sequences
        all_draft_indices: List of pre-built n-gram indices (one per draft)
        min_context: Minimum context size

    Returns:
        Tuple of (draft_idx, draft_position, context_length, num_candidates) for longest match, or None if no match
    """
    best_draft_idx = None
    best_draft_pos = None
    best_context_len = 0
    total_candidates = 0

    for draft_idx, (draft_tokens, draft_index) in enumerate(zip(all_draft_tokens, all_draft_indices)):
        match = find_longest_match(target_tokens, pos, draft_tokens, draft_index, min_context)

        if match is not None:
            draft_pos, ctx_len, num_candidates = match
            total_candidates += num_candidates

            if ctx_len > best_context_len:
                best_context_len = ctx_len
                best_draft_pos = draft_pos
                best_draft_idx = draft_idx

    if best_draft_idx is not None:
        return (best_draft_idx, best_draft_pos, best_context_len, total_candidates)

    return None


def find_first_match_multi_draft(
    target_tokens: list[int],
    pos: int,
    all_draft_tokens: list[list[int]],
    all_draft_indices: list[dict[tuple[int, ...], list[int]]],
    min_context: int,
) -> tuple[int, int, int, int] | None:
    """
    Find the first matching context across multiple draft sequences.
    Returns as soon as a match is found in any draft.

    Args:
        target_tokens: Target token sequence
        pos: Current position in target (where we want to speculate)
        all_draft_tokens: List of draft token sequences
        all_draft_indices: List of pre-built n-gram indices (one per draft)
        min_context: Minimum context size

    Returns:
        Tuple of (draft_idx, draft_position, context_length, num_candidates) for first match, or None if no match
    """
    for draft_idx, (draft_tokens, draft_index) in enumerate(zip(all_draft_tokens, all_draft_indices)):
        match = find_first_match(target_tokens, pos, draft_tokens, draft_index, min_context)

        if match is not None:
            draft_pos, ctx_len, num_candidates = match
            return (draft_idx, draft_pos, ctx_len, num_candidates)

    return None


def calculate_hit_ratio(
    draft_tokens: list[int],
    target_tokens: list[int],
    k: int,
    tokenizer: AutoTokenizer,
    context_size: int = 3,
    use_longest_match: bool = True,
    use_char_fallback: bool = False,
    char_threshold: int = 5,
    per_token_verification_overhead: float = 0.0,
) -> HitRatioResult:
    """
    Calculate hit ratio for speculative decoding with optional hybrid token/character matching.

    For each position in the target (after context_size tokens), we:
    1. Try token-based matching first (n-gram matching)
    2. If no match found and use_char_fallback=True, fall back to character-based matching
    3. If found, speculate the next k tokens from the draft
    4. Count how many of these match the actual target tokens
    5. Skip forward by the number of accepted tokens (or 1 if none accepted)

    This simulates the actual behavior of speculative decoding where accepted
    tokens are skipped, avoiding redundant speculation attempts.

    Args:
        draft_tokens: Token IDs from the draft response
        target_tokens: Token IDs from the target response
        k: Number of tokens to speculate
        tokenizer: Tokenizer for character-based fallback
        context_size: Minimum context window size for n-gram matching
        use_longest_match: If True, find longest matching context; if False, use first match
        use_char_fallback: If True, use character-based fallback when token matching fails
        char_threshold: Minimum character length for character-based fallback
        per_token_verification_overhead: Cost per token verified (default: 0.0)

    Returns:
        HitRatioResult with hit statistics
    """
    # Build n-gram index only for minimum context size
    draft_index = build_ngram_index(draft_tokens, context_size)

    # Decode texts once for character-based fallback (only if enabled)
    draft_text = None
    target_text = None
    if use_char_fallback:
        draft_text = decode_tokens_to_text(tokenizer, draft_tokens)
        target_text = decode_tokens_to_text(tokenizer, target_tokens)

    # Total tokens to generate (excluding initial context)
    total_target_tokens = max(0, len(target_tokens) - context_size)

    total_positions = 0
    context_found_positions = 0
    matched_positions = 0
    total_hits = 0
    total_speculated = 0
    total_context_length = 0
    total_candidates = 0
    total_verification_cost = 0.0
    hit_lengths: dict[int, int] = defaultdict(int)

    # Select matching function based on configuration
    token_match_fn = find_longest_match if use_longest_match else find_first_match
    char_match_fn = find_longest_char_match if use_longest_match else find_first_char_match

    # Iterate through target positions where we can attempt speculation
    # Skip positions based on accepted speculative tokens
    pos = context_size
    while pos < len(target_tokens):
        total_positions += 1

        # Try token-based matching first
        match = token_match_fn(
            target_tokens, pos, draft_tokens, draft_index,
            min_context=context_size,
        )

        draft_pos = None
        ctx_len = 0
        num_candidates = 0

        if match is not None:
            draft_pos, ctx_len, num_candidates = match
        elif use_char_fallback:
            # Fall back to character-based matching (only if enabled)
            char_match = char_match_fn(
                target_text, pos, target_tokens,
                draft_text, draft_tokens,
                tokenizer, char_threshold,
            )
            if char_match is not None:
                draft_pos, char_match_len = char_match
                # Use 1 as context length for character matches (approximate)
                ctx_len = 1
                num_candidates = 1

        if draft_pos is None:
            # No match found, generate 1 token normally (cost = 1)
            total_verification_cost += 1.0
            pos += 1
            continue

        context_found_positions += 1

        # Get speculated tokens from draft (tokens after the matched context)
        spec_start = draft_pos + ctx_len
        spec_end = min(spec_start + k, len(draft_tokens))
        speculated_tokens = draft_tokens[spec_start:spec_end]

        if not speculated_tokens:
            # Context was found but no tokens to speculate, generate 1 token normally (cost = 1)
            total_verification_cost += 1.0
            pos += 1
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

        # Calculate verification cost for this position
        # Cost = 1 + per_token_verification_overhead * (k - 1)
        # This assumes we always verify k tokens at each matching position
        verification_cost = 1.0 + (k - 1) * per_token_verification_overhead
        total_verification_cost += verification_cost

        # Move forward based on accepted tokens
        if hits > 0:
            # Skip positions covered by accepted speculative tokens
            pos += hits
        else:
            # No tokens accepted, move by 1
            pos += 1

    return HitRatioResult(
        total_positions=total_positions,
        context_found_positions=context_found_positions,
        matched_positions=matched_positions,
        total_hits=total_hits,
        total_speculated=total_speculated,
        total_context_length=total_context_length,
        total_candidates=total_candidates,
        total_target_tokens=total_target_tokens,
        hit_lengths=dict(hit_lengths),
        total_verification_cost=total_verification_cost,
    )


def calculate_hit_ratio_multi_draft(
    all_draft_tokens: list[list[int]],
    target_tokens: list[int],
    k: int,
    tokenizer: AutoTokenizer,
    context_size: int = 3,
    use_longest_match: bool = True,
    use_char_fallback: bool = False,
    char_threshold: int = 5,
    per_token_verification_overhead: float = 0.0,
) -> HitRatioResult:
    """
    Calculate hit ratio for speculative decoding using multiple draft sequences with optional hybrid matching.

    For each position in the target (after context_size tokens), we:
    1. Try token-based matching first (n-gram matching across all drafts)
    2. If no match found and use_char_fallback=True, fall back to character-based matching
    3. If found, speculate the next k tokens from the best draft
    4. Count how many of these match the actual target tokens
    5. Skip forward by the number of accepted tokens (or 1 if none accepted)

    Args:
        all_draft_tokens: List of token ID sequences from draft responses
        target_tokens: Token IDs from the target response
        k: Number of tokens to speculate
        tokenizer: Tokenizer for character-based fallback
        context_size: Minimum context window size for n-gram matching
        use_longest_match: If True, find longest matching context; if False, use first match
        use_char_fallback: If True, use character-based fallback when token matching fails
        char_threshold: Minimum character length for character-based fallback
        per_token_verification_overhead: Cost per token verified (default: 0.0)

    Returns:
        HitRatioResult with hit statistics
    """
    # Build n-gram indices for all drafts
    all_draft_indices = [build_ngram_index(draft_tokens, context_size) for draft_tokens in all_draft_tokens]

    # Decode texts once for character-based fallback (only if enabled)
    all_draft_texts = None
    target_text = None
    if use_char_fallback:
        all_draft_texts = [decode_tokens_to_text(tokenizer, draft_tokens) for draft_tokens in all_draft_tokens]
        target_text = decode_tokens_to_text(tokenizer, target_tokens)

    # Total tokens to generate (excluding initial context)
    total_target_tokens = max(0, len(target_tokens) - context_size)

    total_positions = 0
    context_found_positions = 0
    matched_positions = 0
    total_hits = 0
    total_speculated = 0
    total_context_length = 0
    total_candidates = 0
    total_verification_cost = 0.0
    hit_lengths: dict[int, int] = defaultdict(int)

    # Select matching function based on configuration
    token_match_fn = find_longest_match_multi_draft if use_longest_match else find_first_match_multi_draft
    char_match_fn = find_longest_char_match_multi_draft if use_longest_match else find_first_char_match_multi_draft

    # Iterate through target positions where we can attempt speculation
    # Skip positions based on accepted speculative tokens
    pos = context_size
    while pos < len(target_tokens):
        total_positions += 1

        # Try token-based matching first
        match = token_match_fn(
            target_tokens, pos, all_draft_tokens, all_draft_indices,
            min_context=context_size,
        )

        draft_idx = None
        draft_pos = None
        ctx_len = 0
        num_candidates = 0

        if match is not None:
            draft_idx, draft_pos, ctx_len, num_candidates = match
        elif use_char_fallback:
            # Fall back to character-based matching (only if enabled)
            char_match = char_match_fn(
                target_text, pos, target_tokens,
                all_draft_texts, all_draft_tokens,
                tokenizer, char_threshold,
            )
            if char_match is not None:
                draft_idx, draft_pos, char_match_len = char_match
                # Use 1 as context length for character matches (approximate)
                ctx_len = 1
                num_candidates = 1

        if draft_idx is None or draft_pos is None:
            # No match found, generate 1 token normally (cost = 1)
            total_verification_cost += 1.0
            pos += 1
            continue

        context_found_positions += 1

        # Get speculated tokens from the best draft (tokens after the matched context)
        spec_start = draft_pos + ctx_len
        spec_end = min(spec_start + k, len(all_draft_tokens[draft_idx]))
        speculated_tokens = all_draft_tokens[draft_idx][spec_start:spec_end]

        if not speculated_tokens:
            # Context was found but no tokens to speculate, generate 1 token normally (cost = 1)
            total_verification_cost += 1.0
            pos += 1
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

        # Calculate verification cost for this position
        # Cost = 1 + per_token_verification_overhead * (k - 1)
        # This assumes we always verify k tokens at each matching position
        verification_cost = 1.0 + (k - 1) * per_token_verification_overhead
        total_verification_cost += verification_cost

        # Move forward based on accepted tokens
        if hits > 0:
            # Skip positions covered by accepted speculative tokens
            pos += hits
        else:
            # No tokens accepted, move by 1
            pos += 1

    return HitRatioResult(
        total_positions=total_positions,
        context_found_positions=context_found_positions,
        matched_positions=matched_positions,
        total_hits=total_hits,
        total_speculated=total_speculated,
        total_context_length=total_context_length,
        total_candidates=total_candidates,
        total_target_tokens=total_target_tokens,
        hit_lengths=dict(hit_lengths),
        total_verification_cost=total_verification_cost,
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
    use_all_rollouts: bool = False  # If True, use all other rollouts as drafts; if False, use only rollout_id=0
    use_char_fallback: bool = False  # If True, use character-based fallback when token matching fails
    char_threshold: int = 5  # Minimum character length for character-based fallback matching
    per_token_verification_overhead: float = 0.0  # Cost per token verified


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
    total_target_tokens: int = 0
    total_verification_cost: float = 0.0
    hit_lengths: dict[int, int] = field(default_factory=dict)  # Distribution of hit lengths

    @property
    def overall_hit_ratio(self) -> float:
        """Overall hit ratio across all comparisons."""
        if self.total_speculated == 0:
            return 0.0
        return self.total_hits / self.total_speculated

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

    @property
    def overall_speedup(self) -> float:
        """
        Overall speedup compared to non-speculative generation.

        Speedup = (total tokens to generate) / (total generation cost with verification overhead)

        Logic:
        - Without speculation: Need 1 generation step per token = total_target_tokens steps
        - With speculation: Need total_verification_cost steps accounting for verification overhead
        - Speedup shows how many times faster speculative decoding is

        Example: If speedup = 2.0, speculative decoding is 2x faster
        """
        if self.total_verification_cost == 0:
            return 0.0
        return self.total_target_tokens / self.total_verification_cost


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
    if config.use_all_rollouts:
        # Mode: Use all other rollouts as drafts for each target rollout
        if len(instance.rollouts) < 2:
            logger.warning(f"Instance {instance.instance_id} has fewer than 2 rollouts")
            return []

        results: list[HitRatioResult] = []

        # For each rollout, use all other rollouts as drafts
        for target_rollout in instance.rollouts:
            # Get all other rollouts as drafts
            draft_rollouts = [r for r in instance.rollouts if r.rollout_id != target_rollout.rollout_id]

            # Tokenize all drafts
            all_draft_tokens = [tokenize_text(tokenizer, r.assistant_content) for r in draft_rollouts]

            # Tokenize target
            target_tokens = tokenize_text(tokenizer, target_rollout.assistant_content)

            result = calculate_hit_ratio_multi_draft(
                all_draft_tokens=all_draft_tokens,
                target_tokens=target_tokens,
                k=config.k,
                tokenizer=tokenizer,
                context_size=config.context_size,
                use_longest_match=config.use_longest_match,
                use_char_fallback=config.use_char_fallback,
                char_threshold=config.char_threshold,
                per_token_verification_overhead=config.per_token_verification_overhead,
            )
            results.append(result)

        return results
    else:
        # Mode: Use only rollout_id=0 as draft for all other rollouts
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
                tokenizer=tokenizer,
                context_size=config.context_size,
                use_longest_match=config.use_longest_match,
                use_char_fallback=config.use_char_fallback,
                char_threshold=config.char_threshold,
                per_token_verification_overhead=config.per_token_verification_overhead,
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
        total_target_tokens=0,
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
        instance_target_tokens = sum(r.total_target_tokens for r in instance_results)
        instance_verification_cost = sum(r.total_verification_cost for r in instance_results)

        # Aggregate hit_lengths distribution
        for result in instance_results:
            for hit_len, count in result.hit_lengths.items():
                if hit_len not in aggregated.hit_lengths:
                    aggregated.hit_lengths[hit_len] = 0
                aggregated.hit_lengths[hit_len] += count

        aggregated.num_instances += 1
        aggregated.num_comparisons += len(instance_results)
        aggregated.total_positions += instance_positions
        aggregated.total_context_found_positions += instance_context_found
        aggregated.total_matched_positions += instance_matched
        aggregated.total_hits += instance_hits
        aggregated.total_speculated += instance_speculated
        aggregated.total_context_length += instance_context_len
        aggregated.total_candidates += instance_candidates
        aggregated.total_target_tokens += instance_target_tokens
        aggregated.total_verification_cost += instance_verification_cost

    return aggregated


def print_results(results: AggregatedResults, config: AnalysisConfig) -> None:
    """Print analysis results."""
    print("\n" + "=" * 80)
    print("SPECULATIVE DECODING HIT RATIO ANALYSIS")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Context size (n-gram): {config.context_size}")
    print(f"Speculation length (k): {config.k}")
    print(f"Per-token verification overhead: {config.per_token_verification_overhead}")
    match_strategy = "longest match" if config.use_longest_match else "first match"
    print(f"Matching strategy: {match_strategy}")
    rollout_mode = "all other rollouts" if config.use_all_rollouts else "rollout_id=0 only"
    print(f"Draft rollout mode: {rollout_mode}")
    char_fallback_status = "enabled" if config.use_char_fallback else "disabled"
    print(f"Character fallback: {char_fallback_status}")
    if config.use_char_fallback:
        print(f"  Character threshold: {config.char_threshold} chars")
    print("-" * 80)

    print(f"\nOverall Metrics:")
    print(f"  Speedup:                    {results.overall_speedup:.4f}x")
    print(f"  Hit Ratio:                  {results.overall_hit_ratio:.4f}")
    print(f"  Context found rate:         {results.context_found_rate:.4f}")
    print(f"  Usable context rate:        {results.match_rate:.4f}")
    print(f"  Avg context length:         {results.avg_context_length:.2f}")
    print(f"  Instances analyzed:         {results.num_instances}")
    print(f"  Total comparisons:          {results.num_comparisons}")
    print(f"  Total target tokens:        {results.total_target_tokens}")
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
    print("  - Speedup: How many times faster than non-speculative generation")
    print("    Formula: (total target tokens) / (total positions evaluated)")
    print("  - Hit Ratio: Fraction of speculated tokens that matched")
    print("  - Context found rate: Fraction of evaluated positions where context was found (regardless of tokens to speculate)")
    print("  - Usable context rate: Fraction of evaluated positions where context was found AND tokens were available to speculate")
    print("  - Avg context length: Average longest matching context length per usable match (tokens)")
    print("  - Total positions evaluated: Positions where speculation was attempted (skips accepted tokens)")
    print("  - Hit length distribution: Cumulative count of cases where >= N consecutive tokens matched")
    print("    (e.g., 'Length 3' shows how many cases had >= 3 tokens matching)")
    print("")
    print("Matching strategy:")
    if config.use_char_fallback:
        print("  - Hybrid token/character matching:")
        print("    1. First tries token-based n-gram matching")
        print(f"    2. If no match, falls back to character-based matching (min {config.char_threshold} chars)")
        if config.use_longest_match:
            print("    3. Longest match: finds the longest matching context")
        else:
            print("    3. First match: returns as soon as a match is found")
    else:
        print("  - Token-based matching only:")
        print("    Uses n-gram token matching")
        if config.use_longest_match:
            print("    Longest match: finds the longest matching context")
        else:
            print("    First match: returns as soon as a match is found")
    print("")
    print("Draft rollout mode:")
    if config.use_all_rollouts:
        print("  - All other rollouts: For each rollout, all other rollouts are used as drafts")
        if config.use_longest_match:
            print("    Longest match strategy finds the best match across all drafts")
        else:
            print("    First match strategy returns as soon as a match is found in any draft")
    else:
        print("  - Single draft (rollout_id=0): Only the first rollout is used as draft for all others")
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
            "use_all_rollouts": config.use_all_rollouts,
            "use_char_fallback": config.use_char_fallback,
            "char_threshold": config.char_threshold,
            "per_token_verification_overhead": config.per_token_verification_overhead,
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
            "total_target_tokens": results.total_target_tokens,
            "total_verification_cost": results.total_verification_cost,
            "speedup": results.overall_speedup,
            "hit_ratio": results.overall_hit_ratio,
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
        "--use-all-rollouts",
        action="store_true",
        default=False,
        help="If set, use all other rollouts as drafts for each target (default: use only rollout_id=0)",
    )
    parser.add_argument(
        "--use-char-fallback",
        action="store_true",
        default=False,
        help="If set, use character-based fallback when token matching fails (default: False)",
    )
    parser.add_argument(
        "--char-threshold",
        type=int,
        default=5,
        help="Minimum character length for character-based fallback matching (default: 5)",
    )
    parser.add_argument(
        "--per-token-verification-overhead",
        type=float,
        default=0.0,
        help="Cost per token verified (default: 0.0)",
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
        use_all_rollouts=args.use_all_rollouts,
        use_char_fallback=args.use_char_fallback,
        char_threshold=args.char_threshold,
        per_token_verification_overhead=args.per_token_verification_overhead,
    )

    logger.info(f"Loading rollouts from {args.input}")
    instances = load_rollouts(args.input)

    if not instances:
        logger.error("No instances loaded")
        return 1

    match_strategy = "longest match" if config.use_longest_match else "first match"
    rollout_mode = "all other rollouts" if config.use_all_rollouts else "rollout_id=0 only"
    logger.info(f"Running analysis with k={config.k}, context_size={config.context_size}, strategy={match_strategy}, rollout_mode={rollout_mode}")
    results = run_analysis(instances, tokenizer, config)

    print_results(results, config)

    if args.output:
        save_results(results, config, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
