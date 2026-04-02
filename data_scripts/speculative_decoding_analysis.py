#!/usr/bin/env python3
"""
Speculative Decoding Hit Ratio Analysis

This script analyzes the potential hit ratio for n-gram based speculative decoding.
Given multiple rollouts per prompt, it can operate in two modes:
1. Single draft mode (default): Uses the first rollout (rollout_id=0) as a draft for all others
   - If no rollout_id=0 exists, uses only self-lookup for all rollouts
2. All rollouts mode (--use-all-rollouts): For each rollout, uses all other rollouts as drafts
   - If only one rollout exists, uses only self-lookup for that rollout

Features:
- Supports multi-turn conversations (system, user, assistant messages)
- Uses chat templates for proper tokenization of conversations
- Performs self-lookup: also searches within the current rollout up to the current position
  (ensuring at least k tokens are available after any matched prefix)
- For each assistant turn, uses the full conversation context from other rollouts as drafts

The matching scheme:
- Uses token-based n-gram matching with longest match strategy
- Searches in both other rollouts (full content) and current rollout (up to current position)
- Ensures at least k speculative tokens are available after any matched prefix
- Optionally, when using --use-all-rollouts --use-two-prefix-algorithm, computes dynamic k from suffix:
  1. First, finds matches using longest match strategy (same as default)
  2. For each match found, tries to find a second prefix with longest common end portion (>= context_size)
  3. Computes longest common suffix between tokens AFTER the two prefixes in their rollouts
  4. Uses max(k, suffix_length) as dynamic k (minimum value is k, can be larger)
  5. If no second prefix found or suffix length is 0, uses original k value
- Speculates the next dynamic_k tokens from the draft as candidates
- Counts how many of these speculated tokens match the actual target tokens

Usage:
    python speculative_decoding_analysis.py --input all_rollouts.jsonl
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --model /path/to/model
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --k 4
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --context-size 4 --k 5
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-all-rollouts
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --use-all-rollouts --use-two-prefix-algorithm
    python speculative_decoding_analysis.py --input all_rollouts.jsonl --limit-current-to-previous-turn
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
    messages: list[dict[str, str]]  # Full conversation history with role and content


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


@dataclass
class AssistantTurn:
    """Represents a single assistant turn in a conversation."""
    turn_index: int  # Index of this assistant turn (0-based)
    start_token_idx: int  # Start index of assistant tokens in full sequence
    end_token_idx: int  # End index of assistant tokens in full sequence (exclusive)
    tokens: list[int]  # Token IDs for this assistant turn only


@dataclass
class TokenizedRollout:
    """Represents a tokenized rollout with full sequence and assistant turn boundaries."""
    instance_id: int
    rollout_id: int
    full_tokens: list[int]  # All tokens including system, user, and all assistant turns
    assistant_turns: list[AssistantTurn]  # Information about each assistant turn


def tokenize_messages_with_turns(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    instance_id: int,
    rollout_id: int,
) -> TokenizedRollout:
    """
    Tokenize messages using chat template and identify assistant turn boundaries.

    This function processes messages incrementally, tokenizing the conversation
    up to each assistant turn to identify which tokens belong to each assistant response.

    Args:
        tokenizer: HuggingFace tokenizer
        messages: List of message dictionaries with 'role' and 'content'
        instance_id: Instance ID for this rollout
        rollout_id: Rollout ID

    Returns:
        TokenizedRollout with full token sequence and assistant turn information
    """
    assistant_turns = []
    assistant_turn_index = 0

    # Tokenize the full conversation
    full_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
    )

    # Process messages incrementally to find assistant turn boundaries
    current_pos = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Tokenize conversation up to (but not including) this assistant message
            if i > 0:
                prefix_messages = messages[:i]
                prefix_tokens = tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=True,
                    add_generation_prompt=False
                )
                start_idx = len(prefix_tokens)
            else:
                start_idx = 0

            # Tokenize conversation up to and including this assistant message
            messages_with_assistant = messages[:i+1]
            tokens_with_assistant = tokenizer.apply_chat_template(
                messages_with_assistant,
                tokenize=True,
                add_generation_prompt=False
            )
            end_idx = len(tokens_with_assistant)

            # Extract tokens for this assistant turn
            assistant_tokens = tokens_with_assistant[start_idx:end_idx]

            assistant_turns.append(AssistantTurn(
                turn_index=assistant_turn_index,
                start_token_idx=start_idx,
                end_token_idx=end_idx,
                tokens=assistant_tokens,
            ))
            assistant_turn_index += 1

    return TokenizedRollout(
        instance_id=instance_id,
        rollout_id=rollout_id,
        full_tokens=full_tokens,
        assistant_turns=assistant_turns,
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_rollouts(input_file: str) -> dict[int, InstanceRollouts]:
    """
    Load rollouts from JSONL file and group by instance_id.

    Expected format per line:
    {"instance_id": int, "rollout_id": int, "messages": [{"role": "system/user/assistant", "content": ...}, ...]}
    """
    instances: dict[int, InstanceRollouts] = {}

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                instance_id = data["instance_id"]
                rollout_id = data["rollout_id"]
                messages = data["messages"]

                rollout = Rollout(
                    instance_id=instance_id,
                    rollout_id=rollout_id,
                    messages=messages,
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
    max_draft_length: int | None = None,
    min_tokens_after: int = 0,
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
        max_draft_length: If specified, only consider draft tokens up to this position (exclusive)
        min_tokens_after: Minimum number of tokens required after the matched context

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

        # Adjust draft_pos to point to start of the extended context
        final_draft_pos = draft_pos - (ctx_len - min_context)

        # Check if the match end position is within the allowed draft length
        match_end_pos = final_draft_pos + ctx_len
        if max_draft_length is not None and match_end_pos > max_draft_length:
            continue

        # Check if enough tokens available after the match for speculation
        # AND that those tokens are also within the allowed draft length
        if min_tokens_after > 0:
            required_length = match_end_pos + min_tokens_after
            if required_length > len(draft_tokens):
                continue
            # For current rollout, ensure speculation tokens are also within bounds
            if max_draft_length is not None and required_length > max_draft_length:
                continue

        if ctx_len > best_context_len:
            best_context_len = ctx_len
            best_draft_pos = final_draft_pos

    if best_draft_pos is not None:
        return (best_draft_pos, best_context_len, num_candidates)

    return None


def find_longest_common_suffix(tokens1: list[int], tokens2: list[int]) -> int:
    """
    Find the length of the longest common suffix between two token sequences.

    Args:
        tokens1: First token sequence
        tokens2: Second token sequence

    Returns:
        Length of the longest common suffix
    """
    suffix_len = 0
    min_len = min(len(tokens1), len(tokens2))

    for i in range(1, min_len + 1):
        if tokens1[-i] == tokens2[-i]:
            suffix_len = i
        else:
            break

    return suffix_len


def find_longest_common_end_portion(prefix1: list[int], prefix2: list[int], min_length: int) -> int:
    """
    Find the longest common end portion between two prefixes.
    The common end portion must be at least min_length.

    Args:
        prefix1: First prefix
        prefix2: Second prefix
        min_length: Minimum length of common end portion

    Returns:
        Length of the longest common end portion (0 if less than min_length)
    """
    max_len = min(len(prefix1), len(prefix2))

    # Start from min_length and find the longest match
    best_len = 0
    for length in range(min_length, max_len + 1):
        if prefix1[-length:] == prefix2[-length:]:
            best_len = length
        else:
            # No longer matching, stop
            break

    return best_len


def find_longest_match_multi_draft(
    target_tokens: list[int],
    pos: int,
    all_draft_tokens: list[list[int]],
    all_draft_indices: list[dict[tuple[int, ...], list[int]]],
    min_context: int,
    max_draft_lengths: list[int | None] | None = None,
    min_tokens_after: int = 0,
) -> tuple[int, int, int, int] | None:
    """
    Find the longest matching context across multiple draft sequences.

    Args:
        target_tokens: Target token sequence
        pos: Current position in target (where we want to speculate)
        all_draft_tokens: List of draft token sequences
        all_draft_indices: List of pre-built n-gram indices (one per draft)
        min_context: Minimum context size
        max_draft_lengths: Optional list of max lengths (one per draft, or None for no limit)
        min_tokens_after: Minimum number of tokens required after the matched context

    Returns:
        Tuple of (draft_idx, draft_position, context_length, num_candidates) for longest match, or None if no match
    """
    best_draft_idx = None
    best_draft_pos = None
    best_context_len = 0
    total_candidates = 0

    for draft_idx, (draft_tokens, draft_index) in enumerate(zip(all_draft_tokens, all_draft_indices)):
        max_length = None
        if max_draft_lengths is not None and draft_idx < len(max_draft_lengths):
            max_length = max_draft_lengths[draft_idx]

        match = find_longest_match(
            target_tokens, pos, draft_tokens, draft_index, min_context,
            max_draft_length=max_length,
            min_tokens_after=min_tokens_after,
        )

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


def compute_dynamic_k_from_suffix(
    best_draft_idx: int,
    best_draft_pos: int,
    best_context_len: int,
    all_draft_tokens: list[list[int]],
    all_matches: list[tuple[int, int, int, int]],
    min_context: int,
    k: int,
) -> int:
    """
    Compute dynamic k based on longest common suffix between two prefixes.

    Algorithm:
    1. Use the already-found longest prefix (best_draft_idx)
    2. Find a second prefix with longest common end portion
    3. Compute longest common suffix between tokens AFTER the two prefixes
    4. Return max(k, suffix_length) as dynamic k (minimum value is k)
    5. If no valid second prefix or suffix is 0, return original k

    Args:
        best_draft_idx: Index of the draft with longest prefix
        best_draft_pos: Position of the longest prefix in that draft
        best_context_len: Length of the longest prefix
        all_draft_tokens: List of all draft token sequences
        all_matches: List of all matches found (draft_idx, draft_pos, ctx_len, num_candidates)
        min_context: Minimum context size for common end portion
        k: Original k value (also minimum value for dynamic k)

    Returns:
        dynamic_k: max(k, suffix_length) - always at least k
    """
    # Extract the first prefix
    prefix1 = all_draft_tokens[best_draft_idx][best_draft_pos:best_draft_pos + best_context_len]

    # Find the second prefix with longest common end portion
    best_second_draft_idx = None
    best_second_draft_pos = None
    best_second_context_len = 0
    best_end_portion_len = 0
    best_suffix_len = 0

    for draft_idx, draft_pos, ctx_len, _ in all_matches:
        # Skip the first draft
        if draft_idx == best_draft_idx:
            continue

        # Extract this prefix
        prefix2 = all_draft_tokens[draft_idx][draft_pos:draft_pos + ctx_len]

        # Find the longest common end portion between the prefixes
        end_portion_len = find_longest_common_end_portion(prefix1, prefix2, min_context)

        if end_portion_len >= min_context:
            # Get tokens AFTER the prefixes in their respective rollouts
            tokens_after_prefix1 = all_draft_tokens[best_draft_idx][best_draft_pos + best_context_len:]
            tokens_after_prefix2 = all_draft_tokens[draft_idx][draft_pos + ctx_len:]

            # Find the longest common suffix between tokens after the prefixes
            suffix_len = find_longest_common_suffix(tokens_after_prefix1, tokens_after_prefix2)

            # Select the prefix with longest end portion, or longest suffix if tied
            if (end_portion_len > best_end_portion_len or
                (end_portion_len == best_end_portion_len and suffix_len > best_suffix_len)):
                best_end_portion_len = end_portion_len
                best_second_draft_idx = draft_idx
                best_second_draft_pos = draft_pos
                best_second_context_len = ctx_len
                best_suffix_len = suffix_len

    # Return dynamic k (minimum value is k)
    if best_second_draft_idx is None or best_suffix_len == 0:
        # No valid second prefix found or suffix is 0, use original k
        return k
    else:
        # Return max of k and suffix length (dynamic k is at least k)
        return max(k, best_suffix_len)


def calculate_hit_ratio(
    draft_tokens: list[int],
    target_tokens: list[int],
    k: int,
    context_size: int = 3,
    per_token_verification_overhead: float = 0.0,
) -> HitRatioResult:
    """
    Calculate hit ratio for speculative decoding using token-based n-gram matching.

    For each position in the target (after context_size tokens), we:
    1. Try token-based n-gram matching (longest match)
    2. If found, speculate the next k tokens from the draft
    3. Count how many of these match the actual target tokens
    4. Skip forward by the number of accepted tokens (or 1 if none accepted)

    This simulates the actual behavior of speculative decoding where accepted
    tokens are skipped, avoiding redundant speculation attempts.

    Args:
        draft_tokens: Token IDs from the draft response
        target_tokens: Token IDs from the target response
        k: Number of tokens to speculate
        context_size: Minimum context window size for n-gram matching
        per_token_verification_overhead: Cost per token verified (default: 0.0)

    Returns:
        HitRatioResult with hit statistics
    """
    # Build n-gram index only for minimum context size
    draft_index = build_ngram_index(draft_tokens, context_size)

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

    # Iterate through target positions where we can attempt speculation
    # Skip positions based on accepted speculative tokens
    pos = context_size
    while pos < len(target_tokens):
        total_positions += 1

        # Try token-based matching (longest match)
        match = find_longest_match(
            target_tokens, pos, draft_tokens, draft_index,
            min_context=context_size,
        )

        draft_pos = None
        ctx_len = 0
        num_candidates = 0

        if match is not None:
            draft_pos, ctx_len, num_candidates = match

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
    context_size: int = 3,
    per_token_verification_overhead: float = 0.0,
    use_two_prefix_algorithm: bool = False,
    current_rollout_tokens: list[int] | None = None,
    target_start_offset: int = 0,
    max_current_rollout_length: int | None = None,
) -> HitRatioResult:
    """
    Calculate hit ratio for speculative decoding using multiple draft sequences.

    For each position in the target (after context_size tokens), we:
    1. Find matches using longest match strategy across all drafts
    2. If current_rollout_tokens is provided, also search in the current rollout
       up to the specified limit (ensuring k tokens available after match)
    3. If use_two_prefix_algorithm is True, compute dynamic k based on suffix analysis
       - dynamic k = max(k, suffix_length), so always at least k
    4. Otherwise, use original k
    5. Speculate the next dynamic_k tokens from the best draft
    6. Count how many of these match the actual target tokens
    7. Skip forward by the number of accepted tokens (or 1 if none accepted)

    Args:
        all_draft_tokens: List of token ID sequences from draft responses
        target_tokens: Token IDs from the target response
        k: Number of tokens to speculate (also minimum value for dynamic k)
        context_size: Minimum context window size for n-gram matching
        per_token_verification_overhead: Cost per token verified (default: 0.0)
        use_two_prefix_algorithm: If True, compute dynamic k from suffix analysis (default: False)
        current_rollout_tokens: Optional full token sequence from current rollout (for self-lookup)
        target_start_offset: Offset where target_tokens start in current_rollout_tokens
        max_current_rollout_length: If provided, fixed max length for current rollout lookup;
                                    otherwise computed dynamically as target_start_offset + pos

    Returns:
        HitRatioResult with hit statistics
    """
    # Build n-gram indices for all drafts
    all_draft_indices = [build_ngram_index(draft_tokens, context_size) for draft_tokens in all_draft_tokens]

    # Build index for current rollout if provided
    current_rollout_index = None
    if current_rollout_tokens is not None:
        current_rollout_index = build_ngram_index(current_rollout_tokens, context_size)

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

    # Iterate through target positions where we can attempt speculation
    # Skip positions based on accepted speculative tokens
    pos = context_size
    while pos < len(target_tokens):
        total_positions += 1

        # Prepare draft lists (include current rollout if provided)
        drafts_to_search = list(all_draft_tokens)
        draft_indices_to_search = list(all_draft_indices)
        max_draft_lengths = [None] * len(all_draft_tokens)

        if current_rollout_tokens is not None and current_rollout_index is not None:
            # Add current rollout with restriction
            # Two modes:
            # 1. If max_current_rollout_length is provided (limit to previous turn):
            #    Use fixed length (typically start of current assistant turn)
            # 2. Otherwise (default): dynamically use target_start_offset + pos
            #    (all content up to current position)
            if max_current_rollout_length is not None:
                max_current_length = max_current_rollout_length
            else:
                max_current_length = target_start_offset + pos
            drafts_to_search.append(current_rollout_tokens)
            draft_indices_to_search.append(current_rollout_index)
            max_draft_lengths.append(max_current_length)

        # Always use longest match algorithm to find matches
        # Need to collect all matches if use_two_prefix_algorithm is True
        if use_two_prefix_algorithm:
            # Collect all matches for suffix computation
            all_matches = []
            best_draft_idx = None
            best_draft_pos = None
            best_context_len = 0
            num_candidates_total = 0

            for draft_idx, (draft_tokens, draft_index) in enumerate(zip(drafts_to_search, draft_indices_to_search)):
                max_length = max_draft_lengths[draft_idx] if draft_idx < len(max_draft_lengths) else None
                match = find_longest_match(
                    target_tokens, pos, draft_tokens, draft_index, context_size,
                    max_draft_length=max_length,
                    min_tokens_after=k,
                )

                if match is not None:
                    draft_pos, ctx_len, num_candidates = match
                    all_matches.append((draft_idx, draft_pos, ctx_len, num_candidates))
                    num_candidates_total += num_candidates

                    if ctx_len > best_context_len:
                        best_context_len = ctx_len
                        best_draft_pos = draft_pos
                        best_draft_idx = draft_idx

            if best_draft_idx is None:
                # No match found
                total_verification_cost += 1.0
                pos += 1
                continue

            # Compute dynamic k based on suffix analysis
            dynamic_k = compute_dynamic_k_from_suffix(
                best_draft_idx, best_draft_pos, best_context_len,
                drafts_to_search, all_matches, context_size, k
            )

            draft_idx = best_draft_idx
            draft_pos = best_draft_pos
            ctx_len = best_context_len
            num_candidates = num_candidates_total
        else:
            # Use find_longest_match_multi_draft (more efficient when not computing suffix)
            match = find_longest_match_multi_draft(
                target_tokens, pos, drafts_to_search, draft_indices_to_search,
                min_context=context_size,
                max_draft_lengths=max_draft_lengths,
                min_tokens_after=k,
            )

            if match is None:
                # No match found
                total_verification_cost += 1.0
                pos += 1
                continue

            draft_idx, draft_pos, ctx_len, num_candidates = match
            dynamic_k = k  # Use original k

        context_found_positions += 1

        # Get speculated tokens from the best draft (tokens after the matched context)
        spec_start = draft_pos + ctx_len
        spec_end = min(spec_start + dynamic_k, len(drafts_to_search[draft_idx]))
        speculated_tokens = drafts_to_search[draft_idx][spec_start:spec_end]

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
        # Cost = 1 + per_token_verification_overhead * (dynamic_k - 1)
        verification_cost = 1.0 + (dynamic_k - 1) * per_token_verification_overhead
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
    use_all_rollouts: bool = False  # If True, use all other rollouts as drafts; if False, use only rollout_id=0
    per_token_verification_overhead: float = 0.0  # Cost per token verified
    use_two_prefix_algorithm: bool = False  # If True, compute dynamic k from suffix analysis (only applies when use_all_rollouts=True)
    limit_current_to_previous_turn: bool = False  # If True, limit current rollout lookup to end of previous turn; if False, use up to current position


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
    Analyze a single instance's rollouts with multi-turn support.

    For each assistant turn in each rollout:
    - Uses the full tokenized content (all messages) from other rollouts as drafts
    - Compares only the tokens from the current assistant turn

    Args:
        instance: Collection of rollouts for this instance
        tokenizer: HuggingFace tokenizer to use
        config: Analysis configuration

    Returns:
        List of HitRatioResults (one per assistant turn per rollout)
    """
    # Tokenize all rollouts with turn information
    tokenized_rollouts = []
    for rollout in instance.rollouts:
        tokenized = tokenize_messages_with_turns(
            tokenizer=tokenizer,
            messages=rollout.messages,
            instance_id=rollout.instance_id,
            rollout_id=rollout.rollout_id,
        )
        tokenized_rollouts.append(tokenized)

    results: list[HitRatioResult] = []

    if config.use_all_rollouts:
        # Mode: Use all other rollouts as drafts for each target rollout
        if len(tokenized_rollouts) < 1:
            logger.warning(f"Instance {instance.instance_id} has no rollouts")
            return []

        if len(tokenized_rollouts) == 1:
            logger.info(f"Instance {instance.instance_id} has only 1 rollout, using only self-lookup")

        # For each rollout, use all other rollouts as drafts
        for target_rollout in tokenized_rollouts:
            # Get all other rollouts as drafts (use their full tokens)
            # If there's only one rollout, this will be empty (self-lookup only)
            draft_rollouts = [r for r in tokenized_rollouts if r.rollout_id != target_rollout.rollout_id]
            all_draft_tokens = [r.full_tokens for r in draft_rollouts]

            # Analyze each assistant turn in the target rollout
            for assistant_turn in target_rollout.assistant_turns:
                # For this assistant turn, we need to compare its tokens against drafts
                # The target tokens are just this assistant turn's tokens
                target_tokens = assistant_turn.tokens

                if not target_tokens:
                    continue

                # Determine max length for current rollout lookup
                max_current_length = None
                if config.limit_current_to_previous_turn:
                    # Limit to end of previous turn (start of current turn)
                    max_current_length = assistant_turn.start_token_idx

                result = calculate_hit_ratio_multi_draft(
                    all_draft_tokens=all_draft_tokens,
                    target_tokens=target_tokens,
                    k=config.k,
                    context_size=config.context_size,
                    per_token_verification_overhead=config.per_token_verification_overhead,
                    use_two_prefix_algorithm=config.use_two_prefix_algorithm,
                    current_rollout_tokens=target_rollout.full_tokens,
                    target_start_offset=assistant_turn.start_token_idx,
                    max_current_rollout_length=max_current_length,
                )
                results.append(result)

        return results
    else:
        # Mode: Use only rollout_id=0 as draft for all other rollouts (if available)
        draft_rollout = None
        for r in tokenized_rollouts:
            if r.rollout_id == 0:
                draft_rollout = r
                break

        # Determine which rollouts to verify
        if draft_rollout is None:
            # No draft rollout, analyze all rollouts using only self-lookup
            logger.info(f"Instance {instance.instance_id} has no draft rollout, using only self-lookup")
            verification_rollouts = tokenized_rollouts
            all_draft_tokens = []
        else:
            # Draft rollout exists, verify all other rollouts
            verification_rollouts = [r for r in tokenized_rollouts if r.rollout_id != 0]
            if not verification_rollouts:
                logger.warning(f"Instance {instance.instance_id} has no verification rollouts")
                return []
            all_draft_tokens = [draft_rollout.full_tokens]

        # Analyze each assistant turn in each verification rollout
        for target_rollout in verification_rollouts:
            for assistant_turn in target_rollout.assistant_turns:
                # For this assistant turn, compare its tokens against the draft
                target_tokens = assistant_turn.tokens

                if not target_tokens:
                    continue

                # Determine max length for current rollout lookup
                max_current_length = None
                if config.limit_current_to_previous_turn:
                    # Limit to end of previous turn (start of current turn)
                    max_current_length = assistant_turn.start_token_idx

                # Use multi_draft function with draft(s) + current rollout
                result = calculate_hit_ratio_multi_draft(
                    all_draft_tokens=all_draft_tokens,
                    target_tokens=target_tokens,
                    k=config.k,
                    context_size=config.context_size,
                    per_token_verification_overhead=config.per_token_verification_overhead,
                    use_two_prefix_algorithm=False,  # Not applicable in single draft mode
                    current_rollout_tokens=target_rollout.full_tokens,
                    target_start_offset=assistant_turn.start_token_idx,
                    max_current_rollout_length=max_current_length,
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
    print(f"Matching strategy: longest match")
    rollout_mode = "all other rollouts" if config.use_all_rollouts else "rollout_id=0 only"
    print(f"Draft rollout mode: {rollout_mode}")
    current_lookup_mode = "up to previous turn" if config.limit_current_to_previous_turn else "up to current position"
    print(f"Current rollout self-lookup: {current_lookup_mode}")
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
    print("  - Token-based n-gram matching (longest match)")
    print("    Uses n-gram token matching to find the longest matching context")
    print("  - Self-lookup enabled: Also searches within current rollout")
    if config.limit_current_to_previous_turn:
        print("    Limit: Up to end of previous turn (start of current assistant turn)")
    else:
        print("    Limit: Up to current position (includes partial current turn generation)")
    print("    (ensuring at least k tokens available after any matched prefix)")
    print("")
    print("Draft rollout mode:")
    if config.use_all_rollouts:
        print("  - All other rollouts: For each rollout, all other rollouts are used as drafts")
        print("    If only one rollout exists, uses only self-lookup for that rollout")
        print("  - Additionally includes current rollout (self-lookup) with position restrictions")
        if config.use_two_prefix_algorithm:
            print("    Dynamic k computation: ENABLED")
            print("      1. Finds matches using longest match strategy")
            print("      2. Finds second prefix with longest common end portion (>= context_size)")
            print("      3. Computes longest common suffix between tokens AFTER the two prefixes")
            print("      4. Uses max(k, suffix_length) as dynamic k (minimum is k, can be larger)")
            print("      5. If no second prefix or suffix is 0, uses original k value")
        else:
            print("    Dynamic k computation: DISABLED (using fixed k)")
    else:
        print("  - Single draft (rollout_id=0): The first rollout is used as draft for all others")
        print("    If no rollout_id=0 exists, uses only self-lookup for all rollouts")
        print("  - Additionally includes current rollout (self-lookup) with position restrictions")
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
            "use_all_rollouts": config.use_all_rollouts,
            "per_token_verification_overhead": config.per_token_verification_overhead,
            "use_two_prefix_algorithm": config.use_two_prefix_algorithm,
            "limit_current_to_previous_turn": config.limit_current_to_previous_turn,
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
        "--use-all-rollouts",
        action="store_true",
        default=False,
        help="If set, use all other rollouts as drafts for each target (default: use only rollout_id=0)",
    )
    parser.add_argument(
        "--per-token-verification-overhead",
        type=float,
        default=0.0,
        help="Cost per token verified (default: 0.0)",
    )
    parser.add_argument(
        "--use-two-prefix-algorithm",
        action="store_true",
        default=False,
        help="If set, compute dynamic k from suffix analysis (only applies when --use-all-rollouts is True) (default: False)",
    )
    parser.add_argument(
        "--limit-current-to-previous-turn",
        action="store_true",
        default=False,
        help="If set, limit current rollout self-lookup to content up to end of previous turn; otherwise uses content up to current position (default: False)",
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
        use_all_rollouts=args.use_all_rollouts,
        per_token_verification_overhead=args.per_token_verification_overhead,
        use_two_prefix_algorithm=args.use_two_prefix_algorithm,
        limit_current_to_previous_turn=args.limit_current_to_previous_turn,
    )

    logger.info(f"Loading rollouts from {args.input}")
    instances = load_rollouts(args.input)

    if not instances:
        logger.error("No instances loaded")
        return 1

    rollout_mode = "all other rollouts" if config.use_all_rollouts else "rollout_id=0 only"
    logger.info(f"Running analysis with k={config.k}, context_size={config.context_size}, strategy=longest match, rollout_mode={rollout_mode}")
    results = run_analysis(instances, tokenizer, config)

    print_results(results, config)

    if args.output:
        save_results(results, config, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
