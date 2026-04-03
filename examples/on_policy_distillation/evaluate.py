#!/usr/bin/env python3
"""
Evaluation script for math problem solving using HuggingFace models or vLLM server.

This script evaluates a model on the dapo-math-17k dataset by:
1. Loading prompts from the dataset
2. Generating responses using either an HF model or vLLM server
3. Parsing the final answer from the response
4. Comparing against the ground truth label

Usage Examples:

# HuggingFace mode (loads model locally)
python evaluate.py --num-samples 10

# vLLM mode (uses running vLLM server) with parallel workers
python evaluate.py --vllm --port 9000 --model-name qwen3 --num-samples 10 --workers 20

# vLLM mode with custom settings and workers
python evaluate.py --vllm --port 8000 --model-name my-model --temperature 0.8 --max-new-tokens 4096 --workers 50

# Save detailed results
python evaluate.py --vllm --num-samples 100 --output results/eval_results.json

# Save passing rollouts (first successful rollout per prompt) to a JSONL file
python evaluate.py --vllm --num-samples 100 --attempts 3 --save-passing-rollouts --passing-rollouts-output passing_rollouts.jsonl

# Save all rollouts to a JSONL file
python evaluate.py --vllm --num-samples 100 --attempts 3 --save-all-rollouts --passing-rollouts-output all_rollouts.jsonl
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

# Add slime to the system path
script_dir = Path(__file__).resolve().parent
slime_root = script_dir.parent.parent  # Go up two levels from examples/on_policy_distillation
sys.path.insert(0, str(slime_root))

# Import the math answer extraction utility from slime
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer

# Conditional imports for HuggingFace mode (only imported when needed)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Prompt prefix and suffix for math problems
PROMPT_PREFIX = "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
PROMPT_SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def add_prompt_wrapper(content: str) -> str:
    """Add prefix and suffix to prompt if they don't already exist."""
    # Check if prefix already exists
    if not content.strip().startswith("Solve the following math problem"):
        content = PROMPT_PREFIX + content

    # Check if suffix already exists
    if not content.strip().endswith("Remember to put your answer on its own line after \"Answer:\"."):
        content = content + PROMPT_SUFFIX

    return content


def get_prompt_text(prompt: Any) -> str:
    """Extract plain prompt text for logging and output files."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        for message in reversed(prompt):
            if message.get("role") == "user":
                return message.get("content", "")
        return prompt[0].get("content", "")
    return str(prompt)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace and convert to lowercase for comparison
    normalized = str(answer).strip().lower()
    # Remove common formatting
    normalized = normalized.replace(",", "").replace(" ", "")
    return normalized


def answers_match(pred_answer: str, true_answer: str) -> bool:
    """
    Check if predicted answer matches the true answer.
    Handles various numeric and string formats.
    """
    if pred_answer is None or true_answer is None:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(pred_answer)
    true_norm = normalize_answer(true_answer)

    # Direct string match
    if pred_norm == true_norm:
        return True

    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        true_num = float(true_norm)
        # Allow small floating point differences
        return abs(pred_num - true_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def load_dataset(dataset_path: str, num_samples: int = None) -> list[dict[str, Any]]:
    """Load the JSONL dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    samples = []

    with open(dataset_path, 'r') as f:
        for idx, line in enumerate(f):
            if num_samples is not None and idx >= num_samples:
                break
            data = json.loads(line)
            samples.append(data)

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    sample_idx: int = 0,
    add_wrapper: bool = False,
) -> str:
    """Generate a response from the model."""
    logger.info(f"[Sample {sample_idx}] Preparing prompt...")

    # Apply chat template if the prompt is in the expected format
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Prompt is already in chat format - wrap the content of the last user message
        messages = prompt.copy()
        if messages[-1].get("role") == "user":
            if add_wrapper:
                messages[-1]["content"] = add_prompt_wrapper(messages[-1]["content"])
    else:
        # Wrap plain text in user message and optionally add prefix/suffix
        content = add_prompt_wrapper(prompt) if add_wrapper else prompt
        messages = [{"role": "user", "content": content}]

    logger.info(f"[Sample {sample_idx}] Tokenizing input...")
    # Tokenize with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt"
    ).to(model.device)

    input_length = inputs.shape[1]
    logger.info(f"[Sample {sample_idx}] Input tokens: {input_length}, Generating up to {max_new_tokens} new tokens...")

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - start_time

    output_length = outputs.shape[1]
    generated_tokens = output_length - input_length
    logger.info(f"[Sample {sample_idx}] Generated {generated_tokens} tokens in {gen_time:.2f}s ({generated_tokens/gen_time:.1f} tokens/s)")

    # Decode only the generated part (excluding the prompt)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    logger.info(f"[Sample {sample_idx}] Response length: {len(response)} chars")
    logger.info(f"[Sample {sample_idx}] Response preview: {response[:100]}...")

    return response, input_length, generated_tokens


def generate_response_vllm(
    prompt: str,
    port: int = 9000,
    model_name: str = "qwen3",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    sample_idx: int = 0,
    add_wrapper: bool = False,
) -> str:
    """Generate a response using vLLM server API."""
    logger.info(f"[Sample {sample_idx}] Preparing vLLM API request...")

    # Prepare messages format
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Prompt is already in chat format - wrap the content of the last user message
        messages = prompt.copy()
        if messages[-1].get("role") == "user":
            if add_wrapper:
                messages[-1]["content"] = add_prompt_wrapper(messages[-1]["content"])
    else:
        # Wrap plain text in user message and optionally add prefix/suffix
        content = add_prompt_wrapper(prompt) if add_wrapper else prompt
        messages = [{"role": "user", "content": content}]

    # Prepare API request
    url = f"http://localhost:{port}/v1/chat/completions"

    # Non-thinking payload for qwen3 0.6B
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    # # Thinking payload for qwen3 0.6B
    # payload = {
    #     "model": model_name,
    #     "messages": messages,
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    #     "top_k": 20,
    #     "min_p": 0.0,
    #     "max_tokens": max_new_tokens,
    # }


    logger.info(f"[Sample {sample_idx}] Sending request to vLLM server at {url}...")
    logger.info(f"[Sample {sample_idx}] Model: {model_name}, max_tokens: {max_new_tokens}, temperature: {temperature}")

    # Send request
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        response_data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"[Sample {sample_idx}] vLLM API request failed: {e}")
        raise

    gen_time = time.time() - start_time

    # Extract response text
    if "choices" in response_data and len(response_data["choices"]) > 0:
        response_text = response_data["choices"][0]["message"]["content"]

        # Extract token usage stats
        input_tokens = 0
        output_tokens = 0
        if "usage" in response_data:
            usage = response_data["usage"]
            total_tokens = usage.get("total_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            input_tokens = prompt_tokens
            output_tokens = completion_tokens

            logger.info(f"[Sample {sample_idx}] Generated {completion_tokens} tokens in {gen_time:.2f}s ({completion_tokens/gen_time:.1f} tokens/s)")
            logger.info(f"[Sample {sample_idx}] Total tokens used: {total_tokens}")
        else:
            logger.info(f"[Sample {sample_idx}] Request completed in {gen_time:.2f}s")

        logger.info(f"[Sample {sample_idx}] Response length: {len(response_text)} chars")
        logger.info(f"[Sample {sample_idx}] Response preview: {response_text[:100]}...")

        return response_text, input_tokens, output_tokens
    else:
        raise ValueError(f"Unexpected response format from vLLM API: {response_data}")


def process_single_sample(
    idx: int,
    sample: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    use_vllm: bool,
    vllm_port: int,
    vllm_model_name: str,
    add_wrapper: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Process a single sample and return the result."""
    prompt = sample["prompt"]
    prompt_text = get_prompt_text(prompt)
    true_answer = sample["label"]
    logger.info(f"[Sample {idx}] True answer: {true_answer}")

    try:
        if use_vllm:
            response, input_tokens, output_tokens = generate_response_vllm(
                prompt=prompt,
                port=vllm_port,
                model_name=vllm_model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                sample_idx=idx,
                add_wrapper=add_wrapper,
            )
        else:
            response, input_tokens, output_tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                sample_idx=idx,
                add_wrapper=add_wrapper,
            )

        # Extract answer from response
        logger.info(f"[Sample {idx}] Extracting answer from response...")
        pred_answer = extract_boxed_answer(response)
        logger.info(f"[Sample {idx}] Extracted answer: {pred_answer}")

        # Check if correct
        is_correct = answers_match(pred_answer, true_answer)
        logger.info(f"[Sample {idx}] Correct: {is_correct} (pred='{pred_answer}' vs true='{true_answer}')")

        # Store result
        result = {
            "idx": idx,
            "prompt": prompt_text,
            "response": response,
            "predicted_answer": pred_answer,
            "true_answer": true_answer,
            "correct": is_correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        return result

    except Exception as e:
        logger.error(f"[Sample {idx}] ERROR: {e}", exc_info=True)
        return {
            "idx": idx,
            "prompt": prompt_text,
            "response": None,
            "predicted_answer": None,
            "true_answer": true_answer,
            "error": str(e),
            "correct": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def write_attempt_results_file(
    attempt_results: dict[int, dict[str, Any]],
    attempt_num: int,
    output_dir: str,
) -> Path:
    """Write one JSONL file for a single attempt."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    attempt_file = output_path / f"attempt_{attempt_num}.jsonl"
    with open(attempt_file, "w") as f:
        for idx in sorted(attempt_results):
            result = attempt_results[idx]
            record = {
                "idx": result.get("idx", idx),
                "prompt": result.get("prompt"),
                "response": result.get("response"),
                "extracted_answer": result.get("predicted_answer"),
                "true_answer": result.get("true_answer"),
                "correct": result.get("correct"),
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
            }
            if "error" in result:
                record["error"] = result["error"]
            f.write(json.dumps(record) + "\n")

    logger.info(f"Attempt {attempt_num} results written to {attempt_file}")
    return attempt_file


def evaluate(
    model_path: str,
    dataset_path: str,
    num_samples: int = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    device: str = "cpu",
    use_vllm: bool = False,
    vllm_port: int = 9000,
    vllm_model_name: str = "qwen3",
    vllm_workers: int = 20,
    attempts: int = 1,
    add_wrapper: bool = False,
    attempt_output_dir: str | None = None,
) -> dict[str, Any]:
    """Run evaluation on the dataset."""

    model = None
    tokenizer = None

    if use_vllm:
        # vLLM mode - test connection to server
        logger.info(f"Using vLLM mode - connecting to server at localhost:{vllm_port}")
        logger.info(f"Model name: {vllm_model_name}")

        # Test connection
        try:
            test_url = f"http://localhost:{vllm_port}/v1/models"
            response = requests.get(test_url, timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to vLLM server")
            logger.info(f"Available models: {response.json()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            logger.error(f"Make sure vLLM server is running on port {vllm_port}")
            raise
    else:
        # HuggingFace mode - load model
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers and torch are required for non-vLLM mode. "
                "Please install them with: pip install torch transformers"
            )

        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"Tokenizer loaded successfully")

        # Load model
        logger.info(f"Loading model from {model_path}...")
        logger.info(f"Using torch_dtype=bfloat16, device_map=auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        logger.info(f"Model loaded successfully on device: {model.device}")

    # Load dataset
    samples = load_dataset(dataset_path, num_samples)
    logger.info(f"Will evaluate on {len(samples)} samples")
    logger.info(f"Running {attempts} attempt(s)")
    if attempt_output_dir:
        logger.info(f"Per-attempt outputs will be written to {attempt_output_dir}")

    # Initialize tracking structures for multiple attempts
    per_sample_results = {i: [] for i in range(len(samples))}  # Track all attempts per sample
    all_results = []

    logger.info("="*60)
    logger.info("Starting evaluation...")
    if use_vllm and vllm_workers > 1:
        logger.info(f"Using parallel processing with {vllm_workers} workers")
    logger.info("="*60)

    # Run evaluation for multiple attempts
    attempt_times = []
    attempt_p90_times = []
    for attempt_num in range(attempts):
        logger.info(f"\n{'='*60}")
        logger.info(f"ATTEMPT {attempt_num + 1}/{attempts}")
        logger.info(f"{'='*60}\n")
        attempt_start_time = time.time()
        attempt_results = {}
        sample_completion_times = []  # Track when each sample completes

        # Use parallel processing for vLLM mode with multiple workers
        if use_vllm and vllm_workers > 1:
            logger.info(f"Processing {len(samples)} samples in parallel with {vllm_workers} workers...")

            with ThreadPoolExecutor(max_workers=vllm_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(
                        process_single_sample,
                        idx=idx,
                        sample=sample,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        use_vllm=use_vllm,
                        vllm_port=vllm_port,
                        vllm_model_name=vllm_model_name,
                        add_wrapper=add_wrapper,
                        model=model,
                        tokenizer=tokenizer,
                    ): idx
                    for idx, sample in enumerate(samples)
                }

                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    attempt_results[idx] = result
                    per_sample_results[idx].append(result)
                    all_results.append(result)
                    completed += 1

                    # Track completion time for this sample
                    completion_time = time.time() - attempt_start_time
                    sample_completion_times.append(completion_time)

                    # Log progress
                    logger.info(f"Attempt {attempt_num + 1} - Completed {completed}/{len(samples)}")

        else:
            # Sequential processing for HuggingFace mode or single worker vLLM
            for idx, sample in enumerate(samples):
                logger.info(f"\nAttempt {attempt_num + 1} - Processing sample {idx + 1}/{len(samples)}")

                result = process_single_sample(
                    idx=idx,
                    sample=sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    use_vllm=use_vllm,
                    vllm_port=vllm_port,
                    vllm_model_name=vllm_model_name,
                    add_wrapper=add_wrapper,
                    model=model,
                    tokenizer=tokenizer,
                )

                per_sample_results[idx].append(result)
                all_results.append(result)
                attempt_results[idx] = result

                # Track completion time for this sample
                completion_time = time.time() - attempt_start_time
                sample_completion_times.append(completion_time)

        if attempt_output_dir:
            write_attempt_results_file(
                attempt_results=attempt_results,
                attempt_num=attempt_num + 1,
                output_dir=attempt_output_dir,
            )

        # Record time for this attempt
        attempt_time = time.time() - attempt_start_time
        attempt_times.append(attempt_time)

        # Calculate P90 time (time when 90% of samples completed)
        if sample_completion_times:
            sample_completion_times.sort()
            p90_index = int(len(sample_completion_times) * 0.9)
            # Ensure index is valid (at least 1, at most len-1)
            p90_index = max(0, min(p90_index, len(sample_completion_times) - 1))
            p90_time = sample_completion_times[p90_index]
            attempt_p90_times.append(p90_time)
            logger.info(f"\nAttempt {attempt_num + 1} completed in {attempt_time:.2f}s (P90: {p90_time:.2f}s)")
        else:
            attempt_p90_times.append(0.0)
            logger.info(f"\nAttempt {attempt_num + 1} completed in {attempt_time:.2f}s")

    # Compute final metrics
    total = len(samples)

    # Pass@k: number of samples that got at least one correct answer across all attempts
    pass_at_k = sum(1 for idx in range(total) if any(r.get("correct", False) for r in per_sample_results[idx]))

    # Avg@k: average number of correct answers per sample across all attempts
    avg_at_k = sum(sum(1 for r in per_sample_results[idx] if r.get("correct", False)) for idx in range(total)) / total if total > 0 else 0

    # Overall accuracy across all attempts
    total_correct = sum(1 for r in all_results if r.get("correct", False))
    overall_accuracy = total_correct / len(all_results) * 100 if all_results else 0

    # Calculate token statistics from all attempts
    total_input_tokens = sum(r.get("input_tokens", 0) for r in all_results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in all_results)
    avg_input_tokens = total_input_tokens / len(all_results) if all_results else 0
    avg_output_tokens = total_output_tokens / len(all_results) if all_results else 0

    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"Total samples: {total}")
    logger.info(f"Number of attempts: {attempts}")
    logger.info(f"Pass@k (samples with >=1 correct): {pass_at_k}/{total}")
    logger.info(f"Avg@k (average correct per sample across {attempts} attempts): {avg_at_k:.2f}")
    logger.info(f"Overall accuracy (all attempts): {overall_accuracy:.2f}%")
    logger.info(f"Average input tokens: {avg_input_tokens:.1f}")
    logger.info(f"Average output tokens: {avg_output_tokens:.1f}")
    logger.info(f"Total input tokens: {total_input_tokens}")
    logger.info(f"Total output tokens: {total_output_tokens}")
    logger.info("="*60)

    summary = {
        "total_samples": total,
        "num_attempts": attempts,
        "pass_at_k": pass_at_k,
        "pass_at_k_percentage": pass_at_k / total * 100 if total > 0 else 0,
        "avg_at_k": avg_at_k,
        "overall_accuracy_percentage": overall_accuracy,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "mode": "vllm" if use_vllm else "huggingface",
        "model_path": model_path if not use_vllm else f"vllm://{vllm_model_name}",
        "dataset_path": dataset_path,
    }

    if use_vllm:
        summary["vllm_port"] = vllm_port
        summary["vllm_model_name"] = vllm_model_name

    return {
        "summary": summary,
        "results": all_results,
        "samples": samples,
        "per_sample_results": per_sample_results,
        "attempt_times": attempt_times,
        "attempt_p90_times": attempt_p90_times,
    }


def write_pass_rate_files(
    samples: list[dict[str, Any]],
    per_sample_results: dict[int, list[dict[str, Any]]],
    attempts: int,
    output_dir: str = ".",
) -> None:
    """
    Write 3 output files based on pass rates when multiple attempts are used.

    Files created:
    1. {output_dir}/pass_rate_1.0.jsonl - Samples with 100% pass rate
    2. {output_dir}/pass_rate_0.0.jsonl - Samples with 0% pass rate
    3. {output_dir}/pass_rate_partial.jsonl - Samples with 0 < pass_rate < 1, sorted by pass rate (lowest first)
    """
    if attempts <= 1:
        logger.info("Skipping pass rate files - only 1 attempt")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_pass = []
    no_pass = []
    partial_pass = []

    # Categorize samples by pass rate
    for idx in range(len(samples)):
        sample = samples[idx].copy()

        # Calculate pass rate for this sample
        correct_count = sum(1 for r in per_sample_results[idx] if r.get("correct", False))
        pass_rate = correct_count / attempts

        # Add pass_rate to sample
        sample["pass_rate"] = pass_rate

        # Categorize
        if pass_rate == 1.0:
            full_pass.append(sample)
        elif pass_rate == 0.0:
            no_pass.append(sample)
        else:
            partial_pass.append(sample)

    # Sort partial pass by pass_rate (lowest first)
    partial_pass.sort(key=lambda x: x["pass_rate"])

    # Write files
    files_written = []

    # Write full pass file
    if full_pass:
        full_pass_file = output_path / "pass_rate_1.0.jsonl"
        with open(full_pass_file, 'w') as f:
            for sample in full_pass:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Written {len(full_pass)} samples to {full_pass_file}")
        files_written.append(full_pass_file)

    # Write no pass file
    if no_pass:
        no_pass_file = output_path / "pass_rate_0.0.jsonl"
        with open(no_pass_file, 'w') as f:
            for sample in no_pass:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Written {len(no_pass)} samples to {no_pass_file}")
        files_written.append(no_pass_file)

    # Write partial pass file (sorted by pass_rate)
    if partial_pass:
        partial_pass_file = output_path / "pass_rate_partial.jsonl"
        with open(partial_pass_file, 'w') as f:
            for sample in partial_pass:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Written {len(partial_pass)} samples to {partial_pass_file} (sorted by pass_rate, lowest first)")
        files_written.append(partial_pass_file)

    logger.info(f"\nPass rate files written:")
    logger.info(f"  Pass rate 1.0: {len(full_pass)} samples")
    logger.info(f"  Pass rate 0.0: {len(no_pass)} samples")
    logger.info(f"  Pass rate partial: {len(partial_pass)} samples")
    logger.info(f"  Total: {len(full_pass) + len(no_pass) + len(partial_pass)} samples")


def write_passing_rollouts(
    samples: list[dict[str, Any]],
    per_sample_results: dict[int, list[dict[str, Any]]],
    output_file: str = "passing_rollouts.jsonl",
) -> None:
    """
    Write the first passing rollout for each prompt to a JSONL file.

    For each prompt, if there is at least one passing rollout, write the first one
    in the format: {"instance_id": idx, "rollout_id": attempt_num, "messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    Prompts with no passing rollouts are skipped.
    Output is sorted by instance_id.
    """
    passing_rollouts = []

    for idx in range(len(samples)):
        sample = samples[idx]

        # Find the first passing rollout for this prompt
        first_passing = None
        first_passing_rollout_id = None
        for rollout_id, result in enumerate(per_sample_results[idx]):
            if result.get("correct", False):
                first_passing = result
                first_passing_rollout_id = rollout_id
                break

        if first_passing is None:
            # No passing rollouts for this prompt, skip it
            continue

        # Get the original prompt text
        prompt = sample["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Prompt is already in chat format, extract user content
            user_content = None
            for message in prompt:
                if message.get("role") == "user":
                    user_content = message.get("content", "")
                    break
            if user_content is None:
                user_content = prompt[0].get("content", "")
        else:
            # Plain text prompt
            user_content = str(prompt)

        # Format as required: instance_id + rollout_id + messages
        record = {
            "instance_id": idx,
            "rollout_id": first_passing_rollout_id,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": first_passing.get("response", "")},
            ],
        }
        passing_rollouts.append(record)

    # Sort by instance_id
    passing_rollouts.sort(key=lambda x: x["instance_id"])

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in passing_rollouts:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Written {len(passing_rollouts)} passing rollouts to {output_path}")
    logger.info(f"  (Skipped {len(samples) - len(passing_rollouts)} prompts with no passing rollouts)")


def write_all_rollouts(
    samples: list[dict[str, Any]],
    per_sample_results: dict[int, list[dict[str, Any]]],
    output_file: str = "all_rollouts.jsonl",
) -> None:
    """
    Write all rollouts for each prompt to a JSONL file.

    For each prompt and each attempt, write a record in the format:
    {"instance_id": idx, "rollout_id": attempt_num, "messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    Output is sorted by (instance_id, rollout_id).
    """
    all_rollouts = []

    for idx in range(len(samples)):
        sample = samples[idx]

        # Get the original prompt text
        prompt = sample["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # Prompt is already in chat format, extract user content
            user_content = None
            for message in prompt:
                if message.get("role") == "user":
                    user_content = message.get("content", "")
                    break
            if user_content is None:
                user_content = prompt[0].get("content", "")
        else:
            # Plain text prompt
            user_content = str(prompt)

        # Write all rollouts for this prompt
        for rollout_id, result in enumerate(per_sample_results[idx]):
            record = {
                "instance_id": idx,
                "rollout_id": rollout_id,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": result.get("response", "")},
                ],
            }
            all_rollouts.append(record)

    # Sort by (instance_id, rollout_id)
    all_rollouts.sort(key=lambda x: (x["instance_id"], x["rollout_id"]))

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in all_rollouts:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Written {len(all_rollouts)} rollouts to {output_path}")
    logger.info(f"  ({len(samples)} prompts x {len(per_sample_results[0]) if per_sample_results else 0} attempts)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF model on math dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/data/hf_models/Qwen3-0.6B/",
        help="Path to the HuggingFace model (ignored in vLLM mode)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/root/data/datasets/gsm8k/test.jsonl",
        help="Path to the dataset JSONL file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if (HF_AVAILABLE and torch.cuda.is_available()) else "cpu",
        help="Device to use for inference (HuggingFace mode only)",
    )

    # vLLM arguments
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM server mode instead of loading HuggingFace model",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="vLLM server port (default: 9000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3",
        help="Model name for vLLM server (default: qwen3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel workers for vLLM mode (default: 20)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of times to run evaluation on the entire dataset (default: 1)",
    )
    parser.add_argument(
        "--add-prompt-wrapper",
        action="store_true",
        help="Add prefix and suffix to prompts (default: False)",
    )
    parser.add_argument(
        "--attempt-output-dir",
        type=str,
        default=None,
        help="Directory to write one JSONL file per attempt with prompt, response, and extracted answer",
    )
    # Mutually exclusive group for rollout saving options
    rollout_save_group = parser.add_mutually_exclusive_group()
    rollout_save_group.add_argument(
        "--save-passing-rollouts",
        action="store_true",
        help="Save the first passing rollout for each prompt to output file (with rollout_id)",
    )
    rollout_save_group.add_argument(
        "--save-all-rollouts",
        action="store_true",
        help="Save all rollouts for each prompt to output file (with rollout_id)",
    )
    parser.add_argument(
        "--passing-rollouts-output",
        type=str,
        default="passing_rollouts.jsonl",
        help="Output file for rollouts (default: passing_rollouts.jsonl)",
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Math Evaluation Script Starting")
    logger.info("="*60)
    logger.info(f"Mode: {'vLLM' if args.vllm else 'HuggingFace'}")
    if args.vllm:
        logger.info(f"vLLM server: localhost:{args.port}")
        logger.info(f"vLLM model name: {args.model_name}")
        logger.info(f"Workers: {args.workers}")
    else:
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Device: {args.device}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Number of samples: {args.num_samples if args.num_samples else 'all'}")
    logger.info(f"Number of attempts: {args.attempts}")
    logger.info(f"Add prompt wrapper: {args.add_prompt_wrapper}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Sampling mode: {'greedy' if args.greedy else 'sampling'}")
    if args.save_passing_rollouts:
        logger.info(f"Save passing rollouts: {args.passing_rollouts_output}")
    if args.save_all_rollouts:
        logger.info(f"Save all rollouts: {args.passing_rollouts_output}")
    logger.info("="*60 + "\n")

    # Start timing
    start_time = time.time()

    # Run evaluation
    eval_results = evaluate(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        device=args.device,
        use_vllm=args.vllm,
        vllm_port=args.port,
        vllm_model_name=args.model_name,
        vllm_workers=args.workers,
        attempts=args.attempts,
        add_wrapper=args.add_prompt_wrapper,
        attempt_output_dir=args.attempt_output_dir,
    )

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in eval_results["summary"].items():
        if key in ["pass_at_k_percentage", "overall_accuracy_percentage"]:
            print(f"{key:25s}: {value:.2f}%")
        elif key in ["avg_input_tokens", "avg_output_tokens", "avg_at_k"]:
            print(f"{key:25s}: {value:.2f}")
        elif key in ["total_input_tokens", "total_output_tokens"]:
            print(f"{key:25s}: {value}")
        else:
            print(f"{key:25s}: {value}")
    print("="*50 + "\n")

    # Write pass rate files if multiple attempts were used
    if args.attempts > 1:
        # Determine output directory for pass rate files
        if args.output:
            pass_rate_dir = str(Path(args.output).parent)
        else:
            pass_rate_dir = "."

        write_pass_rate_files(
            samples=eval_results["samples"],
            per_sample_results=eval_results["per_sample_results"],
            attempts=args.attempts,
            output_dir=pass_rate_dir,
        )

    # Save rollouts if requested
    if args.save_passing_rollouts:
        write_passing_rollouts(
            samples=eval_results["samples"],
            per_sample_results=eval_results["per_sample_results"],
            output_file=args.passing_rollouts_output,
        )
    elif args.save_all_rollouts:
        write_all_rollouts(
            samples=eval_results["samples"],
            per_sample_results=eval_results["per_sample_results"],
            output_file=args.passing_rollouts_output,
        )

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Detailed results saved to {output_path}")

    # Calculate and print total time taken
    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("TIME SUMMARY")
    print("="*50)
    print(f"Total time: {total_time:.2f}s")

    # Print per-attempt times if available
    if "attempt_times" in eval_results and eval_results["attempt_times"]:
        print("\nPer-attempt times:")
        attempt_times = eval_results["attempt_times"]
        attempt_p90_times = eval_results.get("attempt_p90_times", [])

        for i, attempt_time in enumerate(attempt_times):
            if i < len(attempt_p90_times) and attempt_p90_times[i] > 0:
                print(f"  Attempt {i + 1}: {attempt_time:.2f}s (P90: {attempt_p90_times[i]:.2f}s)")
            else:
                print(f"  Attempt {i + 1}: {attempt_time:.2f}s")

    print("="*50 + "\n")


if __name__ == "__main__":
    main()
