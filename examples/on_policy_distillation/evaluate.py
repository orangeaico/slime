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
) -> str:
    """Generate a response from the model."""
    logger.info(f"[Sample {sample_idx}] Preparing prompt...")

    # Apply chat template if the prompt is in the expected format
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Prompt is already in chat format - wrap the content of the last user message
        messages = prompt.copy()
        if messages[-1].get("role") == "user":
            messages[-1]["content"] = add_prompt_wrapper(messages[-1]["content"])
    else:
        # Wrap plain text in user message and add prefix/suffix
        messages = [{"role": "user", "content": add_prompt_wrapper(prompt)}]

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
) -> str:
    """Generate a response using vLLM server API."""
    logger.info(f"[Sample {sample_idx}] Preparing vLLM API request...")

    # Prepare messages format
    if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
        # Prompt is already in chat format - wrap the content of the last user message
        messages = prompt.copy()
        if messages[-1].get("role") == "user":
            messages[-1]["content"] = add_prompt_wrapper(messages[-1]["content"])
    else:
        # Wrap plain text in user message and add prefix/suffix
        messages = [{"role": "user", "content": add_prompt_wrapper(prompt)}]

    # Prepare API request
    url = f"http://localhost:{port}/v1/chat/completions"

    # # Non-thinking payload for qwen3 1.7B
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
    model: Any = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Process a single sample and return the result."""
    prompt = sample["prompt"]
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
            )
        else:
            response, input_tokens, output_tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                sample_idx=idx,
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
            "prompt": prompt if isinstance(prompt, str) else prompt[0]["content"],
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
            "error": str(e),
            "correct": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }


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

    # Evaluation loop
    results = []
    correct = 0
    total = 0

    logger.info("="*60)
    logger.info("Starting evaluation...")
    if use_vllm and vllm_workers > 1:
        logger.info(f"Using parallel processing with {vllm_workers} workers")
    logger.info("="*60)

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
                results.append(result)

                if result.get("correct", False):
                    correct += 1
                total += 1
                completed += 1

                # Log progress
                current_acc = correct / total * 100
                logger.info(f"Completed {completed}/{len(samples)} | Running accuracy: {correct}/{total} = {current_acc:.2f}%")

        # Sort results by idx to maintain order
        results.sort(key=lambda x: x["idx"])

    else:
        # Sequential processing for HuggingFace mode or single worker vLLM
        for idx, sample in enumerate(samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {idx + 1}/{len(samples)}")
            logger.info(f"{'='*60}")

            result = process_single_sample(
                idx=idx,
                sample=sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                use_vllm=use_vllm,
                vllm_port=vllm_port,
                vllm_model_name=vllm_model_name,
                model=model,
                tokenizer=tokenizer,
            )

            results.append(result)

            if result.get("correct", False):
                correct += 1
            total += 1

            # Log progress summary
            current_acc = correct / total * 100
            logger.info(f"[Sample {idx}] Running accuracy: {correct}/{total} = {current_acc:.2f}%")

    # Compute final metrics
    accuracy = correct / total * 100 if total > 0 else 0

    # Calculate token statistics
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    avg_input_tokens = total_input_tokens / total if total > 0 else 0
    avg_output_tokens = total_output_tokens / total if total > 0 else 0

    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Incorrect: {total - correct}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Average input tokens: {avg_input_tokens:.1f}")
    logger.info(f"Average output tokens: {avg_output_tokens:.1f}")
    logger.info(f"Total input tokens: {total_input_tokens}")
    logger.info(f"Total output tokens: {total_output_tokens}")
    logger.info("="*60)

    summary = {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
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
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF model on math dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/data/hf_models/Qwen3-1.7B/",
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
        default=16384,
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
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Sampling mode: {'greedy' if args.greedy else 'sampling'}")
    logger.info("="*60 + "\n")

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
    )

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for key, value in eval_results["summary"].items():
        if key == "accuracy":
            print(f"{key:20s}: {value:.2f}%")
        elif key in ["avg_input_tokens", "avg_output_tokens"]:
            print(f"{key:20s}: {value:.1f}")
        elif key in ["total_input_tokens", "total_output_tokens"]:
            print(f"{key:20s}: {value}")
        else:
            print(f"{key:20s}: {value}")
    print("="*50 + "\n")

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
