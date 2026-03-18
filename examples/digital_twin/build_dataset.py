#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

import yaml


DEFAULT_CONFIG = "text_simulation/configs/openai_config.yaml"
DEFAULT_FIXED_EVAL_RUN_DIR = "benchmark_runs/20260317_221420"
LABEL_SUFFIX = "_wave4_Q_wave1_3_A.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical digital-twin JSONL dataset for slime.",
    )
    parser.add_argument(
        "--digital-twin-root",
        required=True,
        help="Path to the digital-twin-simulation repository root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where train.jsonl, eval.jsonl, and manifest.json will be written.",
    )
    parser.add_argument(
        "--simulation-config",
        default=DEFAULT_CONFIG,
        help="Config path relative to digital-twin root, or an absolute path.",
    )
    parser.add_argument(
        "--fixed-eval-run-dir",
        default=DEFAULT_FIXED_EVAL_RUN_DIR,
        help="Run directory relative to digital-twin root, or an absolute path, used to source the fixed 50 eval pids.",
    )
    parser.add_argument(
        "--additional-eval-size",
        type=int,
        default=150,
        help="Number of additional random eval pids sampled beyond the fixed eval set.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for sampling the additional eval pids.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def pid_sort_key(pid: str) -> int:
    match = re.search(r"pid_(\d+)", pid)
    if not match:
        raise ValueError(f"Invalid pid: {pid}")
    return int(match.group(1))


def extract_pid(name: str) -> str:
    match = re.search(r"(pid_\d+)", name)
    if not match:
        raise ValueError(f"Could not extract pid from {name}")
    return match.group(1)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / "train.jsonl", output_dir / "eval.jsonl", output_dir / "manifest.json"]
    if not overwrite:
        conflicts = [path for path in existing if path.exists()]
        if conflicts:
            joined = ", ".join(str(path) for path in conflicts)
            raise FileExistsError(f"Output files already exist: {joined}. Use --overwrite to replace them.")


def add_text_simulation_to_syspath(digital_twin_root: Path) -> Path:
    text_simulation_dir = digital_twin_root / "text_simulation"
    if not text_simulation_dir.is_dir():
        raise FileNotFoundError(f"Missing text_simulation directory: {text_simulation_dir}")
    if str(text_simulation_dir) not in sys.path:
        sys.path.insert(0, str(text_simulation_dir))
    return text_simulation_dir


def load_pipeline_modules(digital_twin_root: Path):
    add_text_simulation_to_syspath(digital_twin_root)
    batch_convert_personas = importlib.import_module("batch_convert_personas")
    convert_question_json_to_text = importlib.import_module("convert_question_json_to_text")
    create_text_simulation_input = importlib.import_module("create_text_simulation_input")
    return batch_convert_personas, convert_question_json_to_text, create_text_simulation_input


def load_system_instruction(config_path: Path) -> str:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    system_instruction = config.get("system_instruction")
    if not isinstance(system_instruction, str) or not system_instruction.strip():
        raise ValueError(f"system_instruction missing or empty in config: {config_path}")
    return system_instruction


def collect_fixed_eval_pids(run_dir: Path) -> list[str]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Fixed eval run directory not found: {run_dir}")

    pids = sorted(
        (
            entry.name
            for entry in run_dir.iterdir()
            if entry.is_dir() and re.fullmatch(r"pid_\d+", entry.name)
        ),
        key=pid_sort_key,
    )
    if not pids:
        raise ValueError(f"No pid directories found in fixed eval run directory: {run_dir}")
    return pids


def build_prompt_directory(
    digital_twin_root: Path,
    prompt_work_dir: Path,
) -> Path:
    batch_convert_personas, convert_question_json_to_text, create_text_simulation_input = load_pipeline_modules(
        digital_twin_root
    )

    persona_json_dir = digital_twin_root / "data/mega_persona_json/mega_persona"
    answer_blocks_dir = digital_twin_root / "data/mega_persona_json/answer_blocks"
    persona_text_dir = prompt_work_dir / "text_personas"
    question_text_dir = prompt_work_dir / "text_questions"
    combined_prompt_dir = prompt_work_dir / "text_simulation_input"

    batch_convert_personas.batch_convert_personas(
        persona_json_dir=str(persona_json_dir),
        output_text_dir=str(persona_text_dir),
        variant="full",
    )

    question_text_dir.mkdir(parents=True, exist_ok=True)
    for question_path in sorted(answer_blocks_dir.glob(f"*{LABEL_SUFFIX.replace('_wave1_3_A.json', '_wave4_A.json')}"), key=lambda p: pid_sort_key(extract_pid(p.name))):
        convert_question_json_to_text.process_json_file(
            input_file=str(question_path),
            output_dir=str(question_text_dir),
            include_reasoning=False,
        )

    create_text_simulation_input.create_combined_prompts(
        persona_text_dir=str(persona_text_dir),
        question_prompts_dir=str(question_text_dir),
        output_combined_prompts_dir=str(combined_prompt_dir),
    )

    return combined_prompt_dir


def load_prompt_texts(prompt_dir: Path) -> dict[str, str]:
    prompt_texts: dict[str, str] = {}
    for prompt_path in sorted(prompt_dir.glob("pid_*_prompt.txt"), key=lambda p: pid_sort_key(extract_pid(p.name))):
        pid = extract_pid(prompt_path.name)
        prompt_texts[pid] = prompt_path.read_text(encoding="utf-8")
    if not prompt_texts:
        raise ValueError(f"No prompt files found in {prompt_dir}")
    return prompt_texts


def load_labels(answer_blocks_dir: Path, pids: set[str]) -> dict[str, object]:
    labels: dict[str, object] = {}
    for pid in sorted(pids, key=pid_sort_key):
        label_path = answer_blocks_dir / f"{pid}{LABEL_SUFFIX}"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for {pid}: {label_path}")
        with label_path.open(encoding="utf-8") as handle:
            labels[pid] = json.load(handle)
    return labels


def choose_eval_pids(
    all_pids: list[str],
    fixed_eval_pids: list[str],
    additional_eval_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    all_pid_set = set(all_pids)
    fixed_eval_set = set(fixed_eval_pids)
    missing = sorted(fixed_eval_set - all_pid_set, key=pid_sort_key)
    if missing:
        raise ValueError(f"Fixed eval pids missing from generated dataset: {missing}")

    remaining = sorted(all_pid_set - fixed_eval_set, key=pid_sort_key)
    if additional_eval_size > len(remaining):
        raise ValueError(
            f"Requested {additional_eval_size} additional eval pids, but only {len(remaining)} are available."
        )

    rng = random.Random(seed)
    sampled = sorted(rng.sample(remaining, additional_eval_size), key=pid_sort_key)
    eval_pids = sorted(list(fixed_eval_set) + sampled, key=pid_sort_key)
    train_pids = sorted(all_pid_set - set(eval_pids), key=pid_sort_key)
    return train_pids, eval_pids


def build_record(
    pid: str,
    split: str,
    prompt_text: str,
    system_instruction: str,
    label_filename: str,
    label_data: object,
) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text},
        ],
        "label": label_data,
        "metadata": {
            "pid": pid,
            "split": split,
            "label_json_filename": label_filename,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def write_manifest(
    manifest_path: Path,
    *,
    digital_twin_root: Path,
    simulation_config: Path,
    fixed_eval_run_dir: Path,
    split_seed: int,
    additional_eval_size: int,
    fixed_eval_pids: list[str],
    eval_pids: list[str],
    train_pids: list[str],
) -> None:
    manifest = {
        "dataset_name": "digital_twin_canonical_v1",
        "digital_twin_root": str(digital_twin_root),
        "simulation_config": str(simulation_config),
        "fixed_eval_run_dir": str(fixed_eval_run_dir),
        "split_seed": split_seed,
        "fixed_eval_pid_count": len(fixed_eval_pids),
        "additional_eval_size": additional_eval_size,
        "counts": {
            "train": len(train_pids),
            "eval": len(eval_pids),
            "total": len(train_pids) + len(eval_pids),
        },
        "fixed_eval_pids": fixed_eval_pids,
        "eval_pids": eval_pids,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    args = parse_args()

    digital_twin_root = Path(args.digital_twin_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    simulation_config = resolve_path(digital_twin_root, args.simulation_config).resolve()
    fixed_eval_run_dir = resolve_path(digital_twin_root, args.fixed_eval_run_dir).resolve()
    answer_blocks_dir = digital_twin_root / "data/mega_persona_json/answer_blocks"

    if not digital_twin_root.is_dir():
        raise FileNotFoundError(f"digital-twin root not found: {digital_twin_root}")
    if not simulation_config.is_file():
        raise FileNotFoundError(f"Simulation config not found: {simulation_config}")
    if not answer_blocks_dir.is_dir():
        raise FileNotFoundError(f"Answer blocks directory not found: {answer_blocks_dir}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)
    system_instruction = load_system_instruction(simulation_config)
    fixed_eval_pids = collect_fixed_eval_pids(fixed_eval_run_dir)

    with tempfile.TemporaryDirectory(prefix="digital_twin_builder_") as temp_dir:
        prompt_dir = build_prompt_directory(
            digital_twin_root=digital_twin_root,
            prompt_work_dir=Path(temp_dir),
        )
        prompt_texts = load_prompt_texts(prompt_dir)

    all_pids = sorted(prompt_texts.keys(), key=pid_sort_key)
    labels = load_labels(answer_blocks_dir=answer_blocks_dir, pids=set(all_pids))
    train_pids, eval_pids = choose_eval_pids(
        all_pids=all_pids,
        fixed_eval_pids=fixed_eval_pids,
        additional_eval_size=args.additional_eval_size,
        seed=args.split_seed,
    )

    train_rows = [
        build_record(
            pid=pid,
            split="train",
            prompt_text=prompt_texts[pid],
            system_instruction=system_instruction,
            label_filename=f"{pid}{LABEL_SUFFIX}",
            label_data=labels[pid],
        )
        for pid in train_pids
    ]
    eval_rows = [
        build_record(
            pid=pid,
            split="eval",
            prompt_text=prompt_texts[pid],
            system_instruction=system_instruction,
            label_filename=f"{pid}{LABEL_SUFFIX}",
            label_data=labels[pid],
        )
        for pid in eval_pids
    ]

    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "eval.jsonl", eval_rows)
    write_manifest(
        output_dir / "manifest.json",
        digital_twin_root=digital_twin_root,
        simulation_config=simulation_config,
        fixed_eval_run_dir=fixed_eval_run_dir,
        split_seed=args.split_seed,
        additional_eval_size=args.additional_eval_size,
        fixed_eval_pids=fixed_eval_pids,
        eval_pids=eval_pids,
        train_pids=train_pids,
    )

    print(f"Wrote train split: {output_dir / 'train.jsonl'} ({len(train_rows)} rows)")
    print(f"Wrote eval split: {output_dir / 'eval.jsonl'} ({len(eval_rows)} rows)")
    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
