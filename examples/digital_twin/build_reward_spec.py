#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = "examples/digital_twin/reward_spec.json"
DEFAULT_BENCHMARK_CSV = "data/wave_csv/wave_4_numbers_anonymized.csv"
DEFAULT_COLUMN_MAPPING = "evaluation/column_mapping.csv"
DEFAULT_ANSWER_BLOCKS = "data/mega_persona_json/answer_blocks"
DECILE_GROUP_COLUMNS = {
    "QID164_GROUP": {"Q164", "Q166"},
    "QID168_GROUP": {"Q168", "Q170"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the static reward spec used by the digital-twin slime reward.",
    )
    parser.add_argument(
        "--digital-twin-root",
        required=True,
        help="Path to the digital-twin-simulation repository root.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON path, absolute or relative to the slime repository root.",
    )
    parser.add_argument(
        "--benchmark-csv",
        default=DEFAULT_BENCHMARK_CSV,
        help="Benchmark CSV path relative to digital-twin root, or an absolute path.",
    )
    parser.add_argument(
        "--column-mapping",
        default=DEFAULT_COLUMN_MAPPING,
        help="Column mapping CSV path relative to digital-twin root, or an absolute path.",
    )
    parser.add_argument(
        "--answer-blocks-dir",
        default=DEFAULT_ANSWER_BLOCKS,
        help="Answer blocks directory relative to digital-twin root, or an absolute path.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def add_evaluation_to_syspath(digital_twin_root: Path) -> None:
    evaluation_dir = digital_twin_root / "evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Missing evaluation directory: {evaluation_dir}")
    if str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))


def extract_importid_mapping(benchmark_csv: Path) -> dict[str, str]:
    with benchmark_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = []
        for index, row in enumerate(reader):
            rows.append(row)
            if index >= 5:
                break

    if len(rows) < 2:
        raise ValueError(f"Benchmark CSV is too short to extract ImportIds: {benchmark_csv}")

    headers = rows[0]
    import_row = None
    for row in rows[1:6]:
        importid_count = sum(1 for cell in row if cell.startswith("{") and '"ImportId"' in cell)
        if importid_count > len(row) * 0.3:
            import_row = row
            break

    if import_row is None:
        raise ValueError(f"Could not find ImportId row in benchmark CSV: {benchmark_csv}")

    mapping: dict[str, str] = {}
    for header, cell in zip(headers, import_row, strict=False):
        if cell.startswith("{") and '"ImportId"' in cell:
            import_id = json.loads(cell).get("ImportId")
            if import_id:
                mapping[str(import_id).upper()] = str(header).upper()

    return mapping


def extract_manual_mapping(column_mapping_csv: Path) -> dict[str, str]:
    mapping_df = pd.read_csv(column_mapping_csv)
    mapping: dict[str, str] = {}

    for _, row in mapping_df.iterrows():
        wave4_col = str(row["wave4_column_name"]).strip().upper()
        input_col = str(row["input_column_name"]).strip().upper()
        if wave4_col and input_col and wave4_col != "NAN" and input_col != "NAN":
            mapping[input_col] = wave4_col

    return mapping


def collect_decile_thresholds(
    answer_blocks_dir: Path,
    input_to_benchmark_column: dict[str, str],
) -> dict[str, list[float]]:
    from json2csv import AnswerExtractor, ExtractionMode

    extractor = AnswerExtractor(ExtractionMode.NUMERIC)
    grouped_values: dict[str, list[float]] = {group: [] for group in DECILE_GROUP_COLUMNS}

    for answer_path in sorted(answer_blocks_dir.glob("*_wave4_Q_wave1_3_A.json")):
        answers = extractor.extract_from_file(str(answer_path), include_text_labels=False)
        for input_column, value in answers.items():
            benchmark_column = input_to_benchmark_column.get(str(input_column).upper())
            if benchmark_column is None:
                continue

            for group_name, group_columns in DECILE_GROUP_COLUMNS.items():
                if benchmark_column not in group_columns:
                    continue
                try:
                    grouped_values[group_name].append(float(value))
                except (TypeError, ValueError):
                    pass

    thresholds: dict[str, list[float]] = {}
    for group_name, values in grouped_values.items():
        if not values:
            raise ValueError(f"No values found for decile group {group_name}")
        thresholds[group_name] = np.percentile(values, np.arange(10, 100, 10)).astype(float).tolist()

    return thresholds


def main() -> int:
    args = parse_args()

    slime_root = Path(__file__).resolve().parents[2]
    digital_twin_root = Path(args.digital_twin_root).expanduser().resolve()
    output_path = resolve_path(slime_root, args.output).resolve()
    benchmark_csv = resolve_path(digital_twin_root, args.benchmark_csv).resolve()
    column_mapping_csv = resolve_path(digital_twin_root, args.column_mapping).resolve()
    answer_blocks_dir = resolve_path(digital_twin_root, args.answer_blocks_dir).resolve()

    if not digital_twin_root.is_dir():
        raise FileNotFoundError(f"digital-twin root not found: {digital_twin_root}")
    if not benchmark_csv.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {benchmark_csv}")
    if not column_mapping_csv.is_file():
        raise FileNotFoundError(f"Column mapping CSV not found: {column_mapping_csv}")
    if not answer_blocks_dir.is_dir():
        raise FileNotFoundError(f"Answer blocks directory not found: {answer_blocks_dir}")

    add_evaluation_to_syspath(digital_twin_root)
    from mad_accuracy_evaluation import get_default_column_ranges

    input_to_benchmark_column = extract_importid_mapping(benchmark_csv)
    input_to_benchmark_column.update(extract_manual_mapping(column_mapping_csv))

    column_ranges = {
        str(column).upper(): float(max_value - min_value)
        for column, (min_value, max_value) in get_default_column_ranges().items()
    }
    decile_thresholds = collect_decile_thresholds(answer_blocks_dir, input_to_benchmark_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "digital_twin_reward_v1",
        "source": {
            "digital_twin_root": str(digital_twin_root),
            "benchmark_csv": str(benchmark_csv),
            "column_mapping_csv": str(column_mapping_csv),
            "answer_blocks_dir": str(answer_blocks_dir),
        },
        "input_to_benchmark_column": dict(sorted(input_to_benchmark_column.items())),
        "column_ranges": dict(sorted(column_ranges.items())),
        "decile_thresholds": decile_thresholds,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote reward spec: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
