#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SPEC_PATH = Path(__file__).with_name("reward_spec.json")
DECILE_GROUP_BY_COLUMN = {
    "Q164": "QID164_GROUP",
    "Q166": "QID164_GROUP",
    "Q168": "QID168_GROUP",
    "Q170": "QID168_GROUP",
}


@dataclass
class GoldContext:
    questions: list[dict[str, Any]]
    gold_columns: dict[str, float]


_GOLD_CONTEXT_CACHE: dict[str, GoldContext] = {}


@lru_cache(maxsize=1)
def load_reward_spec() -> dict[str, Any]:
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(
            f"Missing reward spec at {SPEC_PATH}. Generate it with "
            "examples/digital_twin/build_reward_spec.py before training."
        )

    with SPEC_PATH.open(encoding="utf-8") as handle:
        raw_spec = json.load(handle)

    input_to_benchmark = {
        str(input_col).upper(): str(benchmark_col).upper()
        for input_col, benchmark_col in raw_spec["input_to_benchmark_column"].items()
    }
    column_ranges = {str(col).upper(): float(value) for col, value in raw_spec["column_ranges"].items()}
    decile_thresholds = {
        str(group): [float(threshold) for threshold in thresholds]
        for group, thresholds in raw_spec["decile_thresholds"].items()
    }

    return {
        "version": raw_spec.get("version", "digital_twin_reward_v1"),
        "input_to_benchmark_column": input_to_benchmark,
        "column_ranges": column_ranges,
        "decile_thresholds": decile_thresholds,
    }


def _label_blocks(sample) -> list[dict[str, Any]]:
    label = sample.label
    if isinstance(label, str):
        label = json.loads(label)
    if not isinstance(label, list):
        raise TypeError(f"Expected sample.label to be a list of blocks, got {type(label)}")
    return label


def _gold_cache_key(sample) -> str:
    metadata = sample.metadata or {}
    if metadata.get("label_json_filename"):
        return str(metadata["label_json_filename"])
    if metadata.get("pid"):
        return str(metadata["pid"])
    return json.dumps(_label_blocks(sample), sort_keys=True)


def _extract_questions(blocks: Any) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    blocks_list = blocks if isinstance(blocks, list) else [blocks]

    for block in blocks_list:
        if not isinstance(block, dict):
            continue

        if "Questions" in block:
            questions.extend(_flatten_questions(block["Questions"]))
        elif "Elements" in block:
            for element in block["Elements"]:
                if isinstance(element, dict) and "Questions" in element:
                    questions.extend(_flatten_questions(element["Questions"]))
        else:
            questions.append(block)

    return questions


def _flatten_questions(questions_data: Any) -> list[dict[str, Any]]:
    if isinstance(questions_data, list):
        return [question for question in questions_data if isinstance(question, dict)]
    if isinstance(questions_data, dict):
        return [question for question in questions_data.values() if isinstance(question, dict)]
    return []


def _non_db_questions(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [question for question in _extract_questions(blocks) if question.get("QuestionType") != "DB"]


def _is_valid_number(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _is_in_range(value: Any, min_value: float, max_value: float) -> bool:
    if not _is_valid_number(value):
        return False
    numeric_value = float(value)
    return min_value <= numeric_value <= max_value


def _validate_matrix_response(response: Any, question: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    if len(response["SelectedByPosition"]) != len(response["SelectedText"]):
        return False
    if not all(isinstance(pos, (int, str)) for pos in response["SelectedByPosition"]):
        return False
    if not all(isinstance(text, str) for text in response["SelectedText"]):
        return False
    return True


def _validate_single_choice_response(response: Any, question: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    if not isinstance(response["SelectedByPosition"], (int, str)):
        return False
    if isinstance(response["SelectedByPosition"], str):
        try:
            int(response["SelectedByPosition"])
        except ValueError:
            return False
    if not isinstance(response["SelectedText"], str):
        return False
    return True


def _validate_slider_response(response: Any, question: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    if "Values" not in response:
        return False
    if not isinstance(response["Values"], list):
        return False
    if not all(_is_valid_number(value) for value in response["Values"]):
        return False

    constraints = question.get("NumericConstraints", {})
    if "MinValue" in constraints and "MaxValue" in constraints:
        if not all(_is_in_range(value, constraints["MinValue"], constraints["MaxValue"]) for value in response["Values"]):
            return False
    return True


def _validate_text_entry_response(response: Any, question: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False
    if "Text" not in response:
        return False
    if not isinstance(response["Text"], str):
        return False
    return True


def _validate_response(predicted_question: Any, question: dict[str, Any]) -> bool:
    if not isinstance(predicted_question, dict):
        return False

    question_type = predicted_question.get("QuestionType") or predicted_question.get("Question Type")
    answers = predicted_question.get("Answers")

    if not question_type or not answers:
        return False

    validators = {
        "Matrix": _validate_matrix_response,
        "Single Choice": _validate_single_choice_response,
        "Slider": _validate_slider_response,
        "Text Entry": _validate_text_entry_response,
    }
    validation_func = validators.get(question_type)
    if validation_func is None:
        return False

    return validation_func(answers, question)


def _get_item_ids_and_text(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    if "Statements" in question or "StatementsID" in question:
        item_ids = [str(item_id) for item_id in question.get("StatementsID", [])]
        item_text = [str(text) for text in question.get("Statements", [])]
    else:
        item_ids = [str(item_id) for item_id in question.get("RowsID", [])]
        item_text = [str(text) for text in question.get("Rows", [])]

    if not item_ids and item_text:
        item_ids = [str(index + 1) for index in range(len(item_text))]

    return item_ids, item_text


def _extract_numeric_input_columns(question: dict[str, Any], answer_override: dict[str, Any] | None = None) -> dict[str, Any]:
    qid = str(question.get("QuestionID", "")).upper()
    if not qid:
        return {}

    answer_data = answer_override if answer_override is not None else question.get("Answers", {})
    if not isinstance(answer_data, dict) or not answer_data:
        return {}

    question_type = question.get("QuestionType")
    settings = question.get("Settings", {})

    if question_type == "MC":
        selector = settings.get("Selector", "")
        if selector in {"SAVR", "SAHR"}:
            selected_position = answer_data.get("SelectedByPosition", "")
            if isinstance(selected_position, (int, str)) and selected_position not in {"", None}:
                try:
                    return {qid: int(selected_position)}
                except ValueError:
                    selected_text = answer_data.get("SelectedText", "")
                    return {qid: selected_text}
            selected_text = answer_data.get("SelectedText", "")
            return {qid: selected_text}

        selected = answer_data.get("Selected", [])
        if not isinstance(selected, list):
            return {}
        return {f"{qid}_{str(choice_id).upper()}": 1 for choice_id in selected}

    if question_type == "Matrix":
        answers: dict[str, Any] = {}
        item_ids, _item_text = _get_item_ids_and_text(question)
        selected = answer_data.get("SelectedByPosition", [])
        selected_text = answer_data.get("SelectedText", [])

        for index, item_id in enumerate(item_ids):
            key = f"{qid}_{item_id.upper()}"
            if selected and index < len(selected):
                value = selected[index]
                try:
                    answers[key] = int(value)
                except (ValueError, TypeError):
                    if selected_text and index < len(selected_text):
                        answers[key] = selected_text[index]
                    else:
                        answers[key] = None
            else:
                answers[key] = None
        return answers

    if question_type == "Slider":
        answers: dict[str, Any] = {}
        values = answer_data.get("Values", [])
        if not isinstance(values, list):
            return {}

        has_multiple = len(values) > 1 and any(
            field in question for field in ("Statements", "StatementsID", "Rows", "RowsID")
        )
        if has_multiple:
            item_ids, _item_text = _get_item_ids_and_text(question)
            for index, value in enumerate(values):
                if index < len(item_ids):
                    answers[f"{qid}_{item_ids[index].upper()}"] = float(value) if value is not None else None
            return answers

        if values:
            return {qid: float(values[0]) if values[0] is not None else None}
        return {}

    if question_type == "TE":
        answers: dict[str, Any] = {}
        selector = settings.get("Selector", "")
        if selector in {"SL", "ML"} and any(
            field in question for field in ("Statements", "StatementsID", "Rows", "RowsID")
        ):
            item_ids, _item_text = _get_item_ids_and_text(question)
            texts = answer_data.get("Text", [])
            if isinstance(texts, list):
                for index, text in enumerate(texts):
                    if index < len(item_ids):
                        answers[f"{qid}_{item_ids[index].upper()}"] = str(text) if text is not None else None
                return answers
            return {f"{qid}_TEXT": str(texts) if texts else None}

        text = answer_data.get("Text", "")
        return {f"{qid}_TEXT": str(text) if text else None}

    if question_type == "CS":
        answers: dict[str, Any] = {}
        item_ids, _item_text = _get_item_ids_and_text(question)
        values = answer_data.get("Values", [])
        if isinstance(values, list):
            for index, value in enumerate(values):
                if index < len(item_ids):
                    try:
                        answers[f"{qid}_{item_ids[index].upper()}"] = float(value) if value is not None else None
                    except (ValueError, TypeError):
                        answers[f"{qid}_{item_ids[index].upper()}"] = str(value) if value is not None else None
            return answers

        text = answer_data.get("Text", "")
        return {qid: str(text) if text else None}

    value = answer_data.get("Value")
    if value is not None:
        try:
            return {qid: float(value)}
        except (ValueError, TypeError):
            return {qid: str(value)}

    values = answer_data.get("Values", [])
    if isinstance(values, list) and values and values[0] is not None:
        try:
            return {qid: float(values[0])}
        except (ValueError, TypeError):
            return {qid: str(values[0])}

    text = answer_data.get("Text", "")
    return {qid: str(text) if text else None}


def _map_input_to_benchmark_columns(input_columns: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    mapping = spec["input_to_benchmark_column"]
    column_ranges = spec["column_ranges"]

    for input_column, value in input_columns.items():
        benchmark_column = mapping.get(str(input_column).upper())
        if benchmark_column is None:
            continue
        if benchmark_column not in column_ranges:
            continue
        mapped[benchmark_column] = value

    return mapped


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _assign_decile(value: float, thresholds: list[float]) -> float:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return float(index + 1)
    return 10.0


def _normalize_for_scoring(column: str, value: Any, spec: dict[str, Any]) -> float | None:
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return None

    decile_group = DECILE_GROUP_BY_COLUMN.get(column.upper())
    if decile_group:
        thresholds = spec["decile_thresholds"].get(decile_group)
        if not thresholds:
            return None
        numeric_value = _assign_decile(numeric_value, thresholds)

    return float(numeric_value)


def _prepare_gold_context(sample) -> GoldContext:
    spec = load_reward_spec()
    questions = _non_db_questions(_label_blocks(sample))

    gold_input_columns: dict[str, Any] = {}
    for question in questions:
        gold_input_columns.update(_extract_numeric_input_columns(question))

    gold_benchmark_columns = _map_input_to_benchmark_columns(gold_input_columns, spec)
    gold_scorable_columns: dict[str, float] = {}
    for column, value in gold_benchmark_columns.items():
        normalized_value = _normalize_for_scoring(column, value, spec)
        if normalized_value is not None:
            gold_scorable_columns[column] = normalized_value

    return GoldContext(questions=questions, gold_columns=gold_scorable_columns)


def _get_gold_context(sample) -> GoldContext:
    cache_key = _gold_cache_key(sample)
    context = _GOLD_CONTEXT_CACHE.get(cache_key)
    if context is None:
        context = _prepare_gold_context(sample)
        _GOLD_CONTEXT_CACHE[cache_key] = context
    return context


def _parse_response_json(response_text: str) -> tuple[bool, Any]:
    if not isinstance(response_text, str):
        return False, None

    stripped = response_text.strip()
    if not stripped:
        return False, None

    try:
        return True, json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```json\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match is not None:
        try:
            return True, json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            return False, None

    return False, None


def _score_prediction(sample, parsed_output: Any, json_valid: bool) -> dict[str, Any]:
    spec = load_reward_spec()
    gold_context = _get_gold_context(sample)
    predicted_columns: dict[str, Any] = {}
    valid_question_count = 0

    parsed_questions = parsed_output if isinstance(parsed_output, dict) else {}

    for index, question in enumerate(gold_context.questions, start=1):
        predicted_question = parsed_questions.get(f"Q{index}")
        if not _validate_response(predicted_question, question):
            continue

        valid_question_count += 1
        predicted_answers = predicted_question["Answers"]
        input_columns = _extract_numeric_input_columns(question, answer_override=predicted_answers)
        predicted_columns.update(_map_input_to_benchmark_columns(input_columns, spec))

    expected_columns = gold_context.gold_columns
    per_column_scores: list[float] = []

    for column, gold_value in expected_columns.items():
        predicted_value = _normalize_for_scoring(column, predicted_columns.get(column), spec)
        if predicted_value is None:
            per_column_scores.append(0.0)
            continue

        column_range = spec["column_ranges"][column]
        if column_range <= 0:
            per_column_scores.append(0.0)
            continue

        per_column_scores.append(1.0 - abs(predicted_value - gold_value) / column_range)

    benchmark_accuracy = (
        sum(per_column_scores) / len(per_column_scores) if per_column_scores else 0.0
    )
    num_expected_questions = len(gold_context.questions)
    format_valid_ratio = (
        valid_question_count / num_expected_questions if num_expected_questions else 0.0
    )

    return {
        "score": format_valid_ratio * benchmark_accuracy,
        "json_valid": 1.0 if json_valid else 0.0,
        "format_valid_ratio": format_valid_ratio,
        "benchmark_accuracy": benchmark_accuracy,
        "num_valid_questions": valid_question_count,
        "num_expected_questions": num_expected_questions,
        "num_scored_columns": len(expected_columns),
    }


async def reward_func(args, sample, **kwargs):
    json_valid, parsed_output = _parse_response_json(sample.response)
    if not json_valid:
        gold_context = _get_gold_context(sample)
        return {
            "score": 0.0,
            "json_valid": 0.0,
            "format_valid_ratio": 0.0,
            "benchmark_accuracy": 0.0,
            "num_valid_questions": 0,
            "num_expected_questions": len(gold_context.questions),
            "num_scored_columns": len(gold_context.gold_columns),
        }

    return _score_prediction(sample, parsed_output, json_valid=True)
