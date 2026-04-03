import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import yaml

_single_slow_session_id: str | None = None
_single_slow_lock = threading.Lock()

DEFAULT_HARDCODED_PROGRAM: dict[str, Any] = {
    "action_cycle": ["create"],
    "file_path_prefix": "/testbed/slime_dummy_file",
    "submit": {
        "enabled": True,
        "base_turn": 2,
        "spread": 4,
        "task_completed": "true",
        "message": "Submitting after deterministic hardcoded debug loop.",
    },
    "slowdown": {
        "enabled": False,
        "selection_mode": "hash",
        "hash_mod": 3,
        "hash_threshold": 1,
        "submit_extra_turns": 0,
        "sleep_seconds": 0,
        "until_turn": 0,
    },
    "templates": {
        "create_discussion": "Hardcoded debug action: create file.",
        "create_file_text": "session={session_id}\\ninstance={instance_id}\\nturn={turn}\\naction=create\\n",
        "str_replace_discussion": "Hardcoded debug action: str_replace file.",
        "str_replace_new_text": "session={session_id}\\ninstance={instance_id}\\nturn={turn}\\naction=str_replace\\n",
        "finish_discussion": "Hardcoded debug action: submit at staggered turn.",
    },
}


def _stable_hash_mod(value: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def _format_template(template: str, context: dict[str, Any], *, default: str = "") -> str:
    try:
        return template.format(**context)
    except Exception:
        return default or template


def load_hardcoded_program(program_path: Path) -> dict[str, Any]:
    if not program_path.exists():
        raise FileNotFoundError(f"Hardcoded program file not found: {program_path}")

    if program_path.suffix.lower() == ".json":
        loaded = json.loads(program_path.read_text())
    else:
        loaded = yaml.safe_load(program_path.read_text())

    if not isinstance(loaded, dict):
        raise ValueError(f"Hardcoded program must be a mapping/dict: {program_path}")

    program: dict[str, Any] = copy.deepcopy(DEFAULT_HARDCODED_PROGRAM)
    for key, value in loaded.items():
        if key in {"submit", "slowdown", "templates"} and isinstance(value, dict):
            program[key].update(value)
        else:
            program[key] = value

    action_cycle = program.get("action_cycle")
    if action_cycle is None:
        action_mix = str(program.get("action_mix", "")).strip().lower()
        if action_mix == "alternate_create_str_replace":
            action_cycle = ["create", "str_replace"]
        elif action_mix in {"create", "str_replace"}:
            action_cycle = [action_mix]

    if isinstance(action_cycle, str):
        action_cycle = [action_cycle]

    if not isinstance(action_cycle, list):
        action_cycle = ["create"]

    normalized_cycle = []
    for action in action_cycle:
        normalized_action = str(action).strip().lower()
        if normalized_action in {"create", "str_replace"}:
            normalized_cycle.append(normalized_action)

    if not normalized_cycle:
        normalized_cycle = ["create"]
    program["action_cycle"] = normalized_cycle
    return program


def build_hardcoded_program_message(
    program: dict[str, Any],
    *,
    session_id: str,
    instance_id: str,
    turn: int,
    state: dict[str, Any],
    logger: Any | None = None,
) -> str:
    session_short = "".join(ch for ch in session_id if ch.isalnum())[:8] or "session"
    instance_short = "".join(ch for ch in instance_id if ch.isalnum())[:16] or "instance"
    templates = program.get("templates", {})
    if not isinstance(templates, dict):
        templates = {}

    submit_cfg = program.get("submit", {})
    if not isinstance(submit_cfg, dict):
        submit_cfg = {}

    slowdown_cfg = program.get("slowdown", {})
    if not isinstance(slowdown_cfg, dict):
        slowdown_cfg = {}
    slow_enabled = bool(slowdown_cfg.get("enabled", False))
    slow_selection_mode = str(slowdown_cfg.get("selection_mode", "hash")).strip().lower()
    slow_hash_mod = max(1, int(slowdown_cfg.get("hash_mod", 3)))
    slow_hash_threshold = max(0, min(slow_hash_mod, int(slowdown_cfg.get("hash_threshold", 1))))
    is_slow_session = False
    if slow_enabled:
        if slow_selection_mode in {"first_session_once", "first_session_only", "single_session"}:
            global _single_slow_session_id
            with _single_slow_lock:
                if _single_slow_session_id is None:
                    _single_slow_session_id = session_id
                    if logger is not None:
                        logger.info(
                            "[SlimeLLM] Selected deterministic slow session: "
                            f"{session_id[:8]} (selection_mode=first_session_once)"
                        )
                is_slow_session = session_id == _single_slow_session_id
        else:
            is_slow_session = _stable_hash_mod(session_id, slow_hash_mod) < slow_hash_threshold
    slow_submit_extra_turns = max(0, int(slowdown_cfg.get("submit_extra_turns", 0)))
    slow_sleep_seconds = max(0, int(slowdown_cfg.get("sleep_seconds", 0)))
    slow_until_turn = max(0, int(slowdown_cfg.get("until_turn", 0)))

    submit_enabled = bool(submit_cfg.get("enabled", True))
    submit_base_turn = max(1, int(submit_cfg.get("base_turn", 2)))
    submit_spread = max(1, int(submit_cfg.get("spread", 4)))
    submit_turn = submit_base_turn + _stable_hash_mod(session_id, submit_spread)
    if is_slow_session:
        submit_turn += slow_submit_extra_turns

    file_path_prefix = str(program.get("file_path_prefix", "/testbed/slime_dummy_file")).rstrip()
    if not file_path_prefix:
        file_path_prefix = "/testbed/slime_dummy_file"
    path = f"{file_path_prefix}_{session_short}_{turn}.txt"

    context = {
        "session_id": session_id,
        "session_short": session_short,
        "instance_id": instance_id,
        "instance_short": instance_short,
        "turn": turn,
        "submit_turn": submit_turn,
        "is_slow": is_slow_session,
        "sleep_seconds": slow_sleep_seconds,
        "path": path,
    }

    if submit_enabled and turn >= submit_turn:
        finish_discussion = _format_template(
            str(templates.get("finish_discussion", DEFAULT_HARDCODED_PROGRAM["templates"]["finish_discussion"])),
            context,
        )
        finish_message = _format_template(
            str(submit_cfg.get("message", DEFAULT_HARDCODED_PROGRAM["submit"]["message"])),
            context,
            default=str(DEFAULT_HARDCODED_PROGRAM["submit"]["message"]),
        )
        task_completed = str(submit_cfg.get("task_completed", "true")).strip().lower()
        if task_completed not in {"true", "false", "partial"}:
            task_completed = "true"
        default_finish_message = (
            f"DISCUSSION\n{finish_discussion}\n\n"
            "<function=finish>\n"
            f"<parameter=message>{finish_message}</parameter>\n"
            f"<parameter=task_completed>{task_completed}</parameter>\n"
            "</function>"
        )
        finish_message_template = templates.get("finish_message_template")
        if isinstance(finish_message_template, str) and finish_message_template.strip():
            context["finish_message"] = finish_message
            context["task_completed"] = task_completed
            return _format_template(finish_message_template, context, default=default_finish_message)
        return default_finish_message

    if is_slow_session and slow_sleep_seconds > 0 and turn <= slow_until_turn:
        slow_discussion = _format_template(
            str(
                templates.get(
                    "slow_discussion",
                    "Hardcoded debug slow-path action for {instance_id} at turn {turn}.",
                )
            ),
            context,
        )
        default_slow_command = (
            f"sleep {slow_sleep_seconds} && cat > {path} <<'EOF'\n"
            f"session={session_id}\n"
            f"instance={instance_id}\n"
            f"turn={turn}\n"
            "action=slow_path\n"
            "EOF"
        )
        slow_command = _format_template(
            str(templates.get("slow_command", default_slow_command)),
            context,
            default=default_slow_command,
        )
        context["slow_command"] = slow_command
        state["last_create_path"] = path
        state["last_text"] = (
            f"session={session_id}\ninstance={instance_id}\nturn={turn}\naction=slow_path\n"
        )
        default_slow_message = (
            f"DISCUSSION\n{slow_discussion}\n\n"
            "<function=bash>\n"
            f"<parameter=command>{slow_command}</parameter>\n"
            "</function>"
        )
        slow_message_template = templates.get("slow_message_template")
        if isinstance(slow_message_template, str) and slow_message_template.strip():
            return _format_template(slow_message_template, context, default=default_slow_message)
        return default_slow_message

    action_cycle = program.get("action_cycle", ["create"])
    if not isinstance(action_cycle, list) or not action_cycle:
        action_cycle = ["create"]
    action = str(action_cycle[(turn - 1) % len(action_cycle)]).strip().lower()

    if action == "str_replace" and not state.get("last_create_path"):
        action = "create"

    if action == "str_replace":
        target_path = str(state.get("last_create_path"))
        old_text = str(state.get("last_text", "")).strip()
        if not old_text:
            old_text = _format_template(
                str(templates.get("create_file_text", DEFAULT_HARDCODED_PROGRAM["templates"]["create_file_text"])),
                context,
                default="action=create",
            )
        new_text = _format_template(
            str(templates.get("str_replace_new_text", DEFAULT_HARDCODED_PROGRAM["templates"]["str_replace_new_text"])),
            context,
            default=f"turn={turn}\\naction=str_replace\\n",
        )
        context["path"] = target_path
        context["old_text"] = old_text
        context["new_text"] = new_text
        str_replace_discussion = _format_template(
            str(templates.get("str_replace_discussion", DEFAULT_HARDCODED_PROGRAM["templates"]["str_replace_discussion"])),
            context,
        )
        state["last_text"] = new_text
        default_str_replace_message = (
            f"DISCUSSION\n{str_replace_discussion}\n\n"
            "<function=str_replace_editor>\n"
            "<parameter=command>str_replace</parameter>\n"
            f"<parameter=path>{target_path}</parameter>\n"
            f"<parameter=old_str>{old_text}</parameter>\n"
            f"<parameter=new_str>{new_text}</parameter>\n"
            "</function>"
        )
        str_replace_message_template = templates.get("str_replace_message_template")
        if isinstance(str_replace_message_template, str) and str_replace_message_template.strip():
            return _format_template(str_replace_message_template, context, default=default_str_replace_message)
        return default_str_replace_message

    create_file_text = _format_template(
        str(templates.get("create_file_text", DEFAULT_HARDCODED_PROGRAM["templates"]["create_file_text"])),
        context,
        default=f"session={session_id}\\nturn={turn}\\naction=create\\n",
    )
    context["create_file_text"] = create_file_text
    create_discussion = _format_template(
        str(templates.get("create_discussion", DEFAULT_HARDCODED_PROGRAM["templates"]["create_discussion"])),
        context,
    )
    state["last_create_path"] = path
    state["last_text"] = create_file_text
    default_create_message = (
        f"DISCUSSION\n{create_discussion}\n\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>create</parameter>\n"
        f"<parameter=path>{path}</parameter>\n"
        f"<parameter=file_text>{create_file_text}</parameter>\n"
        "</function>"
    )
    create_message_template = templates.get("create_message_template")
    if isinstance(create_message_template, str) and create_message_template.strip():
        return _format_template(create_message_template, context, default=default_create_message)
    return default_create_message


class HardcodedProgramRuntime:
    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._program: dict[str, Any] | None = None
        self._program_path: Path | None = None
        self._state: dict[str, Any] = {}

    def build_message(
        self,
        *,
        program_path: Path,
        session_id: str,
        instance_id: str,
        turn: int,
    ) -> str:
        if self._program is None or self._program_path != program_path:
            self._program = load_hardcoded_program(program_path)
            self._program_path = program_path
            self._logger.info(f"[SlimeLLM] Loaded hardcoded program from {program_path}")
        return build_hardcoded_program_message(
            self._program,
            session_id=session_id,
            instance_id=instance_id,
            turn=turn,
            state=self._state,
            logger=self._logger,
        )
