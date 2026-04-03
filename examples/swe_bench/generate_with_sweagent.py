"""Multi-turn generate function for SWE-bench using SWE-agent infrastructure.

This module properly integrates SWE-agent's DefaultAgent to leverage:
- Response parsing via ToolHandler
- Trajectory saving in SWE-agent format
- State management and error handling
- Automatic requerying on format errors

The integration uses a custom LLM model that queries sglang for on-policy distillation.
"""

import asyncio
import copy
import fcntl
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Set up SWE-agent environment variables
os.environ.setdefault("SWE_AGENT_CONFIG_ROOT", "/root/swe_livup")
os.environ.setdefault("SWE_AGENT_CACHE_ROOT", "/root/repo/slime/outputs/swe_agent_cache")
os.environ.setdefault("SWE_AGENT_TRAJECTORY_DIR", "/root/repo/slime/outputs/swe_agent_trajectories")

# Create required directories
os.makedirs("/root/repo/slime/outputs/swe_agent_cache", exist_ok=True)
os.makedirs("/root/repo/slime/outputs/swe_agent_trajectories", exist_ok=True)

# Add swe_livup to Python path
sys.path.insert(0, "/root/swe_livup")

import yaml
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.agent.models import AbstractModel, InstanceStats, ModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import GithubRepoConfig, PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.types import History
from sweagent.utils.log import get_logger
from swerex.deployment.config import DockerDeploymentConfig
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

from examples.swe_bench.hardcoded_programs.hardcoded_program_mode import HardcodedProgramRuntime
from examples.swe_bench.rollout_hooks import (
    is_abort_resumable_for_partial_rollout,
    mask_previous_response_tokens,
    should_attempt_partial_resume,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample


class AbortedException(Exception):
    """Exception raised when LLM request is aborted by the server."""

    pass


# Enable logging for swe-agent
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

_swerex_wait_patch_applied = False


def _read_pipe_nonblocking(pipe, max_bytes: int = 65536) -> str:
    """Best-effort non-blocking read from a subprocess pipe."""
    if pipe is None:
        return ""
    try:
        fd = pipe.fileno()
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        try:
            data = pipe.read(max_bytes)
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
    except Exception:
        return ""

    if not data:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return str(data)


def _apply_swerex_runtime_wait_patch() -> None:
    """Patch SWE-ReX async wait path to avoid event-loop blocking and timeout deadlocks."""
    global _swerex_wait_patch_applied
    if _swerex_wait_patch_applied:
        return

    try:
        from swerex.deployment import docker as swerex_docker_mod
        from swerex.utils import wait as swerex_wait_mod
    except Exception as e:
        logger.warning(f"[Slime-SWE] Failed to import SWE-ReX patch targets: {e}")
        return

    async def _async_wait_until_alive(function, timeout: float = 10.0, function_timeout: float | None = 0.1, sleep: float = 0.25):
        end_time = time.time() + timeout
        n_attempts = 0
        await_response = None
        while time.time() < end_time:
            await_response = await function(timeout=function_timeout)
            if await_response:
                return
            await asyncio.sleep(sleep)
            n_attempts += 1
        last_response_message = await_response.message if await_response else None
        msg = (
            f"Runtime did not start within {timeout}s (tried to connect {n_attempts} times). "
            f"The last await response was:\n{last_response_message}"
        )
        raise TimeoutError(msg)

    async def _patched_docker_wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _async_wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime_timeout)
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout. Container output (truncated).")
            stdout_text = _read_pipe_nonblocking(getattr(self._container_process, "stdout", None))
            stderr_text = _read_pipe_nonblocking(getattr(self._container_process, "stderr", None))
            if stdout_text:
                self.logger.error(stdout_text)
            if stderr_text:
                self.logger.error(stderr_text)
            assert self._container_process is not None
            await self.stop()
            raise e

    swerex_wait_mod._wait_until_alive = _async_wait_until_alive
    swerex_docker_mod._wait_until_alive = _async_wait_until_alive
    swerex_docker_mod.DockerDeployment._wait_until_alive = _patched_docker_wait_until_alive
    _swerex_wait_patch_applied = True
    logger.info("[Slime-SWE] Applied SWE-ReX async wait/deadlock patch")


_apply_swerex_runtime_wait_patch()

# Startup limiter for Docker container boots.
# We keep this configurable so rollout can run multiple starts in parallel
# while still allowing users to cap startup pressure on shared machines.
_docker_startup_semaphore = None
_docker_startup_concurrency = None

# Global registry for partial rollout Docker containers.
# Maps session_id -> {"env": SWEEnv, "agent": DefaultAgent, "instance_id": str, "rollout_id": int, "timestamp": float}
# Note: No lock needed - asyncio event loop serializes access automatically.
_partial_rollout_containers = {}


def _resolve_docker_startup_concurrency(args) -> int:
    """Resolve startup parallelism for SWE Docker environments."""
    raw_value = getattr(args, "swe_docker_startup_concurrency", 4)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 4
    return max(1, value)


def _get_docker_startup_semaphore(args) -> asyncio.Semaphore:
    """Get/create startup semaphore with current configured concurrency."""
    global _docker_startup_semaphore, _docker_startup_concurrency

    concurrency = _resolve_docker_startup_concurrency(args)
    if _docker_startup_semaphore is None or _docker_startup_concurrency != concurrency:
        _docker_startup_concurrency = concurrency
        _docker_startup_semaphore = asyncio.Semaphore(concurrency)
        logger.info(
            f"[Slime-SWE] Docker startup concurrency configured: {_docker_startup_concurrency}"
        )

    return _docker_startup_semaphore


def _resolve_docker_startup_timeout_seconds(args) -> float:
    """Resolve SWE Docker startup timeout in seconds."""
    raw_value = getattr(args, "swe_docker_startup_timeout_seconds", 900)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 900.0
    return max(30.0, value)


async def cleanup_partial_rollout_containers(max_age_hours: float = 1.0):
    """
    Clean up stale partial rollout containers.

    Containers older than max_age_hours are closed to prevent resource leaks.
    This should be called periodically (e.g., at the start of each rollout).

    Args:
        max_age_hours: Maximum age in hours before cleanup (default: 1 hour)
    """
    if not _partial_rollout_containers:
        return

    import time

    logger.info(f"[Cleanup] Checking {len(_partial_rollout_containers)} partial rollout containers...")

    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600)
    to_remove = []

    for session_id, info in _partial_rollout_containers.items():
        container_age_hours = (current_time - info["timestamp"]) / 3600
        instance_id = info.get("instance_id", "unknown")

        if info["timestamp"] < cutoff_time:
            logger.info(
                f"[Cleanup] Removing stale container: {instance_id} "
                f"(session: {session_id[:8]}, age: {container_age_hours:.1f}h > {max_age_hours}h)"
            )
            try:
                await info["env"].deployment.stop()
                to_remove.append(session_id)
            except Exception as e:
                logger.error(f"[Cleanup] Error closing {instance_id} (session: {session_id[:8]}): {e}")
                # Remove from registry anyway to prevent perpetual errors
                to_remove.append(session_id)

    for session_id in to_remove:
        del _partial_rollout_containers[session_id]

    if to_remove:
        logger.info(
            f"[Cleanup] Removed {len(to_remove)} stale containers. "
            f"Remaining: {len(_partial_rollout_containers)}"
        )
    else:
        logger.info(f"[Cleanup] No stale containers found")


async def cleanup_all_partial_rollout_containers():
    """
    Clean up ALL partial rollout containers immediately.

    Use this when shutting down or when you want to force cleanup of all containers.
    """
    if not _partial_rollout_containers:
        logger.info("[Cleanup] No partial rollout containers to clean up")
        return

    logger.info(f"[Cleanup] Cleaning up ALL {len(_partial_rollout_containers)} partial rollout containers...")

    for session_id, info in list(_partial_rollout_containers.items()):
        instance_id = info.get("instance_id", "unknown")
        try:
            await info["env"].deployment.stop()
            logger.debug(f"[Cleanup] Closed container: {instance_id} (session: {session_id[:8]})")
        except Exception as e:
            logger.error(f"[Cleanup] Error closing {instance_id} (session: {session_id[:8]}): {e}")

    # Clear the entire registry
    _partial_rollout_containers.clear()
    logger.info("[Cleanup] ✓ All partial rollout containers cleaned up")


class SlimeLLMModel(AbstractModel):
    """Custom LLM model that uses sglang chat completions API for slime training.

    This model integrates with SWE-agent's DefaultAgent while using sglang for
    LLM queries to enable on-policy distillation with slime.
    """

    def __init__(self, config: ModelConfig, tools):
        self.config = config
        self.tools = tools
        self.stats = InstanceStats()
        self.logger = get_logger("slime-llm-model")

        # sglang server configuration
        self.sglang_url = None  # Will be set from args
        self.sglang_model_name = None  # Will be set from args
        self.tokenizer = None  # Will be set from state
        self.sample_session_id: str | None = None
        self.sample_instance_id: str | None = None
        self._hardcoded_program_runtime = HardcodedProgramRuntime(logger=self.logger)

    def reset_stats(self):
        """Reset statistics for a new instance."""
        self.stats = InstanceStats()

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 since we don't track costs for sglang."""
        return 0

    def query(self, history: History, action_prompt: str = "> ") -> dict:
        """Query the LLM using sglang's chat completions API.

        Args:
            history: Conversation history (SWE-agent format)
            action_prompt: Action prompt (unused, kept for compatibility)

        Returns:
            dict with 'message' key containing the LLM response text
        """
        hardcoded_mode = str(getattr(self.args, "swe_hardcoded_response_mode", "none")).strip().lower()
        if hardcoded_mode == "program":
            from slime.rollout.sglang_rollout import GenerateState

            state = GenerateState(self.args)
            if state.aborted:
                self.logger.info(f"[SlimeLLM] ⚠ Request aborted, raising AbortedException")
                raise AbortedException("LLM request aborted by server")

            hardcoded_program_path = Path(
                getattr(
                    self.args,
                    "swe_hardcoded_program_path",
                    "/root/repo/slime/examples/swe_bench/hardcoded_programs/hardcoded_program.yaml",
                )
            )
            turn = self.stats.api_calls + 1
            session_id = self.sample_session_id or f"session_{id(self)}"
            instance_id = self.sample_instance_id or "unknown_instance"
            hardcoded_message = self._hardcoded_program_runtime.build_message(
                program_path=hardcoded_program_path,
                session_id=session_id,
                instance_id=instance_id,
                turn=turn,
            )

            self.stats.api_calls += 1
            self.logger.debug(f"[SlimeLLM] Total API calls: {self.stats.api_calls}")
            self.logger.debug(f"[SlimeLLM] Hardcoded response length: {len(hardcoded_message)} chars")
            return {"message": hardcoded_message}

        if hardcoded_mode not in {"", "none"}:
            raise ValueError(
                f"Unsupported swe_hardcoded_response_mode={hardcoded_mode!r}. "
                "Use one of: none, program."
            )

        # Normal flow if no hardcoded response
        # Convert history to messages format
        messages = self._history_to_messages(history)

        # Build payload for sglang chat completions API
        payload = {
            "model": self.sglang_model_name,
            "messages": messages,
            "temperature": getattr(self.config, "temperature", 0.7),
            "max_tokens": 4096,
            "top_p": getattr(self.config, "top_p", 1.0),
        }

        self.logger.debug(f"[SlimeLLM] Querying sglang with {len(messages)} messages")
        self.logger.debug(f"[SlimeLLM] Last message preview: {messages[-1]['content'][:200] if messages else 'None'}...")
        self.logger.debug(f"[SlimeLLM] Temperature: {payload['temperature']}, Top-p: {payload['top_p']}")

        # Make synchronous call to sglang (handle different event loop contexts)
        # Three cases to handle:
        # 1. Running in Ray's main thread with uvloop running → Need separate thread
        # 2. Running in Ray's main thread with no loop running → Use asyncio.run()
        # 3. Running in worker thread (agent loop) with no loop → Use asyncio.run()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Case 1: We're in Ray's main thread with uvloop running
                # Can't use asyncio.run() here, need to run in a separate thread
                self.logger.debug(f"[SlimeLLM] Running loop detected, using separate thread for async query")
                import concurrent.futures

                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._async_query(payload))
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_new_loop)
                    response = future.result()
            else:
                # Case 2: Loop exists but not running - use asyncio.run()
                self.logger.debug(f"[SlimeLLM] Event loop exists but not running, using asyncio.run()")
                response = asyncio.run(self._async_query(payload))
        except RuntimeError as e:
            # Case 3: No event loop in this thread (we're in the agent loop thread)
            # This is expected when running in ThreadPoolExecutor - just use asyncio.run()
            self.logger.debug(f"[SlimeLLM] No event loop in thread (expected in worker thread), using asyncio.run()")
            response = asyncio.run(self._async_query(payload))

        # Extract response from chat completion format
        if "choices" not in response or len(response["choices"]) == 0:
            self.logger.error(f"[SlimeLLM] ✗ Invalid response from sglang API: {response}")
            raise RuntimeError("Invalid response from sglang API")

        choice = response["choices"][0]
        message_content = choice["message"]["content"]

        self.logger.debug(f"[SlimeLLM] ✓ Received response: {len(message_content)} chars")
        self.logger.debug(f"[SlimeLLM] Response preview: {message_content[:300]}...")

        # Check for abort (following the pattern from examples/search-r1/generate_with_search.py)
        if "meta_info" in response and "finish_reason" in response["meta_info"]:
            finish_type = response["meta_info"]["finish_reason"].get("type")
            if finish_type == "abort":
                self.logger.info(f"[SlimeLLM] ⚠ Request was aborted by server")
                raise AbortedException("LLM request aborted by server")

        # Build output dict
        output_dict = {"message": message_content}

        # Handle tool calls if present (for function calling mode)
        if "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
            output_dict["tool_calls"] = choice["message"]["tool_calls"]
            self.logger.debug(f"[SlimeLLM] Tool calls present: {len(choice['message']['tool_calls'])}")

        # Update stats
        self.stats.api_calls += 1
        self.logger.debug(f"[SlimeLLM] Total API calls: {self.stats.api_calls}")

        return output_dict

    async def _async_query(self, payload: dict) -> dict:
        """Make async HTTP request to sglang."""
        return await post(self.sglang_url, payload)

    def _history_to_messages(self, history: History) -> list[dict[str, Any]]:
        """Convert SWE-agent History to OpenAI-compatible messages format.

        Args:
            history: SWE-agent history (list of HistoryItem dicts)

        Returns:
            List of message dicts with proper roles
        """
        history = copy.deepcopy(history)

        def get_role(history_item: dict) -> str:
            """Convert SWE-agent role to OpenAI role."""
            if history_item["role"] == "system":
                # Check if model config wants system messages converted to user
                convert = getattr(self.config, "convert_system_to_user", False)
                return "user" if convert else "system"
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)

            # Build message based on content type
            if role == "tool":
                # Tool response format
                message = {
                    "role": role,
                    "content": history_item["content"],
                    "tool_call_id": history_item["tool_call_ids"][0],
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                # Assistant message with tool calls
                message = {
                    "role": role,
                    "content": history_item["content"],
                    "tool_calls": tool_calls
                }
            else:
                # Standard message
                message = {
                    "role": role,
                    "content": history_item["content"]
                }

            # Add cache control if present (for prompt caching)
            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]

            messages.append(message)

        return messages


async def _async_env_start(env: SWEEnv, state: GenerateState = None):
    """Start SWEEnv in async context.

    This replicates env.start() but uses await instead of asyncio.run()
    to work within an async event loop.

    Args:
        env: SWEEnv instance
        state: GenerateState for abort checking (optional)
    """
    # Check for abort before starting deployment (slowest operation)
    if state and state.aborted:
        logger.info(f"[SWE-agent Init] ⚠ Abort detected, skipping deployment start")
        raise AbortedException("Aborted during Docker setup")

    # Step 1: Start deployment (boot Docker container)
    logger.info(f"[SWE-agent Init] Starting deployment...")
    env._chook.on_start_deployment()
    await env.deployment.start()
    logger.info(f"[SWE-agent Init] ✓ Deployment started")

    # Check for abort after deployment starts
    if state and state.aborted:
        logger.info(f"[SWE-agent Init] ⚠ Abort detected after deployment, skipping remaining setup")
        raise AbortedException("Aborted during Docker setup")

    # Step 2: Create bash session
    logger.debug(f"[SWE-agent Init] Creating bash session...")
    await env.deployment.runtime.create_session(
        CreateBashSessionRequest(startup_source=["/root/.bashrc"], startup_timeout=10)
    )
    logger.info(f"[SWE-agent Init] ✓ Bash session created")

    # Step 3: Set environment variables
    env_vars = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PIP_PROGRESS_BAR": "off",
        "PAGER": "cat"
    }
    for key, value in env_vars.items():
        await _async_communicate(env, f"export {key}={value}", check="ignore")
    logger.info(f"[SWE-agent Init] ✓ Environment variables set")

    # Step 4: cd to root
    logger.debug(f"[SWE-agent Init] Changing to root directory...")
    await _async_communicate(env, "cd /", check="raise")

    # Step 5: Handle repository setup
    if env.repo is not None:
        if isinstance(env.repo, PreExistingRepoConfig):
            logger.debug(f"[SWE-agent Init] Repository '{env.repo.repo_name}' already exists in container")
        elif isinstance(env.repo, GithubRepoConfig):
            logger.info(f"[SWE-agent Init] Checking if repo {env.repo.repo_name} exists...")
            folders_output = await _async_communicate(env, "ls", check="raise")
            folders = folders_output.split("\n")

            if env.repo.repo_name not in folders:
                # Check for abort before expensive git clone
                if state and state.aborted:
                    logger.info(f"[SWE-agent Init] ⚠ Abort detected before git clone, skipping")
                    raise AbortedException("Aborted during Docker setup")

                logger.info(f"[SWE-agent Init] Cloning repository {env.repo.repo_name}...")
                env._chook.on_copy_repo_started(repo=env.repo)

                github_token = os.getenv("GITHUB_TOKEN", "")
                url = env.repo._get_url_with_token(github_token) if github_token else env.repo.github_url
                base_commit = env.repo.base_commit

                clone_commands = " && ".join([
                    f"git clone --filter=blob:none {url} /{env.repo.repo_name}",
                    f"cd /{env.repo.repo_name}",
                    f"git checkout {base_commit}",
                ])
                await _async_communicate(env, clone_commands, timeout=env.repo.clone_timeout, check="raise")
                logger.info(f"[SWE-agent Init] ✓ Repository cloned")

        # Step 6: Reset repository to clean state
        logger.debug(f"[SWE-agent Init] Resetting repository to clean state...")
        startup_commands = [
            f"cd /{env.repo.repo_name}",
            "export ROOT=$(pwd -P)",
            *env.repo.get_reset_commands(),
        ]
        await _async_communicate(
            env,
            " && ".join(startup_commands),
            check="raise",
            timeout=120,
        )
        logger.info(f"[SWE-agent Init] ✓ Repository reset complete")

    # Step 7: Run post-startup commands
    logger.debug(f"[SWE-agent Init] Running {len(env._post_startup_commands)} post-startup commands...")
    for command in env._post_startup_commands:
        await _async_communicate(env, command, check="raise", timeout=env.post_startup_command_timeout)

    logger.info(f"[SWE-agent Init] ✓ Environment started successfully!")


async def _async_communicate(env: SWEEnv, command: str, timeout: int = 30, check: str = "ignore") -> str:
    """Execute command in SWEEnv using async context."""
    # Map check parameter to swerex format
    if check == "raise":
        rex_check = "raise"
    elif check == "silent":
        rex_check = "silent"
    else:
        rex_check = "ignore"

    # Execute command using runtime
    result = await env.deployment.runtime.run_in_session(
        BashAction(command=command, timeout=timeout, check=rex_check)
    )

    return result.output


async def _async_write_text_file(env: SWEEnv, path: str, content: str, timeout: int = 120) -> None:
    """Write a text file inside the container using a heredoc command."""
    payload = content if content.endswith("\n") else f"{content}\n"
    marker = f"SLIME_PATCH_{uuid.uuid4().hex}"
    command = (
        f"cat > {shlex.quote(path)} <<'{marker}'\n"
        f"{payload}"
        f"{marker}\n"
    )
    await _async_communicate(env, command, timeout=timeout, check="raise")


async def _apply_mirror_patches_if_needed(env: SWEEnv, sample: Sample) -> list[str]:
    """Apply optional mirror_patch/test_mirror_patch from sample metadata.

    Returns:
        List of patch labels that were successfully applied.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    mirror_patch = metadata.get("mirror_patch")
    test_mirror_patch = metadata.get("test_mirror_patch")

    patches: list[tuple[str, str | None]] = [
        ("mirror patch", mirror_patch),
        ("test mirror patch", test_mirror_patch),
    ]
    if not any(patch and str(patch).strip() for _, patch in patches):
        return []

    if not env.repo or not env.repo.repo_name:
        logger.warning("[Slime-SWE] No repo information available; skipping mirror patch application")
        return []

    repo_dir = f"/{env.repo.repo_name}"
    applied_labels: list[str] = []

    for patch_label, patch_content in patches:
        if not patch_content or not str(patch_content).strip():
            continue

        patch_text = str(patch_content)
        if not patch_text.endswith("\n"):
            patch_text += "\n"
        patch_path = f"/tmp/slime_{patch_label.replace(' ', '_')}_{uuid.uuid4().hex}.patch"

        logger.info(f"[Slime-SWE] Applying {patch_label} for {metadata.get('instance_id', 'unknown')}")
        try:
            await _async_write_text_file(env, patch_path, patch_text, timeout=120)
            apply_cmd = (
                f"cd {shlex.quote(repo_dir)} && "
                f"git apply -3 --whitespace=fix --recount {shlex.quote(patch_path)}"
            )
            await _async_communicate(env, apply_cmd, timeout=120, check="raise")
            applied_labels.append(patch_label)
            logger.info(f"[Slime-SWE] ✓ {patch_label} applied successfully")
        except Exception as e:
            logger.error(f"[Slime-SWE] Failed to apply {patch_label}: {e}")
            raise
        finally:
            await _async_communicate(env, f"rm -f {shlex.quote(patch_path)}", check="ignore")

    if not applied_labels:
        return []

    status_output = await _async_communicate(
        env,
        f"cd {shlex.quote(repo_dir)} && git status --porcelain",
        timeout=30,
        check="raise",
    )
    if status_output.strip():
        commit_cmd = (
            f"cd {shlex.quote(repo_dir)} && rm -rf .git && git init &&"
            "git config user.email 'sweagent@example.com' && "
            "git config user.name 'SWE Agent' && "
            "git add -A && "
            "git commit -m 'Apply mirror patch(es)'"
        )
        await _async_communicate(env, commit_cmd, timeout=60, check="raise")
        logger.info(f"[Slime-SWE] Committed mirror patch baseline")
    else:
        logger.warning(
            "[Slime-SWE] No repo changes detected after mirror patch application; "
            "patches may already be applied or have no effect"
        )

    return applied_labels


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn agent loop for SWE-bench using SWE-agent infrastructure.

    This function properly integrates with SWE-agent's DefaultAgent to leverage:
    - Response parsing via ToolHandler.parse_actions()
    - Trajectory saving in SWE-agent format
    - State management and error recovery
    - Automatic requerying on format/syntax errors

    Args:
        args: Training arguments containing sglang configuration
        sample: Sample containing prompt and metadata
        sampling_params: Sampling parameters for generation

    Returns:
        Sample with response, tokens, and loss_mask filled in
    """
    state = GenerateState(args)

    # Extract instance information
    instance_id = sample.metadata.get("instance_id", "unknown")
    repo_name = sample.metadata.get("repo", "test/repo")
    base_commit = sample.metadata.get("base_commit", "HEAD")
    image_name = sample.metadata.get("image_name", None)
    problem_statement = sample.prompt
    problem_stmt = TextProblemStatement(
        id=instance_id,
        text=problem_statement,
    )
    output_dir = Path("/root/repo/slime/outputs/swe_agent_trajectories") / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared agent config once so both fresh and resumed paths have turn budget.
    config_path = Path("/root/swe_livup/config/test_xml_v2.yaml")
    with open(config_path) as f:
        swe_config_yaml = yaml.safe_load(f)
    agent_config_dict = swe_config_yaml.get("agent", {})
    model_config_dict = agent_config_dict.get("model", {}) if isinstance(agent_config_dict, dict) else {}
    max_turns = int((
        model_config_dict.get("per_instance_call_limit")
        if isinstance(model_config_dict, dict)
        else None
    ))
    previous_turns_used = max(
        0,
        int(
            sample.metadata.get(
                "swe_turns_used",
                sample.metadata.get("total_turn_count", sample.metadata.get("turn_count", 0)),
            )
            or 0
        ),
    )
    prior_response_length = int(getattr(sample, "response_length", 0) or 0)

    # Generate unique session_id if not present (for container tracking)
    if sample.session_id is None:
        import uuid
        sample.session_id = str(uuid.uuid4())

    # Check if we can resume from a partial rollout container
    env = None
    agent = None
    resuming_partial = False
    env_ready_for_resume = False
    agent_ready_for_resume = False
    system_prompt = None
    instance_prompt = None

    if should_attempt_partial_resume(args, sample) and sample.session_id in _partial_rollout_containers:
        container_info = _partial_rollout_containers[sample.session_id]
        env = container_info["env"]
        agent = container_info["agent"]
        if hasattr(agent, "model"):
            agent.model.sample_session_id = sample.session_id
            agent.model.sample_instance_id = instance_id

        logger.info(
            f"[Slime-SWE] Resuming from partial rollout: {instance_id} "
            f"(session: {sample.session_id[:8]}, "
            f"previous rollout: {container_info['rollout_id']}, "
            f"existing response length: {sample.response_length}, "
            f"used_turns={previous_turns_used}/{max_turns})"
        )

        # Remove from registry - we'll re-add if it becomes partial again
        del _partial_rollout_containers[sample.session_id]

        # Verify container is still alive - if dead, discard this partial sample
        try:
            # Quick health check - try to run a simple command
            test_result = await _async_communicate(env, "echo test", timeout=5, check="ignore")
            if "test" not in test_result:
                raise RuntimeError("Container health check failed - unexpected output")

            resuming_partial = True
            env_ready_for_resume = True
            logger.info(f"[Slime-SWE] ✓ Container health check passed")
        except Exception as e:
            logger.warning(
                f"[Slime-SWE] Container dead for {instance_id}, discarding partial sample: {e}"
            )
            # Close dead container
            try:
                await env.deployment.stop()
            except:
                pass

            # Mark sample as aborted - do NOT retry
            sample.status = Sample.Status.ABORTED
            sample.metadata["discard_reason"] = f"Dead container: {e}"
            return sample
    else:
        logger.info(f"[Slime-SWE] Starting new instance: {instance_id}")

    try:
        # Check for abort before starting expensive Docker setup
        if state.aborted:
            logger.info(f"[Slime-SWE] ⚠ Abort detected before Docker setup, skipping instance")
            sample.status = Sample.Status.ABORTED
            return sample

        # If resuming partial, skip Docker setup (already have env and agent)
        if resuming_partial:
            logger.info(f"[Slime-SWE] Skipping Docker setup - using existing container")
            # Jump to agent loop execution (env and agent already set)
        else:
            # 1. Create repo config:
            # Pre-built images have the repo in /testbed
            repo_config = PreExistingRepoConfig(
                repo_name="testbed",
                base_commit=base_commit,
                reset=True,
            )
            working_dir = "/testbed"
            logger.info(f"[Slime-SWE] Using pre-existing repo in SWE-bench image")

            # 2. Create Docker deployment config
            startup_timeout_seconds = _resolve_docker_startup_timeout_seconds(args)
            if image_name:
                deployment_config = DockerDeploymentConfig(
                    image=image_name,
                    pull="never",
                    startup_timeout=startup_timeout_seconds,
                    python_standalone_dir=None,
                )
                logger.info(
                    f"[Slime-SWE] Using Docker image: {image_name} "
                    f"(startup_timeout={startup_timeout_seconds:.0f}s)"
                )
            else:
                deployment_config = DockerDeploymentConfig(
                    image="python:3.11",
                    startup_timeout=startup_timeout_seconds,
                )
                logger.warning(
                    f"[Slime-SWE] No image_name, using default python:3.11 "
                    f"(startup_timeout={startup_timeout_seconds:.0f}s)"
                )

            env_config = EnvironmentConfig(
                repo=repo_config,
                deployment=deployment_config,
            )

            # 3. Create SWEEnv
            logger.debug(f"[Slime-SWE] Creating SWEEnv...")
            env = SWEEnv.from_config(env_config)

            # Check for abort before expensive Docker startup
            if state.aborted:
                logger.info(f"[Slime-SWE] ⚠ Abort detected before Docker startup, skipping")
                sample.status = Sample.Status.ABORTED
                return sample

            startup_limiter = _get_docker_startup_semaphore(args)
            logger.info(
                f"[Slime-SWE] Waiting for Docker startup slot "
                f"(limit={_docker_startup_concurrency})..."
            )
            async with startup_limiter:
                # Check abort again after acquiring slot (might have been aborted while waiting)
                if state.aborted:
                    logger.info(f"[Slime-SWE] ⚠ Abort detected after acquiring startup slot, skipping startup")
                    sample.status = Sample.Status.ABORTED
                    return sample

                logger.info(
                    f"[Slime-SWE] Startup slot acquired for {instance_id} "
                    f"(session: {sample.session_id[:8]})"
                )
                await _async_env_start(env, state)  # Pass state for abort checking
                env_ready_for_resume = True
                logger.info(f"[Slime-SWE] ✓ Environment ready, releasing startup slot")

            # Apply mirror/test mirror patches from metadata before agent starts.
            applied_patch_labels = await _apply_mirror_patches_if_needed(env, sample)
            if applied_patch_labels and isinstance(sample.metadata, dict):
                sample.metadata["mirror_patches_applied"] = applied_patch_labels

            # 4. Build run-specific SWE-agent config
            agent_config_run_dict = copy.deepcopy(agent_config_dict)

            # Override model config with sglang settings
            agent_config_run_dict["model"] = {
                "name": "sglang",
                "temperature": sampling_params.get("temperature", 0.7) if isinstance(sampling_params, dict) else 0.7,
                "top_p": sampling_params.get("top_p", 1.0) if isinstance(sampling_params, dict) else 1.0,
            }

            # Create DefaultAgentConfig from YAML
            agent_config = DefaultAgentConfig(**agent_config_run_dict)

            logger.debug(f"[Slime-SWE] Loaded agent config from YAML")

            # 5. Create custom model instance
            # We need to manually create the model since we're using a custom class
            model = SlimeLLMModel(config=agent_config.model, tools=agent_config.tools)
            model.args = args  # Store args for abort checking
            model.sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
            model.sglang_model_name = "/root/data/hf_models/Qwen3-1.7B"
            model.tokenizer = state.tokenizer
            model.sample_session_id = sample.session_id
            model.sample_instance_id = instance_id

            logger.debug(f"[Slime-SWE] Created SlimeLLMModel")

            # 6. Create agent manually (bypass from_config to use our custom model)
            from sweagent.tools.tools import ToolHandler

            agent = DefaultAgent(
                templates=agent_config.templates,
                tools=ToolHandler(agent_config.tools),
                history_processors=agent_config.history_processors,
                model=model,
                max_requeries=agent_config.max_requeries,
                action_sampler_config=agent_config.action_sampler,
                interventions=agent_config.interventions,
            )

            logger.info(f"[Slime-SWE] Created DefaultAgent")

            # 7. Build initial prompts (before agent initialization)
            system_template = agent_config.templates.system_template
            instance_template = agent_config.templates.instance_template

            # Replace placeholders
            from sweagent.tools.utils import generate_command_docs
            command_docs = generate_command_docs(
                commands=agent.tools.config.commands,
                subroutine_types=[],
            )
            system_prompt = system_template.replace("{{command_docs}}", command_docs)
            instance_prompt = instance_template.replace("{{working_dir}}", working_dir)
            instance_prompt = instance_prompt.replace("{{problem_statement}}", problem_statement)

            logger.debug(f"[Slime-SWE] Built system and instance prompts")

            logger.debug(f"[Slime-SWE] Setting up agent for background thread execution...")

        # Define the agent loop function that runs in a separate thread
        # This allows swe-agent to use asyncio.run() freely without conflicts
        def run_agent_loop_sync(system_prompt_arg: str | None, instance_prompt_arg: str | None, max_turns: int):
            """Run the agent loop in a synchronous context (separate thread).

            Args:
                system_prompt_arg: System prompt for the agent
                instance_prompt_arg: Instance-specific prompt
                max_turns: Maximum number of turns to run
            """
            nonlocal agent_ready_for_resume

            # Set agent state (replaces agent.setup() which uses asyncio.run())
            agent._env = env
            agent._problem_statement = problem_stmt
            agent._output_dir = output_dir

            # Set trajectory path for save_trajectory()
            agent.traj_path = output_dir / f"{instance_id}.traj"
            logger.debug(f"[Slime-SWE] Trajectory will be saved to: {agent.traj_path}")

            if resuming_partial:
                agent_ready_for_resume = True
                logger.info(
                    f"[Slime-SWE] Continuing resumed agent state "
                    f"(history_len={len(agent.history) if hasattr(agent, 'history') else 'unknown'})"
                )
            else:
                # CRITICAL: Install tools to make commands like str_replace_editor available
                # This adds tool bin directories to PATH in the container
                logger.debug(f"[Slime-SWE] Installing agent tools (this adds bins to PATH)...")
                try:
                    agent.tools.install(env)
                    logger.info(f"[Slime-SWE] ✓ Tools installed successfully")
                except Exception as e:
                    logger.error(f"[Slime-SWE] ✗ Failed to install tools: {e}")
                    logger.exception(e)
                    raise

                if system_prompt_arg is None or instance_prompt_arg is None:
                    raise RuntimeError("Missing prompts for fresh agent initialization")

                # Initialize history with system and instance prompts as HistoryItem entries
                agent.history = [
                    {
                        "role": "system",
                        "content": system_prompt_arg,
                        "message_type": "observation",
                        "agent": agent.name,
                    },
                    {
                        "role": "user",
                        "content": instance_prompt_arg,
                        "message_type": "observation",
                        "agent": agent.name,
                    },
                ]
                # Reset model stats for this instance
                agent.model.reset_stats()
                agent_ready_for_resume = True
                logger.info(f"[Slime-SWE] ✓ Agent initialized in background thread, starting multi-turn loop")

            logger.info(f"[Slime-SWE] Max turns: {max_turns}")

            # Run agent loop
            turn_count = 0
            aborted = False

            while turn_count < max_turns:
                if state.aborted:
                    logger.info(
                        f"[Slime-SWE] ⚠ Abort flag detected before turn {turn_count + 1}, stopping agent loop"
                    )
                    aborted = True
                    break

                try:
                    logger.info(f"[Slime-SWE] ========== Turn {turn_count + 1}/{max_turns} ==========")
                    logger.debug(f"[Slime-SWE] History length before step: {len(agent.history)} items")

                    step_output = agent.step()

                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Step completed")
                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Action: {step_output.action[:200] if step_output.action else 'None'}...")
                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Observation: {len(step_output.observation)} chars")
                    logger.debug(f"[Slime-SWE] Turn {turn_count + 1}: Done flag: {step_output.done}")
                    # logger.debug(f"[Slime-SWE] Turn {turn_count + 1}: Observation preview: {step_output.observation[:500]}...")

                    turn_count += 1

                    if state.aborted:
                        logger.info(
                            f"[Slime-SWE] ⚠ Abort flag detected after turn {turn_count}, marking sample as partial"
                        )
                        aborted = True
                        break

                    # Check if done
                    if step_output.done:
                        logger.info(f"[Slime-SWE] ✓ Agent signaled completion at turn {turn_count}")
                        break

                except AbortedException as e:
                    # LLM request was aborted by server (time-bounded rollout)
                    logger.info(f"[Slime-SWE] ⚠ Agent loop aborted at turn {turn_count + 1}: {e}")
                    aborted = True
                    break

                except Exception as e:
                    if state.aborted:
                        logger.info(
                            f"[Slime-SWE] ⚠ Exception after abort signal at turn {turn_count + 1}: "
                            f"{type(e).__name__}: {e}"
                        )
                        aborted = True
                    else:
                        logger.error(f"[Slime-SWE] ✗ Error in turn {turn_count + 1}: {type(e).__name__}: {e}")
                        logger.exception(e)
                    break

            # Save trajectory
            logger.info(f"[Slime-SWE] Saving trajectory...")
            try:
                agent.save_trajectory()
                logger.info(f"[Slime-SWE] ✓ Trajectory saved to {agent.traj_path}")
            except Exception as e:
                logger.error(f"[Slime-SWE] ✗ Failed to save trajectory: {e}")
                logger.exception(e)
                # Continue anyway - we can still extract trajectory data later

            return turn_count, aborted

        # 8. Run agent loop in a separate thread without blocking the asyncio event loop
        # This is the key: swe-agent expects to run in a sync context, so we give it one
        logger.debug(f"[Slime-SWE] Running agent loop in background thread (swe-agent needs sync context)")

        remaining_turn_budget = max(0, max_turns - previous_turns_used)
        logger.info(
            f"[Slime-SWE] Configuration: total_max_turns={max_turns}, "
            f"previous_turns={previous_turns_used}, current_turn_budget={remaining_turn_budget}, "
            f"temperature={sampling_params.get('temperature', 0.7) if isinstance(sampling_params, dict) else 0.7}"
        )

        if remaining_turn_budget > 0:
            loop = asyncio.get_running_loop()
            turn_count, aborted = await loop.run_in_executor(
                None,
                run_agent_loop_sync,
                system_prompt,
                instance_prompt,
                remaining_turn_budget,
            )
        else:
            logger.info(
                f"[Slime-SWE] Turn budget already exhausted for {instance_id} "
                f"({previous_turns_used}/{max_turns}); skipping further agent turns."
            )
            turn_count, aborted = 0, False

        logger.info(f"[Slime-SWE] ✓ Agent loop completed in background thread ({turn_count} turns, aborted={aborted})")

        # 10. Extract trajectory data (also for aborted runs so partial state can be resumed)
        logger.info(f"[Slime-SWE] Extracting trajectory data...")
        traj_data = agent.get_trajectory_data()
        trajectory = traj_data["trajectory"]
        history = traj_data["history"]
        info = traj_data["info"]

        logger.info(f"[Slime-SWE] Trajectory: {len(trajectory)} steps")
        logger.info(f"[Slime-SWE] History: {len(history)} messages")
        logger.debug(f"[Slime-SWE] Info keys: {list(info.keys())}")
        logger.debug(f"[Slime-SWE] Traj data keys: {list(traj_data.keys())}")

        # 11. Convert trajectory to Sample format for slime training
        logger.info(f"[Slime-SWE] Converting trajectory to training sample...")

        # Use history from traj_data - it already contains the full conversation
        # with proper role/content formatting
        conversation_list = list(history)  # Make a copy

        # Ensure we only include messages up to the last assistant message
        # Find the last assistant message
        last_assistant_idx = -1
        for i in range(len(conversation_list) - 1, -1, -1):
            if conversation_list[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        # Truncate to last assistant message
        if last_assistant_idx >= 0:
            conversation_list = conversation_list[:last_assistant_idx + 1]
            logger.info(f"[Slime-SWE] Truncated conversation to {len(conversation_list)} messages (up to last assistant)")
        else:
            logger.warning(f"[Slime-SWE] No assistant messages found in history!")

        # Tokenize using the same approach as SFT dataset (_process_example function)
        # This ensures proper formatting with chat template tags (im_start, im_end, etc.)
        full_tokens = []
        full_loss_mask = []  # Covers entire sequence (for SFT format)

        logger.debug(f"[Slime-SWE] Tokenizing conversation message-by-message...")
        for i, msg in enumerate(conversation_list):
            # Use apply_chat_template for each message to get proper formatting
            seg_ids = state.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False
            )
            full_tokens.extend(seg_ids)

            # Track which tokens are trainable (assistant) vs masked (user/system)
            if msg["role"] == "assistant":
                full_loss_mask.extend([1] * len(seg_ids))
                logger.debug(f"[Slime-SWE] Message {i+1} ({msg['role']}): {len(seg_ids)} tokens (trainable)")
            else:
                full_loss_mask.extend([0] * len(seg_ids))
                logger.debug(f"[Slime-SWE] Message {i+1} ({msg['role']}): {len(seg_ids)} tokens (masked)")

        # Reconstruct full text for debugging/inspection
        full_text = state.tokenizer.decode(full_tokens, skip_special_tokens=False)

        # Calculate prompt tokens (everything before first assistant response)
        prompt_tokens = []
        for i, msg in enumerate(conversation_list):
            if msg["role"] == "assistant":
                break
            seg_ids = state.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_generation_prompt=False
            )
            prompt_tokens.extend(seg_ids)

        # Extract response-only loss mask (framework expects loss_mask to cover only response tokens)
        # This is different from SFT dataset format which covers the entire sequence
        response_start_idx = len(prompt_tokens)
        loss_mask = full_loss_mask[response_start_idx:]  # Only response tokens
        if resuming_partial and args.mask_offpolicy_in_partial_rollout and prior_response_length > 0:
            # Keep only newly generated tokens trainable after resume.
            # The prefix corresponds to tokens generated before this rollout step.
            loss_mask = mask_previous_response_tokens(loss_mask, prior_response_length)

        logger.info(f"[Slime-SWE] Total tokens: {len(full_tokens)}, Prompt tokens: {len(prompt_tokens)}, Response tokens: {len(full_tokens) - len(prompt_tokens)}, Loss mask length (response only): {len(loss_mask)}")

        # 12. Extract final patch
        logger.info(f"[Slime-SWE] Extracting final patch...")
        patch = info.get("submission", "") if isinstance(info, dict) else ""

        sample.metadata["patch"] = patch
        logger.info(
            f"[Slime-SWE] ✓ Extracted patch: {len(patch)} chars)"
        )

        # 13. Store in sample
        logger.info(f"[Slime-SWE] Preparing final sample...")

        # Decode prompt text from prompt tokens
        prompt_text = state.tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        # Decode response text (everything after prompt)
        response_tokens = full_tokens[len(prompt_tokens):]
        response_text = state.tokenizer.decode(response_tokens, skip_special_tokens=False)

        # If rollout-level abort was signaled while this sample was still in-flight,
        # keep non-submitted responses resumable instead of finalizing as completed.
        exit_status = str(info.get("exit_status", "")).strip().lower() if isinstance(info, dict) else ""
        rollout_abort_for_partial_resume = (
            state.aborted
            and not aborted
            and len(response_tokens) > 0
            and "submitted" not in exit_status
        )
        if rollout_abort_for_partial_resume:
            logger.info(
                f"[Slime-SWE] Rollout abort signal: converting in-flight non-submitted sample "
                f"to ABORTED for partial resume: {instance_id} (exit_status={exit_status!r})"
            )
            aborted = True

        sample.tokens = full_tokens
        sample.response_length = len(response_tokens)
        sample.response = response_text
        sample.loss_mask = loss_mask
        sample.prompt = prompt_text
        sample.status = Sample.Status.ABORTED if aborted else Sample.Status.COMPLETED

        # Store trajectory in metadata for later analysis
        sample.metadata["trajectory"] = trajectory
        sample.metadata["info"] = info
        total_turn_count = previous_turns_used + turn_count
        sample.metadata["turn_count"] = turn_count
        sample.metadata["total_turn_count"] = total_turn_count
        sample.metadata["swe_turns_used"] = total_turn_count
        sample.metadata["swe_max_turns"] = max_turns
        sample.metadata["swe_turn_budget_remaining"] = max(0, max_turns - total_turn_count)
        sample.metadata["swe_turn_budget_exhausted"] = total_turn_count >= max_turns
        sample.metadata["aborted_in_agent_loop"] = aborted
        sample.metadata["rollout_abort_for_partial_resume"] = rollout_abort_for_partial_resume
        sample.metadata["resume_ready"] = is_abort_resumable_for_partial_rollout(
            sample,
            exit_status=exit_status,
            total_turn_count=total_turn_count,
            max_turns=max_turns,
        )

        # Calculate training token statistics
        trainable_tokens = sum(loss_mask)
        masked_tokens = len(loss_mask) - trainable_tokens

        logger.info(f"[Slime-SWE] ✓ Sample prepared successfully:")
        logger.info(
            f"[Slime-SWE]   - Status: {sample.status.value}, "
            f"Total tokens: {len(sample.tokens)}, Prompt tokens: {len(prompt_tokens)}, "
            f"Response tokens: {sample.response_length}, Loss mask length (response only): {len(loss_mask)}, "
            f"Trainable tokens: {trainable_tokens}, Masked tokens: {masked_tokens}, Turns: {turn_count}"
        )

        # 14. Dump sample to outputs directory for inspection
        try:
            import json
            from datetime import datetime

            # Create sample dump directory
            dump_dir = Path("/root/repo/slime/outputs/swe_agent_samples")
            dump_dir.mkdir(parents=True, exist_ok=True)

            # Create filename from instance_id only
            instance_id = sample.metadata.get("instance_id", "unknown")
            safe_instance_id = instance_id.replace("/", "_").replace(":", "_")
            dump_file = dump_dir / f"{safe_instance_id}.json"

            # Prepare sample data for JSON serialization
            sample_data = {
                "metadata": {
                    "instance_id": instance_id,
                    "status": str(sample.status),
                    "turn_count": turn_count,
                    "total_turn_count": total_turn_count,
                    "swe_max_turns": max_turns,
                    "swe_turn_budget_remaining": max(0, max_turns - total_turn_count),
                    "swe_turn_budget_exhausted": total_turn_count >= max_turns,
                    "repo": sample.metadata.get("repo", ""),
                    "base_commit": sample.metadata.get("base_commit", ""),
                    "problem_statement": sample.metadata.get("problem_statement", ""),
                    "patch": sample.metadata.get("patch", ""),
                    "test_patch": sample.metadata.get("test_patch", ""),
                    "hints_text": sample.metadata.get("hints_text", ""),
                },
                "tokenization": {
                    "total_tokens": len(sample.tokens),
                    "prompt_tokens": len(prompt_tokens),
                    "response_tokens": sample.response_length,
                    "trainable_tokens": trainable_tokens,
                    "masked_tokens": masked_tokens,
                },
                "conversation_list": conversation_list,  # Full conversation in SFT format (truncated to last assistant)
                "full_history": history,  # Complete history from SWE-agent (before truncation)
                "formatted_conversation": "\n\n".join([
                    f"[{msg['role'].upper()}]\n{msg['content'][:500]}{'...' if len(msg['content']) > 500 else ''}"
                    for msg in conversation_list
                ]),  # Human-readable conversation
                "response": {
                    "text": sample.response,
                    "tokens_count": sample.response_length,
                },
                "trajectory": {
                    "steps": [
                        {
                            "step_num": i + 1,
                            "action": step.get("action", ""),
                            "response": step.get("response", ""),
                            "observation": step.get("observation", ""),
                            "response_length": len(step.get("response", "")),
                            "observation_length": len(step.get("observation", "")),
                        }
                        for i, step in enumerate(trajectory)
                    ],
                    "total_steps": len(trajectory),
                },
                "loss_mask": {
                    "response_only": {
                        "length": len(loss_mask),
                        "trainable_positions": sum(loss_mask),
                        "masked_positions": len(loss_mask) - sum(loss_mask),
                        "mask_preview": loss_mask[:100] if len(loss_mask) > 100 else loss_mask,
                    },
                    "full_sequence": {
                        "length": len(full_loss_mask),
                        "trainable_positions": sum(full_loss_mask),
                        "masked_positions": len(full_loss_mask) - sum(full_loss_mask),
                        "mask_preview": full_loss_mask[:100] if len(full_loss_mask) > 100 else full_loss_mask,
                    },
                    "note": "response_only is what's used for training (framework expects this), full_sequence is for SFT format reference"
                },
                "full_text_preview": {
                    "full_text": full_text,                    
                    "total_length": len(full_text),
                },
                "agent_info": {
                    "exit_status": info.get("exit_status", ""),
                    "model_stats": info.get("model_stats", {}),
                }
            }

            # Write to file with pretty formatting
            with open(dump_file, 'w') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[Slime-SWE] ✓ Sample dumped to {dump_file}")

        except Exception as e:
            logger.error(f"[Slime-SWE] ✗ Failed to dump sample: {e}")

        return sample

    except AbortedException as e:
        # Abort during Docker setup is expected behavior (time-bounded rollout)
        logger.info(f"[Slime-SWE] ⚠ Aborted during setup: {e}")
        sample.status = Sample.Status.ABORTED
        return sample

    except Exception as e:
        logger.error(f"[Slime-SWE] ✗ Fatal error: {type(e).__name__}: {e}")
        logger.exception(e)
        sample.metadata["error"] = str(e)
        sample.metadata["error_type"] = type(e).__name__
        sample.status = Sample.Status.ABORTED
        return sample

    finally:
        # Handle environment cleanup based on sample status
        if env is not None:
            info = sample.metadata.get("info") if isinstance(sample.metadata, dict) else None
            exit_status = str(info.get("exit_status", "")).strip().lower() if isinstance(info, dict) else ""

            # Check if this sample should be kept alive for partial rollout resumption
            should_keep_alive = (
                args.partial_rollout
                and sample.response
                and len(sample.response) > 0
                and env_ready_for_resume
                and agent is not None
                and agent_ready_for_resume
                and bool(sample.metadata.get("resume_ready", False))
            )

            if should_keep_alive:
                # Save environment and agent for next rollout
                import time

                sample.metadata["resume_ready"] = True
                logger.info(
                    f"[Slime-SWE] Keeping Docker container alive for partial rollout: "
                    f"{instance_id} (session: {sample.session_id[:8]}, "
                    f"response_length={sample.response_length}, status={sample.status.value}, "
                    f"exit_status={exit_status})"
                )

                # Store in global registry using session_id (unique per sample)
                _partial_rollout_containers[sample.session_id] = {
                    "env": env,
                    "agent": agent,
                    "instance_id": instance_id,  # For logging/debugging
                    "rollout_id": sample.metadata.get("start_rollout_id", -1),
                    "timestamp": time.time(),
                }

                logger.info(
                    f"[Slime-SWE] ✓ Container saved to registry (key: {sample.session_id[:8]}). "
                    f"Total active containers: {len(_partial_rollout_containers)}"
                )
            else:
                # Close environment for completed/failed samples
                try:
                    logger.info(f"[Slime-SWE] Closing Docker environment...")
                    await env.deployment.stop()
                    logger.info(f"[Slime-SWE] ✓ Environment closed")
                except Exception as e:
                    logger.error(f"[Slime-SWE] ✗ Error closing environment: {e}")
                    logger.exception(e)
