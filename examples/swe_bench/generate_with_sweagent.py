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
import json
import os
import sys
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

# Global lock to serialize Docker container startups
# When multiple generate() calls run concurrently, Docker startups must be serialized
# to avoid resource contention and runtime startup deadlocks
_docker_startup_lock = asyncio.Lock()


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
        # HARDCODED RESPONSE FOR DEBUGGING
        # Check if hardcoded response file exists
        hardcoded_response_path = Path("/root/repo/slime/examples/swe_bench/hardcoded_response_none.txt")
        if hardcoded_response_path.exists():
            self.logger.info(f"[SlimeLLM] Using hardcoded response from {hardcoded_response_path}")

            # Check for abort before returning hardcoded response
            from slime.rollout.sglang_rollout import GenerateState
            state = GenerateState(self.args)
            if state.aborted:
                self.logger.info(f"[SlimeLLM] ⚠ Request aborted, raising AbortedException")
                raise AbortedException("LLM request aborted by server")

            with open(hardcoded_response_path, 'r') as f:
                hardcoded_message = f.read().strip()

            # Update stats
            self.stats.api_calls += 1
            self.logger.debug(f"[SlimeLLM] Total API calls: {self.stats.api_calls}")
            self.logger.debug(f"[SlimeLLM] Hardcoded response length: {len(hardcoded_message)} chars")

            return {"message": hardcoded_message}

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

    # Handle partial rollout: skip samples that were already completed/aborted in previous iteration
    # NOTE: Resuming partial SWE-agent samples is complex (requires Docker state restoration)
    # For now, we skip them and they'll be retried as fresh instances
    if args.partial_rollout and sample.response_length > 0:
        logger.info(
            f"[Slime-SWE] Skipping partial sample from previous rollout "
            f"(SWE-agent resumption not yet implemented): "
            f"{sample.metadata.get('instance_id', 'unknown')}"
        )
        # Return sample as-is (will be filtered out or retried fresh)
        sample.status = Sample.Status.ABORTED
        return sample

    # Extract repo information from metadata
    repo_name = sample.metadata.get("repo", "test/repo")
    base_commit = sample.metadata.get("base_commit", "HEAD")
    instance_id = sample.metadata.get("instance_id", "unknown")
    image_name = sample.metadata.get("image_name", None)
    problem_statement = sample.prompt

    logger.info(f"[Slime-SWE] Starting instance: {instance_id}")
    # logger.info(f"[Slime-SWE] Repository: {repo_name}")
    # logger.info(f"[Slime-SWE] Base commit: {base_commit}")

    # Initialize SWE environment
    env = None
    agent = None

    try:
        # Check for abort before starting expensive Docker setup
        if state.aborted:
            logger.info(f"[Slime-SWE] ⚠ Abort detected before Docker setup, skipping instance")
            sample.status = Sample.Status.ABORTED
            return sample

        # 1. Create repo config
        if image_name and image_name.startswith("swebench/"):
            # Pre-built SWE-bench images have the repo at /testbed
            repo_config = PreExistingRepoConfig(
                repo_name="testbed",
                base_commit=base_commit,
                reset=True,
            )
            working_dir = "/testbed"
            logger.info(f"[Slime-SWE] Using pre-existing repo in SWE-bench image")
        else:
            # Clone from GitHub
            if "/" in repo_name:
                github_url = f"https://github.com/{repo_name}"
            else:
                github_url = repo_name

            repo_config = GithubRepoConfig(
                github_url=github_url,
                base_commit=base_commit,
            )
            working_dir = f"/{repo_config.repo_name}"
            logger.info(f"[Slime-SWE] Will clone from GitHub")

        # 2. Create Docker deployment config
        if image_name:
            deployment_config = DockerDeploymentConfig(
                image=image_name,
                pull="never",
                startup_timeout=300.0,
                python_standalone_dir=None,
            )
            logger.info(f"[Slime-SWE] Using Docker image: {image_name}")
        else:
            deployment_config = DockerDeploymentConfig(
                image="python:3.11",
                startup_timeout=300.0,
            )
            logger.warning(f"[Slime-SWE] No image_name, using default python:3.11")

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

        # Serialize Docker container startups to prevent concurrent health check deadlocks
        logger.info(f"[Slime-SWE] Waiting for Docker startup lock...")
        async with _docker_startup_lock:
            # Check abort again after acquiring lock (might have been aborted while waiting)
            if state.aborted:
                logger.info(f"[Slime-SWE] ⚠ Abort detected after acquiring lock, skipping startup")
                sample.status = Sample.Status.ABORTED
                return sample

            logger.info(f"[Slime-SWE] Lock acquired, starting environment...")
            await _async_env_start(env, state)  # Pass state for abort checking
            logger.info(f"[Slime-SWE] ✓ Environment ready, releasing lock")

        # 4. Load SWE-agent configuration from YAML
        config_path = Path("/root/swe_livup/config/test_xml_v2.yaml")
        with open(config_path) as f:
            swe_config_yaml = yaml.safe_load(f)

        # Parse agent config
        agent_config_dict = swe_config_yaml.get("agent", {})

        # Override model config with sglang settings
        agent_config_dict["model"] = {
            "name": "sglang",
            "temperature": sampling_params.get("temperature", 0.7) if isinstance(sampling_params, dict) else 0.7,
            "top_p": sampling_params.get("top_p", 1.0) if isinstance(sampling_params, dict) else 1.0,
        }

        # Create DefaultAgentConfig from YAML
        agent_config = DefaultAgentConfig(**agent_config_dict)

        logger.debug(f"[Slime-SWE] Loaded agent config from YAML")

        # 5. Create custom model instance
        # We need to manually create the model since we're using a custom class
        model = SlimeLLMModel(config=agent_config.model, tools=agent_config.tools)
        model.args = args  # Store args for abort checking
        model.sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        model.sglang_model_name = "/root/data/hf_models/Qwen3-1.7B"
        model.tokenizer = state.tokenizer

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

        # 8. Manual agent initialization (avoid asyncio.run() conflict)
        problem_stmt = TextProblemStatement(
            id=instance_id,
            text=problem_statement,
        )

        output_dir = Path("/root/repo/slime/outputs/swe_agent_trajectories") / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[Slime-SWE] Setting up agent for background thread execution...")

        # Build initial messages for later use in tokenization
        setup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt}
        ]

        # Define the agent loop function that runs in a separate thread
        # This allows swe-agent to use asyncio.run() freely without conflicts
        def run_agent_loop_sync(system_prompt_arg: str, instance_prompt_arg: str, max_turns: int):
            """Run the agent loop in a synchronous context (separate thread).

            Args:
                system_prompt_arg: System prompt for the agent
                instance_prompt_arg: Instance-specific prompt
                max_turns: Maximum number of turns to run
            """
            # Set agent state (replaces agent.setup() which uses asyncio.run())
            agent._env = env
            agent._problem_statement = problem_stmt
            agent._output_dir = output_dir

            # Set trajectory path for save_trajectory()
            agent.traj_path = output_dir / f"{instance_id}.traj"
            logger.debug(f"[Slime-SWE] Trajectory will be saved to: {agent.traj_path}")

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

            logger.info(f"[Slime-SWE] ✓ Agent initialized in background thread, starting multi-turn loop")
            logger.info(f"[Slime-SWE] Max turns: {max_turns}")

            # Run agent loop
            turn_count = 0
            aborted = False

            while turn_count < max_turns:
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

        # 8. Run agent loop in a separate thread to avoid event loop conflicts
        # This is the key: swe-agent expects to run in a sync context, so we give it one
        logger.debug(f"[Slime-SWE] Running agent loop in background thread (swe-agent needs sync context)")

        max_turns = agent_config_dict.get("max_turns", 5)
        logger.info(f"[Slime-SWE] Configuration: max_turns={max_turns}, temperature={sampling_params.get('temperature', 0.7) if isinstance(sampling_params, dict) else 0.7}")

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_agent_loop_sync, system_prompt, instance_prompt, max_turns)
            turn_count, aborted = future.result()  # Wait for completion

        logger.info(f"[Slime-SWE] ✓ Agent loop completed in background thread ({turn_count} turns, aborted={aborted})")

        # Check if agent loop was aborted (following pattern from examples/search-r1/generate_with_search.py)
        if aborted:
            logger.info(f"[Slime-SWE] ⚠ Agent loop was aborted, marking sample as ABORTED")
            sample.status = Sample.Status.ABORTED
            return sample

        # 10. Extract trajectory data
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

        logger.info(f"[Slime-SWE] Total tokens: {len(full_tokens)}, Prompt tokens: {len(prompt_tokens)}, Response tokens: {len(full_tokens) - len(prompt_tokens)}, Loss mask length (response only): {len(loss_mask)}")

        # 12. Extract final patch
        logger.info(f"[Slime-SWE] Extracting final git diff patch...")
        try:
            patch = await _async_communicate(env, "git diff", timeout=10, check="ignore")
            sample.metadata["patch"] = patch
            logger.info(f"[Slime-SWE] ✓ Extracted patch: {len(patch)} chars")
        except Exception as e:
            logger.error(f"[Slime-SWE] ✗ Failed to extract patch: {e}")
            sample.metadata["patch"] = info.get("submission", "")

        # 13. Store in sample
        logger.info(f"[Slime-SWE] Preparing final sample...")

        # Decode prompt text from prompt tokens
        prompt_text = state.tokenizer.decode(prompt_tokens, skip_special_tokens=False)

        # Decode response text (everything after prompt)
        response_tokens = full_tokens[len(prompt_tokens):]
        response_text = state.tokenizer.decode(response_tokens, skip_special_tokens=False)

        sample.tokens = full_tokens
        sample.response_length = len(response_tokens)
        sample.response = response_text
        sample.loss_mask = loss_mask
        sample.prompt = prompt_text
        sample.status = Sample.Status.COMPLETED

        # Store trajectory in metadata for later analysis
        sample.metadata["trajectory"] = trajectory
        sample.metadata["info"] = info
        sample.metadata["turn_count"] = turn_count

        # Calculate training token statistics
        trainable_tokens = sum(loss_mask)
        masked_tokens = len(loss_mask) - trainable_tokens

        logger.info(f"[Slime-SWE] ✓ Sample prepared successfully:")
        logger.info(f"[Slime-SWE]   - Total tokens: {len(sample.tokens)}, Prompt tokens: {len(prompt_tokens)}, Response tokens: {sample.response_length}, Loss mask length (response only): {len(loss_mask)}, Trainable tokens: {trainable_tokens}, Masked tokens: {masked_tokens}, Turns: {turn_count}")

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
        # Always close environment if it was initialized
        if env is not None:
            try:
                logger.info(f"[Slime-SWE] Closing Docker environment...")
                await env.deployment.stop()
                logger.info(f"[Slime-SWE] ✓ Environment closed")
            except Exception as e:
                logger.error(f"[Slime-SWE] ✗ Error closing environment: {e}")
                logger.exception(e)
