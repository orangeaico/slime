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

# Enable debug logging for swe-agent
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

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
        hardcoded_response_path = Path("/root/repo/slime/examples/swe_bench/hardcoded_response.txt")
        if hardcoded_response_path.exists():
            self.logger.info(f"[SlimeLLM] Using hardcoded response from {hardcoded_response_path}")
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


async def _async_env_start(env: SWEEnv):
    """Start SWEEnv in async context.

    This replicates env.start() but uses await instead of asyncio.run()
    to work within an async event loop.
    """
    # Step 1: Start deployment (boot Docker container)
    logger.info(f"[SWE-agent Init] Starting deployment...")
    env._chook.on_start_deployment()
    await env.deployment.start()
    logger.info(f"[SWE-agent Init] ✓ Deployment started")

    # Step 2: Create bash session
    logger.info(f"[SWE-agent Init] Creating bash session...")
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
    logger.info(f"[SWE-agent Init] Changing to root directory...")
    await _async_communicate(env, "cd /", check="raise")

    # Step 5: Handle repository setup
    if env.repo is not None:
        if isinstance(env.repo, PreExistingRepoConfig):
            logger.info(f"[SWE-agent Init] Repository '{env.repo.repo_name}' already exists in container")
        elif isinstance(env.repo, GithubRepoConfig):
            logger.info(f"[SWE-agent Init] Checking if repo {env.repo.repo_name} exists...")
            folders_output = await _async_communicate(env, "ls", check="raise")
            folders = folders_output.split("\n")

            if env.repo.repo_name not in folders:
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
        logger.info(f"[SWE-agent Init] Resetting repository to clean state...")
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
    logger.info(f"[SWE-agent Init] Running {len(env._post_startup_commands)} post-startup commands...")
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
    assert not args.partial_rollout, "Partial rollout is not supported for this function."

    state = GenerateState(args)

    # Extract repo information from metadata
    repo_name = sample.metadata.get("repo", "test/repo")
    base_commit = sample.metadata.get("base_commit", "HEAD")
    instance_id = sample.metadata.get("instance_id", "unknown")
    image_name = sample.metadata.get("image_name", None)
    problem_statement = sample.prompt

    logger.info(f"[Slime-SWE] Starting instance: {instance_id}")
    logger.info(f"[Slime-SWE] Repository: {repo_name}")
    logger.info(f"[Slime-SWE] Base commit: {base_commit}")

    # Initialize SWE environment
    env = None
    agent = None

    try:
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
        logger.info(f"[Slime-SWE] Creating SWEEnv...")
        env = SWEEnv.from_config(env_config)

        # Serialize Docker container startups to prevent concurrent health check deadlocks
        logger.info(f"[Slime-SWE] Waiting for Docker startup lock...")
        async with _docker_startup_lock:
            logger.info(f"[Slime-SWE] Lock acquired, starting environment...")
            await _async_env_start(env)
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

        logger.info(f"[Slime-SWE] Loaded agent config from YAML")

        # 5. Create custom model instance
        # We need to manually create the model since we're using a custom class
        model = SlimeLLMModel(config=agent_config.model, tools=agent_config.tools)
        model.sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        model.sglang_model_name = "/root/data/hf_models/Qwen3-1.7B"
        model.tokenizer = state.tokenizer

        logger.info(f"[Slime-SWE] Created SlimeLLMModel")

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

        logger.info(f"[Slime-SWE] Built system and instance prompts")

        # 8. Manual agent initialization (avoid asyncio.run() conflict)
        problem_stmt = TextProblemStatement(
            id=instance_id,
            text=problem_statement,
        )

        output_dir = Path("/root/repo/slime/outputs/swe_agent_trajectories") / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Slime-SWE] Setting up agent for background thread execution...")

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
            logger.info(f"[Slime-SWE] Installing agent tools (this adds bins to PATH)...")
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

            while turn_count < max_turns:
                try:
                    logger.info(f"[Slime-SWE] ========== Turn {turn_count + 1}/{max_turns} ==========")
                    logger.debug(f"[Slime-SWE] History length before step: {len(agent.history)} items")

                    step_output = agent.step()

                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Step completed")
                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Action: {step_output.action[:200] if step_output.action else 'None'}...")
                    logger.info(f"[Slime-SWE] Turn {turn_count + 1}: Observation: {len(step_output.observation)} chars")
                    logger.debug(f"[Slime-SWE] Turn {turn_count + 1}: Done flag: {step_output.done}")
                    logger.debug(f"[Slime-SWE] Turn {turn_count + 1}: Observation preview: {step_output.observation[:500]}...")

                    turn_count += 1

                    # Check if done
                    if step_output.done:
                        logger.info(f"[Slime-SWE] ✓ Agent signaled completion at turn {turn_count}")
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

            return turn_count

        # 8. Run agent loop in a separate thread to avoid event loop conflicts
        # This is the key: swe-agent expects to run in a sync context, so we give it one
        logger.info(f"[Slime-SWE] Running agent loop in background thread (swe-agent needs sync context)")

        max_turns = agent_config_dict.get("max_turns", 3)
        logger.info(f"[Slime-SWE] Configuration: max_turns={max_turns}, temperature={sampling_params.get('temperature', 0.7) if isinstance(sampling_params, dict) else 0.7}")

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_agent_loop_sync, system_prompt, instance_prompt, max_turns)
            turn_count = future.result()  # Wait for completion

        logger.info(f"[Slime-SWE] ✓ Agent loop completed in background thread ({turn_count} turns)")

        # 10. Extract trajectory data
        logger.info(f"[Slime-SWE] Extracting trajectory data...")
        traj_data = agent.get_trajectory_data()
        trajectory = traj_data["trajectory"]
        history = traj_data["history"]
        info = traj_data["info"]

        logger.info(f"[Slime-SWE] Trajectory: {len(trajectory)} steps")
        logger.info(f"[Slime-SWE] History: {len(history)} messages")
        logger.debug(f"[Slime-SWE] Info keys: {list(info.keys())}")

        # 11. Convert trajectory to Sample format for slime training
        logger.info(f"[Slime-SWE] Converting trajectory to training sample...")

        # Build full conversation text from history
        # We need to reconstruct the tokenized conversation with proper loss masking
        full_text = ""
        full_tokens = []
        loss_mask = []

        # Build initial messages (already created during agent setup)
        initial_messages = setup_messages

        logger.debug(f"[Slime-SWE] Tokenizing initial prompt with {len(initial_messages)} messages...")
        # Tokenize initial prompt
        prompt_text = state.tokenizer.apply_chat_template(
            initial_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompt_tokens = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        full_text = prompt_text
        full_tokens = prompt_tokens
        loss_mask = [0] * len(prompt_tokens)  # Don't train on prompt

        logger.debug(f"[Slime-SWE] Prompt: {len(prompt_tokens)} tokens")

        # Add each step's response and observation
        logger.debug(f"[Slime-SWE] Processing {len(trajectory)} trajectory steps...")
        for i, step in enumerate(trajectory):
            # Model response (train on this)
            response = step.get("response", "")
            if response:
                response_tokens = state.tokenizer(response, add_special_tokens=False)["input_ids"]
                full_text += response
                full_tokens += response_tokens
                loss_mask += [1] * len(response_tokens)
                logger.debug(f"[Slime-SWE] Step {i+1}: Added response ({len(response_tokens)} tokens, trainable)")

            # Observation (don't train on this)
            observation = step.get("observation", "")
            if observation:
                # Format observation using next_step_template
                next_step_template = agent_config.templates.next_step_template
                formatted_obs = next_step_template.replace("{{observation}}", observation)

                obs_tokens = state.tokenizer(formatted_obs, add_special_tokens=False)["input_ids"]
                full_text += formatted_obs
                full_tokens += obs_tokens
                loss_mask += [0] * len(obs_tokens)
                logger.debug(f"[Slime-SWE] Step {i+1}: Added observation ({len(obs_tokens)} tokens, masked)")

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
        sample.tokens = full_tokens
        sample.response_length = len(full_tokens) - len(prompt_tokens)
        sample.response = full_text[len(prompt_text):]
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
        logger.info(f"[Slime-SWE]   - Total tokens: {len(sample.tokens)}")
        logger.info(f"[Slime-SWE]   - Prompt tokens: {len(prompt_tokens)} (masked)")
        logger.info(f"[Slime-SWE]   - Response tokens: {sample.response_length}")
        logger.info(f"[Slime-SWE]   - Trainable tokens: {trainable_tokens}")
        logger.info(f"[Slime-SWE]   - Masked tokens: {masked_tokens}")
        logger.info(f"[Slime-SWE]   - Turns completed: {turn_count}")

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
