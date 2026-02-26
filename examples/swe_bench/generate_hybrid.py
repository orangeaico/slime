"""Hybrid approach: Use SWE-agent's Tools infrastructure with custom agent loop.

This combines:
- Async Docker setup and multi-turn loop (custom)
- SWE-agent's Tools for action parsing/execution (reused)
- SWE-agent's trajectory format for saving (reused)
"""

import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path

# Set up SWE-agent environment variables
os.environ.setdefault("SWE_AGENT_CONFIG_ROOT", "/root/swe_livup")
os.environ.setdefault("SWE_AGENT_CACHE_ROOT", "/tmp/swe_agent_cache")
os.environ.setdefault("SWE_AGENT_TRAJECTORY_DIR", "/tmp/swe_agent_trajectories")

# Create required directories
os.makedirs("/tmp/swe_agent_cache", exist_ok=True)
os.makedirs("/tmp/swe_agent_trajectories", exist_ok=True)

# Add swe_livup to Python path
sys.path.insert(0, "/root/swe_livup")

import yaml
from sweagent.environment.repo import GithubRepoConfig, PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.tools.utils import generate_command_docs
from sweagent.utils.log import get_logger
from swerex.deployment.config import DockerDeploymentConfig
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

logger = get_logger(__name__)


async def _async_env_start(env: SWEEnv):
    """Start SWEEnv in async context - replicates env.start() with await."""
    # Step 1: Start deployment
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
    await _async_communicate(env, "cd /", check="raise")

    # Step 5: Handle repository
    if env.repo is not None:
        if isinstance(env.repo, PreExistingRepoConfig):
            logger.info(f"[SWE-agent Init] Repository already exists in container")
        elif isinstance(env.repo, GithubRepoConfig):
            folders_output = await _async_communicate(env, "ls", check="raise")
            folders = folders_output.split("\n")

            if env.repo.repo_name not in folders:
                logger.info(f"[SWE-agent Init] Cloning repository...")
                env._chook.on_copy_repo_started(repo=env.repo)

                github_token = os.getenv("GITHUB_TOKEN", "")
                url = env.repo._get_url_with_token(github_token) if github_token else env.repo.github_url

                clone_commands = " && ".join([
                    f"git clone --filter=blob:none {url} /{env.repo.repo_name}",
                    f"cd /{env.repo.repo_name}",
                    f"git checkout {env.repo.base_commit}",
                ])
                await _async_communicate(env, clone_commands, timeout=env.repo.clone_timeout, check="raise")
                logger.info(f"[SWE-agent Init] ✓ Repository cloned")

        # Step 6: Reset repository
        logger.info(f"[SWE-agent Init] Resetting repository...")
        startup_commands = [
            f"cd /{env.repo.repo_name}",
            "export ROOT=$(pwd -P)",
            *env.repo.get_reset_commands(),
        ]
        await _async_communicate(env, " && ".join(startup_commands), check="raise", timeout=120)
        logger.info(f"[SWE-agent Init] ✓ Repository reset complete")

    # Step 7: Post-startup commands
    for command in env._post_startup_commands:
        await _async_communicate(env, command, check="raise", timeout=env.post_startup_command_timeout)

    logger.info(f"[SWE-agent Init] ✓ Environment started successfully!")


async def _async_communicate(env: SWEEnv, command: str, timeout: int = 30, check: str = "ignore") -> str:
    """Execute command in SWEEnv using async context."""
    rex_check = "raise" if check == "raise" else ("silent" if check == "silent" else "ignore")
    result = await env.deployment.runtime.run_in_session(
        BashAction(command=command, timeout=timeout, check=rex_check)
    )
    return result.output


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn agent loop using SWE-agent's Tools infrastructure.

    This is a hybrid approach that:
    - Uses custom async Docker setup and agent loop
    - Uses SWE-agent's Tools for parsing and execution
    - Saves trajectories in SWE-agent format
    """
    assert not args.partial_rollout, "Partial rollout not supported"

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
    model_name = "/root/data/hf_models/Qwen3-1.7B"

    # Extract metadata
    repo_name = sample.metadata.get("repo", "test/repo")
    base_commit = sample.metadata.get("base_commit", "HEAD")
    instance_id = sample.metadata.get("instance_id", "unknown")
    image_name = sample.metadata.get("image_name", None)
    problem_statement = sample.prompt

    logger.info(f"[Slime-SWE] Starting instance: {instance_id}")

    # Initialize environment
    env = None
    tools = None
    trajectory = []

    try:
        # 1. Setup Docker environment (async)
        if image_name and image_name.startswith("swebench/"):
            repo_config = PreExistingRepoConfig(repo_name="testbed", base_commit=base_commit, reset=True)
        else:
            github_url = f"https://github.com/{repo_name}" if "/" in repo_name else repo_name
            repo_config = GithubRepoConfig(github_url=github_url, base_commit=base_commit)

        deployment_config = DockerDeploymentConfig(
            image=image_name or "python:3.11",
            pull="never" if image_name else "missing",
            startup_timeout=300.0,
            python_standalone_dir=None,
        )

        env_config = EnvironmentConfig(repo=repo_config, deployment=deployment_config)
        env = SWEEnv.from_config(env_config)

        await _async_env_start(env)
        logger.info(f"[Slime-SWE] ✓ Environment ready")

        # 2. Setup Tools (SWE-agent infrastructure)
        config_path = Path("/root/swe_livup/config/test_xml_v2.yaml")
        with open(config_path) as f:
            swe_config = yaml.safe_load(f)

        # Create tool configuration from YAML
        tool_config_dict = swe_config.get("agent", {}).get("tools", {})
        tool_config = ToolConfig(**tool_config_dict)
        tools = ToolHandler(tools=tool_config)

        logger.info(f"[Slime-SWE] ✓ Tools loaded: {len(tools.config.commands)} commands")

        # 3. Build initial message history
        system_template = swe_config.get("agent", {}).get("templates", {}).get("system_template", "")
        instance_template = swe_config.get("agent", {}).get("templates", {}).get("instance_template", "")

        working_dir = "/testbed" if isinstance(repo_config, PreExistingRepoConfig) else f"/{repo_config.repo_name}"
        instance_prompt = instance_template.replace("{{working_dir}}", working_dir)
        instance_prompt = instance_prompt.replace("{{problem_statement}}", problem_statement)

        # Replace command docs placeholder
        # Generate command documentation from the loaded commands
        command_docs = generate_command_docs(
            commands=tools.config.commands,
            subroutine_types=[],  # No subroutines for now
        )
        system_prompt = system_template.replace("{{command_docs}}", command_docs)

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt}
        ]

        # For tokenization
        prompt_text = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Response tracking
        response = ""
        response_token_ids = []
        loss_mask = []

        # 4. Multi-turn agent loop
        max_turns = swe_config.get("agent", {}).get("max_turns", 20)
        for turn_idx in range(max_turns):
            # Query LLM
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": sampling_params.get("temperature", 0.7) if isinstance(sampling_params, dict) else 0.7,
                "max_tokens": sampling_params.get("max_new_tokens", 2048) if isinstance(sampling_params, dict) else 2048,
                "top_p": sampling_params.get("top_p", 1.0) if isinstance(sampling_params, dict) else 1.0,
            }

            logger.info(f"\n[Turn {turn_idx}] Querying LLM with {len(messages)} messages")
            llm_response = await post(url, payload)

            if "choices" not in llm_response or len(llm_response["choices"]) == 0:
                logger.error(f"Invalid LLM response: {llm_response}")
                break

            choice = llm_response["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            cur_response = choice["message"]["content"]

            logger.info(f"[Turn {turn_idx}] Model response ({len(cur_response)} chars)")

            # Add to messages
            messages.append({"role": "assistant", "content": cur_response})

            # Tokenize
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_mask += [1] * len(cur_response_token_ids)

            # Parse action using SWE-agent's Tools
            try:
                # Create a mock model response dict for parsing
                model_output = {"message": cur_response}
                if "tool_calls" in choice["message"]:
                    model_output["tool_calls"] = choice["message"]["tool_calls"]

                thought, action = tools.parse_actions(model_output)

                logger.info(f"[Turn {turn_idx}] Parsed action: {action[:100]}...")

                # Check for exit conditions
                if "submit" in action.lower() or "exit" in action.lower():
                    logger.info(f"[Turn {turn_idx}] Agent signaled completion")
                    # Execute submit if present
                    if "submit" not in action.lower():
                        action = "submit"

                # Execute action using env.communicate
                execution_start = time.time()
                observation = await _async_communicate(
                    env,
                    action,
                    timeout=tools.config.execution_timeout,
                    check="ignore"
                )
                execution_time = time.time() - execution_start

                logger.info(f"[Turn {turn_idx}] Execution complete ({execution_time:.2f}s, {len(observation)} chars)")

                # Add observation to messages
                next_step_template = swe_config.get("agent", {}).get("templates", {}).get("next_step_template", "{{observation}}")
                observation_formatted = next_step_template.replace("{{observation}}", observation)

                messages.append({"role": "user", "content": observation_formatted})

                # Tokenize observation
                obs_tokens = state.tokenizer(observation_formatted, add_special_tokens=False)["input_ids"]
                response += observation_formatted
                response_token_ids += obs_tokens
                loss_mask += [0] * len(obs_tokens)

                # Save trajectory step (as dict, TrajectoryStep is just a TypedDict)
                step = {
                    "action": action,
                    "observation": observation,
                    "response": cur_response,
                    "state": tools.get_state(env=env),
                    "thought": thought,
                    "execution_time": execution_time,
                    "query": messages[:-1],  # Messages before this step
                    "extra_info": {"exit_code": env._last_exit_code},
                }
                trajectory.append(step)

                # Check if submitted
                if "submit" in action.lower():
                    logger.info(f"[Turn {turn_idx}] Submission detected, ending loop")
                    break

            except Exception as e:
                logger.error(f"[Turn {turn_idx}] Error parsing/executing action: {e}")
                logger.exception(e)
                # Add error feedback
                error_msg = f"Error: {str(e)}"
                messages.append({"role": "user", "content": error_msg})
                obs_tokens = state.tokenizer(error_msg, add_special_tokens=False)["input_ids"]
                response += error_msg
                response_token_ids += obs_tokens
                loss_mask += [0] * len(obs_tokens)
                continue

            # Check max length
            if finish_reason == "length":
                logger.warning(f"[Turn {turn_idx}] Hit max length")
                break

        # 5. Extract final patch
        try:
            patch = await _async_communicate(env, "git diff", timeout=10, check="ignore")
            sample.metadata["patch"] = patch
            logger.info(f"[Slime-SWE] Patch extracted: {len(patch)} chars")
        except Exception as e:
            logger.error(f"Failed to extract patch: {e}")
            sample.metadata["patch"] = ""

        # 6. Save trajectory in SWE-agent format
        traj_dir = Path("/tmp/swe_agent_trajectories") / instance_id
        traj_dir.mkdir(parents=True, exist_ok=True)

        traj_file = traj_dir / f"{instance_id}.traj"
        traj_data = {"trajectory": trajectory, "info": sample.metadata}
        traj_file.write_text(json.dumps(traj_data, indent=2))

        logger.info(f"[Slime-SWE] Trajectory saved: {traj_file}")

    except Exception as e:
        logger.error(f"[Slime-SWE] Error: {e}")
        logger.exception(e)
        sample.metadata["error"] = str(e)
        sample.status = Sample.Status.ABORTED
        return sample

    finally:
        if env is not None:
            logger.info(f"[Slime-SWE] Closing environment...")
            await env.deployment.stop()
            logger.info(f"[Slime-SWE] ✓ Environment closed")

    # Store in sample
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.status = Sample.Status.COMPLETED

    return sample
