"""Multi-turn generate function for SWE-bench using SWE-agent environment.

Adapted from slime/examples/search-r1/generate_with_search.py
"""

import os
import re
import sys
from pathlib import Path

import yaml

# Set up SWE-agent environment variables before importing
os.environ.setdefault("SWE_AGENT_CONFIG_ROOT", "/root/swe_livup")
os.environ.setdefault("SWE_AGENT_CACHE_ROOT", "/tmp/swe_agent_cache")
os.environ.setdefault("SWE_AGENT_TRAJECTORY_DIR", "/tmp/swe_agent_trajectories")

# Create required directories
os.makedirs("/tmp/swe_agent_cache", exist_ok=True)
os.makedirs("/tmp/swe_agent_trajectories", exist_ok=True)

# Add swe_livup to Python path
sys.path.insert(0, "/root/swe_livup")

from sweagent.environment.repo import GithubRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

import logging
logger = logging.getLogger(__name__)


async def _async_env_start(env: SWEEnv):
    """Start SWEEnv in async context.

    This replicates env.start() (_init_deployment + reset + post_startup_commands)
    but uses await instead of asyncio.run() to work within an async event loop.
    """
    from swerex.runtime.abstract import CreateBashSessionRequest

    # ===== Replicate _init_deployment() =====
    # Step 1: Start deployment (boot Docker container)
    logger.info(f"[SWE-agent Init] Starting deployment (booting Docker container)...")
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
    # Note: env.set_env_variables() uses asyncio.run() internally, so we set them manually with await
    env_vars = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PIP_PROGRESS_BAR": "off",
        "PAGER": "cat"
    }
    for key, value in env_vars.items():
        await _async_communicate(env, f"export {key}={value}", check="ignore")
    logger.info(f"[SWE-agent Init] ✓ Environment variables set")

    # ===== Replicate reset() =====
    # Step 4: cd to root
    logger.info(f"[SWE-agent Init] Changing to root directory...")
    await _async_communicate(env, "cd /", check="raise")

    # Step 5: Copy/clone repository if needed (_copy_repo)
    if env.repo is not None:
        from sweagent.environment.repo import PreExistingRepoConfig, GithubRepoConfig

        # For PreExistingRepoConfig, the repo is already in the container - skip cloning
        if isinstance(env.repo, PreExistingRepoConfig):
            logger.info(f"[SWE-agent Init] Repository '{env.repo.repo_name}' already exists in container (PreExistingRepoConfig)")
        elif isinstance(env.repo, GithubRepoConfig):
            # For GithubRepoConfig, check if we need to clone
            logger.info(f"[SWE-agent Init] Checking if repo {env.repo.repo_name} exists...")
            folders_output = await _async_communicate(env, "ls", check="raise")
            folders = folders_output.split("\n")

            if env.repo.repo_name not in folders:
                logger.info(f"[SWE-agent Init] Cloning repository {env.repo.repo_name}...")
                env._chook.on_copy_repo_started(repo=env.repo)

                import os
                github_token = os.getenv("GITHUB_TOKEN", "")
                url = env.repo._get_url_with_token(github_token) if github_token else env.repo.github_url
                base_commit = env.repo.base_commit

                logger.info(f"[SWE-agent Init] Git clone {url} @ {base_commit}...")

                # Clone repository
                clone_commands = " && ".join([
                    f"git clone --filter=blob:none {url} /{env.repo.repo_name}",
                    f"cd /{env.repo.repo_name}",
                    f"git checkout {base_commit}",
                ])
                await _async_communicate(env, clone_commands, timeout=env.repo.clone_timeout, check="raise")
                logger.info(f"[SWE-agent Init] ✓ Repository cloned")
        else:
            logger.warning(f"[SWE-agent Init] Unknown repo type {type(env.repo)}, skipping clone")

        # Step 6: Reset repository to clean state (_reset_repository)
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

    # ===== Post-startup commands =====
    # Step 7: Run post-startup commands
    logger.info(f"[SWE-agent Init] Running {len(env._post_startup_commands)} post-startup commands...")
    for command in env._post_startup_commands:
        await _async_communicate(env, command, check="raise", timeout=env.post_startup_command_timeout)

    logger.info(f"[SWE-agent Init] ✓ Environment started successfully!")


async def _async_communicate(env: SWEEnv, command: str, timeout: int = 30, check: str = "ignore") -> str:
    """Execute command in SWEEnv using async context.

    This replicates env.communicate() but uses await instead of asyncio.run().

    Args:
        env: SWEEnv instance
        command: Bash command to execute
        timeout: Timeout in seconds
        check: "ignore" (default), "raise" (raise on error), or "silent"

    Returns:
        Command output as string
    """
    from swerex.runtime.abstract import BashAction

    # Map check parameter to swerex check parameter
    if check == "raise":
        rex_check = "raise"
    elif check == "silent":
        rex_check = "silent"
    else:  # "ignore"
        rex_check = "ignore"

    # Execute command using runtime
    # Note: When check="raise", the runtime automatically raises NonZeroExitCodeError on failure
    result = await env.deployment.runtime.run_in_session(
        BashAction(command=command, timeout=timeout, check=rex_check)
    )

    return result.output


# Load SWE-agent config
def load_swe_agent_config():
    """Load SWE-agent configuration from YAML file."""
    config_path = Path(__file__).parent / "swe_agent_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

# Load config once at module level
_SWE_AGENT_CONFIG = load_swe_agent_config()

# Configuration
SWE_CONFIGS = {
    "max_turns": 2,  # Maximum turns for multi-turn agent loop
    "command_timeout": 30,  # Timeout for bash commands
    # Extract templates from agent config
    "system_template": _SWE_AGENT_CONFIG["agent"]["templates"]["system_template"],
    "instance_template": _SWE_AGENT_CONFIG["agent"]["templates"]["instance_template"],
    "next_step_template": _SWE_AGENT_CONFIG["agent"]["templates"]["next_step_template"],
}


def parse_bash_command(response: str) -> tuple[str | None, bool]:
    """Parse bash command from model response.

    Returns:
        (command, done): Extracted bash command and whether agent is done
    """
    # Check if agent is done
    if "DONE" in response.upper():
        return None, True

    # Try to extract bash command in various formats
    # Format 1: ```bash\ncommand\n```
    bash_block = re.search(r"```bash\n(.*?)\n```", response, re.DOTALL)
    if bash_block:
        return bash_block.group(1).strip(), False

    # Format 2: ```\ncommand\n```
    code_block = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip(), False

    # Format 3: Just assume the response is a command (risky but fallback)
    # Only if it looks like a command (starts with common commands)
    if any(response.strip().startswith(cmd) for cmd in ["ls", "cd", "cat", "grep", "find", "git", "python", "echo", "mkdir", "rm", "mv", "cp", "sed", "awk"]):
        return response.strip(), False

    # No command found
    return None, False


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn agent loop for SWE-bench using SWE-agent environment.

    Args:
        args: Training arguments
        sample: Sample containing prompt and metadata
        sampling_params: Sampling parameters for generation

    Returns:
        Sample with response, tokens, and loss_mask filled in
    """
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize SWE environment from metadata
    env = None
    env_init_error = None

    # Extract repo information from metadata
    repo_name = sample.metadata.get("repo", "test/repo")
    base_commit = sample.metadata.get("base_commit", "HEAD")
    instance_id = sample.metadata.get("instance_id", "unknown")
    image_name = sample.metadata.get("image_name", None)

    logger.info(f"[SWE-agent Init] Starting environment for instance: {instance_id}")
    logger.info(f"[SWE-agent Init] Repository: {repo_name}")
    logger.info(f"[SWE-agent Init] Base commit: {base_commit}")
    logger.info(f"[SWE-agent Init] Docker image: {image_name}")

    try:
        # Create repo config based on whether we have a pre-built SWE-bench image
        if image_name and image_name.startswith("swebench/"):
            # Pre-built SWE-bench images have the repo already inside at /testbed
            # Use PreExistingRepoConfig instead of cloning from GitHub
            from sweagent.environment.repo import PreExistingRepoConfig

            repo_config = PreExistingRepoConfig(
                repo_name="testbed",
                base_commit=base_commit,
                reset=True,  # Reset to base commit on startup
            )
            logger.info(f"[SWE-agent Init] Using pre-existing repo in SWE-bench image: testbed @ {base_commit}")
        else:
            # No pre-built image - clone from GitHub
            if "/" in repo_name:
                github_url = f"https://github.com/{repo_name}"
            else:
                github_url = repo_name

            logger.info(f"[SWE-agent Init] GitHub URL: {github_url}")

            repo_config = GithubRepoConfig(
                github_url=github_url,
                base_commit=base_commit,
            )
            logger.info(f"[SWE-agent Init] Will clone from GitHub")

        logger.info(f"[SWE-agent Init] Created RepoConfig: {repo_config}")

        # Create Docker deployment config with the correct SWE-bench image
        from swerex.deployment.config import DockerDeploymentConfig

        if image_name:
            # Use the pre-built SWE-bench image from metadata
            # SWE-bench images already have everything installed, don't need standalone python
            deployment_config = DockerDeploymentConfig(
                image=image_name,
                pull="never",  # Don't pull - image already exists locally
                startup_timeout=300.0,  # Increased: 180->300s for swe-rex installation via pipx
                python_standalone_dir=None,  # Don't install standalone python - SWE-bench images have it
            )
            logger.info(f"[SWE-agent Init] Using pre-built Docker image: {image_name}")
        else:
            # Fallback to default python:3.11 if no image specified
            deployment_config = DockerDeploymentConfig()
            logger.warning(f"[SWE-agent Init] No image_name in metadata, using default python:3.11")

        env_config = EnvironmentConfig(
            repo=repo_config,
            deployment=deployment_config,
        )
        logger.info(f"[SWE-agent Init] Created EnvironmentConfig")

        logger.info(f"[SWE-agent Init] Calling SWEEnv.from_config()...")
        env = SWEEnv.from_config(env_config)
        logger.info(f"[SWE-agent Init] SWEEnv created: {env}")

        logger.info(f"[SWE-agent Init] Starting environment (will boot Docker container)...")
        # IMPORTANT: We're already in an async context, so we need to call
        # the async initialization manually instead of using env.start()
        # which calls asyncio.run() and fails
        await _async_env_start(env)
        logger.info(f"[SWE-agent Init] ✓ Environment started successfully!")

    except Exception as e:
        # If environment initialization fails, log detailed error but continue with simple generation
        env_init_error = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[SWE-agent Init] ✗ Failed to initialize environment: {env_init_error}")
        logger.exception(e)  # Log full traceback
        sample.metadata["error"] = env_init_error
        # Don't return - continue with simple generation instead

    # Build initial prompt using SWE-agent templates
    # Extract metadata for template formatting
    working_dir = sample.metadata.get("repo", "unknown_repo").split("/")[-1]
    problem_statement = sample.prompt

    # Format the system template (command docs would be filled by SWE-agent tools, we'll leave placeholder)
    system_prompt = SWE_CONFIGS["system_template"].replace("{{command_docs}}",
        "[Bash commands available - ls, cd, cat, grep, find, git, python, etc.]")

    # Format the instance template
    instance_prompt = SWE_CONFIGS["instance_template"].replace("{{working_dir}}", working_dir)
    instance_prompt = instance_prompt.replace("{{problem_statement}}", problem_statement)

    # Combine system and instance prompts
    prompt_text = f"{system_prompt}\n\n{instance_prompt}"

    # Tokenize prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Initialize response tracking
    response = ""
    response_token_ids = []
    loss_mask = []

    try:
        for turn_idx in range(SWE_CONFIGS["max_turns"]):
            # Generate model response
            payload = {
                "text": prompt_text + response,
                "sampling_params": sampling_params,
            }
            output = await post(url, payload)

            # Check for abort
            if output["meta_info"]["finish_reason"]["type"] == "abort":
                sample.status = Sample.Status.ABORTED
                return sample

            cur_response = output["text"]

            # Tokenize current response
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

            # Add to response
            response += cur_response
            response_token_ids += cur_response_token_ids
            loss_mask += [1] * len(cur_response_token_ids)  # Train on model outputs

            # Parse bash command from response
            command, done = parse_bash_command(cur_response)

            if done:
                # Agent finished
                break

            if command is None:
                # No valid command, add feedback and continue
                feedback = "\n[No valid bash command found. Please provide a bash command in ```bash``` blocks or respond with DONE when finished.]\n"
                feedback_tokens = state.tokenizer(feedback, add_special_tokens=False)["input_ids"]
                response += feedback
                response_token_ids += feedback_tokens
                loss_mask += [0] * len(feedback_tokens)  # Don't train on feedback
                continue

            # Execute command in SWE environment (if available)
            if env is not None:
                try:
                    logger.info(f"[SWE-agent Exec] Turn {turn_idx}: Executing command in Docker: {command}")
                    command_output = await _async_communicate(
                        env, command, timeout=SWE_CONFIGS["command_timeout"], check="ignore"
                    )
                    logger.info(f"[SWE-agent Exec] Turn {turn_idx}: Command completed. Output length: {len(command_output)} chars")
                    logger.debug(f"[SWE-agent Exec] Turn {turn_idx}: Output: {command_output[:500]}...")
                except Exception as e:
                    command_output = f"Error executing command: {str(e)}"
                    logger.error(f"[SWE-agent Exec] Turn {turn_idx}: Command failed: {e}")
            else:
                # No environment available, provide mock output
                command_output = f"[SWE environment not available: {env_init_error}]\nMocked command execution for: {command}"
                logger.warning(f"[SWE-agent Exec] Turn {turn_idx}: Using mock execution (env not initialized)")

            # Add command output as observation using next_step_template
            # Format observation using SWE-agent template
            observation = SWE_CONFIGS["next_step_template"].replace("{{observation}}", command_output)
            obs_tokens = state.tokenizer(observation, add_special_tokens=False)["input_ids"]
            response += observation
            response_token_ids += obs_tokens
            loss_mask += [0] * len(obs_tokens)  # Don't train on observations

            # Check if we hit max length
            if output["meta_info"]["finish_reason"]["type"] == "length":
                break

        # Get final patch from environment
        if env is not None:
            try:
                logger.info(f"[SWE-agent Cleanup] Extracting git diff patch...")
                # Get diff of all changes using async version
                patch = await _async_communicate(env, "git diff", timeout=10, check="ignore")
                sample.metadata["patch"] = patch
                logger.info(f"[SWE-agent Cleanup] Patch extracted: {len(patch)} chars")
            except Exception as e:
                logger.error(f"[SWE-agent Cleanup] Failed to extract patch: {e}")
                sample.metadata["patch"] = ""
        else:
            sample.metadata["patch"] = ""
            logger.debug(f"[SWE-agent Cleanup] No patch (env not initialized)")

    finally:
        # Always close environment if it was initialized
        if env is not None:
            logger.info(f"[SWE-agent Cleanup] Closing Docker environment...")
            # Close uses asyncio.run() internally, so we call it directly with await
            await env.deployment.stop()
            logger.info(f"[SWE-agent Cleanup] ✓ Environment closed")

    # Store in sample
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    # Set status based on finish reason
    finish_reason = output["meta_info"]["finish_reason"]["type"]
    if finish_reason == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_reason == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED

    return sample
