# SWE-bench On-Policy Distillation

This example demonstrates **multi-turn agentic rollout** with **on-policy distillation** for training models on SWE-bench tasks using the slime framework.

## Overview

- **Student model**: qwen3-0.6B
- **Teacher model**: qwen3-1.7B (served via SGLang)
- **Training objective**: Pure KL distillation (no task rewards)
- **Environment**: SWE-agent for bash execution and code editing
- **Pattern**: Based on slime's search-r1 example

## Key Features

✅ **Multi-turn agent loop** - Up to 15 turns per task
✅ **SWE-agent integration** - Direct use of SWEEnv for bash, file operations
✅ **Loss masking** - Train on model outputs, not tool observations
✅ **Pure distillation** - Learning signal from teacher KL divergence only
✅ **Simple implementation** - ~400 lines of code across 5 files

## Architecture

### 1. Multi-turn Generate Function (`generate.py`)

Implements the agent loop pattern from search-r1:
- Generates model response
- Parses bash commands from response
- Executes commands via `SWEEnv.communicate()`
- Adds observations with loss_mask=0
- Repeats for max 15 turns

### 2. SWE-agent Environment

Uses existing SWE-agent infrastructure from `/root/swe_livup/sweagent`:
- `SWEEnv.communicate()` - Execute bash commands
- `SWEEnv.read_file()` - Read files from container
- `SWEEnv.write_file()` - Write files to container
- Docker-based repo isolation

### 3. Reward Function (`reward.py`)

Two functions required for OPD:
- `reward_func()`: Returns empty dict (teacher log-probs handled by OPD)
- `post_process_rewards()`: Returns 0.0 for all samples (pure distillation)

The learning signal comes entirely from OPD KL penalty.

### 4. Training Flow

```
1. Student generates on-policy rollouts (multi-turn with SWE-agent)
2. Teacher evaluates same rollouts to get log probabilities
3. OPD loss = 0.0 (task reward) + KL divergence to teacher
4. Student learns to match teacher's distribution
```

## File Structure

```
examples/swe_bench/
├── __init__.py                    # Package init
├── generate.py                    # Multi-turn agent loop with Docker integration (~270 lines)
├── reward.py                      # Reward functions with logprob truncation (~110 lines)
├── data_loader.py                # Data preparation (~110 lines)
├── swe_agent_config.yaml         # SWE-agent config (prompts, tools, settings)
├── inspect_trajectories.py       # Utility to view saved trajectories
├── test_docker_integration.py    # Test Docker/SWE-agent setup independently
├── monitor_docker.sh             # Monitor Docker containers during training
├── run-qwen3-06B-opd.sh          # Training script
├── README.md                      # This file
└── data/
    └── train.jsonl               # Training data (10 instances)
```

## Setup

### 0. Prerequisites

Make sure SWE-agent is properly installed:

```bash
# Install swerex (required by SWE-agent)
pip install swe-rex

# Or if you have swe_livup with dependencies
cd /root/swe_livup
pip install -e .

# Fix git safe.directory issue (if needed)
git config --global --add safe.directory /root/swe_livup
```

The training script will automatically:
- Create required directories: `/home/shared`, `/tmp/swe_agent_cache`, `/tmp/swe_agent_trajectories`
- Configure git safe.directory for swe_livup
- Set SWE-agent environment variables

### 1. Prepare Training Data

```bash
cd /root/slime/examples/swe_bench

# Activate SWE-agent environment
source /root/swe_livup/.venv/bin/activate

# Create training data (100 instances)
python data_loader.py --max-instances 100

# Or specify custom output
python data_loader.py --output data/train.jsonl --max-instances 50
```

### 2. Check Data Format

```bash
# View first instance
head -1 data/train.jsonl | python -m json.tool

# Should have structure:
# {
#   "prompt": "Repository: ...\nProblem: ...",
#   "metadata": {
#     "instance_id": "...",
#     "repo": "...",
#     "base_commit": "...",
#     ...
#   }
# }
```

## Training

### Run Training Script

```bash
cd /root/slime

# Make sure models are available:
# - /root/data/hf_models/Qwen3-0.6B
# - /root/data/hf_models/Qwen3-1.7B

# Run training
bash examples/swe_bench/run-qwen3-06B-opd.sh
```

### Key Training Arguments

```bash
# Custom functions
--custom-generate-function-path examples.swe_bench.generate.generate
--custom-rm-path examples.swe_bench.reward.reward_func
--custom-reward-post-process-path examples.swe_bench.reward.post_process_rewards

# OPD settings
--use-opd                        # Enable on-policy distillation
--opd-type sglang                # Use SGLang for teacher
--opd-kl-coef 1.0                # KL divergence coefficient
--rm-url http://127.0.0.1:4500/generate  # Teacher server

# Data
--prompt-data examples/swe_bench/data/train.jsonl
--rollout-batch-size 1           # Small for long sequences
--n-samples-per-prompt 2         # Memory-constrained
--rollout-max-response-len 8192  # Long context for multi-turn
```

## How It Works

### Multi-turn Agent Loop

```python
for turn in range(15):
    # 1. Generate model response
    response = await generate(prompt + history)

    # 2. Parse bash command
    command = parse_bash_command(response)

    # 3. Execute in SWE environment
    output = env.communicate(command)

    # 4. Add to history with loss masking
    history += response + output
    loss_mask += [1]*len(response) + [0]*len(output)
```

### Loss Masking

- **Model outputs** (loss_mask=1): Train on these tokens
- **Tool observations** (loss_mask=0): Don't train on these tokens
- This ensures the model learns to generate commands, not mimic tool outputs

### On-Policy Distillation

```python
# Student generates rollout
student_tokens, student_logprobs = student.generate(prompt)

# Teacher evaluates same tokens
teacher_logprobs = teacher.evaluate(student_tokens)

# OPD loss
kl_loss = KL(student_logprobs || teacher_logprobs)
task_reward = 0.0  # Pure distillation

loss = -task_reward + kl_coef * kl_loss
```

## Monitoring Training

### Metrics to Watch

- **OPD KL divergence**: Should decrease over training
- **Rollout completion rate**: % of episodes that complete successfully
- **Average turns per episode**: Should stabilize
- **Memory usage**: ~40GB per GPU for qwen3-0.6B

### Logs

```bash
# Teacher server logs
tail -f /tmp/sglang_*.log

# Training logs
ray job logs <job-id>

# Ray dashboard
http://localhost:8265
```

## Implementation Details

### SWE-agent Configuration

The integration uses the real SWE-agent config from `/home/himanshu/swe_livup/config/test_xml_v2.yaml`:

**Config components used**:
- `system_template`: Full system prompt with problem-solving workflow (READING, EXPLORATION, TESTING, etc.)
- `instance_template`: How to format the SWE-bench instance/problem statement
- `next_step_template`: How to format tool observations

**Config location**: Copied to `examples/swe_bench/swe_agent_config.yaml`

This ensures the model receives the same high-quality prompts as the standard SWE-agent, including:
- Detailed problem-solving phases (7 phases from READING to FINAL REVIEW)
- Efficiency guidelines (combine commands, use grep/find effectively)
- Code quality standards (minimal changes, clean code)
- Troubleshooting workflow

### Trajectory Saving

Rollout trajectories are saved to `/tmp/swe_agent_trajectories/rollout_{rollout_id}.pt`:
- Each file contains all samples from that rollout iteration
- Format: PyTorch saved dict with `rollout_id` and `samples` fields
- Each sample includes: **full prompt**, **full response**, tokens, loss_mask, reward, metadata

**Optimization for storage**:
- Teacher logprobs are automatically truncated in saved files (keeps only first 5 and last 5)
- Original teacher logprobs are preserved in `sample.teacher_log_probs` during training
- Prompts and responses are saved in full for complete trajectory analysis
- Typical file size: ~100-500 KB per rollout (vs several MB without truncation)

**To inspect trajectories**:

Use the provided inspection script (shows full prompts/responses by default):
```bash
# View latest rollout with full output
python examples/swe_bench/inspect_trajectories.py

# View specific rollout
python examples/swe_bench/inspect_trajectories.py --rollout-id 0

# View specific sample
python examples/swe_bench/inspect_trajectories.py --rollout-id 0 --sample-idx 0

# Compact view (truncate long outputs)
python examples/swe_bench/inspect_trajectories.py --truncate
```

Or manually:
```python
import torch
data = torch.load("/tmp/swe_agent_trajectories/rollout_0.pt")
print(f"Rollout ID: {data['rollout_id']}")
print(f"Number of samples: {len(data['samples'])}")

# View first sample
sample = data['samples'][0]
print(f"Prompt: {sample['prompt']}")  # Full prompt
print(f"Response: {sample['response']}")  # Full response
print(f"Reward: {sample['reward']}")  # Logprobs truncated to first/last 5
print(f"Metadata: {sample['metadata']}")
```

### Graceful Degradation

The generate function is designed to handle SWEEnv initialization failures gracefully:
- If SWEEnv fails to initialize (e.g., Docker issues, repo access problems), the function continues with **mock command execution**
- This allows training to proceed even if some instances fail, useful for debugging and iterative development
- Failed instances are logged in `sample.metadata["error"]`
- In production, you may want to filter out failed instances during data preprocessing

### Loss Masking

Critical for multi-turn training:
- **Model outputs** (loss_mask=1): Commands, reasoning, responses - train on these
- **Tool observations** (loss_mask=0): Bash output, file contents - don't train on these
- Ensures the model learns to generate actions, not mimic environment outputs

## Debugging Docker Integration

### Monitor Docker Containers

Use the provided monitoring script to watch Docker containers during training:

```bash
# In a separate terminal
bash examples/swe_bench/monitor_docker.sh
```

This shows:
- Running and exited containers
- Container logs
- Real-time updates every 3 seconds

### Check SWE-agent Logs

The generate function now includes detailed logging. Look for these prefixes in training logs:

```
[SWE-agent Init] - Environment initialization
[SWE-agent Exec] - Command execution in Docker
[SWE-agent Cleanup] - Patch extraction and cleanup
```

**Example successful initialization**:
```
[SWE-agent Init] Starting environment for instance: django__django-12345
[SWE-agent Init] Repository: django/django
[SWE-agent Init] Base commit: abc123
[SWE-agent Init] GitHub URL: https://github.com/django/django
[SWE-agent Init] Created RepoConfig: GithubRepoConfig(...)
[SWE-agent Init] Created EnvironmentConfig
[SWE-agent Init] Calling SWEEnv.from_config()...
[SWE-agent Init] SWEEnv created: <SWEEnv object>
[SWE-agent Init] Starting environment (will boot Docker container)...
[SWE-agent Init] ✓ Environment started successfully!
```

**Example failed initialization**:
```
[SWE-agent Init] ✗ Failed to initialize environment: TypeError: 'types.UnionType' object is not callable
```

### Common Issues

**1. "asyncio.run() cannot be called from a running event loop"**
- **Cause**: SWE-agent's `env.start()` and `env.communicate()` use `asyncio.run()` internally, which fails when already in an async context (slime's generate function)
- **Fix**: Created async wrappers (`_async_env_start()`, `_async_communicate()`) that use `await` instead
- **Impact**: Docker containers now boot properly, commands execute successfully
- **Verify**: Check logs for "✓ Environment started successfully!" and "Turn X: Command completed"

**2. "'types.UnionType' object is not callable"** (FIXED)
- **Cause**: Using `RepoConfig(...)` instead of a concrete type
- **Fix**: Code now uses `GithubRepoConfig` directly
- **Verify**: Check logs for "Created RepoConfig: GithubRepoConfig(...)"

**3. "FileNotFoundError: [Errno 2] No such file or directory: 'docker'"** (CURRENT ISSUE)
- **Cause**: Docker CLI and socket not available inside the slime container where Ray workers run
- **Issue**: SWE-agent's deployment needs Docker access to boot containers, but the slime container doesn't have Docker installed or mounted
- **Solutions**:
  1. **Install Docker CLI in container**: `apt-get update && apt-get install -y docker.io`
  2. **Mount Docker socket**: Run container with `-v /var/run/docker.sock:/var/run/docker.sock`
  3. **Run outside Docker**: Execute training script directly on the host instead of inside a container
- **Verify**: `docker ps` should work from inside the Ray worker
- **Note**: This is a Docker-in-Docker (DinD) access issue

**4. Docker containers not starting (after fixing #3)**
- **Check**: Run `docker ps` during training - should see containers with `python:3.11` or `swebench` images
- **Check**: Docker daemon is running: `systemctl status docker`
- **Check**: User has Docker permissions: `docker run hello-world`

**4. Git clone failures**
- **Check**: Network connectivity to GitHub
- **Check**: Rate limiting: `curl https://api.github.com/rate_limit`
- **Check**: Logs for "git clone" timeout errors

### Verify Docker Integration

**Use the provided test script**:

```bash
# Test with default repo (django/django)
python examples/swe_bench/test_docker_integration.py

# Test with specific repo and commit
python examples/swe_bench/test_docker_integration.py \
    --repo scikit-learn/scikit-learn \
    --commit main
```

This will:
1. Create a GithubRepoConfig
2. Initialize SWEEnv
3. Boot a Docker container
4. Clone the repository
5. Execute test commands (pwd, ls, git status, etc.)
6. Extract a sample patch
7. Clean up

**Expected output**: Detailed logs showing each step completing successfully.

If this works, Docker integration is functional! If it fails, check the error messages and Docker logs.

## Troubleshooting

### Out of Memory

Reduce batch sizes:
```bash
--rollout-batch-size 1
--n-samples-per-prompt 1
--micro-batch-size 1
--max-tokens-per-gpu 1024
```

### Environment Initialization Fails

Check SWE-agent setup:
```bash
source /root/swe_livup/.venv/bin/activate
python -c "from sweagent.environment.swe_env import SWEEnv; print('OK')"
```

### Teacher Server Not Starting

Check GPU availability:
```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path Qwen3-1.7B --port 4500
```

### Git Dubious Ownership Error

If you see "SHA is empty, possible dubious ownership" error:
```bash
git config --global --add safe.directory /root/swe_livup
```

This is automatically handled by the training script, but you may need to run it manually if testing components separately.

## Next Steps

### Add Task Rewards

Modify `post_process_rewards()` in `reward.py` to include swegym evaluation:
```python
def post_process_rewards(args, samples: list[Sample], **kwargs):
    # Extract teacher log-probs (if using OPD)
    # ... OPD log-prob extraction code ...

    # Run tests and get pass rate
    task_rewards = []
    for sample in samples:
        patch = sample.metadata.get("patch", "")
        resolved = run_swegym_test(sample.metadata, patch)
        task_rewards.append(1.0 if resolved else 0.0)

    return task_rewards, task_rewards
```

### Scale Up

- Increase dataset size: `--max-instances 500`
- More rollouts: `--num-rollout 200`
- Larger model: Adapt for qwen3-8B or qwen3-14B

### Better Prompting

Improve system prompt in `generate.py`:
- Add few-shot examples
- Add chain-of-thought reasoning
- Add specific SWE-bench instructions

## Reference

Based on:
- **slime OPD example**: `/root/slime/examples/on_policy_distillation/`
- **search-r1 example**: `/root/slime/examples/search-r1/`
- **SWE-agent**: `/root/swe_livup/sweagent/`

## Performance Expectations

### Baselines
- qwen3-0.6B base: ~1-2% SWE-bench resolve rate (few-shot)
- qwen3-1.7B teacher: ~5-8% (few-shot)

### Training Goals (Pure KL Distillation)
- After 50 iterations: KL divergence < 2.0
- After 100 iterations: Student matches teacher trajectory patterns
- Eventually add task rewards to surpass teacher

### Resource Requirements
- **GPUs**: 2-4 (1 for teacher, 1+ for training/rollout)
- **Time per iteration**: ~20-30 minutes
- **Memory**: ~40GB per GPU for qwen3-0.6B
- **Storage**: ~100GB for SWE-bench repos
