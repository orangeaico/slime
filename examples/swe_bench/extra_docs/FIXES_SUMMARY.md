# SWE-bench Docker Integration - Fixes Summary

## Issues Fixed

### 1. ✅ DeploymentNotStartedError
**Problem**: `await env.deployment.runtime.connect()` failed because deployment wasn't started
**Fix**: Call `await env.deployment.start()` before accessing runtime
**File**: `generate.py` - `_async_env_start()` function

### 2. ✅ asyncio.run() errors
**Problem**: SWE-agent uses `asyncio.run()` internally which fails in async context
**Fix**: Created async wrappers that use `await` instead
**Files**:
- `_async_env_start()` - Replicates `env.start()`
- `_async_communicate()` - Replicates `env.communicate()`

### 3. ✅ Docker CLI not available
**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'docker'`
**Fix**: Install Docker CLI in container: `apt-get install docker.io`
**Requirement**: Container must have Docker socket mounted: `-v /var/run/docker.sock:/var/run/docker.sock`

### 4. ✅ Wrong Docker images being used
**Problem**: Using generic `python:3.11` instead of pre-built SWE-bench images
**Fix**:
- Created `add_image_names.py` to add `image_name` field to train.jsonl
- Generate `train_with_images.jsonl` with correct image names
- Pattern: `instance_id` → `swebench/sweb.eval.x86_64.{repo}_1776_{repo}-{issue}`
- Example: `astropy__astropy-14995` → `swebench/sweb.eval.x86_64.astropy_1776_astropy-14995`

### 5. ✅ Trying to clone repositories unnecessarily
**Problem**: Using `GithubRepoConfig` to clone from GitHub, but SWE-bench images already have repos at `/testbed`
**Fix**: Use `PreExistingRepoConfig` when SWE-bench image is detected
**File**: `generate.py` - Check if `image_name.startswith("swebench/")`

### 6. ✅ Container startup timeouts
**Problem**: Containers taking 5+ minutes to start, hitting timeout
**Root Cause**:
- SWE-bench images don't have `swerex-remote` pre-installed
- Fallback command installs swe-rex via pipx (takes 10-20 seconds per container)
- Multiple containers starting simultaneously causes resource contention

**Fixes**:
- Increased `startup_timeout` from 180s to 300s
- Reduced `rollout-batch-size` from 2 to 1
- Reduced `n-samples-per-prompt` from 2 to 1
- Set `pull="never"` to skip image pulling check

## Key Files Modified

### `generate.py`
- Added `_async_env_start()` and `_async_communicate()` async wrappers
- Detect SWE-bench images and use `PreExistingRepoConfig`
- Configure `DockerDeploymentConfig` with correct image and timeouts

### `add_image_names.py` (NEW)
- Script to add `image_name` field to train.jsonl
- Converts instance_id to Docker image name format

### `run-qwen3-06B-opd.sh`
- Updated to use `train_with_images.jsonl`
- Reduced batch sizes to avoid resource contention
- Updated checkpoint path per user modification

## Docker Container Requirements

The slime container must be run with:
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \  # CRITICAL
  -v /home/himanshu/megatron_dir/:/root/repo/ \
  -v /home/shared/megatron_dir:/root/data/ \
  -v /home/himanshu/swe_livup:/root/swe_livup \
  -it slimerl/slime:latest /bin/bash
```

Then inside container:
```bash
apt-get update && apt-get install -y docker.io
```

## How SWE-bench Images Work

1. **Pre-built**: Each instance has its own Docker image (e.g., `swebench/sweb.eval.x86_64.astropy_1776_astropy-14995`)
2. **Repository included**: Repo already cloned at `/testbed` with correct base commit
3. **Python environment**: Has miniconda at `/opt/miniconda3/bin/python3`
4. **No swerex-remote**: Must be installed via pipx on first startup (takes ~15 seconds)
5. **Startup command**: `swerex-remote --auth-token TOKEN || (pip install pipx && pipx run swe-rex --auth-token TOKEN)`

## Current Status

✅ **Docker integration working**
- Containers boot successfully
- Async initialization complete
- Using correct pre-built images
- Repositories available at /testbed

⚠️ **Performance considerations**:
- Each container startup takes ~15-20 seconds (swe-rex installation)
- Recommend batch-size=1 to avoid simultaneous startups
- Total startup time: ~20-30 seconds per sample

## Next Steps

1. Run training with updated configuration
2. Monitor Docker containers: `docker ps`
3. Check logs for successful initialization: `✓ Environment started successfully!`
4. If memory issues occur, further reduce batch sizes or max-tokens-per-gpu

## Testing

To test Docker integration standalone:
```bash
python examples/swe_bench/test_docker_integration.py
```

To verify image conversion:
```python
from examples.swe_bench.add_image_names import instance_id_to_image_name
print(instance_id_to_image_name("astropy__astropy-14995"))
# Output: swebench/sweb.eval.x86_64.astropy_1776_astropy-14995
```
