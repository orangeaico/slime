# SWE-Agent Tools Installation Fix

## Problem

The `str_replace_editor` command was failing with "command not found" error in the Docker container.

### Root Cause

When manually initializing the SWE-agent (to avoid `asyncio.run()` conflicts), we were skipping the critical `agent.tools.install(env)` call that:

1. Uploads tool bundles to `/root/tools/` in the Docker container
2. **Adds tool bin directories to PATH**: `export PATH=/root/tools/{bundle_name}/bin:$PATH`
3. Makes bin scripts executable: `chmod +x /root/tools/{bundle_name}/bin/*`
4. Runs installation scripts: `source install.sh`

Without this call, commands like `str_replace_editor` weren't in the container's PATH, causing "command not found" errors.

## Solution

Added `agent.tools.install(env)` call in the `run_agent_loop_sync()` function (which runs in a separate thread with a clean synchronous context):

```python
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
```

## Key Files

- **Fix Location**: `/home/himanshu/megatron_dir/slime/examples/swe_bench/generate_with_sweagent.py:528-537`
- **Tools Install Method**: `/home/himanshu/swe_livup/sweagent/tools/tools.py:570-631`
- **Tool Config**: `/home/himanshu/swe_livup/tools/str_replace_openrouter/config.yaml`

## How It Works

1. **Tool Bundles** are defined in the SWE-agent config (`test_xml_v2.yaml`):
   ```yaml
   bundles:
     - path: tools/registry
     - path: tools/review_on_submit_m
     - path: tools/str_replace_openrouter  # Contains str_replace_editor
     - path: tools/think
     - path: tools/finish
   ```

2. **Installation Process** (`tools.install()`):
   - Uploads each bundle to `/root/tools/{bundle_name}/` in container
   - For each bundle, runs:
     ```bash
     export PATH=/root/tools/{bundle_name}/bin:$PATH
     chmod +x /root/tools/{bundle_name}/bin/*
     cd /root/tools/{bundle_name} && source install.sh  # if exists
     ```

3. **Result**: Commands like `str_replace_editor`, `bash`, `submit`, `think`, `finish` are now available in the container shell.

## Why This Wasn't Caught Earlier

- Normal SWE-agent flow calls `agent.setup()` which internally calls `tools.install()`
- We couldn't use `agent.setup()` because it uses `asyncio.run()` internally
- We manually initialized the agent state but forgot to install tools
- This is why the tools weren't available in the container

## Testing

Run the training script. The agent should now be able to execute `str_replace_editor` commands without "command not found" errors.

```bash
examples/swe_bench/run-qwen3-06B-opd.sh
```

Expected behavior:
- Tools install successfully with log message: `✓ Tools installed successfully`
- `str_replace_editor view /testbed` executes and returns directory listing
- Agent can now use all configured tools

## Related Issues Fixed

- ✅ Issue #13: RuntimeError "No event loop in thread" (fixed by running agent loop in separate thread)
- ✅ Issue #14: AssertionError in save_trajectory() (fixed by manually setting agent.traj_path)
- ✅ **Issue #15: str_replace_editor command not found (THIS FIX)**
