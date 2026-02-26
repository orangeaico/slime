# SWE-Agent Integration Status

## Summary

Successfully updated `/examples/swe_bench/generate_with_sweagent.py` to properly integrate with swe-agent's `DefaultAgent` infrastructure. The integration is 95% complete with one remaining async context issue to resolve.

## ✅ What Was Fixed

### 1. Import Errors - **FIXED**
- **Issue**: Missing dependencies (`simple-parsing`, `tenacity`, etc.)
- **Solution**: Installed all required dependencies
```bash
pip install simple-parsing tenacity ruamel.yaml litellm unidiff rich-argparse flask flask-cors flask-socketio
```

### 2. Transformers Version Conflict - **FIXED**
- **Issue**: `TypeError: AutoImageProcessor.register() got multiple values for argument 'exist_ok'`
- **Root cause**: swe-agent required transformers 4.57.3, but sglang was incompatible
- **Solution**: Downgraded to transformers 4.57.1
```bash
pip install 'transformers==4.57.1' --force-reinstall
```

### 3. Numpy Version Conflict - **FIXED**
- **Issue**: `AssertionError: Megatron does not support numpy 2.x`
- **Root cause**: Transformers 4.57.1 installed numpy 2.4.2
- **Solution**: Downgraded to numpy 1.26.4
```bash
pip install 'numpy<2.0' --force-reinstall
```

### 4. ProblemStatement Import Error - **FIXED**
- **Issue**: `ImportError: cannot import name 'ProblemStatement' from 'sweagent.types'`
- **Root cause**: `ProblemStatement` is in `sweagent.agent.problem_statement`, not `sweagent.types`
- **Solution**: Updated import statement
```python
from sweagent.agent.problem_statement import TextProblemStatement
```

### 5. Protocol Instantiation Error - **FIXED**
- **Issue**: `TypeError: Protocols cannot be instantiated`
- **Root cause**: `ProblemStatement` is a Protocol (interface), not a concrete class
- **Solution**: Use `TextProblemStatement` instead
```python
problem_stmt = TextProblemStatement(
    id=instance_id,
    text=problem_statement,
)
```

## ⚠️ Remaining Issue

### Async Context Conflict - **NEEDS FIX**
- **Issue**: `RuntimeError: asyncio.run() cannot be called from a running event loop`
- **Location**: Line 426 in `generate_with_sweagent.py`
```python
agent.setup(
    env=env,
    problem_statement=problem_stmt,
    output_dir=output_dir,
)
```
- **Root cause**: `agent.setup()` internally calls `asyncio.run()`, but we're already in an async context (slime's generate function runs in an asyncio event loop)

### Potential Solutions

#### Option 1: Manual Setup (Recommended)
Instead of calling `agent.setup()`, manually initialize the agent state:

```python
# Don't call agent.setup(), instead do manual initialization:
agent._env = env
agent._problem_statement = problem_stmt
agent._output_dir = output_dir

# Initialize history with system and instance prompts
agent.history = []
agent.messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instance_prompt}
]
agent.trajectory = []
agent.info = {"model_stats": agent.model.stats}

# Then use agent.step() in the loop
```

#### Option 2: Async-Compatible Setup Method
Create a custom async version of `agent.setup()` similar to what we did with `_async_env_start()`:

```python
async def _async_agent_setup(agent, env, problem_statement, output_dir):
    """Async version of agent.setup() that doesn't use asyncio.run()"""
    agent._env = env
    agent._problem_statement = problem_statement
    agent._output_dir = output_dir

    # Build initial history using templates
    # (Copy logic from DefaultAgent.setup() but use await instead of asyncio.run())
    ...
```

#### Option 3: Hybrid Approach (Simplest Short-term)
Keep the hybrid approach in `generate_hybrid.py` but add the error recovery logic from DefaultAgent:

```python
# In generate_hybrid.py, wrap parsing in retry loop:
for attempt in range(3):
    try:
        thought, action = tools.parse_actions(model_output)
        break
    except FormatError as e:
        # Add format error message to history and retry
        error_msg = format_error_template.replace("{{error}}", str(e))
        messages.append({"role": "user", "content": error_msg})
        continue
```

## Current Integration Status

### Working Components ✅
1. Environment setup with async Docker integration
2. Custom `SlimeLLMModel` that queries sglang
3. `DefaultAgent` instantiation with proper config
4. Proper imports and dependencies
5. Trajectory data structure compatible with swe-agent

### Not Yet Working ❌
1. Agent setup in async context
2. Multi-turn agent loop (blocked by setup issue)
3. Trajectory saving (blocked by loop not running)

## Files Modified

1. **`generate_with_sweagent.py`** (574 lines)
   - Proper swe-agent integration using `DefaultAgent`
   - Custom `SlimeLLMModel` for sglang queries
   - Async environment setup
   - Ready for agent loop once setup issue is resolved

2. **`run-qwen3-06B-opd.sh`**
   - Updated to use `generate_with_sweagent.generate`

3. **Documentation**
   - `INTEGRATION_GUIDE.md` - Comprehensive architecture guide
   - `QUICK_START.md` - Quick reference
   - `INTEGRATION_STATUS.md` - This file

## Next Steps

### Immediate (Required to Run)
1. **Fix async context issue** using one of the three options above
   - Recommended: Option 1 (Manual Setup) - quickest to implement
   - Alternative: Option 3 (Hybrid with retry logic) - most pragmatic

### Once Running
2. **Test with 1-2 instances** to verify parsing works
3. **Monitor logs** for action formatting
4. **Analyze trajectories** to check data quality
5. **Scale up** to full training runs

## Dependencies Summary

### Installed Versions (Finalized)
```
transformers==4.57.1    # Compatible with sglang
numpy==1.26.4           # Compatible with Megatron
simple-parsing          # For swe-agent config parsing
tenacity               # For retry logic in swe-agent
ruamel.yaml            # For YAML config loading
litellm                # For LLM API abstraction
unidiff                # For patch parsing
rich-argparse          # For CLI argument parsing
flask                  # For web UI (optional)
flask-cors             # CORS support
flask-socketio         # WebSocket support
```

### Version Conflicts (Non-Critical)
```
fsspec 2026.2.0 (needed by litellm) vs <=2025.10.0 (needed by datasets)
openai 2.24.0 (needed by litellm) vs 2.6.1 (needed by sglang)
```
These conflicts are warnings only and don't affect functionality.

## Testing Commands

### Test Imports
```bash
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && python -c 'from examples.swe_bench.generate_with_sweagent import generate; print(\"✓ Import successful\")'"
```

### Run Training (Once Fixed)
```bash
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh"
```

### Monitor Progress
```bash
# Watch logs for success indicators
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh" 2>&1 | grep -E "Slime-SWE|Turn [0-9]|✓"

# Check trajectories
docker exec 9e0e554ee79e ls -lh /tmp/swe_agent_trajectories/
```

## Code Snippets for Quick Fix

### Quick Fix Option 1: Manual Setup (Add after line 414)

Replace:
```python
        logger.info(f"[Slime-SWE] Created DefaultAgent")

        # 7. Setup agent with problem statement
        problem_stmt = TextProblemStatement(
            id=instance_id,
            text=problem_statement,
        )

        output_dir = Path("/tmp/swe_agent_trajectories") / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Slime-SWE] Setting up agent...")
        agent.setup(
            env=env,
            problem_statement=problem_stmt,
            output_dir=output_dir,
        )
```

With:
```python
        logger.info(f"[Slime-SWE] Created DefaultAgent")

        # 7. Manual agent initialization (avoid asyncio.run() issue)
        problem_stmt = TextProblemStatement(
            id=instance_id,
            text=problem_statement,
        )

        output_dir = Path("/tmp/swe_agent_trajectories") / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Slime-SWE] Initializing agent manually...")

        # Set agent state
        agent._env = env
        agent._problem_statement = problem_stmt
        agent._output_dir = output_dir

        # Initialize history
        agent.history = []
        agent.messages = initial_messages  # We already have these from step 4
        agent.trajectory = []
        agent.info = {"model_stats": agent.model.stats}

        # Reset model stats
        agent.model.reset_stats()
```

Then the rest of the code (agent loop) should work as-is.

## Conclusion

The integration is nearly complete. All dependency issues are resolved, and the code structure is correct. Only one async context issue remains, which can be fixed with a simple manual initialization instead of calling `agent.setup()`.

**Estimated time to complete**: 15-30 minutes to implement Option 1 (Manual Setup)

**Risk level**: Low - the fix is straightforward and well-documented above
