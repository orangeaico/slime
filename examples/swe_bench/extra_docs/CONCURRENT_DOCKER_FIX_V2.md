# Concurrent Docker Container Fix V2 (The Real Fix)

## Problem

When `rollout-batch-size` was set to 2 or more, both Docker containers would get stuck and make no progress, even after fixing the event loop blocking issue.

## Root Cause

The issue was with the `tools.install(env)` call inside the thread:

```python
# BROKEN CODE (inside thread):
def run_agent_loop_sync(...):
    agent.tools.install(env)  # ❌ Uses asyncio.run() internally!
    ...
```

**Why this caused deadlock:**

1. `tools.install()` is called from within `run_agent_loop_sync()` which runs in a worker thread
2. `tools.install()` internally calls `asyncio.run(self._upload_bundles(env))` and `asyncio.run(self._check_available_commands(env, ...))`
3. When TWO threads call `asyncio.run()` simultaneously:
   - Both try to interact with the Docker daemon
   - They compete for shared resources (Docker API, file handles)
   - This causes a deadlock

**Location in code:**
```python
# From /home/himanshu/swe_livup/sweagent/tools/tools.py:610-630
def _install_commands(self, env: SWEEnv) -> None:
    """Make sure all commands are available in the container"""
    env.set_env_variables(self.config.env_variables)
    cwd = env.communicate("pwd", check="raise").strip()
    asyncio.run(self._upload_bundles(env))  # ❌ DEADLOCK when concurrent
    for bundle in self.config.bundles:
        ...
    asyncio.run(self._check_available_commands(env, {"PATH": path}))  # ❌ DEADLOCK
```

## Solution

**Two-part fix:**

### 1. Move tools.install() outside the thread
Install tools in the main async context BEFORE spawning the agent loop thread.

### 2. Serialize installations with a lock
Use a global `threading.Lock()` to ensure only ONE installation happens at a time.

### 3. Wrap in run_in_executor
Use `await loop.run_in_executor()` to avoid blocking the event loop.

```python
# Add global lock at module level
import threading
_tools_install_lock = threading.Lock()

# In generate() function:
def install_tools_with_lock():
    """Install tools with lock to serialize across concurrent calls."""
    with _tools_install_lock:
        logger.info(f"Acquired tools install lock, installing...")
        agent.tools.install(env)  # Now serialized!
        logger.info(f"✓ Tools installed successfully")

# Run in executor to avoid blocking event loop
loop = asyncio.get_event_loop()
await loop.run_in_executor(None, install_tools_with_lock)

# Then spawn the agent loop thread (tools already installed)
turn_count = await loop.run_in_executor(
    executor,
    run_agent_loop_sync,
    system_prompt,
    instance_prompt,
    max_turns
)
```

## Why This Works

### Execution Flow with Batch Size = 2

**Before Fix (Deadlocked):**
```
Sample 1: [Thread 1] tools.install() → asyncio.run() → Docker API → [WAITING]
Sample 2: [Thread 2] tools.install() → asyncio.run() → Docker API → [WAITING]
                                                          ↑ Both stuck
```

**After Fix (Serialized):**
```
Sample 1: [Main async] await run_in_executor(install_with_lock) → [LOCK acquired] → tools.install() → ✓ Done
Sample 2: [Main async] await run_in_executor(install_with_lock) → [WAITING for lock] ...
                                                                    ↓
Sample 1: [Thread 1] agent.step() → continues...
Sample 2: [LOCK acquired] → tools.install() → ✓ Done → [Thread 2] agent.step() → continues...
```

### Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| Location | Inside thread | Main async context |
| Serialization | None | Global lock |
| Blocking | Blocks event loop | Non-blocking with run_in_executor |
| Concurrency | Deadlock | Sequential installation, parallel execution |

## Technical Details

### Why Serialization is Necessary

When `asyncio.run()` is called from multiple threads simultaneously:
1. Each creates a new event loop
2. Both try to access Docker daemon
3. Docker API has internal locks/mutexes
4. Resource contention causes deadlock

### Why run_in_executor is Necessary

Without it:
- `agent.tools.install(env)` is called in async context
- It uses `asyncio.run()` which conflicts with running loop
- This would cause: `RuntimeError: asyncio.run() cannot be called from a running event loop`

With it:
- Runs in thread pool, separate from main event loop
- Uses lock to serialize
- Non-blocking await

## Performance Impact

**Before Fix:**
- Deadlock with batch size > 1
- Training hangs indefinitely

**After Fix:**
- Tools installation is sequential (overhead: ~5-10 seconds per sample)
- Agent loop execution is fully parallel
- Total time ≈ (install time × batch size sequentially) + (agent loop time in parallel)
- Still get speedup from parallel agent execution

## Testing

```bash
# Set batch size to 2
--rollout-batch-size 2

# Run training
examples/swe_bench/run-qwen3-06B-opd.sh

# Expected behavior:
# 1. Sample 1 installs tools (5-10 seconds)
# 2. Sample 2 waits for lock
# 3. Sample 2 installs tools (5-10 seconds)
# 4. Both samples run agent loops in parallel
# 5. Both complete successfully
```

## Code Location

- **Lock Definition**: `examples/swe_bench/generate_with_sweagent.py:54-57`
- **Serialized Installation**: `examples/swe_bench/generate_with_sweagent.py:508-528`
- **Thread Spawn**: `examples/swe_bench/generate_with_sweagent.py:624-638`

## Key Lessons

1. **Never call asyncio.run() from multiple threads concurrently** - it causes resource contention
2. **Serialize resource-intensive setup** - use locks for Docker operations, file uploads, etc.
3. **Separate setup from execution** - install tools once, then run agent loop in parallel
4. **Use run_in_executor for blocking operations** - keeps event loop responsive

## Alternative Approaches Considered

1. **Patch tools.install()** - Too invasive, would need to modify SWE-agent internals
2. **Use semaphore with count=N** - Doesn't help, Docker daemon is the bottleneck
3. **Keep in thread with lock** - Doesn't solve event loop blocking issue

The chosen approach (serialize installation in main context) is cleanest and most robust.
