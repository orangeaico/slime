# Docker Startup Comparison: Reference vs My Implementation

## Reference SWE-agent Implementation

**Configuration**:
- Batch size: 1 sample at a time
- Docker command: Same fallback with pipx
- Timeout: 180 seconds
- Image: `swebench/sweb.eval.x86_64.pydata_1776_xarray-3151`

**Timing**:
- Container start: ~immediate (Docker command issued)
- swe-rex installation: ~10-20 seconds (via pipx fallback)
- Total to completion: **< 90 seconds** (completed task including environment setup AND execution)

**Result**: ✅ Completed successfully

**Logs**:
```
🦖 INFO Starting container swebenchsweb.eval.x86_64.pydata_1776_xarray-3151-...
🦖 INFO Starting runtime at 43431
[... 90 seconds later ...]
completed=1 remaining=0 total=1
```

## My Implementation (Current)

**Configuration**:
- Batch size: 4 samples simultaneously (rollout-batch-size=2, n-samples-per-prompt=2)
- Docker command: Same fallback with pipx
- Timeout: 300 seconds (increased)
- Images: `swebench/sweb.eval.x86_64.django_1776_django-10924`, `swebench/sweb.eval.x86_64.astropy_1776_astropy-14995`

**Timing** (as observed):
- Container start: ~immediate (4 Docker containers started)
- swe-rex installation: In progress (verified containers are running)
- Containers responsive: ✅ YES - all 4 ports responding to /health
- Code execution: HUNG at `await env.deployment.start()`

**Logs**:
```
🦖 INFO Starting container swebenchsweb.eval.x86_64.django_1776_django-10924-...
🦖 INFO Starting runtime at 48729
🦖 INFO Starting runtime at 52187
🦖 INFO Starting runtime at 50603
🦖 INFO Starting runtime at 36993
[SWE-agent Init] Starting environment for instance: django__django-10924
[... NO subsequent logs ...]
```

**Container Status** (verified):
```bash
$ curl http://localhost:48729/health
{"detail":"Not authenticated"} ✅

$ docker logs d79b0d1ef552 | tail -5
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit) ✅
```

## Analysis

### What's Working:
1. ✅ Docker containers start successfully
2. ✅ swe-rex installs and runs via pipx fallback
3. ✅ HTTP servers are running and responding on all 4 ports
4. ✅ Using correct SWE-bench images
5. ✅ Using PreExistingRepoConfig for /testbed

### What's NOT Working:
1. ❌ `await env.deployment.start()` appears to hang or take extremely long
2. ❌ No logs after line 242 in generate.py
3. ❌ 4 simultaneous container startups causing resource contention

### Root Cause Hypothesis:

The issue is NOT that containers are slow to start - they ARE starting and running fine.

The issue is likely:

1. **Async timing issue**: `await env.deployment.start()` is not properly awaiting or detecting that the swe-rex server is ready
2. **Connection timeout**: Even though servers are running, the connection check might be failing
3. **Resource contention**: 4 containers starting simultaneously may be overwhelming some resource

### Key Difference:

**Reference**: 1 container at a time → completes in < 90 seconds
**Mine**: 4 containers simultaneously → hangs indefinitely

## Recommendation

**IMMEDIATE FIX**: Reduce batch size to 1:
```bash
--rollout-batch-size 1
--n-samples-per-prompt 1
```

This will:
- Start only 1 container at a time (like reference)
- Reduce resource contention
- Match reference implementation behavior
- Allow proper debugging if still hangs

**DEBUGGING**: If still hangs with batch-size=1, add extensive logging in `_async_env_start()` to see exactly where it's hanging.

## Test Plan

1. Kill all running containers
2. Update run script: batch-size=1, n-samples=1
3. Run again with timeout of 2 minutes
4. Compare timing with reference implementation
5. If still slow, add debug logging in `_async_env_start()` around line 48 where `await env.deployment.start()` is called
