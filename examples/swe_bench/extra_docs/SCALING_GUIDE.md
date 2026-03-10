# SWE-Agent Time-Bounded Rollout - Scaling Guide

## Overview

The time-bounded rollout strategy (`swe_agent_rollout.py`) handles variable-duration SWE-agent tasks efficiently by:
- Processing samples independently (no group blocking)
- Training every N minutes regardless of completion
- Saving partial samples for next iteration
- Over-provisioning to handle variance

## Configuration Examples

### Small Scale Testing (Current Config)
```bash
# For testing with 2 samples
--rollout-batch-size 2
--n-samples-per-prompt 1
--rollout-max-time-minutes 10
--rollout-over-provision-factor 4
--global-batch-size 2
```

**Expected behavior:**
- Submit 8 tasks (2 * 1 * 4)
- Wait 10 minutes max
- Collect 2 completed samples
- Train on 2 samples

### Medium Scale (16 samples)
```bash
# Balanced config for development
--rollout-batch-size 16
--n-samples-per-prompt 2  # 2 attempts per issue
--rollout-max-time-minutes 10
--rollout-over-provision-factor 3
--global-batch-size 16
--over-sampling-batch-size 32
```

**Expected behavior:**
- Submit ~96 tasks (16 * 2 * 3)
- Collect 16 samples in 10 min
- Best-of-2 analysis per issue
- Train on 16 samples

### Large Scale (64 samples, 4 per prompt)
```bash
# Your target configuration
--rollout-batch-size 64
--n-samples-per-prompt 4  # Best-of-4 per issue
--rollout-max-time-minutes 10
--rollout-over-provision-factor 4
--global-batch-size 64
--over-sampling-batch-size 32
--partial-rollout
--mask-offpolicy-in-partial-rollout
```

**Expected behavior:**
- Submit ~1024 tasks (64 * 4 * 4)
- Collect 64 samples in 10 min
- ~960 tasks aborted (partial samples saved)
- Best-of-4 analysis per issue
- Train on 64 samples
- Next iteration: Resume 960 partial samples

## Performance Projections

### Task Duration Distribution
Based on your estimates:
- Min: 5 minutes
- Max: 60 minutes
- Median: ~15 minutes (estimated)

### Completion Rates (10 min window)

**Without over-provisioning (submit exactly 64 tasks):**
```
Time     | Completed | % Complete
---------|-----------|------------
5 min    | 10        | 16%
7 min    | 20        | 31%
10 min   | 35        | 55%   ← Insufficient!
15 min   | 50        | 78%
```
❌ Would only get ~35 samples in 10 min

**With 4x over-provisioning (submit 256 tasks):**
```
Time     | Completed | % Complete | Available
---------|-----------|------------|----------
5 min    | 40        | 16%        | 40
7 min    | 80        | 31%        | 80
10 min   | 140       | 55%        | 140  ← Collect 64
15 min   | 200       | 78%        | (aborted)
```
✅ Get 64+ samples in 10 min, train immediately

### Resource Utilization

**Docker Containers:**
- With Docker lock: Serialized startup (~7s each)
- 256 tasks = 256 * 7s = ~30 min total startup time
- But many run in parallel after startup
- Effective: ~30-50 containers running simultaneously

**GPU Usage:**
- Agent inference: Minimal GPU (using SGLang)
- Training: Full GPU utilization on 64 samples
- ~20% rollout time, 80% training time

## Iteration Timeline Example

### Iteration 0 (Cold Start)
```
t=0min:   Submit 256 tasks (64 issues * 4 attempts)
          ├─ Docker startup serialized
          └─ Agents start working

t=5min:   40 samples completed
t=7min:   80 samples completed
t=10min:  140 samples completed
          ├─ COLLECT 64 BEST SAMPLES
          ├─ Abort 116 remaining tasks
          │  └─ 90 partial samples (had responses)
          └─ START TRAINING (64 samples)

t=15min:  Training complete
```

### Iteration 1 (Warm Start)
```
t=15min:  Submit 90 partial tasks (resume)
          Submit 166 new tasks (64 * 4 - 90 = 166)
          Total: 256 tasks

t=20min:  50 partials complete (continued from iter 0)
          30 new tasks complete
          = 80 samples ready

t=25min:  Time limit (10 min since start)
          ├─ COLLECT 64 BEST SAMPLES
          ├─ Abort remaining
          └─ START TRAINING

t=30min:  Training complete
```

**Efficiency gain:** Partial samples reduce wasted work by ~35%

## Recommended Settings by Scenario

### Scenario 1: Fast Prototyping
**Goal:** Quick iterations, minimal waste
```bash
--rollout-batch-size 8
--n-samples-per-prompt 1
--rollout-max-time-minutes 5
--rollout-over-provision-factor 3
```
- ~24 samples submitted
- ~8 complete in 5 min
- Quick feedback loop

### Scenario 2: Quality Focus
**Goal:** Best-of-N selection, diverse solutions
```bash
--rollout-batch-size 32
--n-samples-per-prompt 8  # 8 attempts per issue!
--rollout-max-time-minutes 15
--rollout-over-provision-factor 2
```
- ~512 samples submitted
- Best-of-8 per issue
- More training diversity

### Scenario 3: Production Scale
**Goal:** Maximum throughput, efficient compute
```bash
--rollout-batch-size 64
--n-samples-per-prompt 4
--rollout-max-time-minutes 10
--rollout-over-provision-factor 4
--partial-rollout
```
- ~1024 samples submitted
- 64 samples every 10 min
- Partial resumption saves compute

## Monitoring and Debugging

### Key Metrics to Track

1. **Completion Rate**
   - How many samples complete in time window?
   - Adjust `over-provision-factor` if too low

2. **Partial Sample Utilization**
   - How many partials resume successfully?
   - Track via "partial_from_rollout" metadata

3. **Task Duration Distribution**
   - Are most tasks 5 min or 60 min?
   - Adjust `rollout-max-time-minutes` accordingly

4. **Training Throughput**
   - Samples per hour
   - GPU utilization during training

### Log Analysis

Look for these logs:
```
[SWE-Agent Rollout X] Starting time-bounded collection: target=64 samples, max_time=10min
[SWE-Agent Rollout X] Collected sample 64/64: instance_id, duration=XXXs
[SWE-Agent Rollout X] Summary: Collected 64, Submitted 256, Partial 180
[SWE-Agent Rollout X] Sample durations: min=5.2s, max=600s, mean=342s
```

### Common Issues

**Problem:** Not enough samples in time window
```
Solution: Increase over-provision-factor (4 → 6)
```

**Problem:** Too many aborted samples (wasted compute)
```
Solution: Increase rollout-max-time-minutes (10 → 15)
```

**Problem:** Docker startup bottleneck
```
Solution: Current serialized startup is necessary for stability.
          Increase over-provision to mask this latency.
```

**Problem:** Memory issues with too many tasks
```
Solution: Code caps at 1000 tasks. Reduce over-provision-factor.
```

## Migration from Current Config

### Current (Blocking Group Mode)
```bash
--rollout-batch-size 2
--n-samples-per-prompt 1
# No time limit
# Waits for slowest sample (up to 60 min)
```

### New (Time-Bounded Mode)
```bash
--rollout-batch-size 2
--n-samples-per-prompt 1
--rollout-max-time-minutes 10
--rollout-over-provision-factor 4
--custom-rollout-function-path examples.swe_bench.swe_agent_rollout.generate_rollout_swe_agent
```

**Impact:**
- Before: 60 min wait for 2 samples
- After: 10 min for 2 samples, train immediately
- 6x faster iterations!

## Next Steps

1. **Test with current config (2 samples)** - Verify basic functionality
2. **Scale to 16 samples** - Validate throughput
3. **Scale to 64 samples** - Production configuration
4. **Enable partial rollout** - Maximize efficiency

## FAQ

**Q: What if all tasks fail in time window?**
A: Script will return whatever completed. Minimum 0 samples. Check your task success rate.

**Q: Can I disable partial rollout?**
A: Yes, remove `--partial-rollout` flag. Aborted samples will be discarded.

**Q: How do I balance rollout vs training time?**
A: Aim for 20-30% rollout, 70-80% training. Adjust `rollout-max-time-minutes`.

**Q: What's the maximum over-provision-factor?**
A: Code caps total tasks at 1000. With batch=64, n=4: max factor = 1000/(64*4) ≈ 3.9

**Q: Do I need to modify my generate function?**
A: No! Your `generate_with_sweagent.py` works as-is. Only rollout collection changes.
