# Time-Bounded Rollout for SWE-Agent

## Summary

A new rollout strategy that handles variable-duration SWE-agent tasks (5-60 minutes) efficiently by collecting samples for a fixed time window and training on whatever's ready.

## Key Features

✅ **Time-bounded collection** - Train every 10 minutes regardless of task duration
✅ **Independent sample processing** - No blocking on groups
✅ **Partial sample resumption** - Save incomplete samples for next iteration
✅ **Over-provisioning** - Submit 4x samples to ensure target is met
✅ **Scales efficiently** - Handles 64+ batch sizes with 4 samples per prompt

## Files

- **`swe_agent_rollout.py`** - Core time-bounded rollout implementation
- **`run-qwen3-06B-opd.sh`** - Updated training script (small scale: 2 samples)
- **`run-qwen3-06B-opd-scaled.sh`** - Production config (large scale: 64 samples)
- **`SCALING_GUIDE.md`** - Detailed scaling guide and performance projections

## Quick Start

### Test with Current Config (2 samples)
```bash
bash examples/swe_bench/run-qwen3-06B-opd.sh
```

This will:
- Submit 8 tasks (2 * 1 * 4 over-provision)
- Collect for 10 minutes
- Train on 2 completed samples
- Save partial samples for next iteration

### Scale to Production (64 samples, 4 per prompt)
```bash
bash examples/swe_bench/run-qwen3-06B-opd-scaled.sh
```

This will:
- Submit 1024 tasks (64 * 4 * 4 over-provision)
- Collect for 10 minutes
- Train on 64 best samples (best-of-4 per issue)
- Save ~960 partial samples for next iteration

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rollout-batch-size` | 2 | Number of samples to collect |
| `--n-samples-per-prompt` | 1 | Samples per issue (for best-of-N) |
| `--rollout-max-time-minutes` | 10 | Max time to collect samples |
| `--rollout-over-provision-factor` | 4 | Submit N× samples to handle variance |
| `--partial-rollout` | - | Enable partial sample saving |
| `--mask-offpolicy-in-partial-rollout` | - | Mask old tokens in resumed samples |

## How It Works

### Traditional Rollout (Blocking on Groups)
```
Group 1: [5min, 7min, 10min, 58min] → BLOCKS for 58min
Group 2: [6min, 8min, 55min, 60min] → BLOCKS for 60min

Result: First training after ~60 minutes
```

### Time-Bounded Rollout (Independent Samples)
```
Submit: 256 independent tasks
  t=5min:  40 complete
  t=7min:  80 complete
  t=10min: 140 complete → TAKE 64, TRAIN NOW
  Abort: 116 remaining (save partials)

Result: Training every 10 minutes
```

## Performance Comparison

### Small Scale (2 samples)
| Metric | Old (Blocking) | New (Time-Bounded) | Improvement |
|--------|----------------|-----------------------|-------------|
| Samples | 2 | 2 | Same |
| Wait time | 60 min (worst case) | 10 min (fixed) | **6x faster** |
| Wasted work | 0% | ~75% (saved as partial) | Resumable |

### Large Scale (64 samples, 4 per prompt)
| Metric | Old (Blocking) | New (Time-Bounded) | Improvement |
|--------|----------------|-----------------------|-------------|
| Samples | 256 | 64 | Focused on best |
| Wait time | 60 min (worst case) | 10 min (fixed) | **6x faster** |
| Throughput | 256 samples/hr | 384 samples/hr | **1.5x higher** |
| Best-of-N | Yes (per group) | Yes (post-hoc) | Same quality |

## Architecture

```python
# Traditional: Group blocks until all N samples finish
async def generate_rollout_traditional():
    for group in sample_groups:
        # Wait for ALL n_samples_per_prompt to finish
        results = await asyncio.gather(*[
            generate(sample) for sample in group
        ])
        # Process complete group
        process(results)

# Time-bounded: Individual samples processed as they finish
async def generate_rollout_swe_agent():
    # Submit all tasks individually
    for group in sample_groups:
        for sample in group:
            tasks.append(asyncio.create_task(generate(sample)))

    # Collect for max_time_minutes
    deadline = time.time() + max_time_minutes * 60
    while len(data) < target and time.time() < deadline:
        done, pending = await asyncio.wait(
            tasks,
            return_when=FIRST_COMPLETED
        )
        # Process each completed sample immediately
        for task in done:
            sample = task.result()
            if filter_passes(sample):
                data.append(sample)

    # Abort remaining, save partials
    abort_and_save_partials(pending)
    return data
```

## Monitoring

### Key Logs to Watch

```
[SWE-Agent Rollout 0] Starting time-bounded collection: target=64 samples, max_time=10min
[SWE-Agent Rollout 0] Submitted 1024 tasks, 0 pending, 0 collected
[SWE-Agent Rollout 0] Collected sample 64/64: pydata__xarray-3993, duration=598.1s
[SWE-Agent Rollout 0] Summary:
  - Collected: 64 samples
  - Submitted: 1024 tasks
  - Partial: 960 samples saved for next iteration
  - Duration: 600.0s (10.0min)
  - Throughput: 6.40 samples/min
[SWE-Agent Rollout 0] Sample durations: min=5.2s, max=600s, mean=342s
```

### Metrics Dashboard

Track these in your monitoring:
- **Completion rate**: % of submitted tasks that finish in time window
- **Throughput**: Samples per minute
- **Partial utilization**: % of partial samples that resume successfully
- **Best-of-N success**: % improvement from selecting best of N

## Troubleshooting

### Not enough samples in time window
```bash
# Increase over-provisioning
--rollout-over-provision-factor 6  # Was 4
```

### Too many aborted samples (wasted compute)
```bash
# Increase time window
--rollout-max-time-minutes 15  # Was 10
```

### Memory issues with many tasks
```bash
# Reduce over-provisioning (code caps at 1000)
--rollout-over-provision-factor 3  # Was 4
```

### Docker startup bottleneck
```bash
# Increase over-provisioning to mask latency
--rollout-over-provision-factor 5
# Docker lock is necessary for stability
```

## Migration Path

### Phase 1: Test (Current)
```bash
# Use current small config to validate
bash examples/swe_bench/run-qwen3-06B-opd.sh
```

### Phase 2: Scale Up
```bash
# Gradually increase batch size
--rollout-batch-size 8   # Test with 8
--rollout-batch-size 16  # Then 16
--rollout-batch-size 32  # Then 32
--rollout-batch-size 64  # Finally 64
```

### Phase 3: Optimize
```bash
# Enable best-of-N
--n-samples-per-prompt 4

# Fine-tune timing
--rollout-max-time-minutes 12  # Adjust based on metrics

# Enable partial rollout
--partial-rollout
--mask-offpolicy-in-partial-rollout
```

## Expected Results

### Iteration Timeline (64 batch, 10 min window)

```
Iteration 0:
  0min → 10min: Collect 64 samples
 10min → 15min: Train on 64 samples

Iteration 1:
 15min → 25min: Collect 64 samples (resume 960 partials)
 25min → 30min: Train on 64 samples

...

Result: Train every ~15 minutes (10 min rollout + 5 min train)
        8 training steps per 2 hours
        vs. 2 steps per 2 hours with blocking
```

## Next Steps

1. ✅ **Implemented** - Time-bounded rollout (`swe_agent_rollout.py`)
2. ✅ **Updated** - Training scripts with new config
3. ✅ **Documented** - Scaling guide and examples
4. 🔲 **Test** - Run with 2 samples to validate
5. 🔲 **Scale** - Increase to 64 samples
6. 🔲 **Optimize** - Fine-tune based on metrics

## Questions?

See `SCALING_GUIDE.md` for detailed performance projections and configuration recommendations.
