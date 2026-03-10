# Quick Start: Proper SWE-Agent Integration

## Summary

I've successfully updated `generate_with_sweagent.py` to properly integrate with swe-agent's `DefaultAgent` infrastructure. This provides:

✅ **Proper action parsing** with automatic error recovery
✅ **Response formatting** that matches swe-agent's expectations
✅ **Trajectory saving** in standard swe-agent format
✅ **State management** and automatic requerying on errors
✅ **Better debugging** with full conversation history

## What Changed

### Before (Hybrid Approach)
- Custom agent loop with manual parsing
- No error recovery - failed on format errors
- Model generating incorrect command formats
- Error: `FormatError: Unexpected argument(s): file_text, path`

### After (Proper Integration)
- Uses swe-agent's `DefaultAgent` directly
- Automatic requerying on format/syntax errors (up to 3 times)
- Proper XML function calling format enforcement
- Full trajectory management with state tracking

## Files Updated

1. **`generate_with_sweagent.py`** - Proper swe-agent integration (574 lines)
   - `SlimeLLMModel` class - Custom LLM model for sglang queries
   - `generate()` function - Uses `DefaultAgent` for multi-turn loop
   - Full trajectory conversion to slime's `Sample` format

2. **`run-qwen3-06B-opd.sh`** - Updated to use new integration
   - Changed: `--custom-generate-function-path examples.swe_bench.generate_with_sweagent.generate`

3. **`INTEGRATION_GUIDE.md`** - Comprehensive guide on how it works

4. **`QUICK_START.md`** - This file

## Dependencies Installed

```bash
pip install simple-parsing tenacity ruamel.yaml litellm unidiff rich-argparse flask flask-cors flask-socketio
```

All dependencies are now installed in the Docker container.

## How to Use

### 1. Run Training

The training script is already updated:

```bash
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh"
```

### 2. Monitor Progress

Watch for these log patterns:

**✅ Success**:
```
[Slime-SWE] Starting instance: django__django-12345
[Slime-SWE] ✓ Environment ready
[Slime-SWE] Created DefaultAgent
[Slime-SWE] Turn 1/20: Action: <function=bash>...
[Slime-SWE] ✓ Trajectory saved
```

**❌ Errors to watch for**:
```
FormatError: Unexpected argument(s): ...
ModuleNotFoundError: No module named 'tenacity'
RuntimeError: Invalid response from sglang API
```

### 3. View Trajectories

```bash
# List saved trajectories
docker exec 9e0e554ee79e ls -lh /tmp/swe_agent_trajectories/

# Inspect a specific trajectory
docker exec 9e0e554ee79e python examples/swe_bench/inspect_trajectories.py --instance-id django__django-12345
```

## Key Architecture

```
┌─────────────────┐
│  SlimeLLMModel  │ (Custom)
│                 │
│ - query()       │ ──────> Queries sglang
│ - _history_to_  │         Returns responses
│   messages()    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ DefaultAgent    │ (SWE-agent)
│                 │
│ - setup()       │ ──────> Initializes environment
│ - step()        │ ──────> Multi-turn loop
│ - parse_actions │ ──────> Parses model output
│ - handle_action │ ──────> Executes commands
│ - save_         │ ──────> Saves trajectories
│   trajectory()  │
└─────────────────┘
         │
         v
┌─────────────────┐
│  Sample Output  │ (For slime training)
│                 │
│ - tokens        │ ──────> Full tokenized conversation
│ - loss_mask     │ ──────> 1 for model outputs, 0 for observations
│ - metadata      │ ──────> Full trajectory + info
└─────────────────┘
```

## Expected Benefits

1. **Fewer Parsing Errors**: Agent automatically retries on format errors
2. **Better Action Quality**: Model learns correct format through retries
3. **Richer Trajectories**: Full conversation with state snapshots
4. **Better Debugging**: Standard `.traj` files viewable with swe-agent tools
5. **Cleaner Code**: Leverages battle-tested swe-agent infrastructure

## Troubleshooting

### Issue: Training still fails with ModuleNotFoundError

**Solution**: Dependencies might not be visible to Ray workers. Check:

```bash
# Verify dependencies in container
docker exec 9e0e554ee79e python -c "import tenacity; import simple_parsing; print('OK')"
```

If fails, the dependencies need to be in the runtime environment passed to Ray workers.

### Issue: Model still generating wrong format

**Solution**: The model needs to learn XML function calling. Try:

1. Lower temperature (0.3-0.5) for more consistent formatting
2. Check system prompt is being sent correctly
3. Add few-shot examples of correct format

### Issue: Docker containers not starting

**Solution**: Check Docker access from within the container:

```bash
docker exec 9e0e554ee79e docker ps
```

If fails, Docker socket may not be mounted correctly.

## Next Steps

1. **Test with 1-2 instances** first to verify integration works
2. **Monitor logs** for parsing errors and action quality
3. **Analyze trajectories** using `inspect_trajectories.py`
4. **Tune prompts** if model isn't following format
5. **Scale up** once stable with 50+ rollouts

## Files to Reference

- **Integration Guide**: `INTEGRATION_GUIDE.md` - Comprehensive explanation
- **Implementation**: `generate_with_sweagent.py` - Full code
- **Training Script**: `run-qwen3-06B-opd.sh` - How to run
- **SWE-agent Config**: `/root/swe_livup/config/test_xml_v2.yaml` - Agent configuration

## Contact

For issues or questions, refer to:
- SWE-agent docs: https://github.com/SWE-agent/SWE-agent
- DefaultAgent source: `/home/himanshu/swe_livup/sweagent/agent/agents.py`
- Slime docs: `/root/slime/examples/on_policy_distillation/`
