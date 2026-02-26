# SWE-Agent Integration Guide for Slime

This document explains the two integration approaches and how to use the proper swe-agent integration.

## Two Integration Approaches

### 1. Hybrid Approach (`generate_hybrid.py`)

**Status**: Currently being used but has parsing errors

**Approach**:
- Custom async Docker setup
- Custom multi-turn agent loop
- Uses SWE-agent's `ToolHandler` for action parsing
- Manual trajectory management

**Issues**:
- Model responses not matching expected format
- Parsing errors: `FormatError: Unexpected argument(s): file_text, path`
- The model generates responses that don't conform to swe-agent's XML function calling format

**Why it fails**:
The hybrid approach parses model output directly with `tools.parse_actions(model_output)` but the model isn't generating the correct XML format that swe-agent expects:

```xml
<!-- Expected format -->
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/path/to/file</parameter>
</function>

<!-- But model might generate -->
{file_text: "...", path: "/path/to/file"}
```

### 2. Proper SWE-Agent Integration (`generate_with_sweagent.py`)

**Status**: ✅ Newly implemented, ready for testing

**Approach**:
- Uses `DefaultAgent` directly
- Leverages full swe-agent infrastructure:
  - Response parsing via `ToolHandler`
  - Trajectory saving in swe-agent format
  - State management
  - Automatic requerying on format/syntax errors
- Custom `SlimeLLMModel` for sglang queries

**Benefits**:
- ✅ Proper action parsing with error recovery
- ✅ Automatic requerying on format errors (up to 3 times)
- ✅ Full trajectory management with swe-agent format
- ✅ State tracking and error handling
- ✅ Uses swe-agent's templates and tool configurations

## How the Proper Integration Works

### Architecture

```
SlimeLLMModel (Custom)          DefaultAgent (SWE-agent)
      |                                  |
      |-- query(history) --------------->|
      |                                  |-- parse_actions()
      |                                  |-- handle_action()
      |                                  |-- save_trajectory()
      |<- return {"message": text} ------|
```

### Key Components

#### 1. SlimeLLMModel

A custom LLM model that extends `AbstractModel` and integrates with sglang:

```python
class SlimeLLMModel(AbstractModel):
    def query(self, history: History, action_prompt: str = "> ") -> dict:
        # Convert swe-agent history to OpenAI messages
        messages = self._history_to_messages(history)

        # Query sglang
        response = await post(self.sglang_url, payload)

        # Return in swe-agent format
        return {"message": response["choices"][0]["message"]["content"]}
```

#### 2. DefaultAgent

Uses swe-agent's `DefaultAgent` which provides:

- **Action Parsing**: `tools.parse_actions()` with configurable parsers (XML, ThoughtAction, etc.)
- **Error Recovery**: Automatic requerying on format errors, bash syntax errors, blocked actions
- **Trajectory Management**: Saves `.traj` files with full conversation history
- **State Tracking**: Records environment state at each step

#### 3. Configuration

Loads from `/root/swe_livup/config/test_xml_v2.yaml`:
- System prompt with problem-solving workflow
- Tool definitions (bash, editor, grep, find, etc.)
- Parser configuration (XML function calling)
- Templates for prompts and observations

### Agent Loop Flow

```python
# Setup
agent.setup(env=env, problem_statement=problem_stmt, output_dir=output_dir)

# Multi-turn loop
while turn_count < max_turns:
    step_output = agent.step()  # Calls model.query() → parse → execute → save

    if step_output.done:
        break

# Save trajectory
agent.save_trajectory()

# Extract data for slime training
traj_data = agent.get_trajectory_data()
trajectory = traj_data["trajectory"]  # List of executed steps
history = traj_data["history"]        # Full conversation
```

### Error Handling

The agent automatically handles:

1. **FormatError**: Model output doesn't match expected format
   - Agent requeries with format error template
   - Up to 3 retries

2. **BashIncorrectSyntaxError**: Invalid bash syntax
   - Agent requeries with syntax error message

3. **BlockedActionError**: Action is on blocklist
   - Agent requeries with blocklist error template

4. **CommandTimeoutError**: Command takes too long
   - Agent interrupts session and continues

### Trajectory Format

Each trajectory step contains:

```python
{
    "action": "bash command executed",
    "observation": "command output",
    "response": "full LLM response text",
    "thought": "agent's reasoning",
    "state": {"working_directory": "/testbed", ...},
    "execution_time": 2.34,
    "query": [...message history sent to LLM...],
    "extra_info": {"exit_code": 0, ...}
}
```

### Conversion to Slime Sample

The integration converts swe-agent's trajectory to slime's `Sample` format:

1. **Tokenization**: Uses slime's tokenizer to convert text → tokens
2. **Loss Masking**:
   - Prompt: `loss_mask = 0` (don't train)
   - Model responses: `loss_mask = 1` (train on these)
   - Observations: `loss_mask = 0` (don't train)
3. **Metadata**: Stores full trajectory for later analysis

## Switching to Proper Integration

### Step 1: Update Training Script

Edit `run-qwen3-06B-opd.sh`:

```bash
# Change this line:
--custom-generate-function-path examples.swe_bench.generate_hybrid.generate

# To this:
--custom-generate-function-path examples.swe_bench.generate_with_sweagent.generate
```

### Step 2: Test with Single Instance

```bash
# Run with 1 rollout to test
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh" | head -1000
```

Look for these log patterns:

**✅ Success indicators**:
```
[Slime-SWE] Starting instance: django__django-12345
[Slime-SWE] ✓ Environment ready
[Slime-SWE] Created DefaultAgent
[Slime-SWE] ✓ Agent setup complete
[Slime-SWE] Turn 1/20: Action: <function=bash>...
[Slime-SWE] ✓ Trajectory saved
[Slime-SWE] ✓ Sample prepared: 2048 tokens, 5 turns
```

**❌ Error indicators**:
```
FormatError: Unexpected argument(s): ...
RuntimeError: Invalid response from sglang API
TimeoutError: Command timed out after 1800s
```

### Step 3: Monitor Trajectories

```bash
# View saved trajectories
ls -lh /tmp/swe_agent_trajectories/

# Inspect a trajectory
python examples/swe_bench/inspect_trajectories.py --instance-id django__django-12345
```

### Step 4: Full Training

If tests pass, run full training:

```bash
docker exec 9e0e554ee79e /bin/bash -c "cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh"
```

## Comparison: Hybrid vs Proper Integration

| Feature | Hybrid Approach | Proper Integration |
|---------|----------------|-------------------|
| **Action Parsing** | Manual with `tools.parse_actions()` | Automatic via `DefaultAgent` |
| **Error Recovery** | None - fails on format error | Automatic requerying (up to 3x) |
| **Format Validation** | Basic | Full validation with multiple parsers |
| **Trajectory Format** | Custom dict | Standard swe-agent `.traj` format |
| **State Management** | Manual | Automatic via `agent.step()` |
| **Template System** | Manual replacement | Automatic via `TemplateConfig` |
| **Tool Installation** | None | Automatic via `ToolHandler.install()` |
| **History Processing** | None | Via `HistoryProcessor` chain |
| **Cost Tracking** | None | Via `InstanceStats` |
| **Replayability** | Limited | Full replay support |

## Expected Improvements

With proper integration, you should see:

1. **Fewer Parsing Errors**: The agent will automatically retry on format errors
2. **Better Action Quality**: The model learns the correct format through retries
3. **Richer Trajectories**: Full conversation history with state snapshots
4. **Better Debugging**: Standard `.traj` files viewable with swe-agent tools
5. **Cleaner Code**: ~400 lines vs ~600 lines in hybrid approach

## Troubleshooting

### Issue: Model still generating wrong format

**Solution**: The model needs to learn the XML function calling format. Check:

1. Is the system prompt being sent correctly?
   - Log the first message in `SlimeLLMModel.query()`

2. Is the model seeing examples of correct format?
   - Add few-shot examples to the prompt

3. Is the temperature too high?
   - Try lower temperature (0.3-0.5) for more consistent formatting

### Issue: Agent gets stuck in retry loops

**Solution**: Reduce `max_requeries` or improve the format error template:

```yaml
# In config.yaml
agent:
  max_requeries: 2  # Reduce from 3

  tools:
    filter:
      format_error_template: |
        Your response was not in the correct format.
        You MUST use this exact format:
        <function=FUNCTION_NAME>
        <parameter=PARAM_NAME>value</parameter>
        </function>
```

### Issue: Trajectories not saving

**Solution**: Check directory permissions:

```bash
docker exec 9e0e554ee79e mkdir -p /tmp/swe_agent_trajectories
docker exec 9e0e554ee79e chmod 777 /tmp/swe_agent_trajectories
```

### Issue: Docker container failures

**Solution**: Check Docker daemon is accessible from within the container:

```bash
# Test Docker access
docker exec 9e0e554ee79e docker ps

# If fails, mount Docker socket
docker run -v /var/run/docker.sock:/var/run/docker.sock ...
```

## Next Steps

1. **Test the Integration**: Run with 1-2 instances first
2. **Monitor Logs**: Check for parsing errors and action quality
3. **Analyze Trajectories**: Use `inspect_trajectories.py` to view agent behavior
4. **Tune Prompts**: Adjust templates if model isn't following format
5. **Scale Up**: Once stable, run full training with 50+ instances

## References

- SWE-agent docs: https://github.com/SWE-agent/SWE-agent
- DefaultAgent source: `/home/himanshu/swe_livup/sweagent/agent/agents.py`
- Config reference: `/home/himanshu/swe_livup/config/test_xml_v2.yaml`
- Slime OPD example: `/root/slime/examples/on_policy_distillation/`
