# SWE-agent Integration - Summary

## What Was Done

I refactored the SWE-bench generation code to properly integrate with SWE-agent's infrastructure, following the principle of **reusing existing code instead of reimplementing it**.

## Problem with Original Approach

The original `generate.py` had several issues:

1. **Simple bash command parsing** - Only extracted commands from ```bash``` blocks
   - No support for XML function calling format
   - No proper error handling or feedback

2. **No feedback generation** - When parsing fails or commands error, no helpful feedback

3. **No state management** - Missing exit states like "submitted", "exit_format", "exit_error"

4. **Limited trajectory saving** - Only saved to .pt file, not SWE-agent's standard format

5. **Wrong message format** - Was concatenating plain text instead of proper conversation format

## New Hybrid Approach

### Architecture

```
generate_hybrid.py
├── Async Docker Setup (Custom)
│   ├── _async_env_start() - Boot Docker, setup environment
│   └── _async_communicate() - Execute commands async
│
├── Multi-turn Agent Loop (Custom)
│   ├── Chat completions API with proper message format
│   ├── Tokenization and loss masking
│   └── Turn-by-turn conversation management
│
└── SWE-agent Infrastructure (Reused)
    ├── ToolHandler - Parse actions from model output
    ├── ToolConfig - Tool configuration from YAML
    ├── Command execution with proper error handling
    └── Trajectory format compatible with SWE-agent
```

### Key Components

#### 1. Proper Chat API Integration (Fixed)
```python
# OLD: Plain text concatenation
payload = {
    "text": prompt_text + response,
    "sampling_params": sampling_params,
}

# NEW: Proper message format
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instance_prompt}
]
payload = {
    "model": model_name,
    "messages": messages,
    "temperature": ...,
    "max_tokens": ...,
}
```

#### 2. SWE-agent Tools Integration (New)
```python
# Load SWE-agent configuration
with open("/root/swe_livup/config/test_xml_v2.yaml") as f:
    swe_config = yaml.safe_load(f)

# Create tool handler
tool_config = ToolConfig(**swe_config["agent"]["tools"])
tools = ToolHandler(config=tool_config)

# Parse actions using SWE-agent's parser
thought, action = tools.parse_actions(model_output)
```

This gives us:
- ✅ XML function calling parser
- ✅ Multiple command formats (bash, str_replace_editor, etc.)
- ✅ Proper error messages on parse failure
- ✅ Blocklist checking
- ✅ State management

#### 3. Proper Template Usage (New)
```python
# Load templates from SWE-agent config
system_template = swe_config["agent"]["templates"]["system_template"]
instance_template = swe_config["agent"]["templates"]["instance_template"]
next_step_template = swe_config["agent"]["templates"]["next_step_template"]

# Replace placeholders
system_prompt = system_template.replace("{{command_docs}}", command_docs)
instance_prompt = instance_template.replace("{{working_dir}}", working_dir)
observation = next_step_template.replace("{{observation}}", raw_observation)
```

#### 4. Trajectory Saving (New)
```python
# Save in SWE-agent TrajectoryStep format
step = TrajectoryStep(
    action=action,
    observation=observation,
    response=cur_response,
    state=tools.get_state(env=env),
    thought=thought,
    execution_time=execution_time,
    query=messages[:-1],
    extra_info={"exit_code": env._last_exit_code},
)
trajectory.append(step)

# Save trajectory file
traj_data = {"trajectory": trajectory, "info": metadata}
traj_file.write_text(json.dumps(traj_data, indent=2))
```

## Benefits

### 1. Proper Response Parsing
- Uses SWE-agent's `XMLFunctionCallingParser`
- Handles multiple command formats
- Provides helpful error messages when parsing fails

### 2. Rich Feedback System
- Format errors: "Your output did not follow the required format..."
- Execution errors: Command output with exit codes
- Timeout errors: "Command timed out after 30 seconds..."
- Blocklist errors: "Command 'rm -rf' is blocked..."

### 3. State Management
- Tracks environment state (working directory, modified files)
- Detects exit conditions (submit, exit, etc.)
- Proper status tracking (submitted, error, timeout)

### 4. Standard Trajectory Format
- Compatible with SWE-agent's trajectory viewer
- Includes all metadata (thought, action, observation, state, timing)
- Can be analyzed with existing SWE-agent tools

### 5. Better LLM Responses
- Uses proper conversation format with roles
- Applies chat template correctly
- Model understands conversation boundaries

## File Structure

### New Files
- `generate_hybrid.py` - Main implementation (hybrid approach)
- `generate_with_sweagent.py` - Earlier attempt at full wrapper (not used)
- `INTEGRATION_SUMMARY.md` - This document
- `REFACTOR_PLAN.md` - Original refactoring plan
- `CHAT_API_MIGRATION.md` - Chat API migration details

### Modified Files
- `run-qwen3-06B-opd.sh` - Updated to use `generate_hybrid.generate`

### Kept Files
- `generate.py` - Old implementation (can be removed later)
- `reward.py` - Unchanged (handles teacher logprobs)
- `data_loader.py` - Unchanged
- `add_image_names.py` - Unchanged

## Testing

To test the new implementation:

```bash
# Start the Docker container
docker exec <container-id> /bin/bash

# Run the training script
cd /root/repo/slime/
bash examples/swe_bench/run-qwen3-06B-opd.sh
```

Expected behavior:
1. Docker containers boot (~12 seconds)
2. Environment setup completes
3. Tools load successfully
4. Agent performs multi-turn interactions
5. Trajectories saved to `/tmp/swe_agent_trajectories/`

## What's Different From Reference SWE-agent

### Custom Parts (For Training)
1. **LLM Query** - Uses sglang chat completions API instead of litellm
2. **Async Event Loop** - Runs in Ray/slime's async context
3. **Tokenization** - Tracks tokens and loss masks for training
4. **Teacher Logprobs** - Collects logprobs for OPD via reward function

### Reused Parts (From SWE-agent)
1. **Tools & Parsing** - Action parsing with proper error handling
2. **Templates** - System, instance, and observation templates
3. **Trajectory Format** - Standard format for analysis
4. **State Management** - Environment state tracking

## Future Improvements

### Could Add
1. **More trajectory files** - .info.log, .raw_model_log, .pred files
2. **Intervention system** - Reproduce issue intervention, etc.
3. **Patch coverage tracking** - Track which lines were viewed/edited
4. **Better error recovery** - Requery on parse failures
5. **Submit detection** - Auto-detect when patch is ready

### Not Needed for Training
1. **Full DefaultAgent wrapper** - Too complex, hybrid is sufficient
2. **Reviewer integration** - Not needed for pure distillation
3. **Summarizer** - Not needed for short contexts
4. **Hook system** - Not needed for training

## Summary

The hybrid approach gives us **~80% of SWE-agent's functionality** while keeping the code **simple and maintainable**. We reuse the most important parts (Tools, parsing, templates, trajectory format) while customizing the parts needed for training (LLM API, async execution, tokenization).

This is the right balance between:
- **Reusing proven code** (SWE-agent's robust infrastructure)
- **Maintaining simplicity** (not wrapping everything)
- **Meeting training needs** (async, tokenization, OPD)
