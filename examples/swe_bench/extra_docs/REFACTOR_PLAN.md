# Refactoring Plan: Integrate SWE-agent Infrastructure

## Current Problems
1. Simple bash command parsing with `parse_bash_command()` - doesn't handle all tool formats
2. No proper feedback for various error scenarios
3. No state management (submitted, exit_format, exit_error, etc.)
4. Basic trajectory saving - only saves to .pt file
5. Missing proper response parsing for different formats (XML, function calling, etc.)

## Proposed Solution

Instead of reimplementing everything, wrap SWE-agent's `DefaultAgent` class and customize only the LLM query part to use sglang's chat completions API.

### Architecture

```
SlimeAgent (Custom)
  └─ Inherits from DefaultAgent
  └─ Overrides model.query() to use sglang chat API
  └─ Keeps all SWE-agent infrastructure:
      - Tools parsing (XMLFunctionCallingParser)
      - Action execution with proper feedback
      - State management
      - Trajectory saving (.traj, .info.log, etc.)
```

### Implementation Steps

1. **Create SlimeAgent class** that extends DefaultAgent
2. **Create SlimeLLMModel class** that implements AbstractModel interface
   - Override `query()` to use sglang chat completions API
   - Keep conversation history in proper message format
3. **Use SWE-agent's infrastructure**:
   - Tools for parsing and execution
   - Template system for prompts
   - Trajectory saving in standard format
   - State management

### Benefits

1. ✅ Proper response parsing with XMLFunctionCallingParser
2. ✅ Rich feedback for various scenarios (format errors, blocked actions, timeouts, etc.)
3. ✅ State management (submitted, exit_format, exit_error, etc.)
4. ✅ Standard trajectory format (.traj, .info.log, .raw_model_log, .pred, etc.)
5. ✅ Minimal custom code - only the LLM API call
6. ✅ Full compatibility with SWE-agent ecosystem

### Code Changes Needed

**New File**: `generate_with_sweagent.py`
- SlimeLLMModel (implements AbstractModel, calls sglang chat API)
- SlimeAgent (extends DefaultAgent if needed)
- generate() function that creates agent and runs it

**Modified**: Remove old `generate.py` or rename to `generate_old.py`

### Key SWE-agent Components to Use

1. **sweagent.agent.agents.DefaultAgent** - Main agent loop
2. **sweagent.agent.models.AbstractModel** - Model interface
3. **sweagent.tools.Tools** - Tool parsing and execution
4. **sweagent.tools.parsing.XMLFunctionCallingParser** - Response parsing
5. **sweagent.environment.swe_env.SWEEnv** - Environment management (already using)

## Next Steps

1. Create SlimeLLMModel that wraps sglang chat completions API
2. Integrate with SWE-agent's DefaultAgent
3. Test that it works with proper parsing and feedback
4. Verify trajectory saving in proper format
