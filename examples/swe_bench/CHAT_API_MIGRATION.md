# Migration to Chat Completions API - Fix for LLM Response Quality

## Problem
The LLM responses were poor quality because the data was being sent as plain concatenated text instead of using proper conversation format with roles (system/user/assistant).

## Root Cause
- Original implementation used `/generate` endpoint with plain text concatenation
- Models trained on conversation format (like Qwen) expect proper role markers and chat template formatting
- Without proper formatting, the model doesn't understand the conversation structure

## Solution
Migrated from `/generate` API to `/v1/chat/completions` API with proper message format.

### Key Changes in `generate.py`

#### 1. API Endpoint Change (line 234)
```python
# Before:
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

# After:
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
```

#### 2. Message Format with Roles (lines 336-340)
```python
# Before: Plain text concatenation
prompt_text = f"{system_prompt}\n\n{instance_prompt}"

# After: Proper message list with roles
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": instance_prompt}
]
```

#### 3. Chat Template Application (lines 342-349)
```python
# Apply chat template to get properly formatted prompt for tokenization
prompt_text = state.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
```

#### 4. API Payload Format (lines 365-386)
```python
# Before: Plain text payload
payload = {
    "text": prompt_text + response,
    "sampling_params": sampling_params,
}

# After: Chat completions payload with proper parameters
payload = {
    "model": model_name,
    "messages": messages,  # List of message dicts with roles
}

# Add sampling parameters properly
if isinstance(sampling_params, dict):
    if "temperature" in sampling_params:
        payload["temperature"] = float(sampling_params["temperature"])
    if "max_new_tokens" in sampling_params:
        payload["max_tokens"] = int(sampling_params["max_new_tokens"])
    if "top_p" in sampling_params:
        payload["top_p"] = float(sampling_params["top_p"])
```

#### 5. Response Extraction (lines 388-391)
```python
# Before: Extract from plain text response
cur_response = output["text"]

# After: Extract from chat completion format
choice = output["choices"][0]
cur_response = choice["message"]["content"]
```

#### 6. Multi-turn Conversation Tracking (lines 391, 416, 445)
```python
# Add each assistant response to messages
messages.append({"role": "assistant", "content": cur_response})

# Add feedback as user message
messages.append({"role": "user", "content": feedback})

# Add observations as user message
messages.append({"role": "user", "content": observation})
```

## Benefits

### 1. Proper Role Separation
- System instructions are clearly marked with `role: "system"`
- User messages (problem statement, observations) have `role: "user"`
- Model responses have `role: "assistant"`

### 2. Chat Template Applied Correctly
- The tokenizer's chat template adds proper special tokens (e.g., `<|im_start|>`, `<|im_end|>` for Qwen)
- These tokens help the model understand conversation boundaries

### 3. Better Model Understanding
- Models trained on conversation datasets expect this format
- Proper formatting improves response quality significantly

### 4. API Compatibility
- Using standard OpenAI-compatible `/v1/chat/completions` endpoint
- Works with sglang and other OpenAI-compatible servers
- Easier to swap LLM backends in the future

## Testing

To test the changes:
```bash
docker exec <container-id> /bin/bash -c 'cd /root/repo/slime/ && bash examples/swe_bench/run-qwen3-06B-opd.sh'
```

Check the logs for:
- `[Turn X] Sending chat request with Y messages` - confirms using chat API
- Model responses should be more coherent and follow instructions better
- No more plain text concatenation

## Reference Implementation
This follows the pattern used by:
- SWE-agent's `sweagent/agent/models.py` - `_history_to_messages()` method
- OpenAI Chat Completions API format
- Standard conversation format used by LiteLLM

## Backward Compatibility
This change is **not backward compatible** with code expecting the `/generate` endpoint format. However, it's the correct approach for:
- Chat models (Qwen, Llama, Mistral, etc.)
- Multi-turn conversations
- Proper conversation history tracking

For base/instruct models that don't use chat format, you would still use `/generate` with plain text, but those models typically have poor performance on agentic tasks.
