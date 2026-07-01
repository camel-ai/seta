# TITO Chat Model Backend + TITO Train Agent

## Context

The current `AgentTrain` drops `reasoning_content` when recording assistant messages to memory. After the model responds with tool calls, `_record_assistant_tool_calls_from_requests()` creates a new `BaseMessage` with only `content` + `tool_calls` — `reasoning_content` is lost. `BaseMessage.to_openai_assistant_message()` also never serializes it. This means the model never sees its prior reasoning on subsequent turns, and the message prefix changes between turns, reducing KV cache hits on Kimi API.

**Goal**: Create a TITO model backend that maintains a per-session message history with raw API responses (preserving everything), and a companion TITO agent — all configurable via YAML.

---

## Architecture

```
Agent (AgentTrainTITO)                  TITOChatModel (BaseModelBackend)
========================                ================================
Manages memory for token               Maintains _session_messages = [
counting, logging, etc.                   {sys_msg},
                                          {user_msg},
Calls model.arun(openai_messages)  →      {raw_asst_response_1},  ← model_dump()
                                          {tool_result_1},
                                          {tool_result_2},
                                          {raw_asst_response_2},
                                          ...
                                        ]

                                        On each arun():
                                        1. Scan agent msgs backward to find last assistant
                                        2. Everything after = new tool/user msgs
                                        3. Append new msgs to _session_messages
                                        4. Call inner._arun(_session_messages)
                                        5. Store raw response via model_dump()
                                        6. Track cache stats from usage
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `seta_env/models/tito_chat_model.py` | **CREATE** | TITO model backend, inherits `BaseModelBackend` |
| `seta_env/agent/tito_train_agent.py` | **CREATE** | TITO agent, extends `AgentTrain` |
| `seta_env/agent/prompt_loader.py` | MODIFY | Register `tito_train_agent` |
| `seta_env/environments/terminal_env.py` | MODIFY | Wrap model with TITO when configured |
| `scripts/evaluation/configs/eval_default_kimi_tito.yaml` | **CREATE** | Config referencing TITO agent + model |

---

## Step 1: `seta_env/models/tito_chat_model.py`

**Inherits from `BaseModelBackend`**, only overrides `_arun` (not `arun`).

The metaclass `ModelBackendMeta` only wraps methods **defined in the class namespace**. Since we only define `_arun`, the metaclass doesn't touch it. The inherited `BaseModelBackend.arun()` handles logging and calls our `_arun()`.

### `__init__(self, inner_model, validate=False)`

- Call `super().__init__(model_type=inner_model.model_type, model_config_dict=inner_model.model_config_dict, token_counter=inner_model.token_counter)` — get proper BaseModelBackend setup (logging, token counting)
- Store `self._inner = inner_model`
- Disable inner model's logging: `inner_model._log_enabled = False` (avoid double-logging; our base arun handles it)
- Init TITO state: `_session_messages = []`, `_cache_stats`, `_validate`

### `reset(self)`
Called at start of each episode. Clears `_session_messages`, resets `_cache_stats`.

### `async _arun(self, messages, response_format=None, tools=None)`

The core TITO logic:

```python
async def _arun(self, messages, response_format=None, tools=None):
    if not self._session_messages:
        # First call in episode: store all messages (system + user prompt)
        self._session_messages = list(messages)
    else:
        # Extract new messages: scan backward to find last assistant msg
        new_msgs = self._extract_new_messages(messages)
        self._session_messages.extend(new_msgs)

    # Optional validation
    if self._validate:
        self._log_validation(messages)

    # Delegate to inner model's _arun (Moonshot-specific API preparation)
    response = await self._inner._arun(
        self._session_messages, response_format, tools
    )

    # Record raw response — NO serialization, use model_dump() to preserve everything
    raw_msg = self._capture_raw_response(response)
    self._session_messages.append(raw_msg)

    # Track cache stats
    self._update_cache_stats(response)

    return response
```

### `_extract_new_messages(self, messages)`

**Backward scan** until hitting the last assistant message. Everything after it = new tool results and user messages.

```python
def _extract_new_messages(self, messages):
    # Find last assistant message by scanning backward
    last_asst_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_asst_idx = i
            break

    if last_asst_idx == -1:
        # No assistant message found — shouldn't happen after first call
        # Fallback: reset and use all messages
        logger.warning("TITO: no assistant message found, resetting session")
        self._session_messages = []
        return messages

    # Everything after the last assistant = new tool/user messages
    new_msgs = messages[last_asst_idx + 1:]
    # Filter out any assistant messages (defensive)
    return [m for m in new_msgs if m.get("role") != "assistant"]
```

### `_capture_raw_response(self, response)`

Use `model_dump()` on the message object to preserve ALL fields — no manual dict construction, no risk of losing fields.

```python
def _capture_raw_response(self, response):
    choice = response.choices[0]
    # model_dump() preserves everything: content, tool_calls, reasoning_content, etc.
    msg = choice.message.model_dump(exclude_none=True)
    msg["role"] = "assistant"  # ensure role is explicit
    return msg
```

### `_update_cache_stats(self, response)`

Extract `cached_tokens` from `usage.prompt_tokens_details`:

```python
def _update_cache_stats(self, response):
    if not response.usage:
        return
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    self._cache_stats["total_prompt_tokens"] += prompt_tokens
    self._cache_stats["total_cached_tokens"] += cached
    self._cache_stats["per_turn_usage"].append({
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": completion_tokens,
    })
```

### `_log_validation(self, agent_messages)`

When `validate=True`, compare agent's messages with TITO's session messages. Log:
- Count mismatch
- Where agent lost `reasoning_content` that TITO preserved
- Role sequence alignment

### Property forwarding

For any property not defined on TITOChatModel (like `token_limit`, `model_config_dict` beyond what super().__init__ set), they are inherited from BaseModelBackend init which copies from inner model's values.

---

## Step 2: `seta_env/agent/tito_train_agent.py`

New file. `AgentTrainTITO` extends `AgentTrain` with:

### Differences from `AgentTrain`

1. **Reset propagation**: Override `reset()` to also call `self.model_backend.reset()` if available (clears TITO session for new episode)

2. **Cache stats extraction**: After the main loop in `_astep_non_streaming_task`, extract TITO cache stats into `meta_info_record`

3. **Per-turn cache logging**: In the per-iteration usage tracking, extract `cached_tokens` from `usage_dict`

### Implementation

```python
class AgentTrainTITO(AgentTrain):
    """AgentTrain with TITO model backend support and cache stats."""

    def reset(self):
        super().reset()
        # Reset TITO model session for new episode
        if hasattr(self, 'model_backend') and hasattr(self.model_backend, 'reset'):
            self.model_backend.reset()

    async def _astep_non_streaming_task(self, input_message, response_format=None):
        # Call parent implementation
        result = await super()._astep_non_streaming_task(input_message, response_format)

        # Extract TITO cache stats into meta_info_record
        if hasattr(self.model_backend, '_cache_stats'):
            stats = self.model_backend._cache_stats
            self.meta_info_record["cached_tokens"] = stats.get("total_cached_tokens", 0)
            total_prompt = stats.get("total_prompt_tokens", 1)
            self.meta_info_record["cache_hit_ratio"] = stats["total_cached_tokens"] / max(1, total_prompt)
            self.meta_info_record["per_turn_cache"] = stats.get("per_turn_usage", [])

        return result
```

Note: The `model_backend` attribute name needs to be verified against `ChatAgent.__init__`. It's set at line 529 of chat_agent.py: `self.model_type = self.model_backend.model_type`, confirming the attribute is `model_backend`.

---

## Step 3: Register in `seta_env/agent/prompt_loader.py`

Add to `_AGENT_REGISTRY`:

```python
_AGENT_REGISTRY: dict[str, str] = {
    "train_agent": "seta_env.agent.train_agent.AgentTrain",
    "tito_train_agent": "seta_env.agent.tito_train_agent.AgentTrainTITO",
}
```

---

## Step 4: Modify `seta_env/environments/terminal_env.py`

In `_reset_agent()` (around line 366-368), after model creation, wrap with TITO if configured:

```python
# Pop TITO keys before ModelFactory.create (they're not valid model kwargs)
tito_enabled = self.model_config.pop('tito_enabled', False)
tito_validate = self.model_config.pop('tito_validate', False)

model = self.model_config.get('model', None)
if model is None:
    from camel.models import ModelFactory
    model = ModelFactory.create(**self.model_config)

# Wrap with TITO model backend if enabled
if tito_enabled:
    from seta_env.models.tito_chat_model import TITOChatModel
    model = TITOChatModel(inner_model=model, validate=tito_validate)

# Existing logging setup (unchanged)
if model is not None:
    model._log_enabled = True
    model._log_dir = str(self.output_path / "CAMEL_LOG_DIR")
```

---

## Step 5: Config `scripts/evaluation/configs/eval_default_kimi_tito.yaml`

```yaml
terminal_env:
  model:
    model_platform: moonshot
    model_type: kimi-k2.5
    tito_enabled: true
    tito_validate: true  # set false after validation
    model_config_dict:
      max_tokens: 4096
      stream: false
      temperature: 1.0

  agent:
    agent: tito_train_agent   # <-- references new agent
    prompt: sys_prompt_base
    max_total_tokens: 28672
    max_completion_tokens: 4096
    max_iteration: 30
    tool_names:
      - shell_exec
      - shell_view
      - shell_wait
      - shell_write_to_process
      - shell_kill_process
      - shell_write_content_to_file
    thinking: true

  runtime:
    env_type: docker
    trial_root: ""
    toolkit: docker

  env:
    reward_fn: pass_ratio
    task_timeouts:
      _reset_env: 300.0
      _reset_agent: 120.0
      agent_astep: 900.0
      _evaluate_completion_sync: 600.0
      _cleanup: null

n_trajs: 1
workers: 16
seed: 42
dataset: seta-env-v2
output_dir: outputs/eval
experiment_name: eval_tito
trial_name: ""
rank: 0
world_size: 1
```

---

## Edge Cases

| Case | Behavior |
|------|----------|
| **First call** | Backward scan finds no assistant → `_session_messages = messages` (all) |
| **Parse error** (user error msg added) | Non-assistant, picked up after backward scan |
| **Image tool results** (extra user msgs) | Non-assistant, picked up naturally |
| **No tool calls** (final answer) | Response stored, loop exits, `reset()` on next episode |
| **Token limit / max iteration** | Loop exits, `reset()` on next episode |
| **History rewritten** (summarization) | Summarization is disabled (`summarize_threshold=None`). If enabled, backward scan still works — finds last assistant, takes everything after |

---

## Why This Design

- **Inherits `BaseModelBackend`**: Satisfies type checks, gets logging/token counting for free. Only overrides `_arun` — metaclass doesn't wrap it (only wraps `arun`/`run` in namespace)
- **Calls `inner._arun()` directly**: Bypasses inner's `InterleavedThinkingMixin.arun()` reasoning injection (unnecessary — TITO messages already have `reasoning_content`). Gets Moonshot-specific `_prepare_request()` (tool schema cleaning, etc.)
- **`model_dump(exclude_none=True)`**: Preserves ALL response fields without manual serialization. No risk of losing `reasoning_content` or any other field.
- **Backward scan**: Robust new-message detection. Doesn't depend on index tracking — just finds the boundary between known (assistant) and new (tool/user) messages.
- **Separate agent class**: `AgentTrainTITO` in its own file, registered in prompt_loader. Config selects it by name. Original `AgentTrain` untouched.

---

## Verification Plan

### 1. Run with validation on `stack_overflow__888`
```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default_kimi_tito.yaml \
    --tasks stack_overflow__888
```

### 2. Check CAMEL_LOG_DIR request logs
Compare request logs between TITO and non-TITO runs:
- TITO: assistant messages should have `reasoning_content` field
- Non-TITO: assistant messages should NOT have it

### 3. Check cache hit metrics in output
In trial output JSON, look at `meta_info_record`:
- `cached_tokens`: should be > 0 on turns 2+
- `cache_hit_ratio`: should increase as conversation grows
- `per_turn_cache`: list showing per-turn breakdown

### 4. Validation logs
With `tito_validate: true`, check logs for:
- "TITO: preserved reasoning_content at position X" (confirms reasoning retention)
- No "TITO validation: count mismatch" warnings (confirms message sync)

### 5. Compare rewards
Run same task with `train_agent` vs `tito_train_agent`. Verify task completion reward is not degraded (should be same or better since model sees its reasoning).
