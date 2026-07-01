# TITO: Token-In Token-Out Pipeline

End-to-end documentation of the TITO (Token-In Token-Out) multi-turn agentic workflow, from the camel agent through the model backend to the miles training session.

## Overview

TITO preserves the model's raw output token IDs across turns so that RL training uses the exact tokens the model generated — no retokenization of assistant content. The key insight: assistant tokens (thinking + tool calls) are the training signal; everything else (tool results, system prompts) is masked during loss computation, so retokenizing those is harmless.

```
Agent (AgentTrainTITO)     TITOChatModel          Miles Session Server       sglang
─────────────────────      ─────────────          ────────────────────       ──────
manages memory,            maintains raw          validates messages,        serves model,
tool execution             session messages       accumulates token IDs      returns token IDs
                           (model_dump)           (LinearTrajectory)         + logprobs
```

---

## Complete Multi-Turn Workflow

### Turn 1: Cold Start

#### Step 1 — Agent prepares messages

**`AgentTrain._astep_non_streaming_task`** (`seta_env/agent/train_agent.py:340`)

The agent enters its main loop. On each iteration:

```python
openai_messages, num_tokens = await self._get_context_with_summarization_async()
# → [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
```

The agent's memory serializes all recorded messages into OpenAI format.

#### Step 2 — Agent calls model

**`AgentTrain._aget_model_response`** (`external/camel/camel/agents/chat_agent.py:3652`)

```python
response = await self.model_backend.arun(openai_messages, response_format, tool_schemas)
```

#### Step 3 — Metaclass preprocessing (bypassed by TITO)

**`BaseModelBackend.arun`** (`external/camel/camel/models/base_model.py:903`)

The `ModelBackendMeta` metaclass wraps `arun` with pre/post processing:

```python
messages = self.preprocess_messages(messages)   # strips <think> tags normally
result = await self._arun(messages, ...)
result = self.postprocess_response(result)      # extracts <think> → reasoning_content
```

**`TITOChatModel.preprocess_messages`** overrides this to be a **no-op** — TITO keeps raw `<think>` tags in messages so the model sees its own prior reasoning.

#### Step 4 — TITO session management

**`TITOChatModel._arun`** (`seta_env/models/tito_chat_model.py:68`)

```python
if not self._session_messages:
    # First call: store all messages
    self._session_messages = [dict(m) for m in messages]
else:
    # Subsequent: extract only NEW messages (tool results)
    new_msgs = self._extract_new_messages(messages)
    self._session_messages.extend(new_msgs)

response = await super()._arun(self._session_messages, ...)

# Store raw response via model_dump() — preserves everything
raw_msg = self._capture_raw_response(response)
self._session_messages.append(raw_msg)
```

Key: `_session_messages` is the source of truth sent to the API. The agent's memory may lose fields (e.g., `reasoning_content`), but TITO's copy is lossless.

#### Step 5 — OpenAI-compatible API call

**`OpenAICompatibleModel._arun`** (`external/camel/camel/models/openai_compatible_model.py:253`)

```python
return await self._async_client.chat.completions.create(
    messages=messages, model=self.model_type, **request_config
)
```

#### Step 6 — Miles session server intercepts (production)

**`sessions.py:104` `chat_completions()`** (`miles/rollout/session/sessions.py`)

In production, the agent's HTTP request goes through the miles session server, not directly to sglang:

**Phase 1 — Prepare (lock held briefly):**
```python
request_messages = body["messages"]

# Inject sglang flags for token tracking
body["logprobs"] = True
body["return_prompt_token_ids"] = True
body["return_meta_info"] = True
body["no_stop_trim"] = False

# TITO pretokenization
pretokenized = session.prepare_pretokenized(
    request_messages, tools=body.get("tools"),
    tito_tokenizer=registry.tito_tokenizer,
)
if pretokenized is not None:
    body["input_ids"] = pretokenized["input_ids"]
```

**Phase 2 — Proxy to sglang (no lock):**
```python
result = await backend.do_proxy(request, "v1/chat/completions", body=body)
```

**Phase 3 — Update state (lock held briefly):**
```python
prompt_token_ids = choice["prompt_token_ids"]           # from sglang
completion_token_ids = [t[1] for t in output_token_logprobs]  # REAL model output

session.update_pretokenized_state(
    request_messages, assistant_message,
    prompt_token_ids=prompt_token_ids,
    completion_token_ids=completion_token_ids,
    max_trim_tokens=tito_tokenizer.max_trim_tokens,
)
```

#### Step 7 — sglang generates

sglang receives `input_ids` (if pretokenized) or `messages` (first turn). With `--tool-call-parser qwen25`, it:

1. Generates raw text: `<think>...\n</think>\n\n<tool_call>\n{"name":"shell_exec",...}\n</tool_call>`
2. **`Qwen25Detector.detect_and_parse`** (`sglang/srt/function_call/qwen25_detector.py:48`) — parses `<tool_call>` blocks via regex + JSON
3. Returns `ChatCompletion` with `message.content` (thinking) + `message.tool_calls` (parsed calls)
4. With miles flags: also returns `prompt_token_ids` and `meta_info.output_token_logprobs`

#### Step 8 — TITO stores raw response

**`TITOChatModel._capture_raw_response`** (`seta_env/models/tito_chat_model.py:114`)

```python
msg = choice.message.model_dump(exclude_none=True)
msg["role"] = "assistant"
if "tool_calls" in msg:
    for tc in msg["tool_calls"]:
        tc.pop("index", None)   # remove sglang-specific field
self._session_messages.append(msg)
```

`model_dump()` preserves ALL fields — `content`, `tool_calls`, `reasoning_content` — unlike the agent's `_record_assistant_tool_calls_from_requests` which creates a new BaseMessage and loses `reasoning_content`.

#### Step 9 — Agent executes tools

**`AgentTrain._astep_non_streaming_task`** (`seta_env/agent/train_agent.py:459-517`)

```python
if tool_call_requests := response.tool_call_requests:
    # Record assistant message in agent memory (⚠️ loses reasoning_content)
    self._record_assistant_tool_calls_from_requests(tool_call_requests, content=response_content)

    # Execute each tool
    for tool_call_request in internal_tool_requests:
        tool_call_record = await self._aexecute_tool(tool_call_request)
        # → calls TerminalToolkit.shell_exec(id, command, block)
        # → docker exec ... bash -c "command"
        # → records tool result in agent memory as {role: "tool", content: output}
```

#### Step 10 — Loop continues

The agent loops back to Step 1. On the next iteration:

- Agent memory has: `[sys, user, assistant₁(stripped), tool₁_result]`
- TITO session has: `[sys, user, assistant₁(RAW with <think>), tool₁_result]`

TITO sends its own session messages to the API, so the model sees its full prior reasoning.

---

### Turn 2+: Pretokenized Prefix Reuse

On subsequent turns, the miles session does incremental tokenization:

#### `prepare_pretokenized` flow

**`LinearTrajectory.prepare_pretokenized`** (`miles/rollout/session/linear_trajectory.py:64`)

1. **Rollback detection** (`_try_detect_and_rollback_to_assistant_checkpoint`): checks if the agent is retrying from an earlier point
2. **`assert_messages_append_only_with_allowed_role`**: validates stored prefix matches, only `tool` role messages appended
3. **`Qwen3TITOTokenizer.merge_tokens`**: computes incremental tokens and merges with prefix

#### `Qwen3TITOTokenizer.merge_tokens` flow

**`tito_tokenizer.py:192`**

```python
def merge_tokens(self, old_messages, new_messages, pretokenized_token_ids, tools):
    incremental = self.tokenize_additional_non_assistant(old_messages, new_messages, tools)
    prefix = list(pretokenized_token_ids)
    if prefix and prefix[-1] == self._im_end_id:
        prefix.append(self._newline_id)   # ← boundary fix
    return prefix + incremental
```

**The boundary fix**: Qwen3 chat template emits `<|im_end|>\n` after every message, but the model stops at `<|im_end|>` without generating the trailing `\n`. `merge_tokens` inserts it.

#### `tokenize_additional_non_assistant` flow

**`tito_tokenizer.py:90`**

Uses a dummy-message diff to compute the exact token IDs for the tool result boundary:

```python
appended_messages = new_messages[len(old_messages):]      # [tool₁]
dummy_assistant = _build_dummy_assistant(appended_messages) # matching tool_call_ids
base = [dummy_user, dummy_assistant]

tokens_without = tokenize(base, add_generation_prompt=False)
tokens_with = tokenize(base + appended_messages, add_generation_prompt=True)
incremental = tokens_with[len(tokens_without):]
# → tokens for: \n<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n
```

#### Result: what gets sent to sglang

```
input_ids = [REAL_tokens_from_prev_turns] + [\n] + [retokenized_tool_boundary + content + gen_prompt]
             ├─ from checkpoint (real)──┤    ↑       ├─── computed by HF tokenizer ───────────────┤
                                        inserted by merge_tokens
```

sglang receives `input_ids` directly, **skips tokenization**, and its radix cache matches the prefix → cache hit.

---

## What's REAL vs Retokenized

```
checkpoint = [sglang_prompt₁ | MODEL_completion₁ | \n | HF_tool₁ | MODEL_completion₂ | ...]
              ├── masked ────┤  ├── TRAIN ───────┤       ├ masked ┤  ├── TRAIN ───────┤
```

| Token source | How obtained | TITO? | Used in training loss? |
|---|---|---|---|
| **Assistant content** (thinking + tool calls) | `completion_token_ids` from sglang `output_token_logprobs` | **True TITO** — real model output | **Yes** — exact tokens + logprobs |
| **Non-assistant content** (tool results, system) | Retokenized via `tokenize_additional_non_assistant` using HF tokenizer | Retokenized | **No** — masked during training |
| **Boundary token** (`\n` after `<|im_end|>`) | Inserted by `Qwen3TITOTokenizer.merge_tokens` | Inserted | **No** — structural |

---

## `compute_session_mismatch` — Diagnostic Check

**`SessionRegistry.compute_session_mismatch`** (`miles/rollout/session/linear_trajectory.py:275`)

A read-only sanity check that compares the accumulated TITO token IDs against a fresh `apply_chat_template` retokenization of `session.messages`.

Uses **`TokenSeqComparator`** (`miles/utils/chat_template_utils/token_seq_comparator.py`) which:
1. Segments both sequences by special tokens (`<|im_start|>`, `<|im_end|>`)
2. Compares aligned segments
3. Classifies mismatches as `ASSISTANT_TEXT` (expected) or `NON_ASSISTANT_TEXT` (bug)

### Known `ASSISTANT_TEXT` mismatches (expected, harmless)

**Qwen3 `<think>` tag injection**: The Qwen3 chat template conditionally adds `<think>\n\n</think>\n\n` before tool calls in assistant messages. However:

- The template only adds it to assistant messages **following a tool response**, not the first assistant after the user message
- The model **always** generates `<think>...</think>` regardless of position
- Result: for some assistant turns, the REAL tokens contain `<think>\n</think>\n\n` but the canonical retokenization doesn't

This is harmless because:
1. Classified as `assistant_text` by the comparator (correctly)
2. The real tokens are what get used for training
3. The retokenization is only for validation, never for training

**JSON argument serialization**: The model may serialize tool call arguments with different whitespace/key ordering than `json.dumps(json.loads(args))`. Same semantic content, different byte representation → different tokens.

---

## Files

| File | Description |
|------|-------------|
| `seta_env/models/tito_chat_model.py` | `TITOChatModel` — inherits `OpenAICompatibleModel`, maintains raw session messages |
| `seta_env/agent/tito_train_agent.py` | `AgentTrainTITO` — extends `AgentTrain` with reset propagation + cache stats |
| `seta_env/agent/prompt_loader.py` | Registry — maps `"tito_train_agent"` → `AgentTrainTITO` |
| `seta_env/environments/terminal_env.py` | Creates `TITOChatModel` when `tito_enabled: true` in config |
| `seta_env/agent/tests/test_tito_e2e.py` | End-to-end test using miles `LinearTrajectory` + `Qwen3TITOTokenizer` |

### Miles dependencies (used for validation, not modified)

| File | Description |
|------|-------------|
| `miles/rollout/session/linear_trajectory.py` | `LinearTrajectory` — session state, rollback, prefix invariant check |
| `miles/rollout/session/sessions.py` | Session server — HTTP proxy between agent and sglang |
| `miles/utils/chat_template_utils/tito_tokenizer.py` | `Qwen3TITOTokenizer` — incremental tokenization with `\n` boundary fix |
| `miles/utils/chat_template_utils/token_seq_comparator.py` | `TokenSeqComparator` — segment-by-segment token comparison |
| `miles/utils/chat_template_utils/template.py` | `apply_chat_template`, `message_matches`, `assert_messages_append_only_with_allowed_role` |
