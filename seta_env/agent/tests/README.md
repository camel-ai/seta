# TITO End-to-End Test

## What it tests

`test_tito_e2e.py` runs a full TITO agent session against a real sglang server and validates every turn against miles' training pipeline invariants.

## How it works

The test mocks the OpenAI async client inside `TITOChatModel` to intercept each chat completion call. Instead of going through `/v1/chat/completions`, each call follows the exact miles session server flow:

```
Agent calls model.arun(messages)
  → TITOChatModel._arun (manages session messages)
    → OpenAICompatibleModel._arun
      → self._async_client.chat.completions.create(...)  ← MOCKED
        │
        ▼
    mock_chat_create():
        │
        ├─ 1. MilesSessionTracker.prepare_request()
        │     └─ LinearTrajectory.prepare_pretokenized()
        │        └─ Qwen3TITOTokenizer.merge_tokens()
        │           └─ inserts \n after <|im_end|>, computes incremental tool tokens
        │     → returns pretokenized input_ids
        │
        ├─ 2. POST sglang /generate (input_ids=..., return_logprob=True)
        │     → REAL output token IDs via output_token_logprobs
        │     → REAL cached_tokens count
        │     → raw text (not tool-parsed)
        │
        ├─ 3. FunctionCallParser + Qwen25Detector
        │     └─ same parser sglang uses with --tool-call-parser qwen25
        │     → parses <tool_call>...</tool_call> blocks from raw text
        │     → extracts normal_text (content) + tool call items
        │
        ├─ 4. MilesSessionTracker.process_response()
        │     └─ LinearTrajectory.update_pretokenized_state()
        │        └─ prefix invariant check with REAL token IDs
        │     → stores checkpoint
        │
        └─ 5. Constructs ChatCompletion (with tool_calls)
              → returns to agent as if it came from /v1/chat/completions
```

## What it validates per turn

| Check | Miles function | What it verifies |
|-------|---------------|-----------------|
| `prepare_pretokenized` | `LinearTrajectory.prepare_pretokenized()` | `message_matches` (stored prefix = request prefix), `assert_messages_append_only_with_allowed_role` (only tool msgs appended), `Qwen3TITOTokenizer.merge_tokens` (boundary `\n` insertion) |
| `update_pretokenized_state` | `LinearTrajectory.update_pretokenized_state()` | Token prefix invariant: accumulated REAL token IDs are a prefix of new prompt + completion |
| `final_mismatch` | `TokenSeqComparator.compare_sequences()` | Full sequence comparison — only `assistant_text` diffs allowed (real model tokens may differ from canonical retokenization) |

## Prerequisites

```bash
# 1. sglang serving Qwen3-8B with tool call parser
bash scripts/evaluation/start_sglang.sh

# 2. Docker daemon running
docker ps
```

## Running

```bash
cd /home/ubuntu/terminal_agent
conda run -n terminal_agent python seta_env/agent/tests/test_tito_e2e.py
```

## Output

```
Turn 1: update_pretokenized_state  PASS (checkpoint 1, 2347 token_ids)
  [sglang] prompt=1494 completion=853 cached=1493 tool_calls=1

Turn 2: prepare_pretokenized       PASS (2423 input_ids)
Turn 2: update_pretokenized_state  PASS (checkpoint 2, 3211 token_ids)
  [sglang] prompt=2423 completion=788 cached=2347 tool_calls=1
...

FINAL MILES SESSION MISMATCH CHECK
  final_mismatch  PASS (N assistant-only diffs — expected, harmless)

RESULTS: 20/20 passed, 0/20 failed
ALL CHECKS PASSED — true TITO with real token IDs
```

The `cached` field shows sglang's radix cache is reusing the TITO prefix across turns.
