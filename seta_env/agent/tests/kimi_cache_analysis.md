# Kimi K2.5 KV Cache Analysis with TITO

Results from running `test_tito_kimi.py` on task `stack_overflow__888` with 10 iterations.

## Per-Turn Cache Stats

| Turn | Prompt | Cached | Hit% | Prev Total | floor(256) | Match? |
|------|--------|--------|------|------------|------------|--------|
| 1 | 1,257 | 0 | 0.0% | cold | - | cold start |
| 2 | 1,616 | 1,024 | 63.4% | 1,324 | 1,280 | ~256-block |
| 3 | 1,809 | 1,536 | 84.9% | 1,677 | 1,536 | exact |
| 4 | 2,208 | 1,792 | 81.2% | 1,894 | 1,792 | exact |
| 5 | 2,572 | 2,048 | 79.6% | 2,261 | 2,048 | exact |
| 6 | 2,721 | 2,560 | 94.1% | 2,691 | 2,560 | exact |
| 7 | 3,252 | 2,560 | 78.7% | 2,780 | 2,560 | exact |
| 8 | 3,692 | 3,072 | 83.2% | 3,381 | 3,328 | ~256-block |
| 9 | 4,183 | 3,584 | 85.7% | 3,818 | 3,584 | exact |
| 10 | 4,423 | 4,096 | 92.6% | 4,274 | 4,096 | exact |
| **Total** | **27,733** | **22,272** | **80.3%** | | | |

**Avg cache hit on turns 2+: 82.6%**

## Additional Findings

- **`reasoning_content` preserved on all 10 turns** — Kimi natively returns this field and TITO's `model_dump()` captures it, unlike the base agent which drops it during `_record_assistant_tool_calls_from_requests`.
- **`cached_tokens` reported at two locations** — both `usage.cached_tokens` (top-level) and `usage.prompt_tokens_details.cached_tokens`. The OpenAI Python SDK only parses `prompt_tokens_details`, but Kimi also provides the top-level field.

## Why Cache Hit Isn't 100%

**It's a Kimi-side implementation detail, not a problem with our TITO implementation.**

Kimi caches KV states in fixed **256-token blocks**. When reporting `cached_tokens`, only **complete blocks** from the previously-computed prefix are counted.

```
Turn N prompt tokens:

[====== cached (complete 256-token blocks) ======][= uncached tail =]
[blk 1][blk 2][blk 3] ... [blk K]                [partial + new tokens]
                                                   ↑ always exists unless
                                                     prompt is exactly
                                                     block-aligned
```

Two sources of uncached tokens each turn:

1. **Block alignment tail** — tokens between the last 256-token block boundary and the end of the previous turn's output. For example, if the previous turn produced 2,691 total tokens, only the first `floor(2691/256) × 256 = 2,560` are cached. The remaining 131 tokens fall in an incomplete block.

2. **New tokens** — each turn appends new tool result tokens that have never been computed before. These always require fresh KV computation.

## How We Know TITO Prefix Is Stable

The proof that our TITO implementation is correct:

1. **`cached_tokens` grows monotonically** — 0 → 1024 → 1536 → 1792 → 2048 → 2560 → 2560 → 3072 → 3584 → 4096. If the message prefix were changing between turns (breaking TITO), cached would drop to 0.

2. **`cached_tokens ≈ floor(prev_total / 256) * 256`** — matches on 7/9 non-cold turns exactly, with the remaining 2 within one block. This means Kimi is caching exactly as many complete blocks as the previous turn's total length allows.

3. **Cache hit ratio increases as conversation grows** — later turns have proportionally more cached prefix relative to new tokens, reaching 94.1% on turn 6 and 92.6% on turn 10.

## Proof Method: Block Alignment Verification

To verify TITO is working, check that `cached_tokens` on turn N equals the previous turn's total tokens rounded down to the nearest 256-token block:

```python
prev_total = prompt_tokens[N-1] + completion_tokens[N-1]
expected_cached = (prev_total // 256) * 256
assert cached_tokens[N] == expected_cached  # ± 1 block tolerance
```

Worked example with Turn 6:

```
Turn 5 finished with:
  prompt_tokens  = 2,572
  completion     = 119
  total          = 2,691

Turn 6 arrives with:
  prompt_tokens  = 2,721  (2,691 from prev + 30 new tool result tokens)
  cached_tokens  = 2,560

Verify: floor(2,691 / 256) × 256 = 10 × 256 = 2,560 ✓

Breakdown:
  [0..2560)     → 10 complete 256-token blocks → CACHED (from turn 5)
  [2560..2691)  → 131 tokens in partial block  → recomputed
  [2691..2721)  → 30 new tool result tokens    → never seen before
```

If TITO were broken (prefix changing between turns), Kimi would see a different token sequence and `cached_tokens` would be 0 or near-0, not `floor(prev_total / 256) * 256`.

### Reproducing

```bash
# Run the test
MOONSHOT_API_KEY=sk-... python seta_env/agent/tests/test_tito_kimi.py

# Check results
cat outputs/test_tito_kimi/kimi_cache_results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
turns = data['per_turn']
prev_total = 0
for i, t in enumerate(turns):
    if i > 0:
        expected = (prev_total // 256) * 256
        actual = t['cached_tokens']
        ok = abs(actual - expected) <= 256
        print(f'Turn {i+1}: expected~={expected} actual={actual} {\"OK\" if ok else \"MISMATCH\"} ')
    prev_total = t['prompt_tokens'] + t['completion_tokens']
"
```

## Comparison: Without TITO

Without TITO, the base `AgentTrain` reconstructs assistant messages from scratch each turn via `_record_assistant_tool_calls_from_requests`, which:
- Drops `reasoning_content`
- Sets `content = ""` (losing thinking text)
- Rebuilds `tool_calls` from `ToolCallRequest` objects

This changes the message content between turns, breaking the prefix stability that KV caching requires. We would expect `cached_tokens = 0` on most turns without TITO, because the message at position N would differ from what was sent in the previous request.
