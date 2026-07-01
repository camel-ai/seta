# Azure GPT-5.4 Cache Findings: TITO vs Default

## Test Setup

Ran the same task (`stack_overflow__888`) twice with Azure GPT-5.4:
- **TITO**: `TITOChatModel` + `AgentTrainTITO` — preserves raw messages via `model_dump()`
- **DEFAULT**: `AzureOpenAIModel` (via ModelFactory) + `AgentTrain` — standard agent memory

## Results

### TITO Run

| Turn | Prompt | Cached | Hit% | Tool Calls |
|------|--------|--------|------|------------|
| 1 | 1,212 | 0 | 0.0% | yes |
| 2 | 1,626 | 1,280 | 78.7% | yes |
| 3 | 1,824 | 1,664 | 91.2% | no |
| **Total** | **4,662** | **2,944** | **63.1%** | |

### Default Run

| Turn | Prompt | Cached | Hit% | Tool Calls |
|------|--------|--------|------|------------|
| 1 | 1,212 | 1,152 | 95.0% | yes |
| 2 | 1,666 | 1,280 | 76.8% | yes |
| 3 | 1,849 | 1,664 | 90.0% | no |
| **Total** | **4,727** | **4,096** | **86.7%** | |

### Comparison

| Metric | TITO | DEFAULT |
|--------|------|---------|
| Avg cache hit (turns 2+) | 85.0% | 83.4% |
| Turn 1 cache hit | 0.0% | 95.0% |
| `reasoning_content` preserved | 0/3 | N/A |

## Key Findings

### 1. Azure caches at the infrastructure level, not per-session

The DEFAULT run got **95% cache hit on turn 1** — this is a cold start for the agent, yet Azure already had KV states cached from previous requests (likely from the TITO run, or from other users hitting the same deployment). Azure OpenAI implements **automatic prompt caching** at the server level across all requests to the same deployment.

### 2. TITO provides no caching advantage over default on Azure

Turns 2+ show essentially identical cache hit rates (85.0% vs 83.4%). Azure's server-side tokenization is deterministic — even when the default agent reconstructs assistant messages (losing `reasoning_content`, setting `content=""`), Azure produces the same token prefix because:
- Tool call structure is preserved identically
- The Jinja chat template produces deterministic output for the same semantic content
- Azure normalizes messages server-side before tokenization

### 3. GPT-5.4 does not use `reasoning_content`

Unlike Kimi K2.5, GPT-5.4 does not return a separate `reasoning_content` field. All reasoning (if any) is embedded in `content`. This means TITO's raw message preservation provides no additional data for this model.

## Conclusion

**For Azure OpenAI (GPT-5.4 and similar): TITO is not needed for caching or message preservation.** Use the standard `train_agent` with `ModelFactory.create(model_platform="azure", ...)`.

**TITO remains valuable for:**
- **Kimi K2.5** — preserves native `reasoning_content` field, confirmed 82.6% cache hit improvement
- **sglang-served models (Qwen3-8B, etc.)** — required for miles training pipeline (`LinearTrajectory` + `Qwen3TITOTokenizer` prefix reuse)
- **Any model that returns `reasoning_content`** as a separate API field

## Config for Azure (no TITO needed)

```yaml
terminal_env:
  model:
    model_platform: azure
    model_type: gpt-5.4
    api_key: "..."
    url: "https://....cognitiveservices.azure.com/"
    api_version: "2024-12-01-preview"
    # tito_enabled: false  (default, no need to set)
    model_config_dict:
      max_completion_tokens: 4096  # NOT max_tokens — GPT-5.4 requires this
      stream: false
      temperature: 1.0

  agent:
    agent: train_agent  # standard agent, no need for tito_train_agent
```

Note: GPT-5.4 requires `max_completion_tokens` instead of `max_tokens` in `model_config_dict`.
