# Agent Notes

Design notes and open work items for the training/eval agents under
[`seta_env/agent/`](../seta_env/agent/). Implementation details and the
turn-by-turn message-flow diagrams live in
[`seta_env/agent/README.md`](../seta_env/agent/README.md).

## Open issues / TODO

### Memory-retaining chat agent

The current `tito_train_agent` and its corresponding sglang model backend
prune everything except tool calls and tool-call results from the model's
context across turns. After a model response, only the tool calls and
tool-call results are retained — all other content (including reasoning
text) is dropped.

For workflows that benefit from carrying reasoning forward (chain-of-thought
continuity, self-correction, multi-step planning that doesn't go through
tools), we want a parallel chat-agent variant that retains the full
assistant response in memory rather than pruning to tool calls only.

**Tasks:**

- Audit how `tito_train_agent` and the sglang `AReaLOpenAICompatibleModel`
  build the per-turn message list, and identify the exact pruning step.
- Implement a sibling agent (e.g. `chat_train_agent`) that retains
  full assistant content across turns.
- Decide whether the pruning vs. retaining behavior should be a flag on a
  single agent class or two separate classes — flag is simpler, two
  classes makes the contract explicit.
- Verify that the sglang backend's `apply_chat_template` path handles the
  retained-content format correctly (this is the existing `! TODO: check
  apply chat template` note in the agent README).
