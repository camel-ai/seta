# TODO

- [x] **Fix `max_tokens` misconfiguration causing `RuntimeError: max_tokens must be greater than the number of prompt tokens`**
  - **Error:** `RuntimeError: max_tokens must be greater than the number of prompt tokens` raised in `external/areal/areal/experimental/openai/client.py:138` when `max_tokens - len(prompt_token_ids) <= 0`.
  - **Cause:** In `scripts/areal/eval.py`, `model_config_dict` passes `max_tokens=max_tokens_per_trajectory` (28672). The areal client interprets `max_tokens` as the total context budget (prompt + completion), so once accumulated tool outputs push the prompt beyond 28672 tokens the budget is exhausted and the call crashes. The actual model context window (`sglang.context_length`) is 32768, so headroom exists but is never used.
  - **Fix:** Either (a) set `max_tokens` to `sglang.context_length` (32768) instead of `max_tokens_per_trajectory`, since `max_completion_tokens=gconfig.max_new_tokens` (4096) already caps the per-turn output; or (b) remove `max_tokens` from `model_config_dict` entirely and rely solely on `max_completion_tokens`, adding graceful truncation of the conversation history before the call when prompt length approaches the context limit.

- [ ] **Fix `task_timeouts` not enforced — `2_run_agent` stage runs unconstrained**
  - **Error:** Log shows `2_run_agent: 905.52s` despite `task_timeouts.agent_astep: 300.0` in config.
  - **Cause:** In `environments/terminal_env.py`, `step()` wraps each stage with `async_timer` for measurement only. The `task_timeouts` from `env_config` is stored but never applied — there is no `asyncio.wait_for()` around any stage call, so all timeouts (`_reset_env`, `agent_astep`, `_evaluate_completion_sync`) are silently ignored.
  - **Fix:** In `step()`, read `task_timeouts` from `self.env_config` and wrap each stage coroutine with `asyncio.wait_for(..., timeout=...)` using the corresponding attribute (`_reset_env` → stage 1, `agent_astep` → stage 2, `_evaluate_completion_sync` → stage 3). Catch `asyncio.TimeoutError` separately from general exceptions so the stage name and configured timeout are logged clearly before falling through to cleanup.
