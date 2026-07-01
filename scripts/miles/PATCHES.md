> **NOTE: V4-DOCKER SPECIFIC** — applies only to the DeepSeek-V4 image builds.

# Local Patches to miles + Megatron-LM (NOT in upstream)

These are surgical fixes applied **directly on each node's local filesystem**
(`/root/miles`, `/root/Megatron-LM`) — neither repo is a git checkout, so
changes here do NOT survive a container rebuild and must be re-applied + fanned
out to all 8 nodes via Ray after every fresh boot.

The companion Ray-fan-out helper lives at
[`scripts/miles/fan_out_node_patches.py`](fan_out_node_patches.py).

---

## Patch 1 — `fp8_simulate_qat(..., block_size=128)` → `(..., 128)`

**Symptom**
```
TypeError: apply() takes no keyword arguments
  File ".../miles_plugins/models/deepseek_v4/ops/v4_indexer.py", line 116
    q = fp8_simulate_qat(q, block_size=128)
  File ".../torch/autograd/function.py", line 581, in apply
    return super().apply(*args, **kwargs)
```

**Why**
`fp8_simulate_qat` is `DeepSeekV4LinearQATFunc.apply` (see
`miles_plugins/models/deepseek_v4/ops/qat.py:21`). It is a
`torch.autograd.Function.apply` bound method — PyTorch's
`Function.apply()` is a C-level dispatcher that **rejects every keyword
argument**. The call must be entirely positional.

Most call sites in the DSV4 codebase already pass `128` / `64` positionally
(compressor.py:164, 168; deepseek_v4.py:242). Two sites were missed.

**Fix — change in two files, BOTH require fan-out:**

| File | Line | Before | After |
|---|---|---|---|
| `/root/miles/miles_plugins/models/deepseek_v4/ops/v4_indexer.py` | 116 | `fp8_simulate_qat(q, block_size=128)` | `fp8_simulate_qat(q, 128)` |
| `/root/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py` | 900 | `self._fp8_simulate_qat(q, block_size=128)` | `self._fp8_simulate_qat(q, 128)` |

**Triggered only when** `MEGATRON_USE_KV_QAT=1` is in the runtime-env-json
passed to `ray job submit` (which it is, in
[`train_v4_flash_milesrouter_r3.sh`](train_v4_flash_milesrouter_r3.sh)). With
`MEGATRON_USE_KV_QAT=0` (default in some setups) neither call site fires.

---

## Patch 2 — `--max-weight-staleness` backport for fully-async training

**Upstream source**: `radixark/miles` GitHub (master branch). Backport
prompted by needing staleness control for `train_v4_flash_async_no_easy.sh`.

**What it adds**
- New CLI flag `--max-weight-staleness N`. When set, the async rollout
  worker discards any completed group whose oldest weight version lags
  the current SGLang engine version by more than `N`, recycles the
  group's samples back to the data buffer via `Sample.reset_for_retry()`,
  and logs `Staleness stats: recycled=…` summary at end of each rollout.
- `Sample.reset_for_retry()` instance method (restores prompt-identity
  fields to dataclass defaults so the sample can be re-sampled).
- `Sample.oldest_weight_version` property (min over `weight_versions`).
- `_CachedWeightVersion` helper that throttle-queries the SGLang router's
  `/model_info` endpoint for the current engine weight version.

**Fix — change in three files, ALL require fan-out:**

| File | Change | Sentinel `[PATCH 2]` comment? |
|---|---|---|
| `/root/miles/miles/utils/types.py` | Add `reset_for_retry()` method + `oldest_weight_version` property to `Sample` class (just above `update_from_meta_info`) | ✅ |
| `/root/miles/miles/utils/arguments.py` | Add `--max-weight-staleness` argparse entry just above `--custom-generate-function-path` | ✅ |
| `/root/miles/examples/fully_async/fully_async_rollout.py` | Replace entire file with upstream version (adds `_CachedWeightVersion`, `group_oldest_weight_version`, staleness branch in `generate_rollout_async`) | ✅ (file header) |

**Silent-fallback behavior**
If the SGLang server doesn't expose `weight_version` in `/model_info`
(or the request times out), `_cached_version.get()` returns `None`. The
filter then short-circuits via `if oldest is not None and
current_engine_version is not None`, and NO groups get filtered. You'll
see this in the rollout log: the `"Staleness filter enabled: …"` line
appears at start, but no `Staleness stats: recycled=…` line at end.
If that happens, either patch SGLang to expose `weight_version` or
drop the `--max-weight-staleness` flag.

**Compatibility**
- Default `--max-weight-staleness=None` → filter inactive → existing
  async runs unaffected.
- Sync rollout (`train.py`) doesn't use this file at all → also unaffected.

---

## Patch 3 — `--dynamic-sampling-filter-path` call site for fully-async

**Why**: Upstream `examples/fully_async/fully_async_rollout.py` (even the
version that ships Patch 2) does not invoke `args.dynamic_sampling_filter_path`
when iterating completed groups — it just does `data.append(group)`. That
means flags like `--dynamic-sampling-filter-path
miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
are LOADED into args but never CALLED in async mode. Sync rollout calls them
at [sglang_rollout.py:404-408](/root/miles/miles/rollout/sglang_rollout.py).

**What it adds** (in `/root/miles/examples/fully_async/fully_async_rollout.py`):
1. Imports `call_dynamic_filter` and `load_function`.
2. Loads the filter once at the top of `generate_rollout_async` (mirroring
   the sync code path).
3. Inserts a filter-evaluation branch AFTER the staleness branch and BEFORE
   `data.append(group)`. On `keep=False`, calls `reset_for_retry()` on every
   sample and recycles the group via `data_buffer.add_samples([group])`
   (same pattern as the staleness recycle, so the worker re-samples those
   prompts under fresh weights instead of the stale aborted ones).
4. End-of-rollout log: `Dynamic-filter stats: dropped=N reasons={…}` —
   poor-man's analog of sync's `MetricGatherer.on_dynamic_filter_drop`
   wandb metric.

**Fix — single-file change, fan-out required:**

| File | Change | Sentinel `[PATCH 3]` comment? |
|---|---|---|
| `/root/miles/examples/fully_async/fully_async_rollout.py` | Imports + filter-call branch in `generate_rollout_async` | ✅ (3 sites) |

**Compatibility**
- Default `--dynamic-sampling-filter-path=None` → no filter loaded → existing
  async runs unaffected (the new code path is gated on `dynamic_filter is not None`).
- Sync rollout doesn't use this file → unaffected.
- Composes with Patch 2's staleness filter (both can be active; staleness
  is evaluated first to avoid wasting a buffer slot on a recycle-then-drop).

**Recommended filter for our run**:
```
--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```
Drops zero-std groups (= every sample got the same reward → no learning
signal). The std-zero contribution is then ZERO from both compute AND
gradient, not just gradient.

---

## Patch 4 — Daytona sandbox leak fix (harbor)

**Symptom**
Persistent `"Failed to create sandbox: No available runners"` errors even when
well under the org's 256-sandbox cap. Errors sustained across the entire
run (1253+ in 90 min at 8×16 concurrency). Daytona API shows 242
error-state sandboxes consuming runner capacity.

**Root cause — sandbox leak cascade**
Three gaps in `harbor/environments/daytona.py` allowed sandboxes to leak:

1. `except Exception:` in `start()` and `_create_sandbox()` does NOT catch
   `asyncio.CancelledError` (a `BaseException` in Python 3.9+). Timeouts
   and cancellations skip all cleanup → sandbox leaked permanently.
2. `CreateSandboxFromSnapshotParams` (the primary production path) does not
   set `auto_stop_interval` or `auto_delete_interval`. Leaked sandboxes
   persist forever on Daytona with no GC — unlike `CreateSandboxFromImageParams`
   which sets `auto_delete_interval=0`.
3. `_create_sandbox` retry (5 attempts, max 30s backoff) causes thundering
   herd: 16 samples in a group all retry at the same instant, amplifying
   API pressure and causing more failures → more leaks.

Over time leaked sandboxes accumulate on Daytona runners, consuming capacity
until `"No available runners"` starts firing even at low concurrency.

**Fix — single file, does NOT require fan-out (env_service-side only):**

| File | Change |
|---|---|
| `/root/data/terminal_agent/external/harbor/src/harbor/environments/daytona.py` | See below |

Changes:
- `except Exception:` → `except BaseException:` in both `_create_sandbox()` and `start()`
  — ensures `CancelledError` triggers orphan cleanup instead of silently leaking
- Add `auto_stop_interval=15, auto_delete_interval=5` to `CreateSandboxFromSnapshotParams`
  — safety net: even if all cleanup code fails, Daytona auto-destroys the sandbox within 20 min
- Import `wait_random` from tenacity
- `_create_sandbox`: `stop_after_attempt(5)` → `stop_after_attempt(10)`,
  `wait_exponential(multiplier=2, min=2, max=30)` → `wait_exponential(..., max=60) + wait_random(0, 15)`
  — per-sample jitter breaks thundering herd; longer retry window covers runner provisioning
- Same retry change for `_create_snapshot_with_retry` inside `build()`

**Result**
| Metric | Before fix | After fix |
|---|---|---|
| "No available runners" errors | 1253 (8×16) / 2923 (12×16) | **0** |
| Peak concurrent Daytona sandboxes | 75 | **192** (full saturation) |
| Dynamic-filter zero_std_0.0 drops (env-failure noise) | 1270 | **0** |
| Runners assigned by Daytona | 16 | **35+** |

**Compatibility**
- No miles changes required. All fixes are in harbor's Daytona backend.
- `auto_stop_interval=15` means a sandbox idles for max 15 min before
  auto-stop. Trajectories that run >15 min are fine — sandbox stays alive
  while actively executing; the timer only counts idle time after the last
  API call.

---

## Patch 5 — `kl_loss` coef-0 guard (NaN-grad crash fix)

**Symptom**
```
RuntimeError: Rank N ... Unexpected result nan
  (found NaN in local grad norm for bucket #0 in backward pass ...)
  .../megatron/core/distributed/param_and_grad_buffer.py:225 in check_grads
```
Training (camel seta, `--use-kl-loss --kl-loss-coef 0.00`) crashed after ~12 steps
once the policy drifted from base (kl≈0.1, train_rollout_logprob_abs_diff≈0.07).

**Why**
`miles/backends/training_utils/loss.py` added the kl term **unconditionally**:
`loss = loss + args.kl_loss_coef * kl_loss`. At coef 0 this is `loss + 0.00*kl_loss`,
which keeps the kl branch in the autograd graph. `compute_approx_kl` (low_var_kl)
does `kl = exp(-log_ratio) - 1 + log_ratio` then `clamp(kl, -10, 10)`: the *value*
is finite, but in the **backward** the clamp grad is 0 for a clamped element while
the upstream grad through `exp()` is Inf → `0*Inf = NaN` grad, then `0.00*NaN = NaN`
into the params. `--bf16` has no optimizer NaN-skip, so the deterministic
`check_for_nan_in_grad` raised. (The kl term is 0-weighted, so it should never have
touched the gradient at all.)

**Fix — single file, fan-out required:**

| File | Change | Sentinel `[PATCH 5]`? |
|---|---|---|
| `/root/miles/miles/backends/training_utils/loss.py` | Guard the loss addition: `if args.kl_loss_coef != 0:` before `loss = loss + args.kl_loss_coef * kl_loss` | ✅ |

`kl_loss` is still computed and still logged (`reported_loss["kl_loss"]`), so the
`train/kl_loss` wandb metric is unchanged — only the 0-weighted NaN is kept out of
the gradient. Mathematically identical training; no step is skipped.

**Compatibility**
- coef != 0 (a real KL penalty) → behaves exactly as before.
- Composes with everything; no arg needed (the launcher just keeps `--use-kl-loss`).
- Added to `fan_out_node_patches.py` PATCHED_FILES; one-off helper
  `_fan_out_patch5_klguard.py`.

---

## Re-applying after a container rebuild

Run the fan-out helper:
```bash
python3 /data/terminal_agent/scripts/miles/fan_out_node_patches.py
```

Or, ad-hoc per-patch via Ray (see commit history for examples).
