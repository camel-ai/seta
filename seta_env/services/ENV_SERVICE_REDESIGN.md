# Env Service — Robustness Redesign Plan

**Status:** Proposal · **Scope:** `seta_env/services/{env_service,env_scheduler}.py` and the
`TerminalEnvironment` / `DockerHarborRuntime` lifecycle they drive · **Author:** redesign review

This document is a *plan only* — no code is changed. It (1) states what the env
service is for, (2) catalogs the concrete robustness failures in the current
implementation with code evidence, (3) lays out the target architecture an
industry-grade sandbox-execution control plane would use, and (4) gives a
phased, low-risk migration path.

---

## 1. What the env service is for

The env service is the **remote execution backend for one RL rollout step (a full
agent trajectory)**. The training engine (areal proxy / `grpo_rollout.py`) does not
run environments itself; it ships a thin task payload to the service and gets back a
reward.

One `POST /step` does the entire trajectory lifecycle:

```
/step request  ──►  resolve task_dir + model endpoint (sglang URL/key)
                ──►  BUILD gate     : ensure image/snapshot exists (single-flight per task)
                ──►  CREATE         : spin up sandbox (Daytona/Docker/Modal) + toolkit
                ──►  RESET AGENT    : model backend + agent + tools
                ──►  RUN            : agent.astep() — the trajectory
                ──►  EVALUATE       : verifier on final sandbox state
                ──►  REWARD         : reward_fn(eval, trajectory, agent_info)
                ──►  CLEANUP        : tear down sandbox
                ──►  return {run_info, reward, sample}
```

Two-tier topology (mirrors `slot_pool_service`):

```
training machine                      CPU node 1            CPU node N
┌───────────────────┐  /step    ┌──────────────────┐    ┌──────────────────┐
│ areal proxy /     │ ────────► │  env_service     │ …  │  env_service     │
│ grpo_rollout      │           │  MAX_SLOTS slots │    │  MAX_SLOTS slots │
│                   │           │  BuildGate       │    │  BuildGate       │
│ env_scheduler     │ ◄──────── │  step executor   │    │  step executor   │
│ (affinity + LB)   │  reward   └────────┬─────────┘    └────────┬─────────┘
└───────────────────┘                    │ sandboxes              │
                                         ▼                        ▼
                              Daytona / Docker / Modal backends (shared quota)
```

The service must hold a **fixed concurrency (`MAX_SLOTS`)** and run for **days** across
many training batches without leaking slots, threads, memory, sandboxes, or snapshot
quota — because a single leaked slot permanently shrinks rollout throughput and a
thread/sandbox leak eventually wedges the whole node.

---

## 2. Failure taxonomy (current implementation)

The current code is correct in intent but is a **stack of point-fixes around three
structural weaknesses**: (A) uncancellable work on threads, (B) truth-by-in-memory-counter,
and (C) per-process reconstruction of global cloud state. Every band-aid below is
evidence of one of those.

### A. Uncancellable work → thread & memory leak (the headline issue)

| # | Problem | Evidence |
|---|---------|----------|
| A1 | **Python threads cannot be cancelled.** A step that blocks in an uninterruptible Daytona/toolkit/httpx call leaves a zombie worker thread even after `STEP_TIMEOUT` releases the asyncio await. | `env_service.py` `_STEP_EXECUTOR` sized `MAX_SLOTS*4` *explicitly to absorb zombies* (lines 106–114); comment "worker thread left to unwind" (1013). `_step_subprocess.py` header: "observed 1800+ threads / 44 GB until uvicorn's accept loop starves → hang." |
| A2 | **The real cure is off by default.** The killable forkserver subprocess (`STEP_USE_SUBPROCESS`) exists but defaults OFF, so production runs the leaky in-thread path. | `_USE_STEP_SUBPROCESS = os.environ.get("STEP_USE_SUBPROCESS","0")=="1"` (67). |
| A3 | **Slot freed ≠ resources freed.** On `STEP_TIMEOUT` the semaphore + `_active_count` are released in `finally`, but the underlying sandbox, httpx client, and thread are still live. Accounting says "free"; reality says "occupied." | `/step` finally (1020–1023) runs even though the thread keeps running (1011–1016). |
| A4 | **`asyncio.run()` per step** spins a brand-new event loop inside a pool thread for every trajectory — heavy, and the loop/thread lifetime is exactly what leaks. | `_run_step_in_isolated_loop` (991–992). |

### B. Truth-by-in-memory-counter → accounting drift

| # | Problem | Evidence |
|---|---------|----------|
| B1 | **~12 module-global mutables are the source of truth**, hand-maintained in one `finally`. Any early return / exception path that misses one decrements drifts the books. | `_active_count`, `_active_per_task`, `_last_used_at`, `_last_admission`, `_build_gate._registry/_timestamps`, `_dataset_locks`, `_eviction_lock`, counters (249–326, 739–740). |
| B2 | **BuildGate cancel path is unsound.** `asyncio.CancelledError` is `BaseException`, so a cancelled builder (client disconnect / outer timeout) is *not* caught by `except Exception`; `state.status` stays `"building"`, `finally` sets the event, waiters wake, see status≠`"failed"` and **fall through to create a sandbox with no built snapshot**. | `BuildGate.ensure_built` (177–209): `except Exception` (185) misses `CancelledError`; waiter only re-raises on `status=="failed"` (208). |
| B3 | **Scheduler slot count is independent of node truth.** `env_scheduler` increments/decrements its own `active_slots`; it never reconciles against the node's authoritative `/health.available_slots`. A scheduler restart zeroes counts while nodes are busy → oversubscription. | `Scheduler.pick_node/release` (63–87); node truth exists but is unused: `/health` returns `available_slots`, `active_steps` (761–775). |
| B4 | **Config swap is unlocked and lossy.** `POST /config` rebinds the `_te_config` global with no lock while `/step` reads it, and drops `_raw_model_config` (so tito keys vanish after a live update). | `update_config` (785–799) vs `_load_terminal_env_config` setting `_raw_model_config` (140). |

### C. Per-process reconstruction of global cloud state → races & churn

| # | Problem | Evidence |
|---|---------|----------|
| C1 | **Snapshot quota is enforced by listing *all* Daytona snapshots on every build**, paginated, under one global lock — O(snapshots) network calls per build, and still racy: two nodes can both pass the check and overrun the per-tier quota. | `_ensure_snapshot_quota` (329–465) paginates `snapshot.list`, reaches into `_build_gate._registry` to evict. |
| C2 | **Four overlapping reconcilers** all compensate for the missing central sandbox/snapshot ledger: startup stale-reap, gentle orphan drain, age-based evict, build-gate reconcile. Each is best-effort and can fight the others. | `_reap_stale_sandboxes` (468), `_drain_orphans_gradually` (524), age-evict (379–411), `_reconcile_build_gate_from_daytona` (557). |
| C3 | **Cleanup is "abandon and hope."** `close()` is bounded to 20s; on overrun the sandbox is left for Daytona's 15-min auto-stop + an "orphan watchdog." Steady leak between GC passes; quota pressure. | `terminal_env.py` `_CLEANUP_TIMEOUT_SECONDS=20` (31), "ABANDONING sandbox" (258–264). |

### D. Cross-cutting protocol & control issues

| # | Problem | Evidence |
|---|---------|----------|
| D1 | **Timeout inversion.** Scheduler client gives up at 900s; node keeps the step to 1800s. Scheduler frees its slot and dispatches another → real oversubscription + duplicate sandboxes. | `env_scheduler STEP_TIMEOUT=900` (33) vs `env_service STEP_TIMEOUT_SECONDS=1800` (72). |
| D2 | **No idempotency.** `/step` keyed by `uid` is not idempotent; any retry (timeout, disconnect) creates a *second* sandbox for the same trajectory. node_manager has exec idempotency but env_service has none. | `/step` (802); contrast node_manager `request_id` cache (`node_manager.py` 578–615). |
| D3 | **No retry / failover / circuit-breaking.** A node error or unreachable node fails the step outright; no re-route, no node ejection. | `env_scheduler.step` returns the error (230–235). |
| D4 | **Unbounded admission parking + global pacer lock.** When full, requests park on the semaphore with no queue bound or fast-503; the pacer holds `_admission_lock` across a sleep, serializing *all* entry globally. | semaphore `async with _slot_semaphore` (938) has no timeout; `_pace_admission` holds lock across sleep (284–289). |
| D5 | **`/cleanup` is a node-wide nuke** (`docker stop $(ps -q)`), unsafe to call while other trajectories run on the node. | `cleanup` (1136–1179). |
| D6 | **Observability is a single leak gauge, not a system.** Thread count + RSS are exposed for an *external, unspecified* watchdog; no per-stage histograms, per-task success, fd count, sandbox count, or tracing correlation. | `/health` (761–775); "external watchdog" referenced (736–738). |
| D7 | **GIL coupling.** Per-step threads don't isolate CPU; a step parsing a megabyte of `demux_log` holds the GIL and stalls the FastAPI accept loop, GC, and drain on the main thread anyway. | rationale comment claims isolation (88–105) but CPU work is GIL-bound. |

---

## 3. Design principles for the target

1. **Crash-only / external-truth.** The process can die at any instant. No resource
   reclamation may depend on in-memory state. The cloud backend (Daytona labels, Docker
   labels) is the **only** source of truth for "what is actually running," and a single
   reconciler drives reality toward desired state. (Replaces C1–C3.)
2. **Kill, don't wait.** Every unit of step work runs in something the supervisor can
   **SIGKILL** and reclaim deterministically — a child process, not a thread. Cancellation
   = process death = guaranteed resource release. (Replaces A1–A4.)
3. **Leases, not counters.** A slot and its sandbox are one **lease** with an owner
   (`uid`), a deadline, and a state. Capacity = count of live leases. Nothing is
   hand-decremented in a `finally`. (Replaces B1–B2.)
4. **One owner per concern.** Snapshot/quota lives in a single-writer image manager;
   sandbox lifecycle lives in the reconciler; routing lives in the scheduler. No object
   reaches into another's privates. (Replaces C1, and the `_build_gate._registry` pokes.)
5. **Idempotent, retryable, observable boundary.** `/step` is idempotent on `uid`;
   timeouts are a strict nested budget; every stage emits a span and a metric.
   (Replaces D1–D3, D6.)
6. **Backpressure is explicit.** Admission is a bounded queue with fast-fail; the
   scheduler routes on the node's *authoritative* free count and never oversubscribes.
   (Replaces B3, D4.)

---

## 4. Target architecture

```
                          ┌────────────────────────── env_service (per node) ───────────────────────────┐
 env_scheduler            │                                                                              │
 (authoritative   /step   │  ┌──────────────┐   lease    ┌──────────────────────────────────────────┐   │
  free-count    ───────►  │  │  Admission   │ ─────────► │  Step Supervisor (process pool)           │   │
  routing,                │  │  (bounded q, │  acquire   │   • 1 long-lived worker proc per slot     │   │
  retry/failover,         │  │   fast 503)  │            │   • runs TerminalEnvironment.step()       │   │
  circuit break)  ◄─────  │  └──────────────┘  release   │   • SIGKILL on deadline ⇒ resources die    │   │
                  reward  │         ▲                     └──────────────┬───────────────────────────┘   │
                          │         │ build? ready?                      │ creates/owns                  │
                          │  ┌──────┴───────┐                           ▼                                │
                          │  │ Image/Snapshot│              ┌────────────────────────┐                   │
                          │  │   Manager     │ ◄─labels──── │   Sandbox Ledger        │                   │
                          │  │ (single-writer│              │  uid → {sandbox_id,     │                   │
                          │  │  per backend  │              │         state, deadline}│                   │
                          │  │  namespace)   │              └───────────┬────────────┘                   │
                          │  └───────────────┘                          │ desired vs actual              │
                          │                                ┌────────────▼────────────┐                   │
                          │                                │   Reconciler / Janitor   │ ── delete ──► backend
                          │                                │  (sole cleanup authority)│                   │
                          │                                └──────────────────────────┘                   │
                          │  /healthz /readyz /metrics  ◄── Watchdog (self-drain+restart on leak signal)  │
                          └──────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Step Supervisor — process-per-slot, kill-based isolation *(cures A, D7)*

Make the killable-subprocess path the **default and only** execution path, but upgrade it
from "fork-per-step" to a **persistent prefork worker pool** (the gunicorn/Firecracker-pool
model):

- Start **`MAX_SLOTS` long-lived worker processes** at boot via `forkserver` (heavy
  imports — camel/harbor/transformers — paid once, as `_step_subprocess.py` already does).
- The supervisor (main process) dispatches one `step()` to one idle worker over a pipe,
  and **owns a hard deadline timer** per dispatch.
- On deadline or cancellation: `worker.kill()` (SIGKILL). The child and *all* its threads,
  httpx clients, and Daytona SDK sockets die with it. The supervisor **respawns a fresh
  worker** to refill the slot. Zero zombie threads, bounded RSS, no `MAX_SLOTS*4` slack.
- The main process now does **only** I/O (accept loop, dispatch, reconcile) — no `asyncio.run`
  in threads, no GIL contention from `demux_log` parsing (it happens in the child). Fixes D7.
- Worker crash (OOM, segfault) is observable as a non-zero exit and converts cleanly to a
  failed step + slot refill, instead of a silent hang.

> Why a pool, not fork-per-step: forkserver fork is cheap but not free; a fixed pool gives
> stable RSS, lets you cap per-worker memory (`RLIMIT_AS`) so one runaway step can't take the
> node, and makes "kill+respawn" the uniform recovery for *every* failure class.

### 4.2 Lease model + Sandbox Ledger *(cures B1, B2, A3, C2)*

Replace the scattered counters with one structure, the **slot lease**:

```python
@dataclass
class Lease:
    uid: str                 # idempotency + ownership key
    task_name: str
    state: str               # QUEUED|BUILDING|CREATING|RUNNING|EVALUATING|CLEANUP|DONE|FAILED
    worker_pid: int | None
    sandbox_id: str | None   # filled the instant the backend returns an id
    created_at: float
    deadline: float          # absolute; enforced by supervisor + reconciler
    error: str | None
```

- A **LeaseTable** (one per node, in-process, optionally mirrored to Redis for
  crash-survival) is the *only* place capacity is computed: `free = MAX_SLOTS - len(live leases)`.
- The worker reports `sandbox_id` back to the supervisor **as soon as the backend returns
  it** (before the agent even runs), so a killed/crashed step never leaves an *unrecorded*
  sandbox. This closes the create-then-crash leak that the four reconcilers chase today.
- `_active_per_task` / `_last_used_at` become **derived views** over the LeaseTable, not
  separately maintained dicts.

### 4.3 Reconciler / Janitor — sole cleanup authority *(cures C2, C3, A3)*

One background loop, the operator pattern, replaces `_reap_stale_sandboxes`,
`_drain_orphans_gradually`, age-evict, and the in-line `close()` timeout dance:

- **Desired state** = sandboxes referenced by a live, non-expired lease.
- **Actual state** = `daytona.list(label=harbor_owner, node=this_node)` (and Docker `ps`).
- Every `RECONCILE_INTERVAL`: delete any sandbox that is *actual ∧ ¬desired* (expired lease,
  crashed worker, prior process lifetime). Because identity comes from **backend labels**,
  this is correct across restarts with zero in-memory dependency — collapsing four
  best-effort loops into one provably-complete one.
- Step `close()` becomes *best-effort and instant*: the trajectory result returns
  immediately and the reconciler guarantees teardown. The 20s "abandon" path stops being a
  silent leak because the janitor *will* collect it on the next pass, bounded by
  `RECONCILE_INTERVAL`, not Daytona's 15-min auto-stop.
- Label sandboxes with `{owner, node_id, lifetime_id, uid, deadline}` so the janitor can
  distinguish *this* process's sandboxes from a peer node's on a shared account.

### 4.4 Image / Snapshot Manager — single writer per backend namespace *(cures C1, B2)*

The snapshot quota races because **N nodes independently mutate one shared Daytona namespace**.
Fix by making the namespace single-writer:

- **Option 1 (preferred): a dedicated Image Manager service** on the training side that owns
  the Daytona snapshot namespace for the whole run. Nodes call `ensure_built(task_name)` →
  it does single-flight build + LRU/quota eviction against a *single* authoritative ledger.
  Nodes never call `snapshot.list/delete`. This is the only way to make quota correct under
  multi-node concurrency; everything else is best-effort.
- **Option 2 (smaller change): shared lease in Redis.** Keep build local but coordinate the
  *quota* via a Redis sorted-set of `(task_name → last_used)` with a distributed lock for
  eviction. Removes the O(snapshots) list-on-every-build and the cross-node overrun.
- Either way, **fix the BuildGate cancel bug now** (B2): catch `BaseException`, and have
  waiters treat any non-`built` terminal status (including a builder that vanished) as
  failure rather than falling through.
- BuildGate state for *declarative* mode (no named snapshots) should short-circuit the whole
  quota machinery (already partially done at line 889) — keep that.

### 4.5 Admission control & backpressure *(cures D4)*

- Replace "park on an unbounded semaphore" with a **bounded queue** (`maxsize ≈ MAX_SLOTS`)
  + a fast `503 + Retry-After` when full. Park time is bounded by a queue-wait deadline so
  the scheduler's client timeout never fires on a request that's merely queued.
- Keep the de-phasing pacer, but make it a **token-bucket that does not hold a lock across
  sleep** — compute the next admission time and `await asyncio.sleep` *without* serializing
  every other admission behind the lock. The leaky-bucket property survives; the global
  bottleneck doesn't.

### 4.6 Scheduler ↔ node contract *(cures B3, D1, D2, D3)*

- **Authoritative free count.** The scheduler polls `/readyz`/`/metrics` (`available_slots`,
  `active_steps`) and routes on the *node's* number, reconciling its optimistic local count
  every few seconds. A scheduler restart re-derives state from nodes instead of zeroing.
- **Strict timeout budget (no inversion):**
  `client_total > scheduler_step > node_step + create + eval + cleanup_margin`.
  Concretely: node `STEP_TIMEOUT` is the *innermost* bound; scheduler timeout = node timeout
  + network margin; client timeout = scheduler + margin. Never the reverse.
- **Idempotency.** `/step` carries `uid`; the node returns the *existing* lease's result (or
  "still running") for a duplicate `uid` instead of creating a second sandbox. Safe retries.
- **Retry + failover + circuit-break.** On node 5xx / connection error, the scheduler retries
  on a *different* node (respecting affinity as a soft preference) and ejects a node that
  trips an error-rate threshold until its `/readyz` recovers.

### 4.7 Config management *(cures B4)*

- Hold config in an immutable snapshot object behind an `asyncio.Lock`/`contextvar`; `/step`
  reads a reference at entry so a mid-flight update can't tear. `POST /config` builds a new
  snapshot atomically and **preserves `_raw_model_config`** (tito keys). Version the config and
  return the version in `/step` responses for traceability.

### 4.8 Lifecycle: startup / drain / shutdown

- **Startup:** start workers → start reconciler → one reconcile pass (adopt/cleanup prior
  lifetime by label) → mark `/readyz` ready. No "gentle drain" heuristic needed; the
  reconciler handles prior-lifetime sandboxes by the desired/actual diff.
- **Graceful drain (SIGTERM):** stop admitting (`/readyz` → draining), let in-flight steps
  finish or hit their deadline, reconcile-delete their sandboxes, then exit. The scheduler
  sees `draining` and stops routing there.
- **Watchdog:** `/healthz` (liveness) vs `/readyz` (capacity) vs `/metrics`. A supervisor
  (systemd/k8s/local) restarts on liveness failure; the service **self-drains-and-restarts**
  when leak signals (thread count, fd count, RSS slope, sandbox-vs-lease skew) breach
  thresholds — turning today's "external unspecified watchdog" into a defined contract.

---

## 5. Concrete surface (proposed)

### Endpoints
| Endpoint | Change |
|----------|--------|
| `POST /step` | Idempotent on `uid`; returns `{run_info, reward, sample, config_version, lease_state}`. Fast `503 + Retry-After` when the admission queue is full. |
| `GET /healthz` | Liveness only (process responsive). |
| `GET /readyz` | `{state: ready|draining, available_slots, max_slots}` — what the scheduler routes on. |
| `GET /metrics` | Prometheus: step rate/errors/duration **per stage**, slots in use, leases by state, worker RSS, fd count, thread count, sandbox-vs-lease skew, build-gate stats, snapshot count. |
| `POST /config` | Atomic, locked, preserves raw model config, returns new version. |
| `POST /drain` | Stop admitting; finish in-flight; reconcile; used by deploys. |
| `POST /cleanup_task` | Keep (snapshot delete + lease forget), routed through Image Manager. |
| ~~`POST /cleanup`~~ | Replace node-wide nuke with **lease-scoped** reset; only the reconciler force-deletes, and only non-leased sandboxes. |

### Step state machine (single source of truth)
```
QUEUED ─► BUILDING ─► CREATING ─► RUNNING ─► EVALUATING ─► CLEANUP ─► DONE
   │          │           │          │            │            │
   └──────────┴───────────┴──────────┴────────────┴────────────┴────► FAILED
   (any state may transition to FAILED on deadline/kill; reconciler GCs its sandbox)
```

---

## 6. Failure-mode → mitigation matrix

| Current failure | Root cause | Mitigation in target |
|---|---|---|
| Zombie threads → 1800 threads/44GB → accept-loop starves (A1, A2, A4) | uncancellable thread work | persistent prefork worker pool; SIGKILL+respawn (§4.1) |
| Slot freed but sandbox/thread alive (A3) | counter ≠ reality | lease + ledger; kill releases everything (§4.1–4.2) |
| Counter drift across return paths (B1) | hand-maintained globals | LeaseTable is the only capacity source (§4.2) |
| Build-gate waiters proceed without snapshot (B2) | `CancelledError` uncaught | catch `BaseException`; non-`built` ⇒ fail (§4.4) |
| Scheduler oversubscribes after restart (B3) | local count, no reconcile | route on node's authoritative free count (§4.6) |
| Snapshot quota overrun across nodes (C1) | N writers, shared namespace | single-writer Image Manager / Redis lock (§4.4) |
| Four reconcilers fighting; orphan sandboxes (C2) | no central ledger | one reconciler off backend labels (§4.3) |
| Cleanup "abandon to auto-stop" leaks 15 min (C3) | inline best-effort close | janitor guarantees teardown ≤ interval (§4.3) |
| Timeout inversion → dup sandboxes (D1) | 900 < 1800 | strict nested budget (§4.6) |
| Retry creates 2nd sandbox (D2) | non-idempotent /step | idempotent on `uid` (§4.6) |
| Node error fails whole step (D3) | no failover | retry/failover/circuit-break (§4.6) |
| Flood parks unbounded; global pacer lock (D4) | semaphore + lock-across-sleep | bounded queue + fast 503; lockless pacer (§4.5) |
| `/cleanup` nukes peers' containers (D5) | node-wide scope | lease-scoped reset (§5) |
| Leak invisible until hang (D6) | single gauge | RED + saturation metrics, self-drain watchdog (§4.8) |
| Main loop stalls on `demux_log` (D7) | GIL on shared process | CPU work in child workers (§4.1) |

---

## 7. Phased migration (low-risk ordering)

Each phase is independently shippable and reduces leak rate; no big-bang rewrite.

**Phase 0 — stop the bleeding (days):**
- Flip `STEP_USE_SUBPROCESS=1` to default; treat the fork-per-step path as the supported one.
- Fix BuildGate `CancelledError` (B2) and the scheduler/node **timeout inversion** (D1).
- Add `/step` idempotency keyed on `uid` (D2).
These are small, surgical, and kill the worst leaks + the two correctness bugs.

**Phase 1 — lease + ledger (1–2 wks):**
- Introduce `LeaseTable`; derive all capacity/`_active_per_task`/`_last_used_at` from it.
- Worker reports `sandbox_id` on create; persist `{uid→sandbox_id, deadline}` (in-proc, opt. Redis).
- Replace the four reconcilers with one label-driven janitor (§4.3); cleanup becomes async-guaranteed.

**Phase 2 — persistent worker pool (1–2 wks):**
- Upgrade the subprocess path to a fixed prefork pool with kill+respawn and `RLIMIT_AS` (§4.1).
- Remove `_STEP_EXECUTOR` and the `MAX_SLOTS*4` slack.

**Phase 3 — control plane hardening (1–2 wks):**
- Bounded admission queue + fast 503; lockless pacer (§4.5).
- Scheduler routes on authoritative free count; add retry/failover/circuit-break (§4.6).
- `/healthz` `/readyz` `/metrics` + self-drain watchdog; lease-scoped `/cleanup` (§4.8, §5).

**Phase 4 — single-writer Image Manager (project-sized):**
- Extract snapshot/quota ownership to one service (or Redis-coordinated lock) (§4.4).
- Only worth it once multi-node snapshot races are the dominant remaining failure.

---

## 8. Keep vs. replace

**Keep (good ideas, right instincts):**
- Per-step isolation intent; the forkserver design in `_step_subprocess.py` (promote it).
- Single-flight BuildGate *concept* (fix the cancel bug; move quota out).
- Admission de-phasing pacer *concept* (make it lockless).
- Scheduler affinity for prefix-cache locality (demote to a soft preference).
- `/health` leak gauges (expand into `/metrics`).
- Best-effort, result-never-blocked cleanup philosophy (back it with a guaranteeing janitor).

**Replace:**
- Thread pool execution → kill-able worker-process pool.
- ~12 module globals → LeaseTable + immutable config snapshot.
- Four ad-hoc reconcilers + inline close-timeout → one label-driven janitor.
- Per-process snapshot quota enforcement → single-writer namespace.
- Optimistic scheduler counting → authoritative, reconciled counting + failover.

---

## 9. Validation strategy

The redesign is only credible if it's tested against the failure modes above, not just the
happy path (`test_stress.py` covers concurrency but not leaks/chaos):

- **Leak soak:** run `MAX_SLOTS` concurrency for ≥24h; assert thread count, fd count, RSS,
  and `sandbox_count − live_leases` stay flat (catches A, C).
- **Chaos kill:** SIGKILL the service mid-`CREATING`/`RUNNING`; assert the reconciler reclaims
  every orphan sandbox within one interval and the slot count recovers (catches A3, C2, C3).
- **Hung-step injection:** stub a toolkit call to block forever; assert SIGKILL+respawn frees
  the slot at the deadline with no zombie (catches A1).
- **Quota race:** drive two nodes at the shared Daytona account past the snapshot cap; assert
  no overrun and no half-built reuse (catches C1, B2).
- **Timeout-budget property test:** assert `client > scheduler > node` holds for all configs
  (catches D1).
- **Idempotency:** replay the same `uid` mid-flight and after completion; assert one sandbox
  and one result (catches D2).

---

## 10. Open decisions (need a call before Phase 1)

1. **Crash-survival store:** in-process LeaseTable (simplest; reconciler recovers from labels)
   vs. Redis-mirrored leases (survives restart without a reconcile gap). Recommend in-process
   + label-driven recovery first; add Redis only if restart gaps hurt.
2. **Image Manager:** centralize now (Option 1) or Redis-coordinate (Option 2)? Recommend
   Option 2 first (smaller), Option 1 if multi-node scaling continues.
3. **Worker pool size vs. memory:** `MAX_SLOTS` persistent workers × peak step RSS must fit the
   node; set `RLIMIT_AS` per worker and size `MAX_SLOTS` from measured peak, not guesswork.
4. **Backend abstraction:** the leak/quota model differs for Docker (local, no snapshot quota)
   vs. Daytona (remote, quota'd) vs. Modal. The reconciler/Image-Manager interfaces should be
   per-backend strategies behind one contract so Docker doesn't pay Daytona's machinery.
```
