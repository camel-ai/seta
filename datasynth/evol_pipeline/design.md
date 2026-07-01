# Evolution Pipeline — Architecture & Design

## Overview

The evolution pipeline generates increasingly challenging synthetic tasks through
iterative evolution of seed tasks. It operates as **three independent stages**
that discover work by scanning the filesystem:

| Stage | Purpose | Discovery |
|-------|---------|-----------|
| **evolve** | Generate evolved task variants from seeds (or prior round output) | Checks `output_dir` for existing `synth_info.json`; skips done tasks |
| **rollout** | Run agent trajectories on any set of Harbor tasks | Scans `tasks_dir` for eligible tasks, checks `rollout_dir` for completed rollouts, fills gaps |
| **verify** | Classify rollout failures as design flaws vs. genuine difficulty | Scans `rollout_dir` for low-pass-rate tasks, checks for existing `traj_judge_report.md`, fills gaps |

Each stage is **independently invocable** and **fully resumable** — safe to
interrupt (Ctrl-C) and re-run at any time. The filesystem is the single source
of truth for progress.

---

## CLI Interface

```bash
python run_pipeline.py config.yaml evolve       # all evolve rounds sequentially
python run_pipeline.py config.yaml rollout      # rollout all tasks_dirs (scan + fill gaps)
python run_pipeline.py config.yaml verify       # verify all tasks_dirs
python run_pipeline.py config.yaml all          # evolve → rollout → verify (sequential)

# Flags
python run_pipeline.py config.yaml rollout --dry-run
python run_pipeline.py config.yaml evolve --generate-summary
```

The `all` mode runs: evolve round 1 → round 2 → … → rollout all targets → verify all targets.

---

## Folder Convention

```
outputs/
├── seeds/                                # original tasks (from HF or local)
│   ├── <task_id>/
│   │   ├── task.toml
│   │   ├── instruction.md
│   │   ├── environment/Dockerfile
│   │   ├── solution/solve.sh
│   │   ├── tests/test_outputs.py
│   │   ├── tests/test.sh
│   │   └── weights.json
│   │   └── (no synth_info.json — raw seed)
│
├── evol_r1_breadth/                      # round 1: CHANGE_CONTEXT
│   ├── <task_id>__b1/
│   │   ├── task.toml, instruction.md, …  # complete Harbor task
│   │   ├── draft_spec.md                 # evolution agent output
│   │   ├── synth_info.json               # status + verdict
│   │   ├── judge_report.md               # self-assessment (7 criteria)
│   │   └── traj_judge_report.md          # verification output (optional)
│   └── summary.csv
│
├── evol_r1_breadth_rollout/              # rollouts for round 1
│   └── kimi_k2/                          # per-model subdirectory
│       └── <task_id>__b1/
│           └── run_1/                    # trajectory results
│
├── evol_r2_depth/                        # round 2: INCREASE_DIFFICULTY
│   └── <task_id>__b1__d1/…              # chained variant IDs
│
├── evol_r2_depth_rollout/                # rollouts for round 2
│   └── kimi_k2/
│       └── <task_id>__b1__d1/run_1/
│
└── seeds_rollout/                        # optional: rollouts for raw seeds
    └── kimi_k2/
        └── <task_id>/run_1/
```

**Naming convention**: `<tasks_folder>_rollout` is the default rollout
directory for any given tasks folder. This is auto-derived when
`rollout_dir` is omitted from the config.

**Variant ID scheme**: `<parent>__<prefix><N>` where prefix is `d` (depth /
increase_difficulty) or `b` (breadth / change_context). Chaining produces
IDs like `402__b1__d1` (seed 402 → change_context → increase_difficulty).

---

## Configuration

See `configs/config.example.yaml` for the canonical example.

### Top-Level Structure

```yaml
huggingface:        # HF repo for on-demand seed download + upload
filter_csv:         # top-level filter — applies to ALL stages at seed task_id level
evolve:             # multi-round evolution settings
rollout:            # agent trajectory rollout settings
verify:             # rollout failure analysis settings
upload:             # periodic HF upload settings
```

### filter_csv (Top-Level)

A CSV file with a single `task_id` column listing the **seed** task IDs to
include (e.g. 125 selected tasks). This filter applies across all stages:

- **evolve**: only evolves seeds in the CSV
- **rollout**: only rolls out tasks whose root seed is in the CSV
- **verify**: only verifies tasks whose root seed is in the CSV

The filter operates on the **root task_id** extracted from variant IDs.
For example, if the CSV contains `402`, then `402`, `402__b1`, and
`402__b1__d1` are all included.

### evolve

Supports multiple sequential rounds in one config via `evolve.rounds`.
Each round's output can chain as the next round's input.

```yaml
evolve:
  n_workers: 4
  skip_timeout: false
  task_timeout_s: 3600
  rounds:
    - name: r1_change_context
      input_dir: outputs/seeds
      output_dir: outputs/evol_r1_breadth
      strategy:
        evol_strategy: breadth
        evol_target: CHANGE_CONTEXT
        max_variants: 1
    - name: r2_increase_difficulty
      input_dir: outputs/evol_r1_breadth
      output_dir: outputs/evol_r2_depth
      strategy:
        evol_strategy: depth
        evol_target: INCREASE_DIFFICULTY
        max_variants: 1
```

Shared settings (`n_workers`, `task_timeout_s`, `skip_timeout`) apply to all
rounds. Per-round settings: `input_dir`, `output_dir`, `strategy`.

For a single-round config, `rounds` is simply a list with one element.

### rollout

Scans one or more task folders, finds tasks missing rollout results,
and runs agent trajectories to fill gaps. Supports multiple models.

```yaml
rollout:
  n_workers: 2
  n_trajs: 1
  tasks_dirs:
    - outputs/evol_r1_breadth
    - outputs/evol_r2_depth
  models:
    - model_config_name: kimi_k2
      model_platform: moonshot
      model_type: kimi-k2.5
      url: https://api.moonshot.ai/v1
      model_config_dict:
        max_tokens: 4096
        temperature: 1.0
```

- `tasks_dirs`: list of task folders to roll out. Each auto-derives its
  rollout_dir as `<tasks_dir>_rollout` unless overridden.
- `models`: list of model configurations. Each model's rollouts are
  namespaced under `<rollout_dir>/<model_config_name>/`.
- Agent, runtime, and env settings are baked in — only model varies per config.

**Unified gap-finding**: works identically for seeds (no synth_info.json)
and evolved tasks (requires `status=done` + `verdict=PASS` in
synth_info.json). No separate seed_rollout configuration needed — just
add `outputs/seeds` to `tasks_dirs`.

### verify

Scans rollout results to find tasks below a pass-rate threshold and runs
a trajectory judge agent to classify failures.

```yaml
verify:
  tasks_dirs:
    - outputs/evol_r2_depth
  model_config_name: kimi_k2
  max_pass_rate: 0.0
```

- `max_pass_rate: 0.0` means only all-fail tasks are verified.
- Judge writes `traj_judge_report.md` into the task folder and updates
  `synth_info.json` with the verdict (`DESIGN_FLAW` or `TOO_HARD`).

---

## Stage Details

### Evolve Stage

Per-task pipeline: `EvolTaskPipeline.run()`

```
Input task → load Harbor files
  → generate variant IDs (e.g. "402__b1")
  → ClaudeEvolAgent.evolve() → draft_spec.md per variant
  → for each non-filtered variant:
      ClaudeDatapointAgent.create() → complete Harbor task + judge_report.md
      parse verdict (PASS / FAIL)
  → write synth_info.json
```

**Evolution strategies**:
- `CHANGE_CONTEXT` (breadth): port to different technology, similar complexity
- `INCREASE_DIFFICULTY` (depth): create harder version, same domain

**Chaining**: `list_input_tasks()` auto-filters to PASS variants when
`synth_info.json` is present in the input dir. Round 2 only evolves
successful round 1 outputs.

### Rollout Stage

For each `tasks_dir` × each `model`:
1. `find_rollout_gaps(tasks_dir, rollout_dir, model_config_name)`
2. Construct `TerminalEnvConfig` from model entry + baked-in agent/runtime/env constants
3. `GRPORollout(cfg).run()` with `n_trajs` trajectories per task
4. Write results to `<rollout_dir>/<model_config_name>/<task_id>/run_N/`

Agent, runtime, and env settings are baked-in constants (`ROLLOUT_AGENT_CONFIG`,
`ROLLOUT_RUNTIME_CONFIG`, `ROLLOUT_ENV_CONFIG` in `evol_config.py`). Only the
model config varies per run.

### Verify Stage

For each `tasks_dir`:
1. Scan `<tasks_dir>_rollout/<model_config_name>/` for completed rollouts
2. Compute pass rate per task
3. Filter to tasks with `pass_rate <= max_pass_rate`
4. Skip tasks with existing `traj_judge_report.md`
5. `ClaudeTrajectoryJudgeAgent.judge()` → writes report + updates synth_info

---

## Resumability

Every stage is safe to stop and resume. The filesystem is the truth:

- **evolve**: `synth_info.json` with `status=done` marks completion. On
  resume, tasks without it (or with `status=in_progress`) are re-queued.
- **rollout**: presence of `run_N/` directories with results marks
  completion. `find_rollout_gaps()` only queues missing ones.
- **verify**: presence of `traj_judge_report.md` marks completion.

Multi-round resume: if stopped mid-round-2, round 1 is already complete.
Resuming re-runs only the remaining round-2 tasks.

---

## Multi-Round Workflow Example

Default pipeline: 125 selected seeds → 2 rounds → kimi rollout → verify.

```
seeds (125 tasks, from seta-env-v2)
  ↓  evolve round 1 (CHANGE_CONTEXT, breadth)
evol_r1_breadth/ (task_id__b1 variants, PASS only chain forward)
  ↓  evolve round 2 (INCREASE_DIFFICULTY, depth)
evol_r2_depth/ (task_id__b1__d1 variants)
  ↓  rollout (kimi, 1 trajectory each round)
evol_r1_breadth_rollout/kimi_k2/
evol_r2_depth_rollout/kimi_k2/
  ↓  verify (all-fail tasks from final round)
evol_r2_depth/<task_id>/traj_judge_report.md
```

---

## Key Files

| File | Purpose |
|------|---------|
| `evol_config.py` | Config dataclasses, YAML loader, validation |
| `evol_orchestrator.py` | Stage runners: `run_evolve()`, `run_rollout()`, `run_verify()` |
| `evol_task_pipeline.py` | Per-task evolve + datapoint pipeline |
| `io_utils.py` | Harbor task loading, variant ID generation, synth_info R/W |
| `run_pipeline.py` | CLI entry point with subcommands |
| `agents/claude_agents.py` | ClaudeEvolAgent, ClaudeDatapointAgent, ClaudeTrajectoryJudgeAgent |
| `agents/traj_judge_prompt.md` | Trajectory judge system prompt |
| `configs/config.example.yaml` | Canonical config example |
