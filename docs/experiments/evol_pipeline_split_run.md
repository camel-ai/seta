# Evolve a Split with Default Config

**Date:** 2026-04-15

Run a 2-round evolution (CHANGE_CONTEXT → INCREASE_DIFFICULTY) with kimi
rollouts and trajectory-judge verification on one of the 4 seed splits.

Run from `datasynth/evol_pipeline/`. All paths relative to that directory.

## 1. Set environment

```bash
export MOONSHOT_API_KEY="..."          # for kimi rollouts (TITO)
export HF_TOKEN="..."                  # for on-demand seed downloads
export ANTHROPIC_API_KEY="..."         # for Claude evol / datapoint / judge agents
cd datasynth/evol_pipeline
```

## 2. Pick a split

```bash
ls configs/filters/split_*_of_4.csv
```

Splits are disjoint subsets of `selected_500.csv` (4 splits × ~125 tasks).
`config.example.yaml` defaults to `split_1_of_4.csv`. To use a different
split, edit the `filter_csv` line in the config or copy the example:

```bash
cp configs/config.example.yaml configs/split_2.yaml
sed -i 's|split_1_of_4.csv|split_2_of_4.csv|' configs/split_2.yaml
```

## 3. Dry-run to preview the queue

```bash
python run_evol_orchestrator.py configs/config.example.yaml --dry-run
```

Shows how many tasks each evolve round will process. Round 1 reads from
`outputs/seeds/` (downloaded from HF on demand); round 2 chains from
round 1's PASS variants.

## 4. Run the pipeline

```bash
python run_evol_orchestrator.py configs/config.example.yaml &> log
```

Three stages run concurrently:

- **evolve** — blocks; 4 workers produce `__b1` then `__b1__d1` variants
- **rollout** — background poller every 30s; picks up new PASS variants
  and runs 1 kimi trajectory per task
- **verify** — background poller every 60s; judges all-fail rollouts
  (writes `traj_judge_report.md` + updates `synth_info.json`)

Safe to interrupt with Ctrl-C and resume — filesystem is source of truth.

## 5. Resume after interruption

```bash
python run_evol_orchestrator.py configs/config.example.yaml &>> log
```

Evolve skips tasks with `status=done` in `synth_info.json`. Rollout skips
tasks with a `reward.txt` under `<rollout_dir>/<model>/<task>/run_N/`.
Verify skips tasks with a `traj_judge` field in `synth_info.json`.

## 6. Run individual stages

```bash
python run_evol_orchestrator.py configs/config.example.yaml evolve    # all rounds
python run_evol_orchestrator.py configs/config.example.yaml rollout   # scan + fill gaps
python run_evol_orchestrator.py configs/config.example.yaml verify    # judge all-fail tasks
```

One-shot standalone runs — no background pollers.

## 7. Inspect results

```
outputs/
├── seeds/                               # raw seed tasks (from HF)
├── evol_r1_breadth/<task>__b1/          # round-1 variants (CHANGE_CONTEXT)
│   ├── draft_spec.md
│   ├── task.toml, instruction.md, environment/, solution/, tests/
│   ├── synth_info.json                  # status, verdict
│   ├── judge_report.md                  # self-assessment (7 criteria)
│   └── traj_judge_report.md             # verify output (optional)
├── evol_r1_breadth_rollout/
│   ├── kimi_k2/<task>__b1/run_1/        # trajectories
│   └── validation/                      # harbor oracle/empty check
├── evol_r2_depth/<task>__b1__d1/        # round-2 variants (INCREASE_DIFFICULTY)
└── evol_r2_depth_rollout/kimi_k2/
```

Quick stats:

```bash
python -c "
import json
from pathlib import Path
for d in ['outputs/evol_r1_breadth', 'outputs/evol_r2_depth']:
    base = Path(d)
    if not base.is_dir(): continue
    done = pass_c = judged = df = 0
    for t in base.iterdir():
        if not t.is_dir() or t.name.startswith(('.','_','validation')): continue
        ip = t / 'synth_info.json'
        if not ip.exists(): continue
        info = json.loads(ip.read_text())
        if info.get('status') == 'done':
            done += 1
            if info.get('verdict') == 'PASS': pass_c += 1
        if info.get('traj_judge'):
            judged += 1
            if info.get('traj_judge') == 'design_flaw': df += 1
    print(f'{d}: {done} done, {pass_c} PASS, {judged} judged ({df} design_flaw)')
"
```

## 8. Upload PASS variants to HF (optional)

Set `huggingface.output_repo` in the config, then:

```bash
python run_evol_orchestrator.py configs/config.example.yaml --upload
```

## Default config summary

`configs/config.example.yaml`:

- `filter_csv: filters/split_1_of_4.csv` (120 tasks)
- `evolve.rounds`: r1_change_context → r2_increase_difficulty
- `evolve.n_workers: 4`, `task_timeout_s: 3600`
- `rollout.models: [kimi_k2]` with `tito_enabled: true`, `n_trajs: 1`
- `verify.max_pass_rate: 0.0` (only all-fail tasks get judged)

Swap split by editing `filter_csv`; swap model by editing `rollout.models`.
