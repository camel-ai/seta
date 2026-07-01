# Seed2Synth Pipeline

Standalone pipeline for synthesizing terminal agent tasks from seed data. Config-driven, supports distributed generation via HuggingFace.

---

## Quick Start: Kaggle Notebooks (1103 tasks)

```bash
# 1. Set HuggingFace token
export HF_TOKEN=<your-hf-token>

# 2. Download Kaggle metadata (~5 sec)
python hf_utils.py --config configs/kaggle_base.yaml download-metadata
# -> outputs/seed_data/kaggle_notebook/metadata.csv (1103 tasks)
```

**Pre-Split Ready**: The 1103 Kaggle tasks are already split into 4 parts in `configs/filters/`:

| Config | Filter | Tasks |
|--------|--------|-------|
| `configs/kaggle_part_01.yaml` | `configs/filters/part_01.csv` | 275 |
| `configs/kaggle_part_02.yaml` | `configs/filters/part_02.csv` | 275 |
| `configs/kaggle_part_03.yaml` | `configs/filters/part_03.csv` | 275 |
| `configs/kaggle_part_04.yaml` | `configs/filters/part_04.csv` | 278 |

```bash
# Run Part 1
python run_orchestrator.py configs/kaggle_part_01.yaml --synth-only

# Dry-run (preview queue)
python run_orchestrator.py configs/kaggle_part_01.yaml --dry-run

# Deploy parts to different machines for parallel synthesis
# Each runs independently, uploads to HF automatically
```

To create custom splits (different n-parts or other sources):

```bash
python hf_utils.py --config configs/kaggle_base.yaml check-and-split \
  --sources kaggle_notebook --n-parts 8 --output-dir configs/filters/
```

---

## Usage

```bash
# Synthesis only
python run_orchestrator.py <config.yaml> --synth-only

# Rollout only (requires model server)
python run_orchestrator.py <config.yaml> --rollout-only

# Full pipeline (synth + rollout)
python run_orchestrator.py <config.yaml>

# Dry-run
python run_orchestrator.py <config.yaml> --dry-run

# Regenerate summary.csv from synth_info.json files
python run_orchestrator.py <config.yaml> --generate-summary

# Override worker count
python run_orchestrator.py <config.yaml> --synth-only --n-synth-workers 4
```

### HF Utilities

```bash
# Download metadata for sources in config
python hf_utils.py --config <config.yaml> download-metadata

# Split tasks for distributed generation
python hf_utils.py --config <config.yaml> check-and-split \
  --sources <source1,source2> --n-parts N --output-dir <dir>

# Upload completed tasks to HF
python hf_utils.py --config <config.yaml> upload-synth
```

### Environment Variables

```bash
export HF_TOKEN=<your-hf-token>           # HuggingFace access
export SGLANG_URL=http://localhost:8000    # Model server (for rollout)
```

### Monitor Progress

```bash
cat outputs/synth_data/kaggle_notebook/summary.csv | head -20
grep "pass" outputs/synth_data/kaggle_notebook/summary.csv | wc -l
```

---

## Configuration

Single YAML file drives everything. See `configs/config.example.yaml` for all options.

```yaml
huggingface:
  seed_repo: camel-ai/seta-env-seed2synth-seed
  synth_repo: camel-ai/seta-env-seed2synth-synth
  token_env: HF_TOKEN

paths:
  seed_data_dir: outputs/seed_data
  synth_data_dir: outputs/synth_data
  rollout_dir: outputs/synth_data_rollouts

sources:                          # which sources to process
  - kaggle_notebook

filter_csv: configs/filters/part_01.csv   # optional: limit to specific tasks

seed_preparation:
  download_all_upfront: false     # true = download all first; false = on-demand

pipeline:
  n_synth_workers: 2
  stage: full                     # full | idea-only | unified
  skip_timeout: false

rollout:
  enabled: false
  n_rollout_workers: 2
  n_trajs: 8
  model_url: "${SGLANG_URL}"     # env var interpolation
  model_config_name: Qwen3-8B_thinking

upload:
  enabled: true
  interval_minutes: 30
```

---

## File Formats

### Input: `metadata.csv` (per source)

All sources use the same schema. Located at `outputs/seed_data/{source}/metadata.csv`.

```
task_id, source, title, category, tags, score, url, filtered, filter_reason
```

Available data:
- `unix_linux_se`: 1,992 tasks
- `stack_overflow`: 1,036 tasks
- `kaggle_notebook`: 1,103 tasks

### Output: `synth_info.json` (per task)

Source of truth for task status. Located at `outputs/synth_data/{source}/{task_id}/synth_info.json`.

```json
{
  "task_id": "10026",
  "source": "unix_linux_se",
  "status": "done",
  "verdict": "pass",
  "stage": "full",
  "idea_time_s": 45.2,
  "datapoint_time_s": 180.5,
  "total_synth_time_s": 225.7,
  "harbor_oracle_passed": true,
  "harbor_empty_failed": true,
  "timestamp": "2026-04-05T00:10:00Z"
}
```

**Status**: `in_progress` | `timeout` | `done`
**Verdict**: `null` | `ditch` | `pass` | `fail`

Resume reads `synth_info.json` directly: done tasks are skipped, in_progress tasks are re-queued.

### Output: `summary.csv` (per source)

Derived from `synth_info.json` files. Regenerate with `--generate-summary`.

### Output: Synth Task Artifacts

Each completed task produces:

```
outputs/synth_data/{source}/{task_id}/
  synth_info.json       # status tracking
  draft_spec.md         # task specification
  instruction.md        # agent instructions
  task.toml             # task config
  solution/solve.sh     # reference solution
  tests/test_*.py       # validation tests
  environment/Dockerfile
  weights.json          # test weights
  judge_report.md       # harbor validation report
```

---

## Directory Structure

```
seed2synth_pipeline/
  run_orchestrator.py          # entry point
  seed2synth_orchestrator.py   # orchestrator core
  seed2task_pipeline.py        # synthesis pipeline (idea -> datapoint agents)
  seed2synth_config.py         # config loader
  hf_utils.py                  # HF download/upload/split
  agents/
    claude_agents.py           # Claude SDK agent implementations
    seed2idea_prompts/         # idea agent prompts per source type
    datapoint_agent_guide/     # datapoint agent instructions
  configs/
    config.example.yaml        # all options documented
    kaggle_base.yaml           # kaggle-only base config
    kaggle_part_0{1..4}.yaml   # pre-built 4-part configs
    filters/
      part_0{1..4}.csv         # pre-split task lists

Runtime data (created outside repo):
  outputs/seed_data/{source}/metadata.csv           # task metadata
  outputs/seed_data/{source}/{task_id}/...          # seed data (downloaded on-demand)
  outputs/synth_data/{source}/{task_id}/...         # synthesis output
  outputs/synth_data/{source}/summary.csv           # aggregated status
  outputs/synth_data_rollouts/{model}/{source}/...  # rollout trajectories
```
