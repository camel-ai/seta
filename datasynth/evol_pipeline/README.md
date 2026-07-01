# Evolution Pipeline

Evolve existing terminal-agent tasks to create harder variants or port them to new domains.

## Usage

```bash
cd datasynth/evol_pipeline

# Edit config
cp configs/config.example.yaml configs/my_run.yaml

# Preview what will run
python run_evol_orchestrator.py configs/my_run.yaml --dry-run

# Run
python run_evol_orchestrator.py configs/my_run.yaml
```

## Strategies

- **INCREASE_DIFFICULTY** — harder version (more steps, edge cases, tighter constraints)
- **CHANGE_CONTEXT** — same complexity, different technology (e.g. nginx → apache2)

One strategy per config run. Add new strategies by creating an adapter in `agents/evol_strategy_prompts/`.

## Chaining

Point the next run's `input_dir` at the previous run's `output_dir`:

```bash
# Turn 1: make harder
python run_evol_orchestrator.py configs/increase_difficulty.yaml

# Turn 2: port to new domains (set input_repo: null, input_dir: previous output)
python run_evol_orchestrator.py configs/change_context_chained.yaml
```

## Cluster Splitting

```bash
python run_evol_orchestrator.py configs/my_run.yaml --generate-filters --n-parts 4
# Give each machine a config with its own filter_csv
```

## CLI

```
--dry-run              Preview queue
--n-evol-workers N     Override parallelism
--generate-summary     Rebuild summary.csv
--generate-filters     Split tasks for multi-machine runs
--upload               Push results to HuggingFace
```
