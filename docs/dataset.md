# Dataset

Datasets are registered in `seta_env/dataset/datasets.yaml`.

```bash
python -m seta_env.dataset.download --list                          # see available
python -m seta_env.dataset.download terminal-bench-core_migrated    # download to dataset/
```

To add a new dataset, add an entry to `seta_env/dataset/datasets.yaml` with `repo` and optional `subfolder`.

## Filtering tasks based on prior eval results

`load_harbor_dataset(<dataset_dir>)` auto-detects `<dataset_dir>/task_filter.txt`
(one `task_id` per line, blank lines and `#` comments ignored) and only loads
the listed tasks. Generate it with `seta_env.dataset.filter_tasks` from an
`evaluated_tasks.csv`. See [evaluation.md](evaluation.md#pre-training-evaluation-and-dataset-filtering)
for the end-to-end workflow.
