# Prompt data for the Harbor recipes

> **The files in this folder are examples only.** They show the prompt format
> and let you smoke-test a launcher end to end. They are taken from the public
> Terminal-Bench 2.1 and Terminal-Bench 4.0 benchmarks, so **do not train on
> them**. For training, build the prompt file from your own task set, for
> example the [SETA-Env](https://huggingface.co/datasets/camel-ai/SETA-Env)
> environments.

Every recipe takes two inputs:

| variable | what it is |
|---|---|
| `PROMPT_DATA` | JSONL, one row per task (format below) |
| `TASKS_DIR` | directory with one Harbor task per sub-directory: `<task>/task.toml`, `instruction.md`, `environment/`, `tests/` |

The Harbor agent server resolves each row's `metadata.instance_id` to
`TASKS_DIR/<instance_id>`, starts that task's sandbox, runs the agent and
verifies the result. The launchers check that every row has a task directory
before they start.

## Row format

```json
{"label": "<task>", "prompt": [{"role": "user", "content": "<task>/instruction.md, verbatim"}], "metadata": {"agent_name": "terminus-2", "instance_id": "<task>"}}
```

See `prompt_template.jsonl`. `metadata.agent_name` is informational; the
recipe's `HARBOR_AGENT_NAME` selects the agent.

## Files

| file | rows | source |
|---|---:|---|
| `prompt_template.jsonl` | 1 | schema placeholder |
| `tb21_example.jsonl` | 5 | [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1) @ `5c8eadf1f393183288fa08b8f73ca9a469cc5e00`, `tasks/` |
| `tb4_example.jsonl` | 3 | [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) release `v4.0.0` (`terminal-bench-prebuilt-v4.0.0.tar.gz`). The benchmark canary strings are kept on purpose |

To run the TB2.1 example, point `TASKS_DIR` at the benchmark tasks:

```bash
git clone https://github.com/harbor-framework/terminal-bench-2-1 tb21
git -C tb21 checkout 5c8eadf1f393183288fa08b8f73ca9a469cc5e00
export PROMPT_DATA=scripts/miles/data/tb21_example.jsonl
export TASKS_DIR=$PWD/tb21/tasks
```

## Build a training prompt file (e.g. from SETA-Env)

1. Get the task directories. SETA-Env environments are Harbor task directories
   (see [docs/dataset.md](../../../docs/dataset.md) for the registered datasets):

```bash
python -m seta_env.dataset.download --list
python -m seta_env.dataset.download seta-env-final     # -> dataset/seta-env-final/<task>/
```
2. Build the JSONL from the task directories:

```bash
python scripts/miles/common/build_prompt_dataset.py build \
  --tasks-dir dataset/seta-env-final --agent-name terminus-2 --output my_train.jsonl
# optional subsets
python scripts/miles/common/build_prompt_dataset.py filter \
  --input my_train.jsonl --output my_subset.jsonl --match '^stack_overflow__' --exclude some-task
```

3. Launch with `PROMPT_DATA=my_train.jsonl TASKS_DIR=$PWD/dataset/seta-env-final`.

GPU tasks cannot run in CPU sandboxes; exclude them with `--exclude`. For
TB4 v4.0.0 these are `fp8-rmsnorm-gemm`, `jax-speedrun-gpu` and `math-eval-grader`.
