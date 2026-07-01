# SFT Training from a HuggingFace Dataset

**Date:** 2026-04-07

Trains a base model on the SFT dataset built by
[`sft_collection_conversion_04_03.md`](sft_collection_conversion_04_03.md).
Run from the repo root. All paths relative.

## Datasets and configs

Two SFT datasets are published, one config per variant:

| Variant | HF dataset | Config |
|---|---|---|
| **thinking** (preserves `<think>…</think>` reasoning) | [camel-ai/seta-sft-kimi-k2.5-thinking](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-thinking) | [`scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml`](../../scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml) |
| **nothink** (reasoning pruned, system prompt suffixed with `/no_think`) | [camel-ai/seta-sft-kimi-k2.5-nothink](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-nothink) | [`scripts/areal_sft/configs/seta_kimi_qwen3_sft_nothink.yaml`](../../scripts/areal_sft/configs/seta_kimi_qwen3_sft_nothink.yaml) |

Both datasets are single-`train`-split `DatasetDict`s. Each row preserves the **full per-trial diagnostic record** so consumers can inspect, filter, or re-tokenize without rerunning the rollouts:

| column | type | meaning |
|---|---|---|
| `task_id`               | str       | task name (without `_t<i>_<hash>` suffix) |
| `trial_uid`             | str       | full trial dir name |
| `reward`                | float     | passed/total tests from `verifier/ctrf.json` |
| `model`                 | str       | rollout model id (e.g. `kimi-k2.5`) |
| `provider_token_counts` | dict      | LLM API counts: prompt / completion / total / cached |
| `local_token_count`     | int       | `len(input_ids)` under the Qwen3-8B tokenizer |
| `n_assistant_tokens`    | int       | `sum(loss_mask)` |
| `n_messages`            | int       | reconstructed conversation length |
| `raw_conv_json`         | str       | full original conv json, serialized |
| `chat_template_str`     | str       | exact rendered template that was tokenized |
| `input_ids`             | list[int] | tokenized full conversation (Qwen3-8B chat template) |
| `loss_mask`             | list[int] | same length, `1` on every assistant span (incl. `<think>` reasoning in the thinking variant + `<tool_call>` segments), `0` elsewhere |

`scripts/areal_sft/seta_sft_dataset.py::get_seta_sft_dataset` resolves `train_dataset.path` as either an HF repo id (`owner/name`) or a local `.jsonl` file. It **projects each row to `(input_ids, loss_mask)` only** at load time, drops every other column, and feeds the AREAL collator the lean two-column shape it expects. It also hash-splits source rows by index using `train_ratio` (default `1.0` → all rows in `train`, `test` empty → AREAL evaluator iterates zero batches and is a no-op). Set `train_dataset.train_ratio=0.95` on the CLI to hold out a 5% eval slice.

For inspection / custom processing the full row is available via plain `datasets.load_dataset`:

```python
from datasets import load_dataset
ds = load_dataset("camel-ai/seta-sft-kimi-k2.5-thinking", split="train")
print(ds[0]["task_id"], ds[0]["reward"], ds[0]["local_token_count"])
print(ds[0]["chat_template_str"][:500])
```

## 1. Launch training (single node, 2 GPUs by default)

Thinking variant:
```bash
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml
```

No-thinking variant:
```bash
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_nothink.yaml
```

Default config: `cluster.n_gpus_per_node=2`, `allocation_mode=d2p1t1`. Override on the CLI for more GPUs (e.g. `cluster.n_gpus_per_node=8 allocation_mode=d8p1t1`). For multi-node on Ray, swap `areal.launcher.local` for `areal.launcher.ray` and add `cluster.n_nodes=N`.

## 2. Override config on the command line (Hydra-style)

```bash
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml \
    train_dataset.batch_size=16 \
    total_train_epochs=1 \
    cluster.fileroot=outputs/areal/my_run
```

Checkpoints + logs land under `${cluster.fileroot}/<experiment>/<trial>/`.
Resume an interrupted run with `recover.mode=auto` on the CLI.

## Notes

- Both datasets are `train`-only. With the loader's default `train_ratio=1.0`, the `valid_dataset` block points at the same dataset, the loader returns an empty Dataset for `test`, and the evaluator iterates zero batches (no-op).
- To enable evaluation, override on the CLI: `train_dataset.train_ratio=0.95 valid_dataset.train_ratio=0.95`. The split is a deterministic per-row-index hash, identical across reruns and process ranks.
- For local-only training (no HF), set `train_dataset.path=outputs/eval/eval_tito/<trial>_merged/sft_thinking.jsonl` (or `sft_no_thinking.jsonl`) — the loader auto-detects the `.jsonl` extension.
