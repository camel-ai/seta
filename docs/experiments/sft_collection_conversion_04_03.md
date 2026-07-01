# SFT Collection & Conversion

**Date:** 2026-04-03 (updated 2026-04-07)

Run from repo root. All paths relative.

## 1. Run eval

```bash
export MOONSHOT_API_KEY="..."
python scripts/evaluation/eval.py --config scripts/evaluation/configs/eval_default_kimi_tito.yaml
```
For Azure GPT-5.4 swap the config to `eval_default_azure_gpt54.yaml` and set `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_BASE_URL` / `AZURE_API_VERSION`.

## 2. (Optional) Resume crashed tasks

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default_kimi_tito.yaml \
    --resume outputs/eval/eval_tito/<trial_name> \
             outputs/eval/eval_tito/<trial_name>_resume
```
Repeat until no tasks remain. Each resume writes to `<first_dir>_resume[_N]/`.

## 3. Merge resume chain

```bash
python -m seta_env.utils.collect_results --merge \
    outputs/eval/eval_tito/<trial_name> \
    outputs/eval/eval_tito/<trial_name>_resume \
    outputs/eval/eval_tito/<trial_name>_resume_2 \
    --output         outputs/eval/eval_tito/<trial_name>_merged \
    --collect-trials move
```
Writes `success.csv` + `failed.csv`, moves all trial dirs into `<merged>/trials/`. Use `copy` to keep originals.

## 4. Build SFT JSONL (smoke first, then full)

**Input:** one trial subdir per trajectory under `<merged>/trials/`. The script reads:
```
<merged>/trials/<task>_t<i>_<hash>/
├── verifier/ctrf.json                       # → reward (passed/total)
├── run_info.json                            # task_id, traj_i, error_info
└── CAMEL_LOG_DIR/<agent_id>/conv_*.json     # largest one = final trajectory
                                             # → request.messages + response.choices[0]
                                             # → response.usage (provider token counts)
```

**Output:** a single JSONL with one row per trial. Per-row fields:

| field | type | source |
|---|---|---|
| `task_id` / `trial_uid` | str | derived from trial dir name |
| `reward` | float | `verifier/ctrf.json` passed/total |
| `model` | str | conv json `model` |
| `provider_token_counts` | dict | conv json `response.usage` |
| `local_token_count`, `n_assistant_tokens`, `n_messages` | int | computed |
| `raw_conv_json`, `chat_template_str` | str | full conv + rendered template |
| `input_ids`, `loss_mask` | list[int] | tokenized + per-token mask (1=assistant trainable, 0=context) |

```bash
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/eval_tito/<trial_name>_merged/trials \
    --output     outputs/eval/eval_tito/<trial_name>_merged/sft.jsonl \
    --model      Qwen/Qwen3-8B --debug
```

`--debug` also writes a sibling tree for eyeballing boundary handling:
```
<merged>/sft.jsonl
<merged>/sft.jsonl.inspect/<trial_uid>/
├── inspect.txt          # per-token boundary dump (first/last 10 of every <|im_start|>..<|im_end|> block)
├── chat_template.txt    # exact rendered template that was tokenized
└── summary.json         # per-trial metrics (reward, token counts, ratios)
```
Drop `--debug` for the full production build (skips per-trial artifact writes).

By default trials with `reward=None` (no `verifier/ctrf.json` — errored / timed out) are dropped; trials with any verified reward are kept. Pass `--min-reward 1.0` to only keep fully-passing rollouts.

## 5. Push to HuggingFace Hub

```bash
export HF_TOKEN="<token>"
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/eval_tito/<trial_name>_merged/trials \
    --output     outputs/eval/eval_tito/<trial_name>_merged/sft.jsonl \
    --model      Qwen/Qwen3-8B \
    --push-to-hub camel-ai/seta-sft-kimi-k2.5
```
Pushed as a single `train` split (no test split) with **all columns preserved**: `task_id`, `trial_uid`, `reward`, `model`, `provider_token_counts`, `local_token_count`, `n_assistant_tokens`, `n_messages`, `raw_conv_json`, `chat_template_str`, `input_ids`, `loss_mask`. The AREAL-shape projection to `(input_ids, loss_mask)` happens at *load time* in [`scripts/areal_sft/seta_sft_dataset.py`](../../scripts/areal_sft/seta_sft_dataset.py), so the trainer still gets the lean two-column shape it expects while downstream consumers can inspect / filter / re-tokenize using the full record.

## Reference: kimi-k2.5 thinking + nothink build commands

The actual commands used to produce the two published kimi-k2.5 SFT datasets from the merged trial dir `outputs/eval/eval_tito/kimi-k2.5_20260406_140126_merged/`:

```bash
# Thinking variant (preserves <think>...</think> reasoning blocks)
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/eval_tito/kimi-k2.5_20260406_140126_merged/trials \
    --output     outputs/eval/eval_tito/kimi-k2.5_20260406_140126_merged/sft_thinking.jsonl \
    --model      Qwen/Qwen3-8B

# No-thinking variant (prunes reasoning_content + appends /no_think to system prompt)
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/eval_tito/kimi-k2.5_20260406_140126_merged/trials \
    --output     outputs/eval/eval_tito/kimi-k2.5_20260406_140126_merged/sft_no_thinking.jsonl \
    --no-thinking
```

Both runs: 1605 trial subdirs scanned → 117 dropped (no `verifier/ctrf.json`) → **1488 rows** written to each JSONL. Reward distribution identical (1112 fully-passing, mean 0.932) — only the assistant content differs.

| | thinking | no-thinking | Δ |
|---|---|---|---|
| total tokens     | 15.69 M | 13.18 M | −16% |
| trainable tokens | 7.79 M (49.6%) | 5.27 M (40.0%) | −32% |
| mean / median tokens / row | 10 544 / 8 884 | 8 856 / 7 434 | — |

The two JSONLs were then projected to `(input_ids, loss_mask)` only and pushed as separate single-`train`-split datasets (with auto-generated dataset cards):

- **[camel-ai/seta-sft-kimi-k2.5-thinking](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-thinking)** — 188 MB parquet, 7.79 M trainable tokens
- **[camel-ai/seta-sft-kimi-k2.5-nothink](https://huggingface.co/datasets/camel-ai/seta-sft-kimi-k2.5-nothink)** — 158 MB parquet, 5.27 M trainable tokens

For training, see [`sft_training_04_07.md`](sft_training_04_07.md).
