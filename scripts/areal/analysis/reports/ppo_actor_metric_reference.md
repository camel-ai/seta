# PPO Actor Metric Reference

Run source:
`scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/complete_history.csv`

W&B logging path:
- Metrics are accumulated in `areal.utils.stats_tracker`.
- They are exported by `stats_tracker.export_all(...)` in `src/tbench_areal_workflow/train.py`.
- They are committed to W&B by `StatsLogger.commit(...)` in `external/areal/areal/utils/stats_logger.py`.

Relevant export / commit code:
- `src/tbench_areal_workflow/train.py:773`
- `external/areal/areal/utils/stats_logger.py:109`
- `external/areal/areal/utils/stats_logger.py:139`

## Suffixes

For tensor stats logged with `stats_tracker.stat(...)`, AReaL records:

- `/avg`: masked mean over the chosen denominator
- `/min`: masked minimum over the chosen denominator
- `/max`: masked maximum over the chosen denominator

The reduction behavior is implemented in:
- `external/areal/areal/utils/stats_tracker.py:170`

## Metric Table

| Chart label | Meaning | Recorded at |
| --- | --- | --- |
| `ppo_actor/correct_n_seqs` | Count of sequences in the batch whose raw task reward is `> 0`. This is a denominator used by other stats, not a PPO loss term. | `external/areal/areal/engine/ppo/actor.py:202` |
| `ppo_actor/incorrect_n_seqs` | Count of sequences in the batch whose raw task reward is `<= 0`. | `external/areal/areal/engine/ppo/actor.py:202` |
| `ppo_actor/n_seqs` | Count of sequences in the PPO batch. | `external/areal/areal/engine/ppo/actor.py:217` |
| `ppo_actor/n_tokens` | Count of all token positions in the PPO batch tensor, including invalid positions before masking. | `external/areal/areal/engine/ppo/actor.py:217` |
| `ppo_actor/n_valid_tokens` | Count of valid loss tokens after the PPO loss mask is applied. | `external/areal/areal/engine/ppo/actor.py:217` |
| `ppo_actor/correct_seq_len/{avg,min,max}` | Sequence length statistics for sequences with raw reward `> 0`. Length is `attention_mask.sum(-1)`. | `external/areal/areal/engine/ppo/actor.py:224` |
| `ppo_actor/incorrect_seq_len/{avg,min,max}` | Sequence length statistics for sequences with raw reward `<= 0`. | `external/areal/areal/engine/ppo/actor.py:227` |
| `ppo_actor/advantages/{avg,min,max}` | Advantage values used by PPO after optional advantage normalization. These are token-level values over valid tokens. | `external/areal/areal/engine/ppo/actor.py:231`, source computation at `external/areal/areal/engine/ppo/actor.py:151` |
| `ppo_actor/kl_rewards/{avg,min,max}` | Token-level KL reward term before task reward is inserted. Computed as `-kl_ctl * kl_estimator(old_logp, ref_logp)`. In this run it is effectively zero because `actor.kl_ctl = 0`. | `external/areal/areal/engine/ppo/actor.py:231`, source computation at `external/areal/areal/engine/ppo/actor.py:135` |
| `ppo_actor/final_reward/{avg,min,max}` | Token-level final reward after adding the sequence task reward near the end of the response. This is averaged over valid tokens, so it is usually much smaller than per-sequence task reward. | `external/areal/areal/engine/ppo/actor.py:231`, source computation at `external/areal/areal/engine/ppo/actor.py:142` |
| `ppo_actor/no_eos_ratios/{avg,min,max}` | Fraction of sequences that hit the maximum sequence length without EOS. Per sequence this is `1.0` when `seqlen == attn_mask.shape[-1]`, else `0.0`. | `external/areal/areal/engine/ppo/actor.py:238` |
| `ppo_actor/task_reward/{avg,min,max}` | Raw per-sequence reward coming from the rollout workflow, before `reward_bias` / `reward_scaling` are applied inside PPO. In this project it is the parsed test `pass_ratio`, with an optional `+1.0` full-completion bonus. | `external/areal/areal/engine/ppo/actor.py:238`, reward source at `src/tbench_areal_workflow/train.py:458` and `src/tbench_areal_workflow/train.py:481` |
| `ppo_actor/prompt_len/{avg,min,max}` | Per-sequence prompt length, computed as `attention_mask.sum(-1) - loss_mask.sum(-1)`. | `external/areal/areal/engine/ppo/actor.py:238` |
| `ppo_actor/seq_len/{avg,min,max}` | Per-sequence total length, computed as `attention_mask.sum(-1)`. | `external/areal/areal/engine/ppo/actor.py:238` |
| `ppo_actor/mask_no_eos_with_zero` | Scalar config flag. If `True`, the terminal task reward is zeroed for sequences with no EOS. This run has it disabled. | `external/areal/areal/engine/ppo/actor.py:246` |
| `ppo_actor/eps_clip` | PPO clip parameter recorded from config. It controls the clip range around importance ratio `1.0`. | `external/areal/areal/engine/ppo/actor.py:246` |
| `ppo_actor/use_dual_clip` | `1` if dual-clip PPO is enabled (`c_clip != None`), else `0`. This run has it disabled. | `external/areal/areal/engine/ppo/actor.py:250` |
| `ppo_actor/behav_imp_weight_cap` | Scalar config value for capping behavior-policy importance weights in decoupled asynchronous training. | `external/areal/areal/engine/ppo/actor.py:255` |
| `ppo_actor/update/update_successful` | Whether the optimizer step succeeded for that PPO update. It becomes `0` when grad norm is non-finite and the update is skipped. | `external/areal/areal/engine/fsdp_engine.py:644` |
| `ppo_actor/update/grad_norm` | Gradient norm after clipping logic in the FSDP engine, before deciding whether the optimizer step is valid. | `external/areal/areal/engine/fsdp_engine.py:644` |
| `ppo_actor/update/lr` | Current learning rate returned by the LR scheduler after the batch update. | `external/areal/areal/engine/fsdp_engine.py:657` |
| `ppo_actor/update/n_tokens` | Token count for the PPO loss-function logging scope. Here `logits.shape[0]` positions are counted. | `external/areal/areal/engine/ppo/actor.py:391` |
| `ppo_actor/update/n_valid_tokens` | Count of valid PPO loss tokens in the loss-function scope. | `external/areal/areal/engine/ppo/actor.py:391` |
| `ppo_actor/update/clipped_tokens` | Count of valid tokens where PPO clipping is active, based on `clip_mask`. | `external/areal/areal/engine/ppo/actor.py:391` |
| `ppo_actor/update/dual_clipped_tokens` | Count of valid tokens where dual clipping is active, based on `dual_clip_mask`. | `external/areal/areal/engine/ppo/actor.py:391` |
| `ppo_actor/update/unclipped_behave_tokens` | Count of valid tokens retained after the behavior-importance-weight cap is applied in decoupled training. | `external/areal/areal/engine/ppo/actor.py:410` |
| `ppo_actor/update/importance_weight/{avg,min,max}` | PPO importance ratio `exp(new_logp - denorm_logprobs)`. With decoupled loss, `denorm_logprobs` is the proximal logprob. | `external/areal/areal/engine/ppo/actor.py:399`, formula at `external/areal/realhf/impl/model/utils/ppo_functional.py:96` |
| `ppo_actor/update/approx_kl/{avg,min,max}` | Logged as `new_logp - denorm_logprobs`. This is AReaL's per-token approximate KL-like drift signal used for PPO monitoring. | `external/areal/areal/engine/ppo/actor.py:399`, formula at `external/areal/realhf/impl/model/utils/ppo_functional.py:131` |
| `ppo_actor/update/new_logp/{avg,min,max}` | New policy log-probabilities of the sampled tokens under the current model. | `external/areal/areal/engine/ppo/actor.py:399` |
| `ppo_actor/update/old_logp/{avg,min,max}` | Logged old log-probabilities used in the PPO update input. In this setup they come from rollout logprobs rolled onto token positions. | `external/areal/areal/engine/ppo/actor.py:399`, prepared at `external/areal/areal/engine/ppo/actor.py:123` |
| `ppo_actor/update/entropy/{avg,min,max}` | Token-level policy entropy of the current logits. Higher means less peaked token distributions. | `external/areal/areal/engine/ppo/actor.py:399`, computed at `external/areal/areal/utils/functional.py:24` |
| `ppo_actor/update/actor_loss/{avg,min,max}` | Token-level detached PPO policy loss contribution before reduction. This is not a direct quality metric. | `external/areal/areal/engine/ppo/actor.py:399`, loss construction at `external/areal/realhf/impl/model/utils/ppo_functional.py:99` |
| `ppo_actor/update/clip_ratio/{avg,min,max}` | Fraction of valid tokens where PPO clipping selected the clipped objective. | `external/areal/areal/engine/ppo/actor.py:399`, mask created at `external/areal/realhf/impl/model/utils/ppo_functional.py:101` |
| `ppo_actor/update/dual_clip_ratio/{avg,min,max}` | Fraction of valid tokens where dual clipping selected the dual-clipped objective. | `external/areal/areal/engine/ppo/actor.py:399`, mask created at `external/areal/realhf/impl/model/utils/ppo_functional.py:107` |
| `ppo_actor/update/behave_imp_weight/{avg,min,max}` | Decoupled-loss behavior importance weight `exp(proximal_logprobs - old_logprobs)` after masking by the cap. It measures rollout-policy staleness relative to the training baseline. | `external/areal/areal/engine/ppo/actor.py:412`, formula at `external/areal/realhf/impl/model/utils/ppo_functional.py:113` |
| `ppo_actor/update/behave_approx_kl/{avg,min,max}` | Per-token behavior drift term `proximal_logprobs - old_logprobs`, masked by the behavior cap. | `external/areal/areal/engine/ppo/actor.py:412`, formula at `external/areal/realhf/impl/model/utils/ppo_functional.py:112` |
| `ppo_actor/update/vocab_min_logits/{avg,min,max}` | Minimum token logit across the full vocabulary for each position. Useful for spotting pathological logit ranges. | `external/areal/areal/engine/ppo/actor.py:417` |
| `ppo_actor/update/vocab_max_logits/{avg,min,max}` | Maximum token logit across the full vocabulary for each position. | `external/areal/areal/engine/ppo/actor.py:417` |

## Rollout-side metrics used with PPO metrics

| Chart label | Meaning | Recorded at |
| --- | --- | --- |
| `rollout/reward` | Raw reward per successful trajectory. In this project it is test `pass_ratio`, optionally plus `1.0` when all tests pass and `encourage_completion_reward=True`. Failed trajectories are not logged here. | `src/tbench_areal_workflow/train.py:595`, reward computed at `src/tbench_areal_workflow/train.py:458` and `src/tbench_areal_workflow/train.py:481` |
| `rollout/num_trajectories_failed` | Number of trajectories in the task batch whose reward evaluation returned `None`. | `src/tbench_areal_workflow/train.py:605` |
| `rollout/num_full_passes` | Number of trajectories with reward exactly `1.0`. In this run this metric is misleading because `encourage_completion_reward=True`, so a fully passing trajectory is changed from `1.0` to `2.0` before logging. | `src/tbench_areal_workflow/train.py:605` |

## Correction Note For This Run

For this specific run, the following metrics are effectively degenerate and should not be used to judge local PPO update size:

- `ppo_actor/update/importance_weight/{avg,min,max}`
- `ppo_actor/update/approx_kl/{avg,min,max}`

Why:

1. the run uses `recompute_logprob = true`
2. the run uses `use_decoupled_loss = true`
3. the run uses `ppo_n_minibatches = 1`
4. `prox_logp` is recomputed immediately before PPO update
5. then `logprobs` in the loss are recomputed again from the same current weights before any optimizer step happens

Relevant code:

- recompute `prox_logp`:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L731)
- loss consumes `prox_logp`:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L350)
- `logprobs` are recomputed in the loss:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L352)
- ratio / approx_kl definition:
  - [`functional.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/utils/functional.py#L303)

So in this run, approximately:

```text
logprobs == proximal_logprobs
importance_weight = exp(logprobs - proximal_logprobs) = 1
approx_kl = logprobs - proximal_logprobs = 0
```

That is why these fields are constant in `complete_history.csv`.

For this run, the more informative drift metrics are:

- `ppo_actor/update/behave_imp_weight/*`
- `ppo_actor/update/behave_approx_kl/*`

because those compare recomputed proximal logprobs against rollout-time `old_logprobs`, not against themselves.
