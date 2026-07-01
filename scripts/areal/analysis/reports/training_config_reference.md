# Training Config Reference

Run source:
`scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/config.csv`

This file is the resolved config snapshot stored by W&B for the actual run. It is more authoritative for this run than the checked-in template yaml.

## Top-level run parameters

| Config key | Run value | Meaning | Primary code path |
| --- | --- | --- | --- |
| `n_trajs` | `16` | Number of trajectories sampled per task during rollout. | `src/tbench_areal_workflow/train.py:496`, `src/tbench_areal_workflow/train.py:524` |
| `max_tokens_per_trajectory` | `28672` | Max total token budget passed into the CAMEL agent per trajectory. This is separate from per-turn generation length. | `src/tbench_areal_workflow/train.py:497`, `src/tbench_areal_workflow/train.py:555` |
| `max_iteration` | `20` | Max number of agent iterations per trajectory. | `src/tbench_areal_workflow/train.py:498`, `src/tbench_areal_workflow/train.py:423` |
| `max_workers` | `256` | Size of the shared thread pool used for Docker / task operations. | `src/tbench_areal_workflow/train.py:499`, `src/tbench_areal_workflow/train.py:522` |
| `async_training` | `True` | Enables asynchronous rollout preparation through `actor.prepare_batch(...)`. | `src/tbench_areal_workflow/train.py:713` |
| `filter_uniform_reward` | `True` | Drops a task entirely if all valid trajectories have the same reward. This changes the effective training distribution. | `src/tbench_areal_workflow/train.py:578` |
| `encourage_completion_reward` | `True` | Adds `+1.0` reward bonus when `pass_ratio == 1.0`, so a full solve becomes reward `2.0`. | `src/tbench_areal_workflow/train.py:481` |
| `train_dataset.batch_size` | `16` | Number of prompts per train step before each prompt expands into `n_trajs` trajectories. | `src/tbench_areal_workflow/train.py:641` |
| `train_dataset.max_length` | `1024` | Dataset-side max prompt length before rollout. | dataset config only; used when building dataloader in `src/tbench_areal_workflow/train.py:635` |
| `total_train_epochs` | `10` | Full passes over the dataset. | `src/tbench_areal_workflow/train.py:698` |
| `rollout.max_concurrent_rollouts` | `8` | Max rollout jobs the scheduler allows concurrently. Lower than `n_trajs=16`, so trajectories are internally queued. | scheduler config passed to rollout engine at `src/tbench_areal_workflow/train.py:648` |
| `rollout.max_head_offpolicyness` | `2` | Allows rollout data to be at most 2 policy versions stale at the head of the queue. Relevant only because `async_training=True`. | rollout config passed to `RemoteSGLangEngine` at `src/tbench_areal_workflow/train.py:648` |
| `gconfig.max_new_tokens` | `10240` | Max new tokens per model turn during agent generation. | `src/tbench_areal_workflow/train.py:554` |
| `gconfig.temperature` | `1.0` | Sampling temperature for rollout generation. | rollout config, and reused by actor at `external/areal/areal/engine/ppo/actor.py:64` |

## PPO / actor parameters

| Config key | Run value | Meaning | Primary code path |
| --- | --- | --- | --- |
| `actor.optimizer.lr` | `1.7e-5` | Actor learning rate. | returned in update stats from `external/areal/areal/engine/fsdp_engine.py:657` |
| `actor.optimizer.weight_decay` | `0.017` | Weight decay applied by Adam optimizer. | optimizer setup inside engine |
| `actor.optimizer.gradient_clipping` | `1.0` | Grad-norm threshold used before stepping the optimizer. | `external/areal/areal/engine/fsdp_engine.py:644` |
| `actor.group_size` | `1` | PPO grouping parameter. With `1`, trajectories are treated independently rather than as a relative multi-sample group. | stored in actor at `external/areal/areal/engine/ppo/actor.py:33` |
| `actor.eps_clip` | `0.4` | PPO ratio clip window. | passed into loss at `external/areal/areal/engine/ppo/actor.py:281`, used at `external/areal/realhf/impl/model/utils/ppo_functional.py:99` |
| `actor.kl_ctl` | `0.0` | Weight of KL reward penalty against the reference policy. This disables KL regularization in `kl_rewards`. | `external/areal/areal/engine/ppo/actor.py:35`, `external/areal/areal/engine/ppo/actor.py:139` |
| `actor.reward_scaling` | `10` | Multiplies the raw sequence reward after biasing. | `external/areal/areal/engine/ppo/actor.py:112` |
| `actor.reward_bias` | `-0.5` | Reward shaping bias added before scaling. Effective shaped reward is `(raw_reward - 0.5) * 10`. | `external/areal/areal/engine/ppo/actor.py:112` |
| `actor.reward_clip` | `20` | Final clamp on shaped reward. | `external/areal/areal/engine/ppo/actor.py:114` |
| `actor.adv_norm` | batch mean/std normalization enabled | Advantage normalization across the batch. This forces `ppo_actor/advantages/avg` near zero by design. | `external/areal/areal/engine/ppo/actor.py:174` |
| `actor.reward_norm` | `None` | No normalization of raw shaped reward before GAE. | `external/areal/areal/engine/ppo/actor.py:117` |
| `actor.discount` | `1` | Discount factor for GAE. Because only a terminal task reward is injected, this keeps credit fully propagated backward. | `external/areal/areal/engine/ppo/actor.py:162` |
| `actor.gae_lambda` | `1` | Lambda for GAE. With no critic and sparse terminal reward, this becomes Monte-Carlo style return propagation. | `external/areal/areal/engine/ppo/actor.py:163` |
| `actor.ppo_n_minibatches` | `1` | Number of PPO minibatches per update. One rollout batch produces one optimizer step. | `external/areal/areal/engine/ppo/actor.py:268` |
| `actor.recompute_logprob` | `True` | Recomputes proximal logprobs on the current policy before training. | `src/tbench_areal_workflow/train.py:731` |
| `actor.use_decoupled_loss` | `True` | Uses decoupled asynchronous PPO loss with proximal logprobs. This changes importance weights to be measured against `prox_logp`, not the original rollout logprobs. | `external/areal/areal/engine/ppo/actor.py:123`, `external/areal/realhf/impl/model/utils/ppo_functional.py:84` |
| `actor.behav_imp_weight_cap` | `5` | Caps behavior importance weights in decoupled training so very stale data does not dominate updates. | `external/areal/realhf/impl/model/utils/ppo_functional.py:121` |
| `actor.importance_sampling_level` | `token` | Importance ratio is computed token-wise, not sequence-wise. | `external/areal/areal/engine/ppo/actor.py:287` |
| `actor.dynamic_sampling` | `False` | Disables dynamic sample filtering / regrouping. | `external/areal/areal/engine/ppo/actor.py:191` |
| `actor.mask_no_eos_with_zero` | `False` | Sequences without EOS still receive the terminal task reward. | `external/areal/areal/engine/ppo/actor.py:144` |
| `actor.max_new_tokens` | `10240` | Max response length assumed by the actor side. | config only; used for overlong penalty logic when enabled |

## Reward interpretation for this run

Raw rollout reward:
- `pass_ratio` from parsed unit tests
- full pass bonus: `+1.0` when `encourage_completion_reward=True`

Code:
- `src/tbench_areal_workflow/train.py:458`
- `src/tbench_areal_workflow/train.py:481`

Shaped PPO reward used in advantage computation:

```text
reward_score = clip((raw_reward + reward_bias) * reward_scaling, -reward_clip, reward_clip)
             = clip((raw_reward - 0.5) * 10, -20, 20)
```

Code:
- `external/areal/areal/engine/ppo/actor.py:112`

Examples for this run:

| Raw reward | Meaning | Shaped reward |
| --- | --- | --- |
| `0.0` | no tests passed | `-5.0` |
| `0.5` | half the tests passed | `0.0` |
| `1.0` | all tests passed, bonus disabled | `5.0` |
| `2.0` | all tests passed, bonus enabled in this run | `15.0` |

## Important run-template mismatch

The checked-in template [`src/tbench_areal_workflow/configs/config_8xh100-qwen3-8b.yaml`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/configs/config_8xh100-qwen3-8b.yaml#L1) is not the exact run snapshot.

Important differences visible in `config.csv`:
- `actor.path` is `camel-ai/tbench-qwen-sft-kimi-thinking-v3-epoch5`, not `Qwen/Qwen3-8B`
- `train_dataset.path` is `synth_data_convert_1376/train.parquet`, not `tbench-tasks_convert/train_filtered_easy.parquet`
- `train_dataset.batch_size` is `16`, not `8`
- `rollout.max_concurrent_rollouts` is `8`, not `16`

For diagnosis, use `config.csv`, not only the checked-in yaml.
