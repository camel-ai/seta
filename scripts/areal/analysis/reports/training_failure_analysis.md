# Training Failure Analysis

Run analyzed:
`scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/complete_history.csv`

Config analyzed:
`scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/config.csv`

## Short answer

I do not see evidence of PPO numerically blowing up around step 30-40.

What I do see is:
- reward becomes noisy after the early rise
- PPO update statistics stay extremely conservative
- the run is configured in a way that makes the training signal sparse, high-variance, and distribution-shifted

So the more likely diagnosis is:
- weak / noisy RL signal
- selection bias from filtering
- GRPO grouping not actually being used even though `n_trajs=16`

not:
- exploding KL
- exploding importance ratios
- clipping saturation
- optimizer divergence

## What the curves say

Observed from the history:

- `rollout/reward` rises early from about `0.49` at step `1` to about `1.02` at step `21`, then oscillates heavily and ends around `0.77`.
- `ppo_actor/task_reward/avg` shows the same pattern: early improvement, then high variance, not a monotonic collapse.
- `ppo_actor/update/importance_weight/avg` stays at `~1.0` for the whole run.
- `ppo_actor/update/approx_kl/avg` stays at `~0.0` for the whole run.
- `ppo_actor/update/clip_ratio/avg` stays at `0.0`.
- `ppo_actor/update/behave_imp_weight/avg` stays near `1.0`, and max stays below the cap `5.0`.
- `ppo_actor/update/grad_norm` is modest for the whole run, mostly around `0.14` to `0.32`, with one spike to `0.70` at step `39`, but updates remain successful.
- `ppo_actor/update/entropy/avg` trends down from about `0.35` early to about `0.11` to `0.20` later, so the policy becomes more confident over time.

Implication:
- the actor is changing only slightly per step
- rollout data is not strongly off-policy
- reward degradation is not explained by PPO stepping too hard

## Important metric caveat

`rollout/num_full_passes` is invalid for this run.

Reason:
- the code counts full passes with `reward == 1.0`
- this run sets `encourage_completion_reward=True`
- then a full pass is changed from `1.0` to `2.0`

Code:
- full-pass counting: `src/tbench_areal_workflow/train.py:605`
- bonus reward: `src/tbench_areal_workflow/train.py:481`

So if you were using `rollout/num_full_passes` to conclude that fully solved tasks disappeared, that conclusion is not reliable.

## Most likely causes

### 1. `n_trajs=16`, but PPO `group_size=1`

This is the biggest config issue.

Your workflow collects `16` trajectories per task:
- `src/tbench_areal_workflow/train.py:515`

But the actor config uses:
- `actor.group_size = 1`

Code:
- config snapshot in `config.csv`
- actor uses `group_size` at `external/areal/areal/engine/ppo/actor.py:33`

Implication:
- you are not using multi-sample relative grouping in the PPO loss
- each trajectory is effectively trained independently
- with sparse terminal reward, that creates much noisier credit assignment

For this task family, that means you pay the cost of sampling `16` trajectories but do not fully exploit them as a grouped relative signal.

### 2. `filter_uniform_reward=True` creates a moving, biased training distribution

Code:
- `src/tbench_areal_workflow/train.py:578`

This discards tasks whenever all valid trajectories have the same reward.

Implication:
- early in training, many uniformly bad tasks get dropped
- later in training, many uniformly easy tasks also get dropped
- the remaining training set is the hardest / most variance-heavy slice

This can easily produce:
- early reward improvement
- later apparent reward drop or plateau

even if the model is still improving on the original task distribution.

This is one of the strongest explanations for “went wrong after 30-40 steps”.

### 3. Reward shaping is very aggressive

Raw reward is test `pass_ratio`, with a bonus for full completion:
- `src/tbench_areal_workflow/train.py:458`
- `src/tbench_areal_workflow/train.py:481`

Then PPO reshapes it as:

```text
(raw_reward - 0.5) * 10
```

Code:
- `external/areal/areal/engine/ppo/actor.py:112`

Implication:
- below `0.5` pass ratio, the shaped reward is negative
- at exactly `0.5`, shaped reward is `0`
- full completion with bonus becomes `15`

That is a large scale for a sparse terminal reward placed near the end of a long sequence.

This is not causing instability in the PPO metrics, but it does make the signal high-variance and heavily winner-take-all.

### 4. Sequence lengths are very long relative to where reward is injected

Observed:
- `ppo_actor/seq_len/avg` mean is about `5487`
- `ppo_actor/prompt_len/avg` mean is about `4146`
- so the average response is still about `1340` tokens

Reward injection happens near the end of the response:
- `external/areal/areal/engine/ppo/actor.py:142`

Implication:
- a single terminal reward has to propagate backward through long trajectories
- with `group_size=1`, this becomes noisy
- `ppo_actor/final_reward/avg` stays near zero because it is averaged over all valid tokens, which is expected for long sequences and sparse terminal reward

This does not prove failure, but it explains weak learning signal.

### 5. No KL anchoring

Config:
- `actor.kl_ctl = 0.0`

Code:
- `external/areal/areal/engine/ppo/actor.py:139`

Implication:
- there is no explicit reward penalty for drifting from the reference policy

In this run, PPO still stays conservative, so this is not the immediate cause of the 30-40 step behavior.
But it removes one stabilizer, which can matter more later if learning rate or reward scale increases.

## Things that are probably not the cause

### Not PPO clipping instability

Evidence:
- `ppo_actor/update/clip_ratio/avg = 0`
- `ppo_actor/update/importance_weight/avg ~ 1`
- `ppo_actor/update/approx_kl/avg ~ 0`

Interpretation:
- the updates are tiny
- clipping is basically never active

### Not off-policy drift from async training

Evidence:
- `behave_imp_weight/avg ~ 1`
- `behave_approx_kl/avg` remains tiny
- cap `5.0` is never close to binding

Interpretation:
- async staleness is present in design, but not large in the recorded data

### Not optimizer explosion

Evidence:
- `update_successful` stays `1`
- grad norm remains finite and modest

## Practical implications

If you want this run family to behave better, the highest-value changes are:

1. Set actor grouping to match the multi-trajectory design.
   - `actor.group_size` should be aligned with the number of trajectories that belong to the same prompt, or otherwise the GRPO-style relative signal is largely lost.

2. Turn off or reduce `filter_uniform_reward` during debugging.
   - Otherwise the reward curve is measured on a shifting subset of tasks.

3. Reduce reward shaping aggressiveness first, then re-evaluate.
   - The current `(reward - 0.5) * 10` plus full-pass bonus is strong.

4. Shorten trajectories or constrain generation more.
   - `gconfig.max_new_tokens=10240` and `max_tokens_per_trajectory=28672` are very large for sparse terminal reward training.

5. Fix `rollout/num_full_passes` before using it operationally.
   - It is currently inconsistent with `encourage_completion_reward=True`.

## Bottom line

My definite read is:

- the run did not “go wrong” because PPO became unstable after step 30-40
- it more likely plateaued into a noisy, biased training regime created by the config

The two strongest config-level culprits are:
- `actor.group_size = 1` despite `n_trajs = 16`
- `filter_uniform_reward = True`

The strongest reward-design contributors are:
- sparse terminal reward only
- aggressive reward shaping with `reward_bias=-0.5`, `reward_scaling=10`
- very long trajectories
