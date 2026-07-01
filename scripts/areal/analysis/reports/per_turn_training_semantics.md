# Per-Turn Training Semantics For This AReaL Agent Run

This note explains the exact training semantics for the run:

- run history: `scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/complete_history.csv`
- run config: `scripts/areal/analysis/wandb_downloads/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train/config.csv`

The key detail is:

- the agent exports `style="individual"`
- each interaction / turn is trained as its own RL sample
- future reward is pushed backward by `apply_reward_discount(...)`
- PPO advantage and loss are then computed inside each exported turn separately

That is different from training one full multi-turn conversation as one single trajectory tensor.

## 1. What the workflow exports

In the terminal-agent workflow:

- final reward is set on the most recent interaction:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L277)
- then reward is propagated backward across earlier turns:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L596)
- then all interactions are exported individually:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L597)

The actual reward propagation logic is in AReaL's interaction cache:

- `set_final_reward(reward)` assigns reward to the latest interaction only:
  - [`cache.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/experimental/openai/cache.py#L19)
- `apply_reward_discount(turn_discount=0.9)` then runs backwards:

```text
reward[i] += reward[i+1] * turn_discount
```

- code:
  - [`cache.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/experimental/openai/cache.py#L24)

`export_interactions(style="individual")` then returns every interaction as a separate training sample:

- [`cache.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/experimental/openai/cache.py#L215)

## 2. What one exported sample means

One exported sample is one model interaction, not the whole conversation.

Each sample contains:

- that turn's prompt tokens
- that turn's response tokens
- token logprobs from rollout
- one scalar reward attached to that turn

That reward is:

- the final task reward if it is the last turn
- or the discounted future reward if it is an earlier turn

So for a 3-turn conversation with final reward `R` and `turn_discount=0.9`, the exported scalar rewards are approximately:

```text
turn 3: R
turn 2: 0.9 * R
turn 1: 0.9^2 * R
```

assuming earlier turns did not already have their own explicit reward.

## 3. What reward is used by PPO in this run

### Raw rollout reward

The terminal-agent reward function computes:

- `pass_ratio`
- and if `encourage_completion_reward=True`, adds `+1.0` when `pass_ratio == 1.0`

Code:

- pass ratio:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L458)
- bonus:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L481)

### Per-turn discounted reward

After `apply_reward_discount(turn_discount=0.9)`, each earlier interaction gets discounted credit from later ones:

- [`cache.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/experimental/openai/cache.py#L24)

### PPO-shaped reward

Inside `compute_advantages`, AReaL reshapes the scalar sample reward:

```python
reward_score = data["rewards"]
reward_score = (reward_score + self.reward_bias) * self.reward_scaling
reward_score = torch.clip(reward_score, max=self.reward_clip, min=-self.reward_clip)
```

Code:

- [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L111)

For this run:

- `reward_bias = -0.5`
- `reward_scaling = 10`
- `reward_clip = 20`

So the actual reward used by PPO is:

```text
shaped_reward = clip((discounted_turn_reward - 0.5) * 10, -20, 20)
```

Important implication:

- because reward discounting happens before PPO shaping, earlier turns can still receive substantial positive or negative shaped reward
- the training is therefore per-turn PPO with discounted terminal labels

## 4. How advantage is computed in this case

For each exported turn sample:

1. A scalar shaped reward is available for that sample.
2. AReaL inserts that scalar reward near the end of the sample's response sequence.
3. Then it runs GAE backward over tokens inside that one sample only.

The relevant code:

- reward insertion:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L142)
- GAE loop:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L151)

This means:

- the advantage is token-level
- but only within that turn's prompt-response sequence
- not over the full multi-turn dialogue as one sequence

So the training signal for an early turn is:

- not “full-trajectory policy gradient over all later tokens”
- but “this turn gets a discounted scalar reward label, then PPO is run on this turn's tokens”

## 5. How PPO loss is computed in this case

The run uses:

- `use_decoupled_loss = true`
- `recompute_logprob = true`
- `importance_sampling_level = token`

So for each valid token in each exported turn sample:

1. current model logprob is recomputed
2. proximal ratio is computed token-wise against `proximal_logprobs`
3. PPO clipped policy loss is formed token-wise
4. behavior importance weighting is applied for stale rollout correction

Code:

- PPO loss entry:
  - [`functional.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/utils/functional.py#L264)
- token-level ratio:
  - [`functional.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/utils/functional.py#L301)
- behavior correction:
  - [`functional.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/utils/functional.py#L328)

Important implication:

- the run is not doing full-conversation GRPO
- and not doing prompt-group relative normalization by default
- it is doing token-level PPO on many individually exported turn samples

## 6. What `ppo_actor/*` means in this per-turn setup

### `ppo_actor/task_reward/*`

This is the per-sample scalar reward before PPO reward shaping.

In this run that means:

- for the last turn: raw final reward
- for earlier turns: discounted future reward after `apply_reward_discount(0.9)`

It is logged from:

- [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L238)

So in this setup, `ppo_actor/task_reward/avg` is not “average final episode reward.”
It is “average per-exported-turn reward label.”

### `ppo_actor/final_reward/*`

This is token-level reward after:

- KL reward term
- scalar task reward insertion near the end token

Logged from:

- [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L231)

Because the reward is sparse and averaged across valid tokens, this value is usually much smaller than `task_reward`.

### `ppo_actor/advantages/*`

These are token-level advantages after optional advantage normalization.

In this run:

- advantage normalization is batch-level, not group-level
- so `advantages/avg` is expected to stay near zero

Code:

- normalization call:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L174)

### `ppo_actor/seq_len/*`, `prompt_len/*`

These are per-exported-turn lengths, not whole-dialogue lengths.

That matters because:

- if one dialogue produces multiple turns, each turn contributes separately
- the chart is averaging over exported turn samples

## 7. How entropy is calculated in this case

### What entropy is

For each valid token position of each exported turn sample, AReaL computes:

```text
entropy_t = -sum_v p(v|context_t) log p(v|context_t)
```

over the whole vocabulary.

Code:

- token entropy computation:
  - [`functional.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/utils/functional.py#L24)
- logged inside PPO update:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L399)

### What `ppo_actor/update/entropy/avg` means

It is:

- the average token entropy
- over all valid loss tokens
- across all exported turn samples in the PPO batch

It is not:

- entropy of one turn
- entropy of the full dialogue
- entropy of the final answer only

It is a pooled average over all valid tokens in all per-turn samples in that update.

### Can you compare entropy across turns?

Not directly from the logged chart, because the chart is aggregated across all valid tokens in the update.

If you wanted “entropy per turn,” you would need extra logging before batch aggregation.

## 8. How grad norm is calculated in this case

### What is actually computed

After the PPO loss has been backpropagated across all microbatches in the update, AReaL computes one global gradient norm over model parameters:

- [`fsdp_engine.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/fsdp_engine.py#L644)

This is not “one grad norm per turn.”

It is:

- one norm over all trainable parameters
- after accumulating gradients from the full PPO batch
- where the full batch already contains many exported turn samples

### What the reported `ppo_actor/update/grad_norm` means

It means:

- the global parameter-gradient norm for the entire actor update step
- after summing contributions from all exported turn samples in that update
- before deciding whether the optimizer step is valid

Code path:

- backward over all microbatches:
  - [`fsdp_engine.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/fsdp_engine.py#L625)
- grad norm:
  - [`fsdp_engine.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/fsdp_engine.py#L644)
- log return:
  - [`fsdp_engine.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/fsdp_engine.py#L657)

### What it is not

It is not:

- grad norm of one parameter
- grad norm of one token
- grad norm of one turn
- grad norm of one trajectory

Each parameter has its own gradient tensor internally, but the logged metric is a single aggregate norm over the whole parameter set.

### Why this matters

If the chart shows a spike in `grad_norm`, that means:

- the whole update batch produced a stronger-than-usual aggregate gradient

It does not let you conclude:

- which exact turn caused it
- which exact parameter caused it

without more detailed instrumentation.

## 9. Why reward scaling still matters here

A common confusion is:

- “If this is GRPO-like multi-trajectory training, shouldn't rewards get normalized away?”

In this run, no.

Reason:

- `reward_norm = None`
- `adv_norm` is batch-level normalization, not reward normalization
- `style="individual"` turns conversations into separate per-turn samples

So reward shaping directly changes:

- the scalar reward attached to each exported turn
- the token-level advantages produced inside each turn
- therefore the batch gradient magnitude and the pressure toward sharpening

## 10. Why this setup can sharpen the policy

Putting the pieces together:

1. final reward is attached only at the end
2. reward is discounted backward across turns
3. each turn is exported and trained separately
4. PPO shaping is applied to each turn reward
5. token-level advantages are computed inside each turn
6. all turn samples are mixed into one PPO update
7. one global grad norm is reported for the whole update

So the system is effectively:

- per-turn PPO
- with discounted final reward labels
- without full-trajectory end-to-end policy optimization
- and in your run, with aggressive reward shaping and no KL anchor

That is a setup where:

- some batches can create large aggregate gradients
- entropy can gradually collapse even if local PPO ratios stay benign

## 11. Practical interpretation of the charts in this specific setup

### If `ppo_actor/task_reward/avg` changes

Interpretation:

- the average reward label over exported turn samples changed
- not necessarily the average final episode success directly

### If `ppo_actor/update/entropy/avg` drops

Interpretation:

- across all valid tokens in all exported turn samples, the policy got more peaked

### If `ppo_actor/update/grad_norm` spikes

Interpretation:

- this PPO update step, aggregating all exported turn samples in the batch, produced a stronger global gradient than usual

### If `importance_weight` and `clip_ratio` remain benign while entropy falls

Interpretation:

- the policy is not jumping too far per PPO step
- but it may still be drifting cumulatively toward sharper token distributions

## 12. Bottom line

For this run, the “devil in the details” summary is:

- training is not one-full-dialogue GRPO
- training is per-turn PPO on `individual` exports
- earlier turns get discounted copies of the final reward
- entropy is averaged over all valid tokens in the update
- grad norm is one global parameter-gradient norm for the whole update

So:

- `entropy/avg` is an update-level pooled token-entropy metric
- `grad_norm` is an update-level pooled parameter-gradient metric
- neither metric is “per turn” in the logged charts, even though the data entering the update is composed of individual turns

## 13. Correction Note About `importance_weight` And `approx_kl`

For this run, these two metrics are effectively degenerate:

- `ppo_actor/update/importance_weight/*`
- `ppo_actor/update/approx_kl/*`

Reason:

- the training loop recomputes `prox_logp` immediately before PPO:
  - [`train.py`](/Users/qijia/Documents/code/terminal_agent/src/tbench_areal_workflow/train.py#L731)
- inside the PPO loss, `logprobs` are recomputed again from the same current actor weights:
  - [`actor.py`](/Users/qijia/Documents/code/terminal_agent/external/areal/areal/engine/ppo/actor.py#L352)
- the run uses only one PPO minibatch / pass per update:
  - run config snapshot

So for this run's update:

```text
logprobs ~= proximal_logprobs
```

and therefore:

```text
importance_weight = exp(logprobs - proximal_logprobs) = 1
approx_kl = logprobs - proximal_logprobs = 0
```

This is not evidence that the metrics are mis-logged.
It is evidence that, under this exact configuration, they are not informative.

The informative drift metrics for this run are instead:

- `behave_imp_weight`
- `behave_approx_kl`

because those compare:

- recomputed proximal logprobs
against
- rollout-time `old_logprobs`
