# Experiment Plan: Fast Iteration on Evol Pipeline

## Test Set

5 tasks from seta-env-harbor (diverse: scripting, config, security, image, systemd):

| Task ID | Domain | Summary |
|---------|--------|---------|
| 1043 | Bash scripting | CPU power profile switcher based on process detection |
| 1202 | System admin | Multi-user sudo policies with fine-grained restrictions |
| 210 | Security | Secure disk wipe with multiple sanitization standards |
| 267 | Image processing | Batch image resizing with subdirectory output |
| 939 | Systemd | Diagnose and fix inconsistent systemd service states |

Filter: `configs/filters/test_5tasks.csv`
Input: `outputs/synth_data/` (already downloaded)

## Experiment Runs

### Run 1: INCREASE_DIFFICULTY

```bash
cd datasynth/evol_pipeline

# Dry run first
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml --dry-run

# Full run (evol + datapoint + harbor validation)
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml
```

**What to check after:**
- `outputs/evol_increase_difficulty/{task_id}__d1/draft_spec.md` — is the evolved task genuinely harder?
- `outputs/evol_increase_difficulty/{task_id}__d1/judge_report.md` — did it PASS all 7 criteria?
- `outputs/evol_increase_difficulty/{task_id}__d1/synth_info.json` — status and timing
- Oracle passes (1.0), empty fails (0.0)?

### Run 2: CHANGE_CONTEXT

```bash
python run_evol_orchestrator.py configs/test_change_context.yaml --dry-run
python run_evol_orchestrator.py configs/test_change_context.yaml
```

**What to check after:**
- `outputs/evol_change_context/{task_id}__b1/draft_spec.md` — is the domain genuinely different?
- Tests use correct commands/syntax for the new technology?
- Evolution fidelity criterion passes?

### Run 3: INCREASE_DIFFICULTY Turn 2 (chain: harder → harder)

```bash
# Only run after Run 1 completes — reads PASS variants from Run 1 output
python run_evol_orchestrator.py configs/test_increase_difficulty_turn2.yaml --dry-run
python run_evol_orchestrator.py configs/test_increase_difficulty_turn2.yaml
```

Produces: `1043__d1` → `1043__d1__d1` (double-hard)

**What to check:**
- Is it genuinely harder than the turn 1 variant?
- Does it avoid repeating the same difficulty increase?
- Does the evol agent reference the turn 1 task (not the original)?

### Run 4: INCREASE_DIFFICULTY after CHANGE_CONTEXT (chain: new domain → harder)

```bash
# Only run after Run 2 completes — reads PASS variants from change_context output
python run_evol_orchestrator.py configs/test_increase_after_context.yaml --dry-run
python run_evol_orchestrator.py configs/test_increase_after_context.yaml
```

Produces: `1043__b1` → `1043__b1__d1` (new domain, then harder)

**What to check:**
- Does it increase difficulty within the *new* domain (not fall back to original)?
- Tests use the new domain's commands/syntax?

### Rollout (GPT-5.4)

Rollout is enabled in all configs. After each run completes, check:
- `outputs/evol_*_rollouts/{model_config_name}/{variant_id}/` — trajectory logs
- Does GPT-5.4 solve the evolved tasks? Compare pass rates across chains.

### Run Order

```
Run 1: test_increase_difficulty.yaml      → outputs/evol_increase_difficulty/
Run 2: test_change_context.yaml           → outputs/evol_change_context/
Run 3: test_increase_difficulty_turn2.yaml → outputs/evol_increase_difficulty_t2/     (needs Run 1)
Run 4: test_increase_after_context.yaml   → outputs/evol_change_context_then_harder/ (needs Run 2)
```

Runs 1 & 2 are independent (parallel). Runs 3 & 4 depend on 1 & 2 respectively.

## Iteration Loop

```
┌─────────────────────────────────────────┐
│  1. Run on 5 tasks                      │
│  2. Inspect outputs (see checklist)     │
│  3. Identify issues                     │
│  4. Fix prompt / strategy / pipeline    │
│  5. Re-run (resumability skips PASS)    │
│     └─ delete synth_info.json to retry  │
│  6. Repeat until quality is good        │
│  7. Scale to full dataset               │
└─────────────────────────────────────────┘
```

## Inspection Checklist

After each run, check these for every variant:

### draft_spec.md (evol agent output)
- [ ] Strategy applied correctly? (harder / different domain)
- [ ] Agent Task DAG has ≥5 steps?
- [ ] Environment setup is realistic?
- [ ] Test design covers the key requirements?
- [ ] External resources referenced where needed?

### Harbor task files (datapoint agent output)
- [ ] All files present? (task.toml, instruction.md, Dockerfile, solve.sh, tests, weights.json)
- [ ] instruction.md doesn't leak the solution?
- [ ] Dockerfile installs correct packages for the domain?
- [ ] solve.sh is realistic (not test-passing tricks)?
- [ ] Tests are deterministic and non-vacuous?

### judge_report.md (self-assessment)
- [ ] All 7 criteria PASS?
- [ ] Evolution fidelity specifically addressed?
- [ ] Oracle passed (1.0), empty failed (0.0)?

### Common failure patterns to watch for
- Evol agent picks a trivial change (not genuinely harder/different)
- Datapoint agent simplifies the draft_spec instead of implementing it fully
- Dockerfile has typos in package names → build fails
- Tests pass on empty (vacuous assertions)
- solve.sh uses hardcoded paths that don't match the Dockerfile
- CHANGE_CONTEXT falls back to the original technology

## Quick Fix Recipes

| Issue | Where to fix | How |
|-------|-------------|-----|
| Evol agent picks trivial changes | `agents/evol_strategy_prompts/*.md` | Add stronger examples of what qualifies |
| Datapoint agent simplifies draft_spec | `agents/datapoint_agent_guide/agent.md` | Strengthen evolution fidelity criterion |
| Oracle fails consistently | `agent.md` Pre-Flight Review section | Add more pre-flight checks |
| Empty passes (vacuous tests) | `agent.md` Test Quality section | Add stricter anti-vacuous rules |
| Wrong packages in Dockerfile | Strategy adapter | Add domain-specific package hints |
| Rollout agent can't solve evolved task | Check if task is too hard | Adjust difficulty guidance in adapter |

## Forcing a Retry

To re-run a specific variant, delete its synth_info.json:
```bash
rm outputs/evol_increase_difficulty/1043__d1/synth_info.json
# Re-run — only this variant will be re-processed
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml
```

To re-run everything:
```bash
rm -rf outputs/evol_increase_difficulty/
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml
```

## Scaling Up

Once the 5-task test produces good results:

```bash
# Generate filter CSVs for the full dataset
python run_evol_orchestrator.py configs/test_increase_difficulty.yaml --generate-filters --n-parts 4

# Run on multiple machines
# Machine 1: filter_csv: configs/filters/part_01.csv
# Machine 2: filter_csv: configs/filters/part_02.csv
# etc.
```
