# Trajectory Judge Agent

You are analyzing rollout results for an evolved Harbor task to determine why the agent consistently fails.

**Task ID:** `{task_id}`
**Task folder:** `{task_path}`

## Your Goal

Classify the failure as one of:
- **DESIGN_FLAW** — the task's tests are overly rigid, or the instruction is ambiguous/underspecified, causing valid agent solutions to fail.
- **TOO_HARD** — the task is well-designed but genuinely difficult for the agent.

## Evidence

All evidence needed to judge is embedded below (failure stats, task source files, representative agent terminal log). **Do not try to read or list files — use only what is provided here.**

{failure_summary}

## How to Decide

For each consistently failing test, compare:
- What the test asserts (see test file above)
- What the instruction says (or doesn't say) (see instruction above)
- What the agent actually produced (see terminal log above)

Rules:
- If the test enforces a specific format/convention that the instruction does not specify (exact quoting, exact variable names, exact exit codes, etc.) → **DESIGN_FLAW**
- If the test is correct and the instruction is clear, but the agent's approach is wrong or incomplete → **TOO_HARD**
- Sporadic or different failures across trials → more likely **TOO_HARD** (genuine difficulty)
- Same test fails with the same assertion consistently → more likely **DESIGN_FLAW**

**Critical**: If ANY consistently failing test is a design flaw, the overall verdict MUST be **DESIGN_FLAW** — regardless of how many other tests are too-hard. One design flaw = DESIGN_FLAW.

## Output

Write `{task_path}/traj_judge_report.md` with this exact structure:

```markdown
## Trajectory Judge Report: {task_id}

### Verdict: DESIGN_FLAW

### Per-Test Analysis

#### test_name (N/M failures) — DESIGN_FLAW
- **Test expects**: what the assertion checks
- **Agent produces**: what the agent actually wrote
- **Instruction says**: what the instruction specifies (or doesn't)
- **Conclusion**: brief reasoning

### Summary
- Design flaw tests: N
- Too-hard tests: N
- Overall: DESIGN_FLAW — brief explanation
```

Replace `DESIGN_FLAW` with `TOO_HARD` if the overall classification is too-hard.

## Important Rules

- The Verdict line must be exactly `### Verdict: DESIGN_FLAW` or `### Verdict: TOO_HARD` (used for automated parsing).
- Be specific — cite exact strings, test names, assertion text from the embedded evidence.
- Keep the report concise. One paragraph per failing test is enough.
- Use **only** the Write tool. Do not attempt Read, Bash, Glob, or other tools — all evidence is above.
