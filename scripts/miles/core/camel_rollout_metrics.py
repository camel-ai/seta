"""Custom rollout-logging hook: per-training-step agent/trajectory metrics to wandb.

Wired via `--custom-rollout-log-function-path camel_rollout_metrics.log_rollout_data`.
Called by miles' RolloutManager (miles/ray/rollout.py:_log_rollout_data) once per
rollout/training step with the TRAINING batch (`samples`, post dynamic-filter).

Each sample carries the full env_service run_info at
`sample.metadata["camel_run_info"]` (set by generate_with_camel.generate), which
includes `agent_summary` (iteration_count, termination_reason, total_tool_calls,
parse_error_count, …) and `error_info`.

NOTE on the timeout metric: the env total-step timeout makes the sample env-FAIL
(status→FAILED) and is filtered out by group_reward_filter before training, so it
does NOT appear here. The AGENT-step timeout (agent_astep) is still evaluated and
KEPT in the batch — so `agent_timeout_ratio` over this batch is the genuine
agent-timeout rate (any "timeout" in termination_reason / error_message here is an
agent timeout, env-step timeouts having already been filtered).

This hook returns False so miles' DEFAULT rollout/* logging still runs; our metrics
are logged under the same `rollout/step` key so they align on the wandb x-axis.
Everything is wrapped so a metric bug can never crash training.
"""
import logging

logger = logging.getLogger(__name__)
_PREFIX = "rollout/agent/"


def _pct(num, den):
    return (float(num) / den) if den else 0.0


def _stats(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    n = len(xs)
    return {
        "mean": sum(xs) / n,
        "p50": xs[n // 2],
        "p90": xs[min(n - 1, int(0.9 * n))],
        "max": xs[-1],
    }


def compute_agent_metrics(run_infos, step_seconds=None):
    """Pure function (unit-testable): list[run_info dict] -> {metric: value}."""
    n = len(run_infos)
    if n == 0:
        return {}
    iters, tool_calls, parse_errs = [], [], []
    n_timeout = n_env_fail = 0
    term = {"task_finished": 0, "max_iteration_reached": 0, "max_tokens_exceeded": 0,
            "agent_timeout": 0, "other": 0}
    for ri in run_infos:
        ri = ri or {}
        a = ri.get("agent_summary") or {}
        ei = ri.get("error_info") or {}
        ic = a.get("iteration_count")
        if isinstance(ic, (int, float)):
            iters.append(ic)
        tc = a.get("total_tool_calls")
        if isinstance(tc, (int, float)):
            tool_calls.append(tc)
        pe = a.get("parse_error_count")
        if isinstance(pe, (int, float)):
            parse_errs.append(pe)
        reason = str(a.get("important_termination_reason") or a.get("termination_reason") or "")
        msg = str(ei.get("error_message") or "")
        is_timeout = ("timeout" in reason.lower()) or ("timeout" in msg.lower())
        if is_timeout:
            n_timeout += 1
        if ei:
            n_env_fail += 1
        # termination bucket
        if is_timeout:
            term["agent_timeout"] += 1
        elif reason in term:
            term[reason] += 1
        else:
            term["other"] += 1

    out = {
        f"{_PREFIX}agent_timeout_ratio": _pct(n_timeout, n),
        f"{_PREFIX}env_error_ratio": _pct(n_env_fail, n),  # ~0 in trained batch (filtered); sanity check
        f"{_PREFIX}batch_size": n,
    }
    for k, v in _stats(iters).items():
        out[f"{_PREFIX}iterations_{k}"] = v
    for k, v in _stats(tool_calls).items():
        out[f"{_PREFIX}tool_calls_{k}"] = v
    if parse_errs:
        out[f"{_PREFIX}parse_errors_mean"] = sum(parse_errs) / len(parse_errs)
    for reason, cnt in term.items():
        out[f"{_PREFIX}term_{reason}_ratio"] = _pct(cnt, n)
    if step_seconds:
        ss = [s for s in step_seconds if isinstance(s, (int, float))]
        if ss:
            for k, v in _stats(ss).items():
                out[f"{_PREFIX}env_step_seconds_{k}"] = v
    return out


_STRICT_TITO_TYPES = ("special_token_count", "special_token_type", "non_assistant_text")


def _log_tito_mismatch_examples(rollout_id, samples, max_examples_per_type=3):
    """Surface the EXACT cause of tito_session_mismatch for root-causing.

    Each mismatch dict carries type/segment_index/expected_text/actual_text/detail
    (see miles/utils/chat_template_utils/token_seq_comparator.py:Mismatch). The rates
    go to wandb (rollout/tito_session_mismatch_rate/*) but the per-session diff is
    discarded — so we log a few concrete examples here. repr() is used deliberately to
    reveal whitespace (the sglang strip bug manifested as a trailing '\\n' difference).

    Strict types (special_*, non_assistant_text) are TITO/chat-template BUGS and logged
    at WARNING; assistant_text is tolerated (inherited prefix tokens) and logged at INFO.
    """
    try:
        by_type = {}  # type -> list[(uid, mismatch_dict)]
        counts = {}   # type -> int
        n_with_mismatch = 0
        for s in samples or []:
            md = getattr(s, "metadata", None) or {}
            if not isinstance(md, dict):
                continue
            mm = md.get("tito_session_mismatch")
            if not mm:
                continue
            n_with_mismatch += 1
            uid = md.get("instance_id") or md.get("uid") or md.get("index") or "?"
            for m in mm:
                t = m.get("type", "?")
                counts[t] = counts.get(t, 0) + 1
                bucket = by_type.setdefault(t, [])
                if len(bucket) < max_examples_per_type:
                    bucket.append((uid, m))
        if not counts:
            return
        logger.warning(
            "tito_mismatch step=%s samples_with_mismatch=%d/%d counts=%s",
            rollout_id, n_with_mismatch, len(samples or []), counts,
        )
        for t, bucket in by_type.items():
            strict = t in _STRICT_TITO_TYPES
            lvl = logger.warning if strict else logger.info
            for uid, m in bucket:
                lvl(
                    "tito_mismatch[%s%s] uid=%s seg=%s detail=%s\n  expected=%r\n  actual  =%r",
                    t, " STRICT-BUG" if strict else "", uid,
                    m.get("segment_index"), m.get("detail"),
                    (m.get("expected_text") or "")[:300],
                    (m.get("actual_text") or "")[:300],
                )
    except Exception as e:
        try:
            logger.warning("tito mismatch-example logging failed (non-fatal): %s", e)
        except Exception:
            pass


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """miles hook. Returns False so default rollout/* logging still runs."""
    _log_tito_mismatch_examples(rollout_id, samples)
    try:
        run_infos, step_seconds = [], []
        for s in samples or []:
            md = getattr(s, "metadata", None) or {}
            if not isinstance(md, dict):
                continue
            run_infos.append(md.get("camel_run_info") or {})
            step_seconds.append(md.get("camel_step_seconds"))
        metrics = compute_agent_metrics(run_infos, step_seconds)
        if metrics:
            from miles.utils.metric_utils import compute_rollout_step
            from miles.utils import tracking_utils
            metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
            tracking_utils.log(args, metrics, step_key="rollout/step")
            logger.info(
                "camel_rollout_metrics step=%s timeout=%.3f iters_mean=%.1f finished=%.3f",
                metrics["rollout/step"], metrics.get(f"{_PREFIX}agent_timeout_ratio", 0.0),
                metrics.get(f"{_PREFIX}iterations_mean", 0.0),
                metrics.get(f"{_PREFIX}term_task_finished_ratio", 0.0),
            )
    except Exception as e:  # never crash training over a metric
        try:
            logger.warning("camel_rollout_metrics hook failed (non-fatal): %s", e)
        except Exception:
            pass
    return False  # keep miles' default rollout/* logging
