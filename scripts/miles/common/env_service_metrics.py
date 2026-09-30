"""Miles rollout-logging hook: per-step agent metrics for the seta env_service rollout.

    --custom-rollout-log-function-path common.env_service_metrics.log_rollout_data

Miles calls it once per training step with the training batch (after the dynamic
filter). Metrics go to the tracking backends under ``rollout/agent/*`` on the same
``rollout/step`` axis as Miles' own rollout metrics: agent turns, tool calls, parse
errors, termination reasons, agent timeouts and env_service step time.

Per-sample input, in ``sample.metadata``:
* ``camel_run_info`` / ``camel_step_seconds``: the env_service run_info and /step time,
  stored by ``common.env_service_generate``;
* otherwise ``agent_metrics`` / ``exit_status`` from ``common.env_service_agent``
  (session-server path), which carry the same agent summary.

Timeouts: an env_service step that exceeds its total budget fails the sample, and the
group filter removes it before training, so it never reaches this hook. An agent that
runs out of its own time budget is still verified and kept, so ``agent_timeout_ratio``
here is the agent-timeout rate of the trained batch.

It also logs a few concrete examples of TITO session mismatches (the Miles session
server's check that the recorded tokens re-render to the same conversation), because
Miles only reports their rates.

Returns False so Miles' default rollout logging still runs. Every step is wrapped so a
metric bug cannot stop training.
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
    """Pure function: list of env_service run_info dicts -> {metric: value}."""
    n = len(run_infos)
    if n == 0:
        return {}
    iters, tool_calls, parse_errs = [], [], []
    n_timeout = n_env_fail = 0
    term = {"task_finished": 0, "max_iteration_reached": 0, "max_tokens_exceeded": 0, "agent_timeout": 0, "other": 0}
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
        if is_timeout:
            term["agent_timeout"] += 1
        elif reason in term:
            term[reason] += 1
        else:
            term["other"] += 1

    out = {
        f"{_PREFIX}agent_timeout_ratio": _pct(n_timeout, n),
        # Close to 0 in a trained batch (the group filter removes env failures); a sanity check.
        f"{_PREFIX}env_error_ratio": _pct(n_env_fail, n),
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


def _run_info_from_metadata(md):
    """(run_info, step_seconds) of one sample, from either rollout path."""
    run_info = md.get("camel_run_info")
    if run_info is not None:
        return run_info, md.get("camel_step_seconds")
    agent_metrics = md.get("agent_metrics")
    if not isinstance(agent_metrics, dict):
        return {}, None
    # env_service_agent flattens run_info.agent_summary into agent_metrics and turns
    # run_info.error_info into exit_status.
    exit_status = md.get("exit_status")
    error_info = {} if exit_status in (None, "Submitted", "Unknown") else {"stage": exit_status}
    return {"agent_summary": agent_metrics, "error_info": error_info}, agent_metrics.get("total_time")


# TITO mismatch types that indicate a chat-template or TITO bug; other types (e.g.
# assistant_text from inherited prefix tokens) are tolerated.
_STRICT_TITO_TYPES = ("special_token_count", "special_token_type", "non_assistant_text")


def _log_tito_mismatch_examples(rollout_id, samples, max_examples_per_type=3):
    """Log a few concrete TITO session mismatches per type.

    Each mismatch carries type / segment_index / expected_text / actual_text / detail
    (Miles' token-sequence comparator). repr() shows whitespace differences, which are
    the usual culprit. Strict types are logged as warnings, the rest as info.
    """
    try:
        by_type = {}  # type -> list[(uid, mismatch)]
        counts = {}
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
            rollout_id,
            n_with_mismatch,
            len(samples or []),
            counts,
        )
        for t, bucket in by_type.items():
            strict = t in _STRICT_TITO_TYPES
            lvl = logger.warning if strict else logger.info
            for uid, m in bucket:
                lvl(
                    "tito_mismatch[%s%s] uid=%s seg=%s detail=%s\n  expected=%r\n  actual  =%r",
                    t,
                    " STRICT" if strict else "",
                    uid,
                    m.get("segment_index"),
                    m.get("detail"),
                    (m.get("expected_text") or "")[:300],
                    (m.get("actual_text") or "")[:300],
                )
    except Exception as e:
        try:
            logger.warning("tito mismatch-example logging failed (non-fatal): %s", e)
        except Exception:
            pass


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Miles hook. Returns False so the default rollout logging still runs."""
    _log_tito_mismatch_examples(rollout_id, samples)
    try:
        run_infos, step_seconds = [], []
        for s in samples or []:
            md = getattr(s, "metadata", None) or {}
            if not isinstance(md, dict):
                continue
            run_info, seconds = _run_info_from_metadata(md)
            run_infos.append(run_info)
            step_seconds.append(seconds)
        metrics = compute_agent_metrics(run_infos, step_seconds)
        if metrics:
            from miles.utils import tracking_utils
            from miles.utils.metric_utils import compute_rollout_step

            metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
            tracking_utils.log(args, metrics, step_key="rollout/step")
            logger.info(
                "env_service_metrics step=%s timeout=%.3f iters_mean=%.1f finished=%.3f",
                metrics["rollout/step"],
                metrics.get(f"{_PREFIX}agent_timeout_ratio", 0.0),
                metrics.get(f"{_PREFIX}iterations_mean", 0.0),
                metrics.get(f"{_PREFIX}term_task_finished_ratio", 0.0),
            )
    except Exception as e:  # never stop training over a metric
        try:
            logger.warning("env_service_metrics hook failed (non-fatal): %s", e)
        except Exception:
            pass
    return False
