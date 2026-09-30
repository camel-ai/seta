#!/usr/bin/env bash
# Run the Harbor agent server in the foreground.
#
# Normally started (in the background, with a health check) by
# harbor_start_server in common/launcher.sh. Can also be run by hand to serve
# trials to an already running Miles job.
#
# Required environment:
#   HARBOR_DIR, HARBOR_VENV   pinned checkout + venv (see harbor_install.sh)
#   TASKS_DIR                 directory with one <task>/task.toml per task
#   TRIALS_DIR                where per-trial artifacts are written
# Backend (one of):
#   SANDBOX_BACKEND           daytona | modal | docker | gke | ... (Harbor environment type)
#   HARBOR_ENVIRONMENT_CONFIG JSON EnvironmentConfig (takes precedence; used for gke)
# Optional:
#   AGENT_SERVER_HOST/PORT, AGENT_MAX_CONCURRENT
#   HARBOR_AGENT_TIMEOUT_MULTIPLIER  scales task agent timeouts (default 12)
#   HARBOR_AGENT_TIMEOUT_SEC         uniform agent-phase budget, overrides task timeouts
#   HARBOR_EXTRA_ARTIFACTS           whitespace-separated --artifact entries
#   HARBOR_EXTRA_COLLECT             newline-separated --collect hooks (needs a Harbor pin with --collect)
set -euo pipefail

: "${HARBOR_DIR:?}" "${HARBOR_VENV:?}" "${TASKS_DIR:?}" "${TRIALS_DIR:?}"
export PYTHONPATH="${HARBOR_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
# LiteLLM needs some key even against a local SGLang endpoint.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

if [[ -n "${HARBOR_ENVIRONMENT_CONFIG:-}" ]]; then
  ENV_ARGS=(--environment-config "${HARBOR_ENVIRONMENT_CONFIG}")
else
  ENV_ARGS=(--environment "${SANDBOX_BACKEND:?set SANDBOX_BACKEND or HARBOR_ENVIRONMENT_CONFIG}")
fi

ARTIFACT_ARGS=()
for artifact in ${HARBOR_EXTRA_ARTIFACTS:-}; do
  ARTIFACT_ARGS+=(--artifact "${artifact}")
done
COLLECT_ARGS=()
if [[ -n "${HARBOR_EXTRA_COLLECT:-}" ]]; then
  while IFS= read -r hook; do
    [[ -n "${hook}" ]] && COLLECT_ARGS+=(--collect "${hook}")
  done <<< "${HARBOR_EXTRA_COLLECT}"
fi

mkdir -p "${TRIALS_DIR}"
exec "${HARBOR_VENV}/bin/python" -m harbor.agent_server \
  --tasks-dir "${TASKS_DIR}" \
  --trials-dir "${TRIALS_DIR}" \
  "${ENV_ARGS[@]}" \
  --host "${AGENT_SERVER_HOST:-0.0.0.0}" --port "${AGENT_SERVER_PORT:-11000}" \
  --max-concurrent "${AGENT_MAX_CONCURRENT:-64}" \
  --agent-timeout-multiplier "${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-12}" \
  ${HARBOR_AGENT_TIMEOUT_SEC:+--agent-timeout-sec "${HARBOR_AGENT_TIMEOUT_SEC}"} \
  ${ARTIFACT_ARGS[@]+"${ARTIFACT_ARGS[@]}"} \
  ${COLLECT_ARGS[@]+"${COLLECT_ARGS[@]}"} \
  --no-access-log
