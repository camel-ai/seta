# shellcheck shell=bash
# Helpers for the scripts that roll out through the seta env_service
# (seta_env/services/env_service.py, see docs/env_service.md). Source after launcher.sh:
#
#   source "${SCRIPT_DIR}/../../common/launcher.sh"
#   source "${SCRIPT_DIR}/../../common/env_service.sh"
#
#   env_service_configure        resolve the settings below and export what the Ray rollout
#                                workers read (CAMEL_*); needs init_run first
#   env_service_check_data       every PROMPT_DATA row has a task directory
#   env_service_start            start the env_service on this node, wait for /health, stop it on exit
#   env_service_stop             stop it (process group), give in-flight steps time to clean up
#
# The env_service runs on the Ray head node. For each rollout it resolves the task
# directory DATASET_ROOT/<dataset>/<instance_id>, starts the sandbox, runs seta's CAMEL
# agent against the model URL it is given and returns the verified reward.
#
# Settings (defaults never override a value the caller already exported):
#   ENV_SERVICE_CONFIG     the recipe's env_service.yaml (agent, model client, sandbox, timeouts)
#   ENV_SERVICE_PYTHON     Python with seta_env, CAMEL and Harbor installed
#                          (default ${SETA_ROOT}/.venv/bin/python, see the recipe README)
#   ENV_SERVICE_HOST       bind address (default 0.0.0.0)
#   ENV_SERVICE_PORT       port (default 8002)
#   ENV_SERVICE_START_TIMEOUT  seconds to wait for /health (default 180)
#   ENV_SERVICE_LOG_LEVEL  uvicorn log level (default info)
#   TASKS_DIR              task directories; DATASET_ROOT = its parent, dataset name = its basename
#   DATASET_ROOT, CAMEL_DATASET_NAME   set these instead of TASKS_DIR to name them directly
#   CAMEL_TRIAL_NAME       trial folder name, default RUN_NAME: trials land in RUN_ROOT/trials/<name>/
#   CAMEL_ENV_SERVICE_URL  URL the Ray rollout workers use (default http://${HEAD_IP}:${ENV_SERVICE_PORT})
# Read by the env_service process itself (and by the Harbor Daytona backend it uses):
#   DAYTONA_API_KEY, DAYTONA_API_URL   Daytona credentials (runtime.env_type: daytona)
#   MAX_SLOTS              concurrent trajectories = live sandboxes (env_service default 16)
#   STEP_TIMEOUT_SECONDS   hard cap on one trajectory, frees its slot (default 1800)
#   STEP_USE_SUBPROCESS    1: run each trajectory in a killable subprocess (default 0)
#   BUILD_TIMEOUT_SECONDS  hard cap on one sandbox image build (default 600)
#   BUILD_CONCURRENCY      concurrent image builds (default 8)
#   INFLIGHT_LEAD          trajectories built ahead of a free slot (default 48)
#   HARBOR_DAYTONA_MAX_CREATES     concurrent Daytona create calls, 0 = unlimited (default 0)
#   CREATE_CAPACITY_MAX_ATTEMPTS   create retries while Daytona has no free runner (default 75)
#   DAYTONA_DECLARATIVE    1: build images through Daytona's declarative image cache instead
#                          of named snapshots (default 0)
#   DAYTONA_SNAPSHOT_EVICT_AGE_HOURS  with named snapshots: delete ones unused this long (default 0 = off)
#   MODEL_TIMEOUT          the agent's client timeout for one model call, seconds (default 180)
#   HF_TOKEN, HF_HOME      passed through

env_service_configure() {
  : "${RUN_ROOT:?call init_run first}" "${RUN_NAME:?call init_run first}"
  : "${ENV_SERVICE_CONFIG:?set ENV_SERVICE_CONFIG to the recipe env_service.yaml}"
  export ENV_SERVICE_CONFIG
  export ENV_SERVICE_PYTHON="${ENV_SERVICE_PYTHON:-${SETA_ROOT}/.venv/bin/python}"
  export ENV_SERVICE_HOST="${ENV_SERVICE_HOST:-0.0.0.0}"
  export ENV_SERVICE_PORT="${ENV_SERVICE_PORT:-8002}"
  if [[ -z "${DATASET_ROOT:-}" || -z "${CAMEL_DATASET_NAME:-}" ]]; then
    : "${TASKS_DIR:?set TASKS_DIR to the task directories referenced by PROMPT_DATA}"
    local tasks_dir="${TASKS_DIR%/}"
    DATASET_ROOT="${DATASET_ROOT:-$(dirname -- "${tasks_dir}")}"
    CAMEL_DATASET_NAME="${CAMEL_DATASET_NAME:-$(basename -- "${tasks_dir}")}"
  fi
  export DATASET_ROOT CAMEL_DATASET_NAME
  export CAMEL_TRIAL_NAME="${CAMEL_TRIAL_NAME:-${RUN_NAME}}"
  export CAMEL_ENV_SERVICE_URL="${CAMEL_ENV_SERVICE_URL:-http://${HEAD_IP:-127.0.0.1}:${ENV_SERVICE_PORT}}"
}

env_service_check_data() {
  : "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL or parquet}"
  [[ -s "${PROMPT_DATA}" ]] || { echo "PROMPT_DATA ${PROMPT_DATA} is missing or empty" >&2; return 2; }
  [[ -d "${DATASET_ROOT}/${CAMEL_DATASET_NAME}" ]] || {
    echo "task directory ${DATASET_ROOT}/${CAMEL_DATASET_NAME} does not exist" >&2
    return 2
  }
  python3 - "${PROMPT_DATA}" "${DATASET_ROOT}/${CAMEL_DATASET_NAME}" <<'PY'
import json, pathlib, sys
path, tasks = sys.argv[1], pathlib.Path(sys.argv[2])
if path.endswith(".parquet"):
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
else:
    rows = [json.loads(line) for line in open(path) if line.strip()]
ids = []
for row in rows:
    md = row.get("metadata") or {}
    if isinstance(md, str):
        md = json.loads(md)
    ids.append(md.get("instance_id", ""))
missing = [i for i in ids if not i or not (tasks / i).is_dir()]
if missing:
    sys.exit(f"{len(missing)} of {len(ids)} prompt rows have no task directory in {tasks}, e.g. {missing[:5]}")
print(f"[env_service] {len(ids)} prompts, all tasks present in {tasks}", file=sys.stderr)
PY
}

env_service_stop() {
  local pid_file="${RUN_ROOT:-}/env_service.pid" pid _
  [[ -s "${pid_file}" ]] || return 0
  pid=$(<"${pid_file}")
  # SIGTERM first so the service runs its shutdown hook; SIGKILL whatever is left after
  # 30 s. Sandboxes of interrupted trajectories stay until Daytona auto-stops and deletes
  # them; common/daytona_cleanup.py removes them at once.
  kill -TERM -- "-${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 -- "-${pid}" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
  log "env_service stopped"
}

env_service_start() {
  env_service_configure
  [[ -x "${ENV_SERVICE_PYTHON}" ]] || {
    echo "ENV_SERVICE_PYTHON=${ENV_SERVICE_PYTHON} is not an executable Python; install seta_env first (see the recipe README)" >&2
    return 2
  }
  [[ -f "${ENV_SERVICE_CONFIG}" ]] || { echo "ENV_SERVICE_CONFIG ${ENV_SERVICE_CONFIG} not found" >&2; return 2; }
  if grep -Eq '^[[:space:]]*env_type:[[:space:]]*daytona' "${ENV_SERVICE_CONFIG}"; then
    : "${DAYTONA_API_KEY:?the env_service runs Daytona sandboxes: set DAYTONA_API_KEY (and DAYTONA_API_URL for a non-default endpoint)}"
  fi
  local health="http://127.0.0.1:${ENV_SERVICE_PORT}/health"
  if curl -sf "${health}" >/dev/null 2>&1; then
    echo "something already answers at ${health}; stop it or set ENV_SERVICE_PORT to a free port" >&2
    return 2
  fi
  local log_file="${ENV_SERVICE_LOG:-${RUN_ROOT}/logs/env_service.log}"
  mkdir -p "$(dirname -- "${log_file}")"
  log "starting env_service on :${ENV_SERVICE_PORT} (config ${ENV_SERVICE_CONFIG}, tasks ${DATASET_ROOT}/${CAMEL_DATASET_NAME}, up to ${MAX_SLOTS:-16} sandboxes)"
  # Its own process group, so env_service_stop also reaches the per-step subprocesses.
  # HARBOR_ROOT is where the env_service writes trials/<CAMEL_TRIAL_NAME>/ and image builds.
  (
    cd "${SETA_ROOT}" &&
      exec setsid env PYTHONPATH="${SETA_ROOT}" ENV_SERVICE_CONFIG="${ENV_SERVICE_CONFIG}" \
        DATASET_ROOT="${DATASET_ROOT}" HARBOR_ROOT="${RUN_ROOT}" \
        "${ENV_SERVICE_PYTHON}" -m uvicorn seta_env.services.env_service:app \
        --host "${ENV_SERVICE_HOST}" --port "${ENV_SERVICE_PORT}" --log-level "${ENV_SERVICE_LOG_LEVEL:-info}"
  ) >"${log_file}" 2>&1 &
  echo "$!" > "${RUN_ROOT}/env_service.pid"
  disown "$!"  # no job-control notice when env_service_stop kills it
  trap 'env_service_stop' EXIT
  local _ pid
  pid=$(<"${RUN_ROOT}/env_service.pid")
  for _ in $(seq 1 "${ENV_SERVICE_START_TIMEOUT:-180}"); do
    if curl -sf "${health}" >/dev/null 2>&1; then
      log "env_service ready at ${CAMEL_ENV_SERVICE_URL}"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "env_service exited during startup; last lines of ${log_file}:" >&2
      tail -n 40 "${log_file}" >&2
      return 1
    fi
    sleep 1
  done
  echo "env_service did not become healthy; see ${log_file}" >&2
  return 1
}
