# shellcheck shell=bash
# Helpers shared by the training scripts in scripts/miles/<model>/.
#
#   source "${SCRIPT_DIR}/../common/launcher.sh"
#
# Run setup
#   load_env_file <dir>          source <dir>/.env (or $ENV_FILE): your keys and paths, see env.example
#   init_run <name>              RUN_ROOT=${RUNS_ROOT}/${RUN_NAME}/{logs,trials,checkpoints,wandb}
#   dry_run_exit <cmd...>        DRY_RUN=1: print the resolved training command and exit
#   configure_ray [cluster]      external Ray head from HEAD_IP or CLUSTER_CONFIG
#   check_git_pin <n> <d> <c>    warn when a checkout is not at the tested commit (STRICT_PINS=1: fail)
#   wandb_args <array>           append Miles W&B flags when WANDB_API_KEY is set
#   write_manifest <files...>    RUN_ROOT/run_manifest.md + copies of the launch files
# Harbor agent-server rollout
#   harbor_install               pinned Harbor checkout + venv (HARBOR_COMMIT)
#   harbor_require_source <f> <re> <why>   fail if the pinned Harbor lacks a feature the script relies on
#   harbor_check_data            every PROMPT_DATA row has a task directory in TASKS_DIR
#   harbor_configure             agent-server environment + sandbox backend (SANDBOX_BACKEND)
#   harbor_start_server          start the agent server, wait for /health, stop it on exit
#
# Harbor inputs: PROMPT_DATA (JSONL, see scripts/miles/data/README.md), TASKS_DIR (Harbor task
# directories), SANDBOX_BACKEND = daytona (default) | modal | docker | gke | any Harbor environment type.
#
# Defaults never override a value the caller already exported.

COMMON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MILES_SCRIPTS_DIR=${MILES_SCRIPTS_DIR:-$(cd -- "${COMMON_DIR}/.." && pwd)}
SETA_ROOT=${SETA_ROOT:-$(cd -- "${MILES_SCRIPTS_DIR}/../.." && pwd)}
export MILES_SCRIPTS_DIR SETA_ROOT

log() { printf '[%s] %s\n' "${LOG_TAG:-miles}" "$*" >&2; }

# Settings file: copy an example's env.example to .env next to it and fill it in.
# Values in the file take precedence over the script defaults. ENV_FILE points
# at a file elsewhere.
load_env_file() {
  local file="${ENV_FILE:-${1:?directory}/.env}"
  if [[ -f "${file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${file}"
    set +a
    log "loaded settings from ${file}"
  elif [[ -n "${ENV_FILE:-}" ]]; then
    echo "ENV_FILE=${ENV_FILE} does not exist" >&2
    return 2
  fi
}

init_run() {
  local name="${1:?run name prefix}" var
  RUNS_ROOT="${RUNS_ROOT:-${PWD}/runs}"
  # The Ray job runs in another working directory: make user paths absolute.
  for var in RUNS_ROOT PROMPT_DATA TASKS_DIR; do
    [[ -z "${!var:-}" ]] || printf -v "${var}" '%s' "$(realpath -m -- "${!var}")"
  done
  RUN_NAME="${RUN_NAME:-${name}-$(date -u +%Y%m%d-%H%M%S)}"
  RUN_ROOT="${RUNS_ROOT}/${RUN_NAME}"
  export RUNS_ROOT RUN_NAME RUN_ROOT
  [[ -z "${PROMPT_DATA:-}" ]] || export PROMPT_DATA
  [[ -z "${TASKS_DIR:-}" ]] || export TASKS_DIR
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/trials" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/wandb"
  log "run folder: ${RUN_ROOT}"
}

dry_run_exit() {
  [[ "${DRY_RUN:-0}" == 1 ]] || return 0
  local cmd
  cmd=$(printf '%q ' "$@")
  # Do not echo secrets: mask the W&B key if it is part of the command.
  [[ -z "${WANDB_API_KEY:-}" ]] || cmd=${cmd//"${WANDB_API_KEY}"/'<WANDB_API_KEY>'}
  printf 'DRY_RUN=1, resolved command:\n%s\n' "${cmd}"
  exit 0
}

configure_ray() {
  local cluster_config="${1:-${CLUSTER_CONFIG:-}}"
  if [[ -z "${HEAD_IP:-}" ]]; then
    [[ -n "${cluster_config}" && -f "${cluster_config}" ]] || {
      echo "set HEAD_IP, or CLUSTER_CONFIG to a cluster YAML (see scripts/miles/cluster.example.yaml)" >&2
      return 2
    }
    HEAD_IP=$(sed -n '/^head:/,/^workers:/s/^[[:space:]]*ip:[[:space:]]*//p' "${cluster_config}" | head -n 1)
    [[ -n "${HEAD_IP}" ]] || { echo "no head.ip in ${cluster_config}" >&2; return 2; }
  fi
  export HEAD_IP MASTER_ADDR="${MASTER_ADDR:-${HEAD_IP}}"
  export RAY_ADDRESS="${RAY_ADDRESS:-http://${HEAD_IP}:${RAY_DASHBOARD_PORT:-8265}}"
  export MILES_SCRIPT_EXTERNAL_RAY=1 MILES_SCRIPT_ENABLE_RAY_SUBMIT=1
  # Set these when the default route is not the interconnect (e.g. bond0, ib0).
  [[ -z "${NCCL_SOCKET_IFNAME:-}" ]] || export NCCL_SOCKET_IFNAME
  [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] || export GLOO_SOCKET_IFNAME
}

check_git_pin() {
  local name="$1" dir="$2" want="$3" have
  have=$(git -C "${dir}" rev-parse HEAD 2>/dev/null || echo unknown)
  [[ "${have}" == "${want}" ]] && return 0
  if [[ "${STRICT_PINS:-0}" == 1 ]]; then
    echo "${name} at ${dir} is ${have}; this script is tested with ${want} (STRICT_PINS=1)" >&2
    return 2
  fi
  log "WARNING: ${name} at ${dir} is ${have}; this script is tested with ${want}"
}

# The key is passed to Miles as --wandb-key (as upstream Miles launchers do), so
# every Ray process can log in; it therefore appears in the submitted job command.
wandb_args() {
  local -n _wandb_out="$1"
  [[ -n "${WANDB_API_KEY:-}" ]] || { log "WANDB_API_KEY unset: W&B logging off"; return 0; }
  _wandb_out+=(--use-wandb --wandb-key "${WANDB_API_KEY}" --wandb-project "${WANDB_PROJECT:-seta}"
               --wandb-group "${RUN_NAME}" --disable-wandb-random-suffix --wandb-dir "${RUN_ROOT}/wandb")
  [[ -z "${WANDB_ENTITY:-}" ]] || _wandb_out+=(--wandb-team "${WANDB_ENTITY}")
}

write_manifest() {
  local snapshot="${RUN_ROOT}/launcher" f
  mkdir -p "${snapshot}"
  {
    printf '# %s\n\n' "${RUN_NAME}"
    printf -- '- started: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- script: %s\n' "$0"
    printf -- '- seta: %s\n' "$(git -C "${SETA_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    [[ -z "${MILES_DIR:-}" ]] || printf -- '- miles: %s\n' "$(git -C "${MILES_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    [[ -z "${HARBOR_COMMIT:-}" ]] || printf -- '- harbor: %s\n' "${HARBOR_COMMIT}"
    [[ -z "${SANDBOX_BACKEND:-}" ]] || printf -- '- sandbox backend: %s\n' "${SANDBOX_BACKEND}"
    if [[ -n "${PROMPT_DATA:-}" && -f "${PROMPT_DATA}" ]]; then
      printf -- '- prompt data: %s (%s rows, sha256 %s)\n' "${PROMPT_DATA}" \
        "$(awk 'NF {n++} END {print n + 0}' "${PROMPT_DATA}")" "$(sha256sum "${PROMPT_DATA}" | awk '{print $1}')"
    fi
    [[ -z "${TASKS_DIR:-}" ]] || printf -- '- tasks: %s\n' "${TASKS_DIR}"
  } > "${RUN_ROOT}/run_manifest.md"
  for f in "$@"; do [[ -f "${f}" ]] && cp "${f}" "${snapshot}/"; done
  log "wrote ${RUN_ROOT}/run_manifest.md"
}

# ---------------------------------------------------------------- Harbor agent server

harbor_install() {
  : "${HARBOR_COMMIT:?the training script must set HARBOR_COMMIT}"
  local home="${HARBOR_HOME:-${HOME}/.cache/seta/harbor}"
  HARBOR_DIR="${HARBOR_DIR:-${home}/${HARBOR_COMMIT}}"
  HARBOR_VENV="${HARBOR_VENV:-${HARBOR_DIR}/.venv}"
  if [[ ! -x "${HARBOR_VENV}/bin/python" ]]; then
    HARBOR_HOME="${home}" HARBOR_COMMIT="${HARBOR_COMMIT}" bash "${COMMON_DIR}/harbor_install.sh"
  fi
  [[ "$(git -C "${HARBOR_DIR}" rev-parse HEAD 2>/dev/null)" == "${HARBOR_COMMIT}" ]] || {
    echo "Harbor at ${HARBOR_DIR} is not at HARBOR_COMMIT=${HARBOR_COMMIT}" >&2
    return 2
  }
  export HARBOR_COMMIT HARBOR_DIR HARBOR_VENV
}

harbor_require_source() {
  local rel="$1" pattern="$2" why="$3"
  grep -qE -- "${pattern}" "${HARBOR_DIR}/${rel}" 2>/dev/null || {
    echo "Harbor ${HARBOR_COMMIT} lacks '${pattern}' in ${rel}: ${why}" >&2
    return 2
  }
}

harbor_check_data() {
  : "${PROMPT_DATA:?set PROMPT_DATA to a prompt JSONL (see scripts/miles/data/README.md)}"
  : "${TASKS_DIR:?set TASKS_DIR to the Harbor task directories referenced by PROMPT_DATA}"
  [[ -s "${PROMPT_DATA}" ]] || { echo "PROMPT_DATA ${PROMPT_DATA} is missing or empty" >&2; return 2; }
  [[ -d "${TASKS_DIR}" ]] || { echo "TASKS_DIR ${TASKS_DIR} is not a directory" >&2; return 2; }
  python3 - "${PROMPT_DATA}" "${TASKS_DIR}" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
tasks = pathlib.Path(sys.argv[2])
missing = [r["metadata"]["instance_id"] for r in rows
           if not (tasks / r["metadata"]["instance_id"] / "task.toml").is_file()]
if missing:
    sys.exit(f"{len(missing)} prompt rows have no task directory in {tasks}, e.g. {missing[:5]}")
print(f"[harbor] {len(rows)} prompts, all tasks present in {tasks}", file=sys.stderr)
PY
}

harbor_configure() {
  : "${RUN_ROOT:?call init_run first}" "${HARBOR_DIR:?call harbor_install first}" "${TASKS_DIR:?set TASKS_DIR}"
  export TASKS_DIR
  export SANDBOX_BACKEND="${SANDBOX_BACKEND:-daytona}"
  export TRIALS_DIR="${TRIALS_DIR:-${RUN_ROOT}/trials}"
  export AGENT_SERVER_HOST="${AGENT_SERVER_HOST:-0.0.0.0}"
  export AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-11000}"
  export AGENT_SERVER_URL="${AGENT_SERVER_URL:-http://${HEAD_IP:-127.0.0.1}:${AGENT_SERVER_PORT}}"
  export AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-model}"
  export AGENT_MAX_CONCURRENT="${AGENT_MAX_CONCURRENT:-64}"
  export HARBOR_AGENT_NAME="${HARBOR_AGENT_NAME:-terminus-2}"
  local external_host=""
  case "${SANDBOX_BACKEND}" in
    daytona)
      : "${DAYTONA_API_KEY:?SANDBOX_BACKEND=daytona needs DAYTONA_API_KEY (and DAYTONA_API_URL for a non-default endpoint)}"
      ;;
    modal)
      [[ -n "${MODAL_TOKEN_ID:-}" || -f "${HOME}/.modal.toml" ]] || {
        echo "SANDBOX_BACKEND=modal needs MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or ~/.modal.toml" >&2
        return 2
      }
      ;;
    gke)
      # Your own GKE cluster: set it up with common/gke_cluster.sh and common/gke_tools_image.sh;
      # common/harbor_gke_environment.py lists every GKE_* variable.
      local gke_missing="" gke_var
      for gke_var in GKE_PROJECT_ID GKE_REGION GKE_CLUSTER_NAME GKE_TOOLS_IMAGE; do
        [[ -n "${!gke_var:-}" ]] || gke_missing+=" ${gke_var}"
      done
      [[ -z "${gke_missing}" ]] || {
        echo "SANDBOX_BACKEND=gke needs${gke_missing} (see common/gke_cluster.sh; GKE_TOOLS_IMAGE comes from common/gke_tools_image.sh)" >&2
        return 2
      }
      if ! command -v gcloud >/dev/null 2>&1 || ! command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
        echo "SANDBOX_BACKEND=gke needs gcloud and gke-gcloud-auth-plugin on PATH where the agent server runs (gcloud components install gke-gcloud-auth-plugin)" >&2
        return 2
      fi
      # Harbor uses the current context of this kubeconfig: one file per cluster (same default as
      # gke_cluster.sh), written on first use.
      export KUBECONFIG="${GKE_KUBECONFIG:-${HOME}/.kube/gke_${GKE_PROJECT_ID}_${GKE_REGION}_${GKE_CLUSTER_NAME}}"
      [[ -s "${KUBECONFIG}" ]] || GKE_PROJECT_ID="${GKE_PROJECT_ID}" GKE_REGION="${GKE_REGION}" \
        GKE_CLUSTER_NAME="${GKE_CLUSTER_NAME}" GKE_KUBECONFIG="${KUBECONFIG}" \
        bash "${COMMON_DIR}/gke_cluster.sh" credentials
      export GKE_ENVIRONMENT_CONFIG="${GKE_ENVIRONMENT_CONFIG:-${RUN_ROOT}/gke-environment.json}"
      # Pod size; "task" keeps each task's own resources. Training scripts set their defaults.
      export SANDBOX_CPUS="${SANDBOX_CPUS:-task}" SANDBOX_MEMORY_MB="${SANDBOX_MEMORY_MB:-task}"
      export SANDBOX_STORAGE_MB="${SANDBOX_STORAGE_MB:-task}"
      "${HARBOR_VENV}/bin/python" "${COMMON_DIR}/harbor_gke_environment.py"
      export HARBOR_ENVIRONMENT_CONFIG="${HARBOR_ENVIRONMENT_CONFIG:-${GKE_ENVIRONMENT_CONFIG}}"
      external_host="${HEAD_IP:-}"
      ;;
  esac
  # Host the agent uses to reach the Miles session servers (rewrites the session
  # URL host). Empty keeps the URL Miles hands out.
  export MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST-${external_host}}"
  export MILES_HOST_IP="${MILES_HOST_IP-${external_host}}"
  # Miles resolves --custom-*-path plugins (common.harbor_*) from scripts/miles.
  case ":${PYTHONPATH:-}:" in
    *":${MILES_SCRIPTS_DIR}:"*) ;;
    *) export PYTHONPATH="${MILES_SCRIPTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

# Cancel in-flight trials (Harbor tears down their sandboxes), then stop the server.
harbor_stop_server() {
  local pid_file="${RUN_ROOT:-}/agent_server.pid"
  [[ -s "${pid_file}" ]] || return 0
  curl -s --max-time 30 -X POST "${AGENT_SERVER_URL}/flush_all" -H 'Content-Type: application/json' \
    -d '{"cancel_tasks": true, "kill_containers": true, "prune": false}' >/dev/null 2>&1 || true
  kill -- "-$(<"${pid_file}")" 2>/dev/null || true
  rm -f "${pid_file}"
}

harbor_start_server() {
  if curl -sf "${AGENT_SERVER_URL}/health" >/dev/null 2>&1; then
    echo "something already answers at ${AGENT_SERVER_URL}; set AGENT_SERVER_PORT to a free port" >&2
    return 2
  fi
  setsid bash "${COMMON_DIR}/harbor_server.sh" >"${RUN_ROOT}/logs/agent_server.log" 2>&1 &
  echo "$!" > "${RUN_ROOT}/agent_server.pid"
  trap 'harbor_stop_server' EXIT
  local _
  for _ in $(seq 1 180); do
    if curl -sf "${AGENT_SERVER_URL}/health" >/dev/null 2>&1; then
      log "Harbor agent server ready at ${AGENT_SERVER_URL} (${SANDBOX_BACKEND}, up to ${AGENT_MAX_CONCURRENT} trials)"
      return 0
    fi
    sleep 2
  done
  echo "Harbor agent server did not become healthy; see ${RUN_ROOT}/logs/agent_server.log" >&2
  return 1
}
