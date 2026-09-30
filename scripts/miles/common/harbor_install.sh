#!/usr/bin/env bash
# Install Harbor at a pinned commit into its own virtualenv (idempotent).
#
# Each training script pins its own HARBOR_COMMIT, so several pins can coexist: every commit gets its own checkout and venv under
# HARBOR_HOME/<commit>. The Miles training environment itself does not need
# Harbor installed; only the agent server runs from this venv.
#
# Usage:
#   HARBOR_COMMIT=<40-char sha> bash scripts/miles/common/harbor_install.sh
#
# Environment:
#   HARBOR_COMMIT   required, full commit SHA
#   HARBOR_REPO     default https://github.com/Michaelsqj/harbor.git
#   HARBOR_HOME     default ~/.cache/seta/harbor
#   HARBOR_EXTRAS   pyproject extras, default "daytona modal gke camel"
#   PYTHON          interpreter for the venv, default python3.12 (Harbor needs >=3.12)
set -euo pipefail

HARBOR_COMMIT="${HARBOR_COMMIT:-${1:-}}"
[[ "${HARBOR_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "HARBOR_COMMIT must be a full 40-character commit SHA (got '${HARBOR_COMMIT}')" >&2
  exit 2
}
HARBOR_REPO="${HARBOR_REPO:-https://github.com/Michaelsqj/harbor.git}"
HARBOR_HOME="${HARBOR_HOME:-${HOME}/.cache/seta/harbor}"
HARBOR_EXTRAS="${HARBOR_EXTRAS:-daytona modal gke camel}"
PYTHON="${PYTHON:-python3.12}"
HARBOR_DIR="${HARBOR_HOME}/${HARBOR_COMMIT}"
HARBOR_VENV="${HARBOR_DIR}/.venv"

if [[ ! -d "${HARBOR_DIR}/.git" ]]; then
  echo "[harbor-setup] fetching ${HARBOR_REPO} @ ${HARBOR_COMMIT}"
  mkdir -p "${HARBOR_DIR}"
  git -C "${HARBOR_DIR}" init -q
  git -C "${HARBOR_DIR}" remote add origin "${HARBOR_REPO}"
  git -C "${HARBOR_DIR}" fetch -q --depth 1 origin "${HARBOR_COMMIT}"
  git -C "${HARBOR_DIR}" checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "${HARBOR_DIR}" rev-parse HEAD)" == "${HARBOR_COMMIT}" ]] || {
  echo "[harbor-setup] ${HARBOR_DIR} is not at ${HARBOR_COMMIT}" >&2
  exit 2
}
[[ -z "$(git -C "${HARBOR_DIR}" status --porcelain --untracked-files=no)" ]] || {
  echo "[harbor-setup] ${HARBOR_DIR} has local modifications; refusing to reuse it" >&2
  exit 2
}

if [[ ! -x "${HARBOR_VENV}/bin/python" ]] || \
   ! "${HARBOR_VENV}/bin/python" -c 'import harbor.agent_server' >/dev/null 2>&1; then
  extras=()
  for extra in ${HARBOR_EXTRAS}; do extras+=(--extra "${extra}"); done
  if command -v uv >/dev/null 2>&1; then
    echo "[harbor-setup] uv sync (${HARBOR_EXTRAS}) -> ${HARBOR_VENV}"
    UV_PROJECT_ENVIRONMENT="${HARBOR_VENV}" uv sync --project "${HARBOR_DIR}" \
      --python "${PYTHON}" --locked --no-dev "${extras[@]}"
  else
    echo "[harbor-setup] uv not found; using ${PYTHON} -m venv + pip"
    "${PYTHON}" -m venv "${HARBOR_VENV}"
    "${HARBOR_VENV}/bin/pip" install -q --upgrade pip
    "${HARBOR_VENV}/bin/pip" install -q -e "${HARBOR_DIR}[$(tr ' ' ',' <<<"${HARBOR_EXTRAS}")]"
  fi
fi
"${HARBOR_VENV}/bin/python" -c 'import harbor.agent_server' || {
  echo "[harbor-setup] harbor.agent_server is not importable at ${HARBOR_COMMIT}" >&2
  exit 2
}
echo "[harbor-setup] ready: HARBOR_DIR=${HARBOR_DIR}"
