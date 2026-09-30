#!/usr/bin/env bash
# Check out the Megatron-Bridge commit the GLM-5.2 PPO script trains with (idempotent).
#
# Michaelsqj/Megatron-Bridge@e91c4923 is radixark/Megatron-Bridge@7f0fb345 plus a
# one-line fix to the GLM-5 TileLang sparse-MLA backward kernel, without which
# LoRA gradients come out NaN. To use a checkout of your own instead, apply
# megatron_bridge_sparse_mla_bwd.patch to it. The critic needs nothing else from
# Megatron-Bridge: the pinned Miles attaches the value head itself and keeps it
# out of the Hugging Face weight load.
#
# The training launcher puts ${MEGATRON_BRIDGE_DIR}/src first on PYTHONPATH on
# every Ray worker, so the directory must exist at the same path on every node:
# run this on each node, or once on a filesystem all nodes mount.
#
# Environment:
#   MEGATRON_BRIDGE_DIR   default /root/Megatron-Bridge-glm5_2
#   MEGATRON_BRIDGE_REPO  default https://github.com/Michaelsqj/Megatron-Bridge.git
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../common/launcher.sh"
load_env_file "${SCRIPT_DIR}"   # MEGATRON_BRIDGE_DIR from .env, if present

MEGATRON_BRIDGE_DIR=${MEGATRON_BRIDGE_DIR:-/root/Megatron-Bridge-glm5_2}
MEGATRON_BRIDGE_REPO=${MEGATRON_BRIDGE_REPO:-https://github.com/Michaelsqj/Megatron-Bridge.git}
MEGATRON_BRIDGE_COMMIT=e91c492328700f9cd59b8d1a82d40a5e54fbd835
KERNEL=src/megatron/bridge/models/glm5/tilelang/tilelang_sparse_mla_bwd.py

if [[ ! -d "${MEGATRON_BRIDGE_DIR}/.git" ]]; then
  echo "[megatron-bridge] fetching ${MEGATRON_BRIDGE_REPO} @ ${MEGATRON_BRIDGE_COMMIT}"
  mkdir -p "${MEGATRON_BRIDGE_DIR}"
  git -C "${MEGATRON_BRIDGE_DIR}" init -q
  git -C "${MEGATRON_BRIDGE_DIR}" remote add origin "${MEGATRON_BRIDGE_REPO}"
fi
if [[ "$(git -C "${MEGATRON_BRIDGE_DIR}" rev-parse HEAD 2>/dev/null)" != "${MEGATRON_BRIDGE_COMMIT}" ]]; then
  [[ -z "$(git -C "${MEGATRON_BRIDGE_DIR}" status --porcelain --untracked-files=no 2>/dev/null)" ]] || {
    echo "[megatron-bridge] ${MEGATRON_BRIDGE_DIR} has local modifications; refusing to move it" >&2
    exit 2
  }
  git -C "${MEGATRON_BRIDGE_DIR}" fetch -q --depth 1 origin "${MEGATRON_BRIDGE_COMMIT}"
  git -C "${MEGATRON_BRIDGE_DIR}" checkout -q --detach FETCH_HEAD
fi
[[ "$(git -C "${MEGATRON_BRIDGE_DIR}" rev-parse HEAD)" == "${MEGATRON_BRIDGE_COMMIT}" ]] || {
  echo "[megatron-bridge] ${MEGATRON_BRIDGE_DIR} is not at ${MEGATRON_BRIDGE_COMMIT}" >&2
  exit 2
}
[[ -z "$(git -C "${MEGATRON_BRIDGE_DIR}" status --porcelain --untracked-files=no)" ]] || {
  echo "[megatron-bridge] ${MEGATRON_BRIDGE_DIR} has local modifications" >&2
  exit 2
}
grep -q 'TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False' "${MEGATRON_BRIDGE_DIR}/${KERNEL}" || {
  echo "[megatron-bridge] ${KERNEL} does not carry the sparse-MLA backward fix" >&2
  exit 2
}
echo "[megatron-bridge] ready: MEGATRON_BRIDGE_DIR=${MEGATRON_BRIDGE_DIR}"
