#!/usr/bin/env bash
# Build Harbor's sandbox tools image, push it to your Artifact Registry and print GKE_TOOLS_IMAGE.
#
# Harbor mounts /opt/harbor-tools of this small image into every GKE sandbox (tmux and timeout
# that run on any Linux task image), so Terminus-2 works without changing task images. The
# Dockerfile is examples/gke-grpo-rollout/tools/ of the pinned Harbor checkout. Build it once per
# registry; the tag is a hash of that folder, so re-running only reprints the digest.
#
#   HARBOR_COMMIT=<pin from the run script> bash gke_tools_image.sh
#
# Settings (environment, then ./.env or ENV_FILE for anything unset):
#   GKE_PROJECT_ID, GKE_REGION      required
#   GKE_REGISTRY_NAME               default harbor-sandboxes (gke_cluster.sh create makes it)
#   HARBOR_DIR or HARBOR_COMMIT     a Harbor checkout, or the commit to fetch into
#                                   HARBOR_HOME/<commit> (default ~/.cache/seta/harbor, where
#                                   harbor_install.sh later adds its virtualenv)
#   GKE_TOOLS_BUILDER=cloudbuild    cloudbuild: build in Cloud Build (needs roles/cloudbuild.builds.editor
#                                   and roles/storage.admin); docker: local Docker build + push
#                                   (needs roles/artifactregistry.writer)
#   DRY_RUN=1                       print the commands only
set -euo pipefail
# shellcheck source=gke_cluster.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/gke_cluster.sh"
gke_settings GKE_PROJECT_ID GKE_REGION

builder="${GKE_TOOLS_BUILDER:-cloudbuild}"
[[ "${builder}" == cloudbuild || "${builder}" == docker ]] ||
  gke_die "GKE_TOOLS_BUILDER must be cloudbuild or docker"
gke_need gcloud "https://cloud.google.com/sdk/docs/install"
[[ "${builder}" != docker ]] || gke_need docker "https://docs.docker.com/engine/install/"

if [[ -z "${HARBOR_DIR:-}" ]]; then
  [[ -n "${HARBOR_COMMIT:-}" ]] ||
    gke_die "set HARBOR_COMMIT (the HARBOR_COMMIT of your run script) or HARBOR_DIR (a Harbor checkout)"
  HARBOR_DIR="${HARBOR_HOME:-${HOME}/.cache/seta/harbor}/${HARBOR_COMMIT}"
fi
tools_dir="${HARBOR_DIR}/examples/gke-grpo-rollout/tools"
if [[ ! -f "${tools_dir}/Dockerfile" && ! -d "${HARBOR_DIR}/.git" && -n "${HARBOR_COMMIT:-}" ]]; then
  # Same checkout layout as harbor_install.sh, which reuses it.
  [[ "${HARBOR_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || gke_die "HARBOR_COMMIT must be a full 40-character commit SHA"
  gke_log "fetching Harbor ${HARBOR_COMMIT} into ${HARBOR_DIR}"
  gke_run mkdir -p "${HARBOR_DIR}"
  gke_run git -C "${HARBOR_DIR}" init -q
  gke_run git -C "${HARBOR_DIR}" remote add origin "${HARBOR_REPO:-https://github.com/Michaelsqj/harbor.git}"
  gke_run git -C "${HARBOR_DIR}" fetch -q --depth 1 origin "${HARBOR_COMMIT}"
  gke_run git -C "${HARBOR_DIR}" checkout -q --detach FETCH_HEAD
fi
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
if [[ -f "${tools_dir}/Dockerfile" ]]; then
  tag="tools-$(cd -- "${tools_dir}" && find . -type f | LC_ALL=C sort | xargs cat | sha256 | cut -c1-12)"
elif [[ "${DRY_RUN:-0}" == 1 ]]; then
  tag="tools-<content-hash>"
else
  gke_die "no Harbor sandbox tools Dockerfile at ${tools_dir}"
fi
image="${GKE_REGISTRY_HOST}/${GKE_PROJECT_ID}/${GKE_REGISTRY_NAME}/harbor-sandbox-tools:${tag}"

if gke_probe gcloud artifacts docker images describe "${image}" --project "${GKE_PROJECT_ID}"; then
  gke_log "${image} already exists"
elif [[ "${builder}" == cloudbuild ]]; then
  gke_run gcloud builds submit "${tools_dir}" --tag "${image}" "${GKE_WHERE[@]}" --quiet
else
  # Sandbox nodes are x86_64; the image carries its own libraries, so the platform must match.
  gke_run gcloud auth configure-docker "${GKE_REGISTRY_HOST}" --quiet
  gke_run docker build --platform linux/amd64 --tag "${image}" "${tools_dir}"
  gke_run docker push "${image}"
fi

digest=$(gke_value "sha256:<digest>" gcloud artifacts docker images describe "${image}" \
  --project "${GKE_PROJECT_ID}" --format 'value(image_summary.digest)')
[[ "${digest}" == sha256:* ]] || gke_die "could not read the digest of ${image}"
gke_log "put this in .env (a digest, so every node runs exactly this build):"
printf 'GKE_TOOLS_IMAGE=%s@%s\n' "${image%:*}" "${digest}"
