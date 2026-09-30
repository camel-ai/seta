#!/usr/bin/env bash
# A GKE cluster for Harbor sandboxes (SANDBOX_BACKEND=gke): create it, get credentials, scale it,
# clean up.
#
# Only the sandboxes run in GKE. The Harbor agent server stays on the Ray head node and starts one
# pod per trial through the Kubernetes API; Cloud Build builds a task's image into Artifact
# Registry the first time the task runs. The cluster needs CPU nodes only.
#
#   bash gke_cluster.sh <command> [args]
#
# Commands
#   create                       enable the APIs; Artifact Registry repository; node service account;
#                                optional VPC and Cloud NAT; regional Standard cluster with Dataplane V2
#                                (enforces NetworkPolicy) and Workload Identity; sandbox node pools at
#                                0 nodes; kubeconfig; namespace. Safe to re-run: existing parts are kept.
#   credentials                  write this cluster's kubeconfig to GKE_KUBECONFIG
#   scale <n> | <pool>=<n> ...   start sandbox nodes: <n> per pool, rounded up to a multiple of the
#                                pool's zones; the autoscaler is pinned to that size
#   down [--yes]                 delete sandbox pods and scale the sandbox pools to zero
#   clear-sandboxes [--yes]      delete Harbor sandbox pods (label app=sandbox), e.g. after a crashed run
#   apply-network-policies [--no-web-egress]
#                                apply gke_network_policies.yaml to GKE_NAMESPACE: sandboxes get DNS and
#                                public web (TCP 80/443) only; --no-web-egress: no internet at all.
#                                Undo: kubectl -n <namespace> delete -f gke_network_policies.yaml
#   status                       cluster, node pools, nodes, sandbox pods, network policies
#   delete [--yes]               delete the cluster (registry, network and service accounts are kept)
# Without --yes those commands only say what they would do. DRY_RUN=1 prints every gcloud and
# kubectl command instead of running it.
#
# Settings: environment variables; anything unset is read from ./.env (or ENV_FILE), so the .env of
# an example folder serves both its run script and this one.
#   GKE_PROJECT_ID, GKE_REGION, GKE_CLUSTER_NAME    required
#   GKE_NAMESPACE=default                  namespace of the sandbox pods
#   GKE_REGISTRY_NAME=harbor-sandboxes     Artifact Registry Docker repository (in GKE_REGION)
#   GKE_KUBECONFIG=~/.kube/gke_<project>_<region>_<cluster>   one kubeconfig per cluster
#   GKE_NODE_POOLS="sandboxes=n2-standard-16"   space-separated <pool>=<machine type>; several pools
#                                          spread sandboxes over machine families (CPU quota is per family)
#   GKE_DISK_SIZE_GB=250, GKE_DISK_TYPE=pd-balanced   sandbox node disk: image cache + sandbox storage
#   GKE_ZONES                              comma-separated zones for all nodes (default: GKE picks three)
#   GKE_SYSTEM_MACHINE_TYPE=e2-standard-2  default pool, 1 node per zone, for kube-system pods
#   GKE_RELEASE_CHANNEL=stable
#   GKE_NETWORK=default, GKE_SUBNETWORK=<GKE_NETWORK>   existing VPC and subnet
#   GKE_CREATE_NETWORK=0                   1: create a custom-mode VPC (default <cluster>-net) and subnet
#                                          (default <cluster>-subnet) with GKE_SUBNET_RANGE=10.0.0.0/20,
#                                          GKE_POD_RANGE=10.4.0.0/14, GKE_SERVICE_RANGE=10.8.0.0/20
#   GKE_PRIVATE_NODES=0                    1: nodes without external IPs, plus Cloud NAT <cluster>-nat so
#                                          sandboxes still reach the internet (GKE_CREATE_NAT=0 to skip);
#                                          GKE_MASTER_CIDR only if your GKE version asks for one
#   GKE_AUTHORIZED_NETWORKS                comma-separated CIDRs allowed to reach the control plane;
#                                          must include the public IP of the head node (Harbor calls the API)
#   GKE_NODE_SERVICE_ACCOUNT=harbor-sandbox-nodes@<project>.iam.gserviceaccount.com
#   GKE_NODE_IMAGE_GC=1                    kubelet removes unused images early (disk 50% -> 40%, after
#                                          5 min): nodes cycle through many large task images
#   GKE_COMPOSE_REGISTRY_READ=1            pods in GKE_NAMESPACE may read the registry: the Docker-in-Docker
#                                          pods of Compose tasks pull their prebuilt image with a Workload
#                                          Identity token
#   GKE_ENABLE_APIS=1
#
# IAM for the identity gcloud uses (gcloud auth list); roles/owner covers everything.
#   create     roles/container.admin, roles/artifactregistry.admin, roles/iam.serviceAccountAdmin,
#              roles/iam.serviceAccountUser, roles/resourcemanager.projectIamAdmin,
#              roles/serviceusage.serviceUsageAdmin (or GKE_ENABLE_APIS=0), and with GKE_CREATE_NETWORK=1
#              or GKE_PRIVATE_NODES=1 roles/compute.networkAdmin
#   scale, down, status, clear-sandboxes, apply-network-policies, credentials, delete
#              roles/container.admin
#   training   (the head node running the agent server) roles/container.developer (pods, exec),
#              roles/artifactregistry.reader, roles/cloudbuild.builds.editor, roles/storage.admin
#              (gcloud stages build sources in gs://<project>_cloudbuild)
# create also grants Cloud Build's service account roles/artifactregistry.writer on the repository
# and roles/logging.logWriter, so task image builds can push; it warns when it cannot look that
# account up.
#
# Needs gcloud with gke-gcloud-auth-plugin and kubectl (gcloud components install
# gke-gcloud-auth-plugin kubectl). This file can also be sourced for its helpers.

GKE_COMMON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

gke_log() { printf '[gke] %s\n' "$*" >&2; }
gke_die() { printf '[gke] %s\n' "$*" >&2; exit 2; }

# A command as copy-pasteable shell, on stderr.
gke_print() {
  local arg line="+"
  for arg in "$@"; do
    if [[ "${arg}" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then line+=" ${arg}"; else line+=" $(printf '%q' "${arg}")"; fi
  done
  printf '%s%s\n' "${line}" "${GKE_PRINT_SUFFIX:-}" >&2
}

# Mutating command: printed with DRY_RUN=1, run otherwise.
gke_run() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    gke_print "$@"
    return 0
  fi
  "$@"
}

# Existence check: quiet; with DRY_RUN=1 printed and reported as absent, so the plan shows every step.
gke_probe() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    GKE_PRINT_SUFFIX="    # existence check" gke_print "$@"
    return 1
  fi
  "$@" >/dev/null 2>&1
}

# Query: prints the command's output; with DRY_RUN=1 prints the command and the placeholder $1.
gke_value() {
  local placeholder="$1"
  shift
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    gke_print "$@"
    printf '%s\n' "${placeholder}"
    return 0
  fi
  "$@"
}

# IAM bindings on a just-created service account can fail until it propagates.
gke_retry() {
  local attempt
  for attempt in 1 2 3 4 5 6; do
    "$@" && return 0
    [[ "${attempt}" == 6 ]] || sleep 10
  done
  return 1
}

gke_need() {
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  command -v "$1" >/dev/null 2>&1 || gke_die "$1 not found on PATH ($2)"
}

# Fill unset GKE_* and HARBOR_* variables from ./.env or ENV_FILE; the environment wins.
gke_load_env_file() {
  local file="${ENV_FILE:-${PWD}/.env}" name value
  [[ -f "${file}" ]] || return 0
  while IFS= read -r -d '' name && IFS= read -r -d '' value; do
    [[ -n "${!name+x}" ]] || export "${name}=${value}"
  done < <(
    set +eu -a
    # shellcheck disable=SC1090
    source "${file}" >/dev/null 2>&1
    for name in $(compgen -v GKE_) HARBOR_COMMIT HARBOR_HOME HARBOR_DIR HARBOR_REPO; do
      [[ -n "${!name+x}" ]] && printf '%s\0%s\0' "${name}" "${!name}"
    done
  )
}

# Required settings, then defaults.
gke_settings() {
  local name missing=""
  gke_load_env_file
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || missing+=" ${name}"
  done
  [[ -z "${missing}" ]] || gke_die "set${missing} (environment or .env; see the header of ${GKE_COMMON_DIR}/gke_cluster.sh)"
  GKE_NAMESPACE="${GKE_NAMESPACE:-default}"
  GKE_REGISTRY_NAME="${GKE_REGISTRY_NAME:-harbor-sandboxes}"
  # shellcheck disable=SC2034  # used by gke_tools_image.sh
  GKE_REGISTRY_HOST="${GKE_REGION:-}-docker.pkg.dev"
  GKE_KUBECONFIG="${GKE_KUBECONFIG:-${HOME}/.kube/gke_${GKE_PROJECT_ID:-}_${GKE_REGION:-}_${GKE_CLUSTER_NAME:-}}"
  GKE_NODE_POOLS="${GKE_NODE_POOLS:-sandboxes=n2-standard-16}"
  GKE_DISK_SIZE_GB="${GKE_DISK_SIZE_GB:-250}"
  GKE_DISK_TYPE="${GKE_DISK_TYPE:-pd-balanced}"
  GKE_SYSTEM_MACHINE_TYPE="${GKE_SYSTEM_MACHINE_TYPE:-e2-standard-2}"
  GKE_RELEASE_CHANNEL="${GKE_RELEASE_CHANNEL:-stable}"
  GKE_CREATE_NETWORK="${GKE_CREATE_NETWORK:-0}"
  if [[ "${GKE_CREATE_NETWORK}" == 1 ]]; then
    GKE_NETWORK="${GKE_NETWORK:-${GKE_CLUSTER_NAME:-}-net}"
    GKE_SUBNETWORK="${GKE_SUBNETWORK:-${GKE_CLUSTER_NAME:-}-subnet}"
  else
    GKE_NETWORK="${GKE_NETWORK:-default}"
    GKE_SUBNETWORK="${GKE_SUBNETWORK:-${GKE_NETWORK}}"
  fi
  GKE_SUBNET_RANGE="${GKE_SUBNET_RANGE:-10.0.0.0/20}"
  GKE_POD_RANGE="${GKE_POD_RANGE:-10.4.0.0/14}"
  GKE_SERVICE_RANGE="${GKE_SERVICE_RANGE:-10.8.0.0/20}"
  GKE_PRIVATE_NODES="${GKE_PRIVATE_NODES:-0}"
  GKE_CREATE_NAT="${GKE_CREATE_NAT:-1}"
  GKE_NODE_SERVICE_ACCOUNT="${GKE_NODE_SERVICE_ACCOUNT:-harbor-sandbox-nodes@${GKE_PROJECT_ID:-}.iam.gserviceaccount.com}"
  GKE_NODE_IMAGE_GC="${GKE_NODE_IMAGE_GC:-1}"
  GKE_COMPOSE_REGISTRY_READ="${GKE_COMPOSE_REGISTRY_READ:-1}"
  GKE_ENABLE_APIS="${GKE_ENABLE_APIS:-1}"
  GKE_WHERE=(--region "${GKE_REGION:-}" --project "${GKE_PROJECT_ID:-}")
  GKE_AR_WHERE=(--location "${GKE_REGION:-}" --project "${GKE_PROJECT_ID:-}")
}

gke_pool_names() {
  local spec
  for spec in ${GKE_NODE_POOLS}; do printf '%s\n' "${spec%%=*}"; done
}

gke_kubectl() { gke_run kubectl --kubeconfig "${GKE_KUBECONFIG}" -n "${GKE_NAMESPACE}" "$@"; }

gke_ensure_kubeconfig() {
  gke_need kubectl "gcloud components install kubectl"
  [[ -s "${GKE_KUBECONFIG}" && "${DRY_RUN:-0}" != 1 ]] || gke_cmd_credentials
}

# ------------------------------------------------------------------------------------ commands

gke_cmd_credentials() {
  [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$(dirname -- "${GKE_KUBECONFIG}")"
  gke_run env KUBECONFIG="${GKE_KUBECONFIG}" \
    gcloud container clusters get-credentials "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}"
  command -v gke-gcloud-auth-plugin >/dev/null 2>&1 || [[ "${DRY_RUN:-0}" == 1 ]] ||
    gke_log "WARNING: gke-gcloud-auth-plugin is not on PATH; kubectl and Harbor need it (gcloud components install gke-gcloud-auth-plugin)"
  gke_log "kubeconfig: ${GKE_KUBECONFIG} (export KUBECONFIG=${GKE_KUBECONFIG} to use kubectl on this cluster)"
}

gke_create_network() {
  gke_probe gcloud compute networks describe "${GKE_NETWORK}" --project "${GKE_PROJECT_ID}" ||
    gke_run gcloud compute networks create "${GKE_NETWORK}" --subnet-mode custom \
      --project "${GKE_PROJECT_ID}" --quiet
  gke_probe gcloud compute networks subnets describe "${GKE_SUBNETWORK}" "${GKE_WHERE[@]}" ||
    gke_run gcloud compute networks subnets create "${GKE_SUBNETWORK}" "${GKE_WHERE[@]}" \
      --network "${GKE_NETWORK}" --range "${GKE_SUBNET_RANGE}" \
      --secondary-range "${GKE_SUBNETWORK}-pods=${GKE_POD_RANGE},${GKE_SUBNETWORK}-services=${GKE_SERVICE_RANGE}" \
      --enable-private-ip-google-access --quiet
}

# Private nodes have no external IP: sandboxes reach package mirrors and registries through NAT.
# Many sandboxes per node open many connections, hence the large per-VM port allocation.
gke_create_nat() {
  local router="${GKE_CLUSTER_NAME}-router" nat="${GKE_CLUSTER_NAME}-nat"
  gke_probe gcloud compute routers describe "${router}" "${GKE_WHERE[@]}" ||
    gke_run gcloud compute routers create "${router}" --network "${GKE_NETWORK}" "${GKE_WHERE[@]}" --quiet
  gke_probe gcloud compute routers nats describe "${nat}" --router "${router}" "${GKE_WHERE[@]}" ||
    gke_run gcloud compute routers nats create "${nat}" --router "${router}" "${GKE_WHERE[@]}" \
      --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges --min-ports-per-vm 4096 --quiet
}

# Nodes pull task images with this account; it gets nothing beyond logging, metrics and registry read.
gke_create_node_service_account() {
  local sa="${GKE_NODE_SERVICE_ACCOUNT}" role
  gke_probe gcloud iam service-accounts describe "${sa}" --project "${GKE_PROJECT_ID}" ||
    gke_run gcloud iam service-accounts create "${sa%%@*}" --display-name "GKE sandbox nodes" \
      --project "${GKE_PROJECT_ID}" --quiet
  for role in roles/logging.logWriter roles/monitoring.metricWriter roles/monitoring.viewer; do
    gke_retry gke_run gcloud projects add-iam-policy-binding "${GKE_PROJECT_ID}" \
      --member "serviceAccount:${sa}" --role "${role}" --condition None --quiet >/dev/null
  done
  gke_retry gke_run gcloud artifacts repositories add-iam-policy-binding "${GKE_REGISTRY_NAME}" \
    "${GKE_AR_WHERE[@]}" --member "serviceAccount:${sa}" \
    --role roles/artifactregistry.reader --quiet >/dev/null
}

# Harbor builds task images with `gcloud builds submit`; the build runs as the project's default
# Cloud Build service account, which must be able to push to the repository and write logs.
gke_grant_cloud_build() {
  local sa
  sa=$(gke_value "projects/${GKE_PROJECT_ID}/serviceAccounts/CLOUD_BUILD_SERVICE_ACCOUNT" \
    gcloud builds get-default-service-account --project "${GKE_PROJECT_ID}" \
    --format 'value(serviceAccountEmail)') || sa=""
  sa="${sa##*/}"
  if [[ -z "${sa}" ]]; then
    gke_log "WARNING: could not look up Cloud Build's service account; grant it roles/artifactregistry.writer on ${GKE_REGISTRY_NAME} and roles/logging.logWriter"
    return 0
  fi
  gke_run gcloud artifacts repositories add-iam-policy-binding "${GKE_REGISTRY_NAME}" \
    "${GKE_AR_WHERE[@]}" --member "serviceAccount:${sa}" \
    --role roles/artifactregistry.writer --quiet >/dev/null
  gke_run gcloud projects add-iam-policy-binding "${GKE_PROJECT_ID}" --member "serviceAccount:${sa}" \
    --role roles/logging.logWriter --condition None --quiet >/dev/null
}

gke_create_cluster() {
  local datapath
  if gke_probe gcloud container clusters describe "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}"; then
    gke_log "cluster ${GKE_CLUSTER_NAME} exists; keeping it"
    datapath=$(gke_value ADVANCED_DATAPATH gcloud container clusters describe "${GKE_CLUSTER_NAME}" \
      "${GKE_WHERE[@]}" --format 'value(networkConfig.datapathProvider)')
    [[ "${datapath}" == ADVANCED_DATAPATH ]] ||
      gke_log "WARNING: ${GKE_CLUSTER_NAME} does not use Dataplane V2; network policies will not be enforced"
    return 0
  fi
  local args=(
    "${GKE_WHERE[@]}" --release-channel "${GKE_RELEASE_CHANNEL}"
    --network "${GKE_NETWORK}" --subnetwork "${GKE_SUBNETWORK}" --enable-ip-alias
    # Dataplane V2 enforces NetworkPolicy (gke_network_policies.yaml); it cannot be enabled later.
    --enable-dataplane-v2
    # Workload Identity: pods get no node credentials from the metadata server.
    --workload-pool "${GKE_PROJECT_ID}.svc.id.goog"
    # Default pool for kube-system (DNS, konnectivity, metrics); sandboxes use the pools below.
    --machine-type "${GKE_SYSTEM_MACHINE_TYPE}" --num-nodes 1 --disk-type pd-balanced --disk-size 50
    --image-type COS_CONTAINERD --service-account "${GKE_NODE_SERVICE_ACCOUNT}"
    --scopes "storage-ro,logging-write,monitoring-write" --workload-metadata GKE_METADATA
    --enable-shielded-nodes --shielded-secure-boot --shielded-integrity-monitoring --quiet
  )
  [[ -z "${GKE_ZONES:-}" ]] || args+=(--node-locations "${GKE_ZONES}")
  [[ "${GKE_CREATE_NETWORK}" != 1 ]] || args+=(
    --cluster-secondary-range-name "${GKE_SUBNETWORK}-pods"
    --services-secondary-range-name "${GKE_SUBNETWORK}-services")
  if [[ "${GKE_PRIVATE_NODES}" == 1 ]]; then
    args+=(--enable-private-nodes)
    [[ -z "${GKE_MASTER_CIDR:-}" ]] || args+=(--master-ipv4-cidr "${GKE_MASTER_CIDR}")
  fi
  if [[ -n "${GKE_AUTHORIZED_NETWORKS:-}" ]]; then
    args+=(--enable-master-authorized-networks --master-authorized-networks "${GKE_AUTHORIZED_NETWORKS}")
  else
    gke_log "GKE_AUTHORIZED_NETWORKS unset: the control plane accepts (authenticated) connections from any IP"
  fi
  gke_run gcloud container clusters create "${GKE_CLUSTER_NAME}" "${args[@]}"
}

gke_create_pool() {
  local pool="$1" machine="$2" node_config=""
  if gke_probe gcloud container node-pools describe "${pool}" --cluster "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}"; then
    gke_log "node pool ${pool} exists; keeping it"
    return 0
  fi
  local args=(
    --cluster "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" --machine-type "${machine}"
    --disk-type "${GKE_DISK_TYPE}" --disk-size "${GKE_DISK_SIZE_GB}" --image-type COS_CONTAINERD
    --num-nodes 0 --service-account "${GKE_NODE_SERVICE_ACCOUNT}"
    --scopes "storage-ro,logging-write,monitoring-write" --workload-metadata GKE_METADATA
    --shielded-secure-boot --shielded-integrity-monitoring --enable-autorepair --enable-autoupgrade
    --max-surge-upgrade 1 --max-unavailable-upgrade 0 --quiet
  )
  if [[ "${GKE_NODE_IMAGE_GC}" == 1 ]]; then
    # Harbor's examples/gke-grpo-rollout/node-gc.yaml: every task has its own multi-GB image, so the
    # default GC (85% disk) lets a node fill up and evict running sandboxes.
    node_config=$(mktemp "${TMPDIR:-/tmp}/gke-node-config.XXXXXX")
    cat > "${node_config}" <<'YAML'
kubeletConfig:
  imageGcHighThresholdPercent: 50
  imageGcLowThresholdPercent: 40
  imageMinimumGcAge: 2m
  imageMaximumGcAge: 5m
YAML
    args+=(--system-config-from-file "${node_config}")
    [[ "${DRY_RUN:-0}" != 1 ]] || { gke_log "${node_config}:"; cat "${node_config}" >&2; }
  fi
  gke_run gcloud container node-pools create "${pool}" "${args[@]}"
  [[ -z "${node_config}" ]] || rm -f "${node_config}"
}

gke_cmd_create() {
  local spec project_number
  gke_need gcloud "https://cloud.google.com/sdk/docs/install"
  gke_need kubectl "gcloud components install kubectl"
  if [[ "${GKE_ENABLE_APIS}" == 1 ]]; then
    gke_run gcloud services enable container.googleapis.com compute.googleapis.com \
      artifactregistry.googleapis.com cloudbuild.googleapis.com iam.googleapis.com \
      --project "${GKE_PROJECT_ID}"
  fi
  # Task images (built by Harbor with Cloud Build) and the sandbox tools image live here.
  gke_probe gcloud artifacts repositories describe "${GKE_REGISTRY_NAME}" "${GKE_AR_WHERE[@]}" ||
    gke_run gcloud artifacts repositories create "${GKE_REGISTRY_NAME}" --repository-format docker \
      "${GKE_AR_WHERE[@]}" --description "Harbor sandbox images" --quiet
  [[ "${GKE_CREATE_NETWORK}" != 1 ]] || gke_create_network
  [[ "${GKE_PRIVATE_NODES}" != 1 || "${GKE_CREATE_NAT}" != 1 ]] || gke_create_nat
  gke_create_node_service_account
  gke_grant_cloud_build
  gke_create_cluster
  for spec in ${GKE_NODE_POOLS}; do
    [[ "${spec}" == *=* ]] || gke_die "GKE_NODE_POOLS entry '${spec}' needs <pool>=<machine type> for create"
    gke_create_pool "${spec%%=*}" "${spec#*=}"
  done
  gke_cmd_credentials
  if [[ "${GKE_NAMESPACE}" != default ]]; then
    gke_probe kubectl --kubeconfig "${GKE_KUBECONFIG}" get namespace "${GKE_NAMESPACE}" ||
      gke_run kubectl --kubeconfig "${GKE_KUBECONFIG}" create namespace "${GKE_NAMESPACE}"
  fi
  if [[ "${GKE_COMPOSE_REGISTRY_READ}" == 1 ]]; then
    project_number=$(gke_value PROJECT_NUMBER gcloud projects describe "${GKE_PROJECT_ID}" --format 'value(projectNumber)')
    gke_retry gke_run gcloud artifacts repositories add-iam-policy-binding "${GKE_REGISTRY_NAME}" \
      "${GKE_AR_WHERE[@]}" \
      --member "principalSet://iam.googleapis.com/projects/${project_number}/locations/global/workloadIdentityPools/${GKE_PROJECT_ID}.svc.id.goog/namespace/${GKE_NAMESPACE}" \
      --role roles/artifactregistry.reader --quiet >/dev/null
  fi
  cat >&2 <<EOF
[gke] cluster ${GKE_CLUSTER_NAME} is set up with its sandbox pools at 0 nodes. Next:
  bash ${GKE_COMMON_DIR}/gke_tools_image.sh              # once: prints GKE_TOOLS_IMAGE for .env
  bash ${GKE_COMMON_DIR}/gke_cluster.sh apply-network-policies   # optional egress control
  bash ${GKE_COMMON_DIR}/gke_cluster.sh scale <nodes>    # before a run; 'down --yes' after it
EOF
}

gke_scale_pool() {
  local pool="$1" want="$2" locations zones per_zone total
  if [[ "${want}" == 0 ]]; then
    gke_run gcloud container clusters update "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" \
      --node-pool "${pool}" --no-enable-autoscaling --quiet
    gke_run gcloud container clusters resize "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" \
      --node-pool "${pool}" --num-nodes 0 --quiet
    return 0
  fi
  # A regional pool's --num-nodes is per zone.
  locations=$(gke_value "${GKE_ZONES:-zone-a,zone-b,zone-c}" gcloud container node-pools describe "${pool}" \
    --cluster "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" --format 'value(locations)')
  IFS=';,' read -r -a zones <<< "${locations}"
  (( ${#zones[@]} > 0 )) || gke_die "could not read the zones of node pool ${pool}"
  per_zone=$(( (want + ${#zones[@]} - 1) / ${#zones[@]} ))
  total=$(( per_zone * ${#zones[@]} ))
  gke_log "${pool}: ${total} nodes (${per_zone} in each of ${#zones[@]} zones)"
  # Autoscaler pinned at that size: it replaces lost nodes in any zone with capacity and never
  # removes nodes from under running sandboxes.
  gke_run gcloud container clusters update "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" --node-pool "${pool}" \
    --enable-autoscaling --total-min-nodes "${total}" --total-max-nodes "${total}" --location-policy ANY --quiet
  gke_run gcloud container clusters resize "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" --node-pool "${pool}" \
    --num-nodes "${per_zone}" --quiet
}

gke_cmd_scale() {
  local spec pool
  [[ $# -gt 0 ]] || gke_die "usage: gke_cluster.sh scale <nodes> | <pool>=<nodes> ..."
  gke_need gcloud "https://cloud.google.com/sdk/docs/install"
  for spec in "$@"; do
    if [[ "${spec}" =~ ^[0-9]+$ ]]; then
      for pool in $(gke_pool_names); do gke_scale_pool "${pool}" "${spec}"; done
    elif [[ "${spec}" =~ ^([a-z0-9-]+)=([0-9]+)$ ]]; then
      gke_scale_pool "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
      gke_die "scale: '${spec}' is not <nodes> or <pool>=<nodes>"
    fi
  done
}

gke_cmd_clear_sandboxes() {
  local yes="${1:-}"
  gke_ensure_kubeconfig
  if [[ "${yes}" != --yes ]]; then
    gke_kubectl get pods -l app=sandbox
    gke_log "nothing deleted. Stop every run using ${GKE_CLUSTER_NAME}, then: bash ${GKE_COMMON_DIR}/gke_cluster.sh clear-sandboxes --yes"
    return 0
  fi
  gke_kubectl delete pods -l app=sandbox --ignore-not-found --wait=true
}

gke_cmd_down() {
  local pool
  if [[ "${1:-}" != --yes ]]; then
    gke_log "would delete sandbox pods and scale node pools $(gke_pool_names | paste -sd, -) of ${GKE_CLUSTER_NAME} to zero"
    gke_log "stop every run using the cluster, then: bash ${GKE_COMMON_DIR}/gke_cluster.sh down --yes"
    return 0
  fi
  gke_need gcloud "https://cloud.google.com/sdk/docs/install"
  gke_ensure_kubeconfig
  gke_kubectl delete pods -l app=sandbox --ignore-not-found --wait=false
  for pool in $(gke_pool_names); do gke_scale_pool "${pool}" 0; done
  gke_log "sandbox pools are at zero; the control plane and the default pool keep billing until 'delete'"
}

gke_cmd_apply_network_policies() {
  local option="${1:-}"
  [[ -z "${option}" || "${option}" == --no-web-egress ]] ||
    gke_die "usage: gke_cluster.sh apply-network-policies [--no-web-egress]"
  gke_ensure_kubeconfig
  gke_kubectl apply -f "${GKE_COMMON_DIR}/gke_network_policies.yaml"
  if [[ "${option}" == --no-web-egress ]]; then
    gke_kubectl delete networkpolicy allow-sandbox-web-egress --ignore-not-found
  fi
  gke_log "applied to namespace ${GKE_NAMESPACE}; set GKE_CLUSTER_DEFAULT_DENY_EGRESS=true if your Harbor pin supports it"
}

gke_cmd_status() {
  local pods
  gke_need gcloud "https://cloud.google.com/sdk/docs/install"
  gke_run gcloud container clusters describe "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" \
    --format 'table(name,status,currentMasterVersion,location,networkConfig.datapathProvider)'
  gke_run gcloud container node-pools list --cluster "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" \
    --format 'table(name,config.machineType,status,autoscaling.enabled,autoscaling.totalMinNodeCount,autoscaling.totalMaxNodeCount,locations.list())'
  gke_ensure_kubeconfig
  gke_run kubectl --kubeconfig "${GKE_KUBECONFIG}" get nodes -L cloud.google.com/gke-nodepool
  pods=$(gke_value "" kubectl --kubeconfig "${GKE_KUBECONFIG}" -n "${GKE_NAMESPACE}" get pods -l app=sandbox -o name)
  printf 'sandbox pods in %s: %s\n' "${GKE_NAMESPACE}" "$(printf '%s' "${pods}" | grep -c . || true)"
  gke_kubectl get networkpolicy
}

gke_cmd_delete() {
  if [[ "${1:-}" != --yes ]]; then
    gke_log "would delete cluster ${GKE_CLUSTER_NAME} (${GKE_REGION}); registry ${GKE_REGISTRY_NAME}, network and service accounts stay"
    gke_log "stop every run using the cluster, then: bash ${GKE_COMMON_DIR}/gke_cluster.sh delete --yes"
    return 0
  fi
  gke_need gcloud "https://cloud.google.com/sdk/docs/install"
  gke_run gcloud container clusters delete "${GKE_CLUSTER_NAME}" "${GKE_WHERE[@]}" --quiet
  [[ "${DRY_RUN:-0}" == 1 ]] || rm -f "${GKE_KUBECONFIG}"
}

gke_main() {
  set -euo pipefail
  local command="${1:-}"
  [[ $# -eq 0 ]] || shift
  case "${command}" in
    create | credentials | scale | down | clear-sandboxes | apply-network-policies | status | delete) ;;
    *)
      sed -n '2,/^# Settings/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//' >&2
      exit 2
      ;;
  esac
  gke_settings GKE_PROJECT_ID GKE_REGION GKE_CLUSTER_NAME
  "gke_cmd_${command//-/_}" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  gke_main "$@"
fi
