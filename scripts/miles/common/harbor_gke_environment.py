"""Write Harbor's GKE environment config (SANDBOX_BACKEND=gke) from GKE_* variables.

harbor_configure in launcher.sh runs this with the Harbor virtualenv's Python and passes the
file to the agent server as --environment-config. Create the cluster with gke_cluster.sh and the
tools image with gke_tools_image.sh.

Required
  GKE_PROJECT_ID, GKE_REGION, GKE_CLUSTER_NAME   regional cluster; registry and builds use GKE_REGION
  GKE_TOOLS_IMAGE                    sandbox tools image (tmux for Terminus-2), from gke_tools_image.sh
  GKE_ENVIRONMENT_CONFIG             output path (the launcher uses $RUN_ROOT/gke-environment.json)
Optional (default)
  GKE_NAMESPACE (default)            namespace of the sandbox pods
  GKE_REGISTRY_NAME (harbor-sandboxes)   Artifact Registry repository for task images
  SANDBOX_CPUS, SANDBOX_MEMORY_MB, SANDBOX_STORAGE_MB (task)   pod size; "task" keeps each task's own
  GKE_POD_PRIVILEGED (auto)          auto: privileged pods only for tasks whose Dockerfile installs a
                                     Docker daemon; false for untrusted tasks; true for all
  GKE_POD_READY_TIMEOUT_SEC          pod start deadline (Harbor's default: 300 s for CPU pods)
  GKE_MEMORY_LIMIT_MULTIPLIER        memory limit = multiplier x request (unset: no limit)
  GKE_CLUSTER_DEFAULT_DENY_EGRESS (false)   true once gke_network_policies.yaml is applied: Harbor
                                     then runs tasks that ask for network_mode = "no-network"
  GKE_CLOUD_BUILD_CONFIG             Cloud Build config for task images (default: Harbor's
                                     examples/gke-grpo-rollout/cloudbuild-buildkit.yaml in HARBOR_DIR)
  GKE_CLOUD_BUILD_LOGS_BUCKET        gs://bucket[/path] for build logs (unset: Cloud Logging only)
  GKE_CLOUD_BUILD_WORKER_POOL        projects/<project>/locations/<region>/workerPools/<pool>
  GKE_COMPOSE_PREBUILT_MANIFEST_PATH  manifest of prebuilt Compose service images; needs both
  GKE_REGISTRY_TOKEN_IMPERSONATOR, GKE_REGISTRY_TOKEN_SERVICE_ACCOUNT   (short-lived registry token
                                     for the inner Docker, minted by the head node)
Some options exist only in newer Harbor commits; they fail here, instead of being ignored, when
the checkout in HARBOR_DIR lacks them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REQUIRED = {
    "GKE_PROJECT_ID": "Google Cloud project of the cluster",
    "GKE_REGION": "region of the cluster, registry and Cloud Build",
    "GKE_CLUSTER_NAME": "the GKE cluster (see common/gke_cluster.sh)",
    "GKE_TOOLS_IMAGE": "sandbox tools image, printed by common/gke_tools_image.sh",
    "GKE_ENVIRONMENT_CONFIG": "output file (set by harbor_configure)",
}
HARBOR_BUILD_CONFIG = "examples/gke-grpo-rollout/cloudbuild-buildkit.yaml"


def fail(message: str) -> None:
    sys.exit(f"[gke] {message}")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def positive_int(name: str) -> int | None:
    value = env(name)
    if value is None:
        return None
    if not value.isdigit() or int(value) < 1:
        fail(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def sandbox_size(name: str) -> int | None:
    value = env(name, "task")
    if value == "task":
        return None
    if not value.isdigit() or int(value) < 1:
        fail(f"{name} must be a positive integer or 'task', got {value!r}")
    return int(value)


def harbor_supports(argument: str) -> bool:
    """Whether the pinned Harbor's GKE environment accepts this keyword argument.

    Older commits take unknown arguments through **kwargs and silently ignore them.
    """
    harbor_dir = env("HARBOR_DIR")
    source = Path(harbor_dir or "", "src/harbor/environments/gke.py")
    if not harbor_dir or not source.is_file():
        return True
    return argument in source.read_text()


def require_support(variable: str, argument: str) -> None:
    if not harbor_supports(argument):
        fail(
            f"{variable} needs a Harbor commit whose GKE environment has '{argument}'; "
            f"HARBOR_COMMIT={env('HARBOR_COMMIT', '?')} does not. Unset {variable} or use a newer commit."
        )


def cloud_build_config(output_dir: Path) -> str:
    """Path of the Cloud Build config Harbor submits task image builds with.

    Harbor's example config names its maintainers' log bucket, which other projects cannot
    write, so a copy without it is written next to the environment config. Without a log bucket
    of yours, build logs go to Cloud Logging only (works with any build service account).
    """
    user_config = env("GKE_CLOUD_BUILD_CONFIG")
    logs_bucket = env("GKE_CLOUD_BUILD_LOGS_BUCKET")
    if user_config:
        source = Path(user_config).expanduser().resolve()
    else:
        harbor_dir = env("HARBOR_DIR")
        if not harbor_dir:
            fail("set GKE_CLOUD_BUILD_CONFIG, or HARBOR_DIR (harbor_install sets it) to use Harbor's config")
        source = Path(harbor_dir, HARBOR_BUILD_CONFIG)
    if not source.is_file():
        fail(f"Cloud Build config not found: {source}")
    if user_config and not logs_bucket:
        return str(source)

    try:
        import yaml
    except ImportError:
        fail("PyYAML is missing; run this with the Harbor virtualenv's python")
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict) or not config.get("steps"):
        fail(f"{source} is not a Cloud Build config (no steps)")
    if not user_config:
        config.pop("logsBucket", None)
    if logs_bucket:
        config["logsBucket"] = logs_bucket if logs_bucket.startswith("gs://") else f"gs://{logs_bucket}"
    if "logsBucket" not in config:
        config.setdefault("options", {}).setdefault("logging", "CLOUD_LOGGING_ONLY")

    def block_strings(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    yaml.SafeDumper.add_representer(str, block_strings)
    target = output_dir / "cloudbuild.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False, width=100_000))
    return str(target)


def main() -> None:
    missing = [f"  {name}: {why}" for name, why in REQUIRED.items() if not env(name)]
    if missing:
        fail("SANDBOX_BACKEND=gke needs:\n" + "\n".join(missing))
    output = Path(env("GKE_ENVIRONMENT_CONFIG")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    region = env("GKE_REGION")

    privileged = env("GKE_POD_PRIVILEGED", "auto").lower()
    if privileged not in {"auto", "true", "false"}:
        fail(f"GKE_POD_PRIVILEGED must be auto, true or false, got {privileged!r}")

    kwargs = {
        "project_id": env("GKE_PROJECT_ID"),
        "cluster_name": env("GKE_CLUSTER_NAME"),
        "region": region,
        "namespace": env("GKE_NAMESPACE", "default"),
        "registry_location": region,
        "registry_name": env("GKE_REGISTRY_NAME", "harbor-sandboxes"),
        "sandbox_tools_image": env("GKE_TOOLS_IMAGE"),
        "cloud_build_config_path": cloud_build_config(output.parent),
        # Builds on the shared pool use small builders (a common default quota); two hours
        # covers compilation-heavy task images.
        "cloud_build_machine_type": "E2_STANDARD_2",
        "cloud_build_disk_size_gb": 200,
        "cloud_build_timeout_sec": 7200,
        # Compose tasks pull sidecar images and wait for health checks on start.
        "compose_up_timeout_sec": 1200,
        "pod_privileged": privileged if privileged == "auto" else privileged == "true",
        # Replace the image's entrypoint as well as its command, so images whose entrypoint
        # exits or expects arguments still keep the pod alive.
        "keepalive": ["sh", "-c", "sleep infinity"],
    }
    if worker_pool := env("GKE_CLOUD_BUILD_WORKER_POOL"):
        kwargs["cloud_build_worker_pool"] = worker_pool
    if (multiplier := env("GKE_MEMORY_LIMIT_MULTIPLIER")) is not None:
        try:
            kwargs["memory_limit_multiplier"] = float(multiplier)
        except ValueError:
            fail(f"GKE_MEMORY_LIMIT_MULTIPLIER must be a number, got {multiplier!r}")
    if (timeout := positive_int("GKE_POD_READY_TIMEOUT_SEC")) is not None:
        require_support("GKE_POD_READY_TIMEOUT_SEC", "pod_ready_timeout_sec")
        kwargs["pod_ready_timeout_sec"] = timeout
    if (deny := env("GKE_CLUSTER_DEFAULT_DENY_EGRESS")) is not None:
        if deny.lower() not in {"true", "false"}:
            fail(f"GKE_CLUSTER_DEFAULT_DENY_EGRESS must be true or false, got {deny!r}")
        if deny.lower() == "true":
            require_support("GKE_CLUSTER_DEFAULT_DENY_EGRESS", "cluster_default_deny_egress")
            kwargs["cluster_default_deny_egress"] = True
    if manifest := env("GKE_COMPOSE_PREBUILT_MANIFEST_PATH"):
        require_support("GKE_COMPOSE_PREBUILT_MANIFEST_PATH", "compose_prebuilt_manifest_path")
        manifest_path = Path(manifest).expanduser().resolve()
        if not manifest_path.is_file():
            fail(f"GKE_COMPOSE_PREBUILT_MANIFEST_PATH not found: {manifest_path}")
        impersonator = env("GKE_REGISTRY_TOKEN_IMPERSONATOR")
        token_account = env("GKE_REGISTRY_TOKEN_SERVICE_ACCOUNT")
        if not impersonator or not token_account:
            fail(
                "GKE_COMPOSE_PREBUILT_MANIFEST_PATH also needs GKE_REGISTRY_TOKEN_IMPERSONATOR and "
                "GKE_REGISTRY_TOKEN_SERVICE_ACCOUNT"
            )
        kwargs["compose_prebuilt_manifest_path"] = str(manifest_path)
        kwargs["registry_token_impersonator"] = impersonator
        kwargs["registry_token_service_account"] = token_account

    config = {
        "type": "gke",
        "force_build": False,
        "delete": True,
        "override_cpus": sandbox_size("SANDBOX_CPUS"),
        "override_memory_mb": sandbox_size("SANDBOX_MEMORY_MB"),
        "override_storage_mb": sandbox_size("SANDBOX_STORAGE_MB"),
        "kwargs": kwargs,
    }
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"[gke] {output}: cluster {kwargs['cluster_name']} ({region}), namespace {kwargs['namespace']}, "
        f"registry {kwargs['registry_name']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
