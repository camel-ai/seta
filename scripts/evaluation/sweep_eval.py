"""Sweep eval.py over a cartesian product of config axes.
Restarts SGLang automatically when model.model_type changes between runs.
Put the model axis first in sweep.yaml to minimise restarts.

Usage:
    python sweep_eval.py configs/sweep.yaml
    python sweep_eval.py configs/sweep.yaml --dry-run
    python sweep_eval.py configs/sweep.yaml grpo.n_trajs=16
"""
import itertools, os, subprocess, sys, time, urllib.request
from pathlib import Path
import yaml

EVAL      = Path(__file__).parent / "eval.py"
ROOT      = Path(__file__).parents[2]
SGLANG_SH = Path(__file__).parent / "start_sglang.sh"
PORT      = os.environ.get("SGLANG_PORT", "30000")


def sglang_ready():
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
        return True
    except Exception:
        return False


def restart_sglang(model):
    subprocess.run(["pkill", "-f", "sglang.launch_server"], check=False)
    time.sleep(3)
    subprocess.Popen(["bash", str(SGLANG_SH), model])
    print(f"  [sglang] waiting for {model} ...", flush=True)
    for _ in range(60):        # up to 5 min
        if sglang_ready():
            print("  [sglang] ready"); return
        time.sleep(5)
    sys.exit("ERROR: SGLang failed to start")


cfg    = yaml.safe_load(open(sys.argv[1]))
base   = (Path(sys.argv[1]).parent / cfg["base_config"]).resolve()
dry    = "--dry-run" in sys.argv
extra  = [a for a in sys.argv[2:] if a != "--dry-run"]
axes   = cfg.get("axes", {})
common = cfg.get("common", {})

runs = []
for combo in itertools.product(*axes.values()):
    label     = "__".join(item["label"] for item in combo)
    overrides = {**common, **{k: v for item in combo for k, v in item.items() if k != "label"}}
    runs.append((label, overrides))

current_model = None
for i, (label, overrides) in enumerate(runs, 1):
    model = overrides.get("terminal_env.model.model_type")
    if model and model != current_model and not dry:
        restart_sglang(model)
        current_model = model

    print(f"\n[{i}/{len(runs)}] {label}")
    cmd = [sys.executable, str(EVAL), "--config", str(base), f"trial_name={label}",
           *[f"{k}={v}" for k, v in overrides.items()], *extra]
    print(" ", " ".join(cmd))
    if not dry:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        subprocess.run(cmd, cwd=str(ROOT), env=env)

subprocess.run(["pkill", "-f", "sglang.launch_server"], check=False)
