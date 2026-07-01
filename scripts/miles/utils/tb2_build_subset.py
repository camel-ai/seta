"""Build a subset parquet of TB2.0 tasks that have < 5 SCORED trials for a given
checkpoint, aggregating across the checkpoint's original run dir AND any prior
backfill dirs. Prints the number of short tasks (last stdout line)."""
import sys, os, glob, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd

ckpt, out = sys.argv[1], sys.argv[2]
BASE = "/data/training_runs/eval-tb2.0-2r7t-sweep-20260602-191645"
FULL = "/data/terminal_agent/dataset/terminal-bench-2.0.parquet"
TARGET = 5

# all run dirs for this checkpoint: original + backfills (…-<ckpt>  and  …-<ckpt>-bf-*)
rundirs = []
for d in glob.glob(f"{BASE}-{ckpt}") + glob.glob(f"{BASE}-{ckpt}-bf-*"):
    if os.path.isdir(d):
        rundirs.append(d)

counts = {}
for rd in rundirs:
    for tdir in glob.glob(f"{rd}/trials/*"):
        if not os.path.isdir(tdir):
            continue
        trial = os.path.basename(tdir); suf = f"_{trial}_"
        for n in os.listdir(tdir):
            if suf not in n:
                continue
            t = n.split(suf)[0]
            p = os.path.join(tdir, n, "run_info.json")
            if not os.path.exists(p):
                continue
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if isinstance(d.get("reward"), (int, float)):
                counts[t] = counts.get(t, 0) + 1

df = pd.read_parquet(FULL)
ids = [str(m["instance_id"]) for m in df["metadata"]]
short = set(i for i in ids if counts.get(i, 0) < TARGET)
mask = [str(m["instance_id"]) in short for m in df["metadata"]]
sub = df[mask].reset_index(drop=True)
sub.to_parquet(out)
# diagnostics to stderr, count to stdout
print(f"ckpt={ckpt} rundirs={len(rundirs)} scored_tasks={len(counts)} short={len(short)} -> {out}", file=sys.stderr)
print(len(short))
