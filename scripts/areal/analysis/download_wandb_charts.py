#!/usr/bin/env python
"""Download all charts and metrics from a wandb run."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path

# Configuration
RUN_PATH = "jayrainton-sambanova-systems/camel-terminal_agent-grpo/runs/camel-terminal_agent-grpo_date-0325-1376_dataset_sft_trained_ckpt-8xh100-qwen3-8b-grpo_train"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wandb_downloads")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# Initialize wandb API
api = wandb.Api()
run = api.run(RUN_PATH)

print("="*80)
print(f"Run: {run.name}")
print(f"ID: {run.id}")
print(f"State: {run.state}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)

# Create output directory
run_output_dir = os.path.join(OUTPUT_DIR, run.id)
os.makedirs(run_output_dir, exist_ok=True)

# Download metrics history
print("\nFetching metrics history...")
history = run.history(samples=1000000)

if not history.empty:
    # Save complete history
    csv_file = os.path.join(run_output_dir, "complete_history.csv")
    history.to_csv(csv_file, index=False)
    print(f"✓ Saved complete history: {csv_file}")
    print(f"  Rows: {len(history)}, Columns: {len(history.columns)}")

    # Save as parquet
    parquet_file = os.path.join(run_output_dir, "complete_history.parquet")
    history.to_parquet(parquet_file, index=False)
    print(f"✓ Saved parquet: {parquet_file}")

    # Generate charts
    print("\nGenerating chart images...")
    charts_dir = os.path.join(run_output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Get numeric columns (metrics)
    numeric_cols = history.select_dtypes(include=['float64', 'int64']).columns
    skip_cols = {'_step', '_runtime', '_timestamp'}
    metric_cols = [col for col in numeric_cols if col not in skip_cols]

    print(f"Found {len(metric_cols)} metrics to plot:")
    for col in metric_cols:
        if history[col].isna().all():
            continue

        data = history[['_step', col]].dropna()
        if len(data) == 0:
            continue

        print(f"  - {col} ({len(data)} points)")

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(data['_step'], data[col], linewidth=2)
        plt.xlabel('Step')
        plt.ylabel(col)
        plt.title(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        safe_name = col.replace('/', '_').replace(' ', '_')
        chart_file = os.path.join(charts_dir, f"{safe_name}.png")
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Charts saved to: {charts_dir}")

# Download logged media files
print("\nChecking for logged media files...")
media_dir = os.path.join(run_output_dir, "media")
os.makedirs(media_dir, exist_ok=True)

try:
    files = run.files()
    media_files = [f for f in files if any(f.name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf'])]

    if media_files:
        print(f"Found {len(media_files)} media files:")
        for file in media_files:
            print(f"  - {file.name}")
            file.download(root=media_dir, replace=True)
        print(f"✓ Media files saved to: {media_dir}")
    else:
        print("  No media files found")
except Exception as e:
    print(f"  Could not fetch media files: {e}")

# Save summary and config
print("\nSaving run summary and config...")

summary = dict(run.summary)
if summary:
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(run_output_dir, "summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

config = dict(run.config)
if config:
    config_df = pd.DataFrame([config])
    config_file = os.path.join(run_output_dir, "config.csv")
    config_df.to_csv(config_file, index=False)
    print(f"✓ Saved config: {config_file}")

print("\n" + "="*80)
print("Download complete!")
print(f"All files saved to: {run_output_dir}")
print("="*80)
