"""Convert a harbor-format dataset folder to a parquet file for AReaL eval.

Usage:
    python scripts/areal/convert_harbor_to_parquet.py \
        --dataset dataset/seta-env-harbor \
        --output dataset/seta-env-harbor.parquet

The output parquet has three columns:
    task_name   — subdirectory name (e.g. "0", "42")
    task_path   — absolute path to the task directory
    instruction — contents of instruction.md
"""
import argparse
from pathlib import Path

from seta_env.dataset import load_harbor_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert harbor dataset to parquet.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to harbor-format dataset folder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet file path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[2] / dataset_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[2] / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_harbor_dataset(dataset_path)
    ds.to_parquet(str(output_path))
    print(f"Saved {len(ds)} tasks to {output_path}")


if __name__ == "__main__":
    main()
