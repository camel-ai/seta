"""Download Inkling-Small and convert it to the Megatron torch_dist checkpoint the actor loads.

Run once, inside the Miles container on one node with 8 GPUs, on storage every node can read at
the same path (or copy both outputs to every node afterwards):

  PYTHONPATH=/root/miles python scripts/miles/examples/inkling_grpo/prepare_model.py --model-root /root/models

Writes <model-root>/Inkling-Small (Hugging Face checkpoint, BF16) and
<model-root>/Inkling-Small_torch_dist. Both steps are resumable: the download skips files already
present and the conversion is skipped once its output is complete.
"""

import os
import shutil
import subprocess
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

HF_REPO = "thinkingmachines/Inkling-Small"
HF_REVISION = "8cc5877b44d343f88b92086aa1fb72897950f06a"  # the revision the run scripts were tested with
MODEL_NAME = "Inkling-Small"
MEGATRON_MODEL_TYPE = "inkling-small"  # scripts/models/inkling-small.sh in Miles


def main(
    model_root: Path = typer.Option(Path("/root/models"), help="output directory for both checkpoints"),
    megatron_path: Path = typer.Option(Path("/root/Megatron-LM"), help="Megatron-LM checkout"),
    revision: str = typer.Option(HF_REVISION, help=f"Hugging Face revision of {HF_REPO}"),
):
    hf_dir = model_root / MODEL_NAME
    if shutil.which("hf") is None:
        raise RuntimeError("the Hugging Face `hf` CLI is missing (pip install -U huggingface_hub)")
    subprocess.run(
        ["hf", "download", HF_REPO, "--revision", revision, "--local-dir", str(hf_dir), "--max-workers", "4"],
        check=True,
    )
    if not (hf_dir / "config.json").is_file():
        raise FileNotFoundError(f"download is incomplete: {hf_dir}")
    if not megatron_path.is_dir():
        raise FileNotFoundError(f"missing Megatron-LM checkout: {megatron_path}")

    import torch

    if torch.cuda.device_count() < 8:
        raise RuntimeError("the TP8/EP8 conversion needs 8 visible GPUs")
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # Keep the single pipeline stage requested below; otherwise the converter spreads the
    # layers over one stage per GPU.
    os.environ["CONVERT_KEEP_PP1"] = "1"
    # Skipped when <model-root>/Inkling-Small_torch_dist is already complete.
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=8,
        hf_checkpoint=str(hf_dir),
        dir_dst=str(model_root),
        megatron_path=str(megatron_path),
        extra_args=(
            "--tensor-model-parallel-size 8 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
            "--sequence-parallel"
        ),
    )
    print(f"ready: {hf_dir} and {model_root / (MODEL_NAME + '_torch_dist')}")


if __name__ == "__main__":
    typer.run(main)
