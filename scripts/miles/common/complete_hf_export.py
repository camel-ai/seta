#!/usr/bin/env python3
"""Complete a Miles torch_dist -> HF export with the weights training never touched.

Called by export_to_hf.sh after Miles' tools/convert_torch_dist_to_hf.py.

The converter writes only what Megatron trained. Some Hugging Face checkpoints
carry more: e.g. a vision-language model's vision tower (``model.visual.*``) or
a multi-token-prediction head (``mtp.*``) that text-only training never loads.
Serving frameworks expect the full checkpoint, so this script copies every
tensor that is in the base checkpoint but missing from the export, verbatim,
into one extra shard, merges the safetensors index, and then checks that the
export has exactly the base checkpoint's tensor names and shapes (a dtype may
only widen bf16 -> fp32, which SGLang casts back on load).

    python complete_hf_export.py --export-dir OUT --base-hf-dir BASE [--prefix model.visual. --prefix mtp.]

``--prefix`` restricts the copy to tensors with those name prefixes; any other
missing tensor is then reported as an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch
from safetensors import safe_open



def shard_headers(hf_dir: Path) -> dict[str, tuple[str, list[int], str]]:
    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    out: dict[str, tuple[str, list[int], str]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(hf_dir / shard, framework="pt") as f:
            for name in f.keys():
                slice_ = f.get_slice(name)
                out[name] = (shard, list(slice_.get_shape()), str(slice_.get_dtype()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--base-hf-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix", action="append", default=[],
        help="only copy missing tensors whose name starts with this prefix (repeatable)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="skip the copy; only compare the export"
    )
    args = parser.parse_args()

    base = shard_headers(args.base_hf_dir)
    export_index_path = args.export_dir / "model.safetensors.index.json"
    export_index = json.loads(export_index_path.read_text())
    exported = set(export_index["weight_map"])

    to_copy = [n for n in base if n not in exported and (not args.prefix or n.startswith(tuple(args.prefix)))]
    if args.verify_only or not to_copy:
        to_copy, copied_bytes, shard_name = [], 0, "(none)"
    else:
        shard_name = "model-untrained-from-base.safetensors"
        copied_bytes = copy_towers(args, base, to_copy, shard_name, export_index, export_index_path)

    got = shard_headers(args.export_dir)
    missing = sorted(set(base) - set(got))
    extra = sorted(set(got) - set(base))
    shape_mismatched = sorted(n for n in base if n in got and base[n][1] != got[n][1])
    # Megatron may keep some parameters in fp32 (e.g. GatedDeltaNet A_log) that
    # the HF checkpoint stores as bf16; SGLang casts on load (param.copy_), so a
    # widening dtype difference is reported, not rejected.
    dtype_widened = sorted(
        n for n in base
        if n in got and base[n][1] == got[n][1] and base[n][2] != got[n][2]
        and (base[n][2], got[n][2]) == ("BF16", "F32")
    )
    dtype_mismatched = sorted(
        n for n in base
        if n in got and base[n][1] == got[n][1] and base[n][2] != got[n][2]
        and n not in dtype_widened
    )
    print(
        f"copied {len(to_copy)} tensors ({copied_bytes / 1e9:.2f} GB) into {shard_name}; "
        f"export has {len(got)} tensors vs base {len(base)}; "
        f"{len(dtype_widened)} tensors widened bf16->fp32 (cast at load)"
    )
    if missing or extra or shape_mismatched or dtype_mismatched:
        print(
            f"missing={missing[:10]} extra={extra[:10]} "
            f"shape_mismatched={shape_mismatched[:10]} dtype_mismatched={dtype_mismatched[:10]}"
        )
        return 1
    print("export tensor names and shapes match the base checkpoint exactly")
    return 0


def copy_towers(args, base, to_copy, shard_name, export_index, export_index_path) -> int:
    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for name in to_copy:
        by_shard.setdefault(base[name][0], []).append(name)
    for shard, names in by_shard.items():
        with safe_open(args.base_hf_dir / shard, framework="pt") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    safetensors.torch.save_file(tensors, args.export_dir / shard_name)
    copied_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    for name in to_copy:
        export_index["weight_map"][name] = shard_name
    export_index["metadata"]["total_size"] += copied_bytes
    export_index_path.write_text(json.dumps(export_index, indent=2) + "\n")
    return copied_bytes


if __name__ == "__main__":
    raise SystemExit(main())
