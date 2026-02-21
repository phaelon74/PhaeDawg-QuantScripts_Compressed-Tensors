#!/usr/bin/env python3
"""
Fix MiniMax-M2.5 W4A16 checkpoint for VLLM compatibility.

VLLM's FusedMoE loader looks up params by name; when it sees weight_scale_inv
it fails (KeyError). The checkpoint has both weight_scale (correct shape) and
weight_scale_inv. We REMOVE weight_scale_inv keys so the loader uses weight_scale.
Do NOT overwrite weight_scale with weight_scale_inv - that causes shape mismatch.

Usage:
    python fix_minimax_vllm_checkpoint.py /path/to/W4A16_GS32/
    python fix_minimax_vllm_checkpoint.py /path/to/W4A16_GS32/ --dry-run  # diagnose only
"""

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(
        description="Remove MoE expert weight_scale_inv keys for VLLM (use weight_scale)."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the quantized checkpoint (e.g. W4A16_GS32/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be removed, do not modify files",
    )
    args = parser.parse_args()

    save_dir = Path(args.checkpoint_dir)
    if not save_dir.is_dir():
        print(f"Error: {save_dir} is not a directory")
        return 1

    st_files = list(save_dir.glob("*.safetensors"))
    if not st_files:
        print(f"No .safetensors files found in {save_dir}")
        return 1

    print(f"Found {len(st_files)} safetensors files in {save_dir}")

    total_removed = 0
    found_any = False
    for st_path in st_files:
        tensors = {}
        keys_to_remove = []
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
                if ("block_sparse_moe.experts." in key or "mlp.experts." in key) and "weight_scale_inv" in key:
                    keys_to_remove.append(key)

        if keys_to_remove:
            found_any = True
            if args.dry_run:
                print(f"\n{st_path.name}: would remove {len(keys_to_remove)} weight_scale_inv keys")
                for k in keys_to_remove[:5]:
                    print(f"  (remove) {k}")
                if len(keys_to_remove) > 5:
                    print(f"  ... and {len(keys_to_remove) - 5} more")
            else:
                for k in keys_to_remove:
                    del tensors[k]
                save_file(tensors, st_path)
                print(f"Removed {len(keys_to_remove)} weight_scale_inv keys from {st_path.name}")
                total_removed += len(keys_to_remove)

    if not found_any:
        # Diagnostic: show sample keys from first file
        print("\nNo weight_scale_inv keys matched.")
        print("Sample keys from first file:")
        with safe_open(st_files[0], framework="pt", device="cpu") as f:
            keys = list(f.keys())
            moe_keys = [k for k in keys if "expert" in k.lower() or "moe" in k.lower()]
            scale_keys = [k for k in keys if "scale" in k.lower()]
            print(f"  MoE-related (first 10): {moe_keys[:10]}")
            print(f"  Scale-related (first 10): {scale_keys[:10]}")
    elif args.dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")
    elif total_removed:
        print(f"\nDone. Removed {total_removed} weight_scale_inv keys total.")
    return 0


if __name__ == "__main__":
    exit(main())
