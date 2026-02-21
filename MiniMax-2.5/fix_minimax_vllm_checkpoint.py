#!/usr/bin/env python3
"""
Fix MiniMax-M2.5 W4A16 checkpoint for VLLM compatibility.

VLLM's FusedMoE expects params named w13_weight_scale / w2_weight_scale, but
compressed-tensors saves weight_scale_inv. This script renames checkpoint keys
so VLLM can load the quantized model.

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
        description="Rename MoE expert scale keys in MiniMax checkpoint for VLLM."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the quantized checkpoint (e.g. W4A16_GS32/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be renamed, do not modify files",
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

    total_renamed = 0
    found_any = False
    for st_path in st_files:
        tensors = {}
        keys_to_rename = []
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
                if ("block_sparse_moe.experts." in key or "mlp.experts." in key) and "weight_scale_inv" in key:
                    keys_to_rename.append(key)

        if keys_to_rename:
            found_any = True
            if args.dry_run:
                print(f"\n{st_path.name}: would rename {len(keys_to_rename)} keys")
                for k in keys_to_rename[:5]:
                    print(f"  {k} -> {k.replace('weight_scale_inv', 'weight_scale')}")
                if len(keys_to_rename) > 5:
                    print(f"  ... and {len(keys_to_rename) - 5} more")
            else:
                for old_key in keys_to_rename:
                    new_key = old_key.replace("weight_scale_inv", "weight_scale")
                    tensors[new_key] = tensors.pop(old_key)
                save_file(tensors, st_path)
                print(f"Renamed {len(keys_to_rename)} keys in {st_path.name}")
                total_renamed += len(keys_to_rename)

    if not found_any:
        # Diagnostic: show sample keys from first file
        print("\nNo keys matched '(block_sparse_moe|mlp).experts.' and 'weight_scale_inv'.")
        print("Sample keys from first file:")
        with safe_open(st_files[0], framework="pt", device="cpu") as f:
            keys = list(f.keys())
            moe_keys = [k for k in keys if "expert" in k.lower() or "moe" in k.lower()]
            scale_keys = [k for k in keys if "scale" in k.lower()]
            print(f"  MoE-related (first 10): {moe_keys[:10]}")
            print(f"  Scale-related (first 10): {scale_keys[:10]}")
    elif args.dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")
    elif total_renamed:
        print(f"\nDone. Renamed {total_renamed} keys total.")
    return 0


if __name__ == "__main__":
    exit(main())
