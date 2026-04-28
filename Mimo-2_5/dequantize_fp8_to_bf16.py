"""
MiMo-V2.5 (and other DeepSeek-V3-style FP8 block-quantized models)
streaming FP8 -> BF16 dequantizer.

Why this exists:
    transformers v5.x's `dequantize=True` path materializes every weight
    as an FP32 intermediate inside `Fp8Dequantize.convert()` and runs 4 of
    these in parallel via a ThreadPoolExecutor (GLOBAL_WORKERS=4 in
    `core_model_loading.py`). For a 310B-param model that pushes peak
    CPU RAM to ~1.3 TB during load — far more than the 768 GB needed for
    the final BF16 model.

    This script bypasses transformers entirely and dequantizes one
    safetensors shard at a time, one weight at a time, with a tight
    memory footprint (~20 GB peak per shard) and writes a vanilla BF16
    safetensors model that any standard loader can consume.

Usage:
    python dequantize_fp8_to_bf16.py <SOURCE_FP8_DIR> <DEST_BF16_DIR>

Output:
    <DEST_BF16_DIR>/
        config.json                       (quantization_config stripped)
        model-XXXXX-of-YYYYY.safetensors  (BF16 weights, no scale_inv)
        model.safetensors.index.json      (regenerated)
        tokenizer*, special_tokens_map.json, generation_config.json, ...

Notes:
    - Block size is read from config.quantization_config.weight_block_size
      (default [128, 128] if absent — matches MiMo-V2.5 / DeepSeek-V3).
    - Tensors that have NO sibling `*.weight_scale_inv` are passed through
      verbatim (their dtype stays whatever the source had — typically BF16
      for norms, FP32 for embeddings of small layers, etc.).
    - Tensors listed in `quantization_config.ignored_layers` are also
      passed through verbatim (those are kept in source dtype on disk).
"""
import argparse
import gc
import json
import os
import shutil
import sys
from collections import defaultdict
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


# Files to copy verbatim from source to destination (everything that is
# NOT a weight shard or the weight index).
_PASSTHROUGH_GLOBS = (
    "tokenizer*", "special_tokens_map.json", "generation_config.json",
    "vocab*", "merges.txt", "added_tokens.json", "chat_template*",
    "preprocessor_config.json", "processor_config.json",
    "configuration_*.py", "modeling_*.py", "image_processing_*.py",
    "video_processing_*.py", "audio_processing_*.py", "processing_*.py",
    "*.tiktoken", "README.md", "LICENSE*", "USE_POLICY*",
)


def dequantize_block_fp8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 2D FP8 block-quantized weight to `out_dtype`.

    Mirrors transformers.integrations.finegrained_fp8.Fp8Dequantize.convert()
    but writes the FP32 intermediate into a destination buffer of the
    target dtype to keep peak memory at ~3 bytes/param instead of 7.
    """
    rows, cols = weight_fp8.shape[-2:]
    block_m, block_n = block_size
    if rows % block_m != 0 or cols % block_n != 0:
        raise ValueError(
            f"weight shape {tuple(weight_fp8.shape)} not divisible by "
            f"block_size {block_size}"
        )

    # Cast FP8 -> FP32 (the intermediate the multiplication needs)
    w_fp32 = weight_fp8.to(torch.float32)
    reshaped = w_fp32.reshape(
        *weight_fp8.shape[:-2],
        rows // block_m, block_m, cols // block_n, block_n,
    )
    expanded_scales = scale_inv.reshape(
        *weight_fp8.shape[:-2],
        rows // block_m, cols // block_n,
    ).unsqueeze(-1).unsqueeze(-3)
    dequantized = (reshaped * expanded_scales).reshape(weight_fp8.shape)
    out = dequantized.to(out_dtype)

    # Drop intermediates eagerly so peak stays low.
    del w_fp32, reshaped, expanded_scales, dequantized
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="Source FP8 model directory")
    ap.add_argument("dst", help="Destination BF16 model directory")
    ap.add_argument(
        "--out-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Output dtype for dequantized weights (default: bfloat16)",
    )
    args = ap.parse_args()

    src, dst = args.src, args.dst
    out_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.out_dtype]

    if not os.path.isdir(src):
        sys.exit(f"src not found: {src}")
    os.makedirs(dst, exist_ok=True)

    # 1. Load + sanity-check config
    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {}) or {}
    block_size = tuple(qc.get("weight_block_size", [128, 128]))
    print(f"Source quant_method : {qc.get('quant_method', '<none>')}")
    print(f"Block size          : {block_size}")
    print(f"Output dtype        : {args.out_dtype}")

    # 2. Find shards
    shards = sorted(glob(os.path.join(src, "model-*.safetensors"))) \
             or sorted(glob(os.path.join(src, "*.safetensors")))
    if not shards:
        sys.exit(f"no .safetensors files found under {src}")
    print(f"Found {len(shards)} shards.")

    # 3. Stream each shard
    new_index_weight_map: dict[str, str] = {}
    total_dequant = 0
    total_passthrough = 0
    for shard_path in shards:
        shard_name = os.path.basename(shard_path)
        out_path = os.path.join(dst, shard_name)
        print(f"\n=== {shard_name} ===")

        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt") as st:
            keys = list(st.keys())
            # Figure out which keys are FP8 weights w/ scale_inv siblings
            scale_keys = {k for k in keys if k.endswith(".weight_scale_inv")}
            paired_weights = {
                k for k in keys
                if k.endswith(".weight") and (k[:-len(".weight")]
                                              + ".weight_scale_inv") in scale_keys
            }
            print(f"  tensors: {len(keys)}  "
                  f"(dequant: {len(paired_weights)}, "
                  f"passthrough: {len(keys) - len(paired_weights) - len(scale_keys)}, "
                  f"scale_inv: {len(scale_keys)})")

            for key in tqdm(keys, desc=f"  {shard_name}", unit="t"):
                if key in scale_keys:
                    # scale_inv: drop, will be rolled into the dequant
                    continue
                if key in paired_weights:
                    scale_key = key[:-len(".weight")] + ".weight_scale_inv"
                    w_fp8 = st.get_tensor(key)
                    scale = st.get_tensor(scale_key)
                    out_tensors[key] = dequantize_block_fp8(
                        w_fp8, scale, block_size, out_dtype
                    )
                    del w_fp8, scale
                    total_dequant += 1
                else:
                    out_tensors[key] = st.get_tensor(key)
                    total_passthrough += 1

        # 4. Save shard, record key -> shard mapping for the index
        save_file(out_tensors, out_path, metadata={"format": "pt"})
        for k in out_tensors:
            new_index_weight_map[k] = shard_name
        bytes_out = os.path.getsize(out_path)
        print(f"  -> wrote {bytes_out/1e9:.2f} GB to {out_path}")

        # 5. Drop everything before next shard
        del out_tensors
        gc.collect()

    # 6. Rebuild model.safetensors.index.json
    total_size = sum(
        os.path.getsize(os.path.join(dst, s)) for s in set(new_index_weight_map.values())
    )
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_index_weight_map,
    }
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)
    print(f"\nWrote model.safetensors.index.json "
          f"({len(new_index_weight_map)} weights, {total_size/1e9:.2f} GB total).")

    # 7. Strip quantization_config from config.json and write
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = args.out_dtype
    cfg["dtype"] = args.out_dtype
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("Wrote config.json (quantization_config stripped, "
          f"dtype={args.out_dtype}).")

    # 8. Copy passthrough files (tokenizer, modeling code, etc.)
    print("\nCopying passthrough files...")
    for pat in _PASSTHROUGH_GLOBS:
        for f in glob(os.path.join(src, pat)):
            if os.path.isfile(f):
                shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
                print(f"  copied {os.path.basename(f)}")

    print(f"\n=== Done ===")
    print(f"Dequantized   : {total_dequant} tensors")
    print(f"Passthrough   : {total_passthrough} tensors")
    print(f"Output        : {dst}")
    print(f"Output size   : {total_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
