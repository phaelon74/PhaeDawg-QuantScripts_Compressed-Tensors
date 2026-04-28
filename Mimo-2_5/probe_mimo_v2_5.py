"""
MiMo-V2.5 model structure probe.

Loads the model on the *meta* device (no real weights, no GPU memory)
and dumps everything we need to write a correct quantization recipe:

  1. Top-level submodules and their classes
  2. All unique module class names that look like MoE / attention /
     vision / audio / MTP / projector / sink components
  3. All unique nn.Linear paths, collapsed to templates (e.g.
     model.layers.{N}.self_attn.q_proj) — these are the actual
     quantization candidates
  4. The MoE block class name (so we can check llm-compressor's registry)
  5. A safetensors-side cross-check of top-level weight prefixes

Usage:
    python probe_mimo_v2_5.py /path/to/MiMo-V2.5 > probe.txt
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", help="Local path to MiMo-V2.5 directory")
    ap.add_argument(
        "--no-meta",
        action="store_true",
        help="If you already have the model loaded, skip meta load and run "
             "the inspect_loaded_model() snippet manually in your REPL.",
    )
    args = ap.parse_args()

    model_path = args.model_path
    print(f"MODEL_PATH: {model_path}")
    print(f"exists: {os.path.isdir(model_path)}")
    print()

    # -------------------------------------------------------------------
    # 1. Config sanity
    # -------------------------------------------------------------------
    print("=" * 70)
    print("CONFIG (raw)")
    print("=" * 70)
    cfg_path = os.path.join(model_path, "config.json")
    cfg = json.load(open(cfg_path))
    for k in [
        "architectures", "model_type", "num_hidden_layers", "hidden_size",
        "intermediate_size", "moe_intermediate_size", "n_routed_experts",
        "n_shared_experts", "num_experts_per_tok", "moe_layer_freq",
        "hybrid_layer_pattern", "tie_word_embeddings", "vocab_size",
    ]:
        v = cfg.get(k, "<missing>")
        if isinstance(v, list) and len(v) > 12:
            v = f"{v[:6]} ... {v[-3:]}  (len={len(v)})"
        print(f"  {k:24s}: {v}")
    print(f"  has vision_config       : {'vision_config' in cfg}")
    print(f"  has audio_config        : {'audio_config' in cfg}")
    print(f"  pre-existing quant_cfg  : {bool(cfg.get('quantization_config'))}")
    if cfg.get("quantization_config"):
        qc = cfg["quantization_config"]
        print(f"    quant_method          : {qc.get('quant_method')}")
        print(f"    activation_scheme     : {qc.get('activation_scheme')}")
        print(f"    ignored_layers count  : {len(qc.get('ignored_layers', []))}")
        print(f"    sample ignored        : {qc.get('ignored_layers', [])[:3]}")
    print(f"  auto_map                : {cfg.get('auto_map')}")
    print()

    # -------------------------------------------------------------------
    # 2. Safetensors weight-key cross-check (no model load)
    # -------------------------------------------------------------------
    print("=" * 70)
    print("SAFETENSORS WEIGHT-KEY PREFIXES (from model.safetensors.index.json)")
    print("=" * 70)
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        keys = list(idx["weight_map"].keys())
        print(f"  total tensors: {len(keys)}")
        # collapse layer indices for templating
        tpl_count = defaultdict(int)
        for k in keys:
            t = re.sub(r"\.\d+\.", ".{N}.", k)
            t = re.sub(r"\.\d+$", ".{N}", t)
            tpl_count[t] += 1
        print(f"  unique templated keys: {len(tpl_count)}")
        print()
        print("  -- Top-level prefixes (segment 0..1) --")
        top = sorted({".".join(k.split(".")[:2]) for k in keys})
        for p in top:
            print(f"    {p}")
        print()
        print("  -- Sample templates per top-prefix --")
        by_top = defaultdict(list)
        for t in tpl_count:
            by_top[t.split(".")[0]].append(t)
        for top, items in sorted(by_top.items()):
            print(f"\n  [{top}] ({len(items)} unique templates)")
            for it in sorted(items)[:25]:
                print(f"    {it}")
            if len(items) > 25:
                print(f"    ... +{len(items) - 25} more")
    else:
        print("  (no .index.json — check single-file safetensors)")
    print()

    # -------------------------------------------------------------------
    # 3. Meta-device model load (no weights, no GPU)
    # -------------------------------------------------------------------
    if args.no_meta:
        print("=" * 70)
        print("Skipped meta-device load (--no-meta).")
        print("Run inspect_loaded_model(model) on your already-loaded model.")
        print("=" * 70)
        return

    print("=" * 70)
    print("META-DEVICE MODEL LOAD (no weights allocated)")
    print("=" * 70)
    try:
        import torch  # noqa: F401
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"config class: {type(config).__name__}")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )
        print(f"model class : {type(model).__name__}")
        print(f"# params (logical): {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"META LOAD FAILED: {type(e).__name__}: {e}")
        print()
        print("Fallback: in your existing Python session where the model is")
        print("already loaded, paste this:")
        print()
        print("    from probe_mimo_v2_5 import inspect_loaded_model")
        print("    inspect_loaded_model(model)")
        return

    inspect_loaded_model(model)


def inspect_loaded_model(model):
    """Dump structure of an already-instantiated MiMoV2 model."""
    import torch.nn as nn

    print()
    print("=" * 70)
    print("TOP-LEVEL CHILDREN")
    print("=" * 70)
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        print(f"  {name:30s} {type(child).__name__:40s}  params={n_params:,}")
    print()

    print("=" * 70)
    print("SECOND-LEVEL CHILDREN  (model.<x>.<y>)")
    print("=" * 70)
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth == 1:
            n_params = sum(p.numel() for p in mod.parameters())
            print(f"  {name:40s} {type(mod).__name__:40s}  params={n_params:,}")
    print()

    print("=" * 70)
    print("CLASSES OF INTEREST  (MoE / Expert / Attention / Vision / Audio / MTP)")
    print("=" * 70)
    keywords = [
        "moe", "expert", "router", "gate", "attn", "attention", "sink",
        "mtp", "visual", "vision", "audio", "decoder", "encoder",
        "projector", "merger", "patch",
    ]
    classes = defaultdict(list)
    for name, mod in model.named_modules():
        cn = type(mod).__name__
        ln = cn.lower()
        if any(k in ln for k in keywords):
            classes[cn].append(name)
    for cn in sorted(classes):
        names = classes[cn]
        print(f"\n  CLASS: {cn}   (count: {len(names)})")
        for n in names[:3]:
            print(f"    {n}")
        if len(names) > 3:
            print(f"    ... +{len(names) - 3} more")
    print()

    print("=" * 70)
    print("UNIQUE nn.Linear PATHS  (templated, → quantization candidates)")
    print("=" * 70)
    tpl_to_paths = defaultdict(list)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            t = re.sub(r"\.\d+\.", ".{N}.", name)
            t = re.sub(r"\.\d+$", ".{N}", t)
            tpl_to_paths[t].append(name)
    print(f"  {len(tpl_to_paths)} unique Linear templates "
          f"({sum(len(v) for v in tpl_to_paths.values())} total Linear modules)")
    print()
    for t in sorted(tpl_to_paths):
        cnt = len(tpl_to_paths[t])
        ex = tpl_to_paths[t][0]
        in_f = getattr(model.get_submodule(ex), "in_features", "?")
        out_f = getattr(model.get_submodule(ex), "out_features", "?")
        print(f"  [{cnt:5d}x] in={in_f:>6} out={out_f:>6}  {t}")
    print()

    print("=" * 70)
    print("HEURISTIC IGNORE-LIST SUGGESTION  (cross-check vs llm-compressor)")
    print("=" * 70)
    suggested = set()
    for t in tpl_to_paths:
        tl = t.lower()
        if "lm_head" in tl:
            suggested.add("lm_head")
        if "visual" in tl or "vision" in tl:
            suggested.add("re:.*visual.*")
        if "audio" in tl or "decoder" in tl:
            suggested.add("re:.*audio.*")
        if "mtp" in tl:
            suggested.add("re:.*mtp.*")
        if t.endswith(".gate") or t.endswith(".gate.{N}"):
            suggested.add("re:.*mlp\\.gate$")
        if "shared_expert_gate" in tl:
            suggested.add("re:.*shared_expert_gate$")
    for p in sorted(suggested):
        print(f"  {p!r},")
    print()
    print("DONE — paste the entire stdout back to the assistant.")


if __name__ == "__main__":
    sys.exit(main())
