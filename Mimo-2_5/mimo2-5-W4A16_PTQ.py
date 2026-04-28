"""
MiMo-V2.5 (XiaomiMiMo/MiMo-V2.5) W4A16 PTQ — Post-Training Quantization

Source model     : MiMoV2ForCausalLM, model_type="mimo_v2"
                   310B total / 15B activated (sparse MoE, 256 routed experts × 8/tok)
                   Native omnimodal: text + vision (MiMoVisionTransformer)
                                              + audio (MiMoAudioEncoder)
                   Hybrid SWA + GA attention, dual RoPE thetas, attention sinks,
                   3-layer MTP, and a dense layer 0 (intermediate=16384).
Source format    : BF16 (already dequantized from Xiaomi's FP8 release via
                   the companion `dequantize_fp8_to_bf16.py` streaming script).
                   Reason: transformers v5.x's in-process `dequantize=True`
                   path keeps both the FP8 source and FP32 dequant
                   intermediates resident across 4 worker threads, pushing
                   peak CPU RAM past 1.3 TB on a 310B model. Pre-dequantizing
                   to disk caps load-time peak at ~700 GB.

Why this recipe :
  * `QuantizationModifier` (RTN), no AWQ / GPTQ / SmoothQuant.
    -> No calibration dataset needed.
    -> No `CalibrationMimoV2MoE` definition exists in
       llm-compressor.modeling yet (only glm4_moe / qwen3_moe / qwen3_5_moe
       / qwen3_vl_moe), so any data-driven algorithm would silently miscalibrate
       the unactivated experts. RTN sidesteps this entirely.
  * W4A16 preset: 4-bit group-128 weights, BF16 activations.
  * `trust_remote_code=True` is mandatory: `mimo_v2` is not yet in
    transformers' CONFIG_MAPPING_NAMES (HF PR #45144 only adds MiMo-V2-Flash,
    not the V2.5 omnimodal variant).
  * `AutoModelForCausalLM` (NOT ImageTextToText): the registered architecture
    is `MiMoV2ForCausalLM`, with `visual` / `audio_encoder` / `speech_embeddings`
    as flat top-level siblings of `model` (verified via probe_mimo_v2_5.py).
    Therefore NO post-save key remapping is needed (unlike Qwen3.6 VLM).

Disk space note  : Saved W4A16 output is ~155 GB (310B params * 0.5 byte/param
                   + scales/zeros). Source FP8 on disk is ~315 GB. No BF16
                   intermediate is materialized to disk.
"""
import argparse

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W4A16 RTN PTQ on XiaomiMiMo/MiMo-V2.5. No calibration data needed. "
        "Source FP8 (E4M3, block 128x128) -> output W4A16 (group-128) "
        "compressed-tensors checkpoint, loadable in vLLM nightly as "
        "MiMoV2ForCausalLM."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")
args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

# MIMO-V2.5 LOAD PLAN
# ===================
# Input is the BF16 model produced by `dequantize_fp8_to_bf16.py`. Its
# config.json has had `quantization_config` stripped, so transformers loads
# straight into vanilla nn.Linear modules with BF16 weights — no FP8
# quantizer involvement, no FP32 intermediates, no FP8Experts MoE bug.
#
# Memory budget (for the 310B MiMo-V2.5):
#   ~620 GB BF16 resident + ~30 GB Python/torch overhead + ~20 GB working
#   buffers during oneshot ≈ 670 GB peak CPU RAM. Fits in 768 GB with
#   ~100 GB headroom.
#
# Device map is CPU. The 4 GPUs are not used during load or during the
# RTN sweep (all CPU). `dispatch_model(model)` after oneshot redistributes
# the much-smaller W4A16 weights (~155 GB, ~40 GB/card) for sample-gen.
#
# `trust_remote_code=True` remains REQUIRED (mimo_v2 not in
# CONFIG_MAPPING_NAMES even on transformers 5.6.2 — HF PR #45144).
print(f"Loading config from: {MODEL_ID}")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if getattr(config, "quantization_config", None) is not None:
    print("  WARNING: source still has a quantization_config. Did you point")
    print("  this script at the FP8 source instead of the BF16 dequantized")
    print("  output? Run dequantize_fp8_to_bf16.py first.")

print(f"Loading model: {MODEL_ID}  (device_map=cpu)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Model class : {type(model).__name__}")
print(f"# params    : {sum(p.numel() for p in model.parameters()):,}")

# =========================
# Quantization recipe — MiMo-V2.5 W4A16 RTN
# =========================
# Verified against probe_mimo_v2_5.py output (23 unique nn.Linear templates,
# 36,382 total Linear modules) and Xiaomi's own FP8 release ignore list.
#
# Quantization candidates (kept):
#   model.layers.{N}.self_attn.qkv_proj           48x   in=4096  out=13568  (fused QKV)
#   model.layers.{N}.mlp.{gate,up,down}_proj      ~2x   layer 1..47 dense MLP-like (none — see note)
#   model.layers.{N}.mlp.experts.{N}.{g,u,d}_proj 12032x each   (47 layers x 256 experts, the bulk)
#
# Skipped (see ignore list below):
#   lm_head, all self_attn.o_proj, layer 0 (dense, 16384 intermediate),
#   MoE router gate (custom Parameter container, not nn.Linear, defensive),
#   visual / audio_encoder / speech_embeddings (multimodal towers),
#   model.mtp.* (multi-token prediction layers).
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        # 1. Output head (always skip; tied/untied — MiMo has tie_word_embeddings=False)
        "lm_head",

        # 2. Mirror Xiaomi's FP8 release: ALL attention output projections.
        #    Their ignored_layers excludes every model.layers.{0..47}.self_attn.o_proj
        #    plus model.decoder.self_attn.o_proj (audio decoder o_proj — not present
        #    in current MiMo-V2.5 module tree but covered by audio_encoder ignore).
        "re:.*self_attn\\.o_proj$",

        # 3. Dense layer 0 (the only non-MoE FFN, intermediate_size=16384).
        #    Per the GLM-4.7 W4A16 example pattern; this layer is the
        #    perception->reasoning bottleneck and disproportionately sensitive
        #    at 4-bit. Comment this line out if you want maximum compression.
        "re:^model\\.layers\\.0\\..*",

        # 4. MoE router gate: MiMoV2MoEGate is a custom Parameter container,
        #    not nn.Linear (verified via probe). This regex is defensive and
        #    won't match anything in practice; harmless to keep.
        "re:.*mlp\\.gate$",

        # 5. Vision tower — MiMoVisionTransformer (~728M params, 28 blocks).
        #    Includes visual.blocks.{N}.{attn.qkv, attn.proj, mlp.{g,u,d}_proj}
        #    and visual.merger.mlp.{0,1}.
        "re:^visual\\..*",

        # 6. Audio encoder — MiMoAudioEncoder (~390M params).
        #    Includes audio_encoder.input_local_transformer.layers.{0..5}.*
        #    (Qwen2DecoderLayer, q/k/v/o_proj + mlp.{g,u,d}_proj) and
        #    audio_encoder.projection.mlp.{0,1}.
        "re:^audio_encoder\\..*",

        # 7. Multi-Token Prediction layers (3 layers, ~329M params).
        #    Their internal Linear paths collapse onto the same templates as
        #    model.layers.{N}.* under regex templating, so we explicitly
        #    exclude the mtp branch.
        "re:.*mtp.*",

        # 8. Speech embeddings (20x nn.Embedding) — not Linear, defensive.
        "re:^speech_embeddings.*",
    ],
)

# =========================
# Apply quantization (no calibration data needed — RTN)
# =========================
print("\n=== Running W4A16 RTN PTQ on MiMo-V2.5 ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only, vision/audio paths untouched)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
prompt = "Hello, I am MiMo. Please introduce yourself in one sentence."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
# No KEY_REMAPS / fix_saved_weight_keys is required for MiMo-V2.5: its
# saved weight keys (model.layers.*, lm_head.weight, visual.*, audio_encoder.*,
# speech_embeddings.{N}.*, model.mtp.*) are already in the canonical layout
# that vLLM's MiMoV2ForCausalLM loader expects.
SAVE_DIR = args.output_path
print(f"Saving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
print("Load in vLLM nightly:")
print("  from vllm import LLM")
print(f"  llm = LLM(model='{SAVE_DIR}', trust_remote_code=True)")
