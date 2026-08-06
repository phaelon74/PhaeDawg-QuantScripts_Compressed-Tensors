"""
Qwen3-VL-30B-A3B-Instruct-Heretic W8A16 PTQ (Post-Training Quantization)

MoE VLM — catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic
(base: Qwen/Qwen3-VL-30B-A3B-Instruct)

  - QuantizationModifier with W8A16 preset (no AWQ, no GPTQ, no calibration data)
  - Qwen3VLMoeForConditionalGeneration + AutoProcessor (official MoE VL pattern)
  - MoE load helper probed across llm-compressor nightlies (load_context /
    load_quantizable_moe / replace_modules_for_calibration / plain load)
  - Vision tower left in BF16; MoE experts ARE quantized; router (mlp.gate) skipped
  - Saves processor so vision stays usable in vLLM / Transformers

  Example:
    python Qwen3-VL-30B-A3B-Heretic-W8A16_PTQ.py /path/to/model /path/to/out-W8A16
"""
import argparse
import contextlib

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

# Transformers v5 compatibility
import transformers.modeling_utils as _tmu

if not hasattr(_tmu, "TORCH_INIT_FUNCTIONS"):
    _tmu.TORCH_INIT_FUNCTIONS = {
        "uniform_": nn.init.uniform_,
        "normal_": nn.init.normal_,
        "trunc_normal_": nn.init.trunc_normal_,
        "constant_": nn.init.constant_,
        "xavier_uniform_": nn.init.xavier_uniform_,
        "xavier_normal_": nn.init.xavier_normal_,
        "kaiming_uniform_": nn.init.kaiming_uniform_,
        "kaiming_normal_": nn.init.kaiming_normal_,
        "uniform": nn.init.uniform,
        "normal": nn.init.normal,
        "xavier_uniform": nn.init.xavier_uniform,
        "xavier_normal": nn.init.xavier_normal,
        "kaiming_uniform": nn.init.kaiming_uniform,
        "kaiming_normal": nn.init.kaiming_normal,
    }

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def _resolve_moe_load_context(model_cls):
    """
    Nightly llm-compressor MoE load APIs have moved around:
      newest  -> llmcompressor.utils.load_context
      prior   -> llmcompressor.modeling.moe.linearize.load_quantizable_moe
      oldest  -> replace_modules_for_calibration (post-load, not a CM)
    Returns (context_manager_or_nullcontext, post_load_fn_or_None, label).
    """
    try:
        from llmcompressor.utils import load_context

        return load_context(model_cls), None, "load_context"
    except ImportError:
        pass

    for import_path in (
        "llmcompressor.modeling.moe.linearize",
        "llmcompressor.modeling.moe",
        "llmcompressor.modeling.linearize",
    ):
        try:
            mod = __import__(import_path, fromlist=["load_quantizable_moe"])
            fn = getattr(mod, "load_quantizable_moe", None)
            if fn is not None:
                return fn(model_cls), None, f"{import_path}.load_quantizable_moe"
        except ImportError:
            continue

    for import_path in (
        "llmcompressor.modeling",
        "llmcompressor.modeling.prepare",
    ):
        try:
            mod = __import__(import_path, fromlist=["replace_modules_for_calibration"])
            fn = getattr(mod, "replace_modules_for_calibration", None)
            if fn is not None:
                return contextlib.nullcontext(), fn, f"{import_path}.replace_modules_for_calibration"
        except ImportError:
            continue

    return (
        contextlib.nullcontext(),
        None,
        "plain from_pretrained (no MoE load helper in this llm-compressor)",
    )

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(
    description=(
        "Run W8A16 PTQ on catplusplus/Qwen3-VL-30B-A3B-Instruct-Heretic "
        "(Qwen3-VL MoE VLM). No calibration data needed. Vision preserved."
    )
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()
MODEL_ID = args.model_path

# =========================
# Model + processor
# =========================
# NOTE: Qwen3-VL-MoE needs transformers>=4.57 (or install from source).
load_cm, post_load_fn, load_label = _resolve_moe_load_context(
    Qwen3VLMoeForConditionalGeneration
)
print(f"MoE load path: {load_label}")
with load_cm:
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", trust_remote_code=True
    )
if post_load_fn is not None:
    model = post_load_fn(model)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe
# =========================
# W8A16: INT8 per-channel symmetric weights, activations untouched.
# Ignore:
#   - lm_head: output head stays BF16
#   - visual.*: vision tower / projector stay BF16 (vision preserved)
#   - mlp.gate: MoE router stays BF16 (vLLM / compressed-tensors convention)
# Experts (gate_proj / up_proj / down_proj) ARE quantized — same as other MoE scripts.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "re:.*lm_head",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# =========================
# Apply quantization (datafree — no calibration dataset)
# =========================
print("\n=== Running W8A16 PTQ (datafree, vision preserved) ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)

SAMPLE_PROMPT = "Hello my name is"
messages = [{"role": "user", "content": SAMPLE_PROMPT}]
prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=[prompt_text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model + processor (vision usable downstream)
# =========================
SAVE_DIR = args.output_path
print(f"Saving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
