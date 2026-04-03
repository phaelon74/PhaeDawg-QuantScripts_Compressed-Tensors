"""
Gemma 4 W8A16 PTQ (Post-Training Quantization)

  - QuantizationModifier with preset scheme (no AWQ, no GPTQ)
  - Gemma4ForConditionalGeneration to preserve VLM weight paths for vLLM
  - No calibration dataset needed
  - W8A16 preset: INT8 per-channel symmetric weights, FP16/BF16 activations

  Gemma 4 31B is a dense multimodal (vision-language) model with:
    - 60-layer transformer decoder (30.7B params)
    - SigLIP vision encoder (~550M params)
    - 256K token context window
    - Hybrid sliding-window / global attention

  Requires: pip install -U transformers llmcompressor compressed-tensors accelerate
  Note: Gemma 4 requires transformers >= 4.52 (or install from source).
"""
import argparse

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from transformers import Gemma4ForConditionalGeneration, AutoProcessor

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

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W8A16 PTQ on Gemma 4 31B. No calibration data needed."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save quantized model.")

args = parser.parse_args()

# =========================
# Model
# =========================
MODEL_ID = args.model_path

# Gemma4ForConditionalGeneration is the native VLM class for Gemma 4.
# It saves weights with the correct language_model.layers.* paths that
# vLLM's Gemma 4 implementation expects, preserving the vision encoder
# and multi-modal projector structure.
# (AutoModelForCausalLM would drop the vision tower weights.)
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)

# Gemma 4 uses AutoProcessor (handles both text tokenization and image
# processing) rather than a plain AutoTokenizer.
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Quantization recipe
# =========================
# W8A16 preset: INT8 per-channel symmetric weights, activations untouched.
#
# Ignore list rationale for Gemma 4 31B:
#   - lm_head: output projection, quantizing it hurts quality
#   - re:.*vision_tower.*: SigLIP vision encoder (~550M params) — keep in
#     original precision to preserve visual understanding fidelity
#   - re:.*multi_modal_projector.*: the linear projector bridging
#     vision embeddings into the language model's hidden space
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*multi_modal_projector.*",
        "re:.*embed_vision.*",
    ],
)


# =========================
# Apply quantization (no calibration data needed for W8A16 PTQ)
# =========================
print("\n=== Running W8A16 PTQ ===")
oneshot(model=model, recipe=recipe)

# =========================
# Quick sanity generation (text-only)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)

SAMPLE_PROMPT = "Hello my name is"

_tok = getattr(processor, "tokenizer", processor)
if getattr(_tok, "chat_template", None):
    messages = [{"role": "user", "content": SAMPLE_PROMPT}]
    prompt_text = _tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt_text], return_tensors="pt")
else:
    inputs = processor(text=[SAMPLE_PROMPT], return_tensors="pt")

input_ids = inputs.input_ids.to(model.device)
input_len = input_ids.shape[-1]
output = model.generate(input_ids, max_new_tokens=100)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = args.output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)