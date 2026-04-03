"""
Gemma 4 W4A16 AWQ (Activation-Aware Weight Quantization)

  - AWQModifier with config_groups for configurable group size (default 32)
  - Custom AWQ mappings for Gemma4ForConditionalGeneration (not in llm-compressor registry)
  - Text calibration via neuralmagic/calibration (LLM split) + AutoTokenizer
  - Gemma4ForConditionalGeneration + AutoProcessor save for vLLM multimodal paths

  Gemma 4 31B: hybrid local/global attention; global layers omit v_proj (shared KV).
  AWQ skips mappings when a target module is missing.

  Example:
    python Gemma4-W4A16_AWQ.py /path/to/gemma-4-31B/ /path/to/out/ \\
      --dataset neuralmagic/calibration --group-size 32

  Requires: pip install -U transformers llmcompressor compressed-tensors accelerate datasets
  Note: Gemma 4 requires transformers >= 4.52 (or install from source).
"""
import argparse

import torch.nn as nn
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, Gemma4ForConditionalGeneration

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
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

# Regex patterns use $ anchors so vision paths like *.q_proj.linear are not matched;
# ignore list also excludes vision_tower / embed_vision.
gemma4_awq_mappings = [
    AWQMapping(
        smooth_layer="re:.*input_layernorm$",
        balance_layers=[
            "re:.*self_attn\\.q_proj$",
            "re:.*self_attn\\.k_proj$",
            "re:.*self_attn\\.v_proj$",
        ],
    ),
    AWQMapping(
        smooth_layer="re:.*self_attn\\.v_proj$",
        balance_layers=["re:.*self_attn\\.o_proj$"],
    ),
    AWQMapping(
        smooth_layer="re:.*post_attention_layernorm$",
        balance_layers=[
            "re:.*mlp\\.gate_proj$",
            "re:.*mlp\\.up_proj$",
        ],
    ),
    AWQMapping(
        smooth_layer="re:.*mlp\\.up_proj$",
        balance_layers=["re:.*mlp\\.down_proj$"],
    ),
]

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(
    description="Gemma 4 W4A16 AWQ with configurable group size and text calibration."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save the quantized model.")
parser.add_argument(
    "--dataset",
    type=str,
    default="neuralmagic/calibration",
    help="HuggingFace dataset id for calibration (default: neuralmagic/calibration).",
)
parser.add_argument(
    "--group-size",
    type=int,
    default=32,
    choices=(32, 64, 128),
    help="Quantization group size (default: 32).",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=512,
    help="Number of calibration samples (default: 512).",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Max sequence length for calibration (default: 2048).",
)
args = parser.parse_args()

MODEL_ID = args.model_path

# =========================
# Model + tokenizer (text calib) + processor (save for VLM)
# =========================
model = Gemma4ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Loaded model: {MODEL_ID}")

# =========================
# Calibration dataset (text-only LLM split)
# =========================
print(
    f"Loading calibration: dataset={args.dataset!r} name=LLM "
    f"split=train[:{args.num_samples}]"
)
ds = load_dataset(
    args.dataset, name="LLM", split=f"train[:{args.num_samples}]"
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return tokenizer(
        example["text"],
        padding=False,
        max_length=args.max_seq_length,
        truncation=True,
    )


ds = ds.map(preprocess, remove_columns=ds.column_names)

# =========================
# AWQ recipe (custom group size via config_groups, not W4A16 preset)
# =========================
recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "re:.*vision_tower.*",
            "re:.*embed_vision.*",
        ],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": args.group_size,
                },
            }
        },
        mappings=gemma4_awq_mappings,
    ),
]

print(f"\n=== Running W4A16 AWQ (group_size={args.group_size}) ===")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=args.max_seq_length,
    num_calibration_samples=args.num_samples,
    tokenizer=tokenizer,
)

# =========================
# Quick sanity generation (text-only)
# =========================
print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
inputs = processor(text=["Hello my name is"], return_tensors="pt")
input_ids = inputs.input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model + full processor
# =========================
SAVE_DIR = args.output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
