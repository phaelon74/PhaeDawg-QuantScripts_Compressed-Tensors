import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier


# =========================
# Load ENV Variables
# =========================
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

SRC_DIR = require_env("SRC_DIR")
DST_DIR = require_env("DST_DIR")

# =========================
# Model (gpt-oss-20b)
# =========================
MODEL_ID = require_env("SRC_DIR")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data (Neural Magic LLM Compression Calibration)
# =========================
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# Render messages to chat-style text (batch)
# The neuralmagic dataset has "messages" field with user/assistant roles
def preprocess(batch):
    rendered = []
    for messages in batch["messages"]:
        # Apply chat template to the messages directly
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered.append(text)
    return {"text": rendered}

ds = ds.map(preprocess, batched=True, num_proc=4)

# Tokenize in batches
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

# =========================
# AWQ recipe with config_groups for gpt-oss-20b
#  - Weight-only INT8 (W8A16 **symmetric**)
#  - group_size: 128
#  - Based on model config.json quantization_config:
#    * Ignore all self-attention layers (model.layers.*.self_attn)
#    * Ignore all MLP router layers (model.layers.*.mlp.router)
#    * Ignore embedding tokens (model.embed_tokens)
#    * Ignore output head (lm_head)
#  - Model has 24 layers (num_hidden_layers: 24)
#  - MoE architecture with 32 experts per layer
# =========================

# Build ignore list for gpt-oss-20b
# According to config.json, we need to ignore:
# 1. All self-attention layers (q_proj, k_proj, v_proj, o_proj)
# 2. All MLP router layers
# 3. Embedding tokens
# 4. Language model head

gpt_oss_ignores = [
    # Embedding layer
    "model.embed_tokens",
    
    # All self-attention layers (24 layers, 0-23)
    # Each layer has: q_proj, k_proj, v_proj, o_proj
]

# Add all self-attention layers for 24 layers
for layer_idx in range(24):
    gpt_oss_ignores.extend([
        f"model.layers.{layer_idx}.self_attn.q_proj",
        f"model.layers.{layer_idx}.self_attn.k_proj",
        f"model.layers.{layer_idx}.self_attn.v_proj",
        f"model.layers.{layer_idx}.self_attn.o_proj",
        # MLP router layer
        f"model.layers.{layer_idx}.mlp.router",
    ])

# Output head
gpt_oss_ignores.append("lm_head")

recipe = [
    AWQModifier(
        ignore=gpt_oss_ignores,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,   # W8A16 (symmetric)
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": False,
                },
            },
        },
    ),
]

# =========================
# Quantize + save (writes quantization_config for vLLM)
# =========================
SAVE_DIR = require_env("DST_DIR")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#    output_dir=SAVE_DIR,
)

# Fix generation config validation issue before saving
if hasattr(model, 'generation_config') and model.generation_config is not None:
    # If temperature is set but do_sample is False, either enable do_sample or remove temperature
    if hasattr(model.generation_config, 'temperature') and model.generation_config.temperature is not None:
        if not getattr(model.generation_config, 'do_sample', False):
            # Set do_sample=True to make temperature valid, or remove temperature
            model.generation_config.do_sample = True

# (Optional redundant save)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

