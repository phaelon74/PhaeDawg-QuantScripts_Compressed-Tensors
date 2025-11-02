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
# Model (GLM / GLM-MoE)
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
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 64
#  - IMPORTANT: do NOT ignore mlp.gate / gate_up_proj (merged layer)
#  - Keep router and output head unquantized
# =========================
moe_ignores = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.shared_experts.gate_proj",
    "model.layers.1.mlp.shared_experts.up_proj",
    "model.layers.1.mlp.shared_experts.down_proj",
    "model.layers.2.mlp.shared_experts.gate_proj",
    "model.layers.2.mlp.shared_experts.up_proj",
    "model.layers.2.mlp.shared_experts.down_proj",
    "model.layers.3.mlp.shared_experts.gate_proj",
    "model.layers.3.mlp.shared_experts.up_proj",
    "model.layers.3.mlp.shared_experts.down_proj",
    "model.layers.4.mlp.shared_experts.gate_proj",
    "model.layers.4.mlp.shared_experts.up_proj",
    "model.layers.4.mlp.shared_experts.down_proj",
    "model.layers.5.mlp.shared_experts.gate_proj",
    "model.layers.5.mlp.shared_experts.up_proj",
    "model.layers.5.mlp.shared_experts.down_proj",
    "model.layers.6.mlp.shared_experts.gate_proj",
    "model.layers.6.mlp.shared_experts.up_proj",
    "model.layers.6.mlp.shared_experts.down_proj",
    "model.layers.7.mlp.shared_experts.gate_proj",
    "model.layers.7.mlp.shared_experts.up_proj",
    "model.layers.7.mlp.shared_experts.down_proj",
    "model.layers.8.mlp.shared_experts.gate_proj",
    "model.layers.8.mlp.shared_experts.up_proj",
    "model.layers.8.mlp.shared_experts.down_proj",
    "model.layers.9.mlp.shared_experts.gate_proj",
    "model.layers.9.mlp.shared_experts.up_proj",
    "model.layers.9.mlp.shared_experts.down_proj",
    "model.layers.10.mlp.shared_experts.gate_proj",
    "model.layers.10.mlp.shared_experts.up_proj",
    "model.layers.10.mlp.shared_experts.down_proj",
    "model.layers.11.mlp.shared_experts.gate_proj",
    "model.layers.11.mlp.shared_experts.up_proj",
    "model.layers.11.mlp.shared_experts.down_proj",
    "model.layers.12.mlp.shared_experts.gate_proj",
    "model.layers.12.mlp.shared_experts.up_proj",
    "model.layers.12.mlp.shared_experts.down_proj",
    "model.layers.13.mlp.shared_experts.gate_proj",
    "model.layers.13.mlp.shared_experts.up_proj",
    "model.layers.13.mlp.shared_experts.down_proj",
    "model.layers.14.mlp.shared_experts.gate_proj",
    "model.layers.14.mlp.shared_experts.up_proj",
    "model.layers.14.mlp.shared_experts.down_proj",
    "model.layers.15.mlp.shared_experts.gate_proj",
    "model.layers.15.mlp.shared_experts.up_proj",
    "model.layers.15.mlp.shared_experts.down_proj",
    "model.layers.16.mlp.shared_experts.gate_proj",
    "model.layers.16.mlp.shared_experts.up_proj",
    "model.layers.16.mlp.shared_experts.down_proj",
    "model.layers.17.mlp.shared_experts.gate_proj",
    "model.layers.17.mlp.shared_experts.up_proj",
    "model.layers.17.mlp.shared_experts.down_proj",
    "model.layers.18.mlp.shared_experts.gate_proj",
    "model.layers.18.mlp.shared_experts.up_proj",
    "model.layers.18.mlp.shared_experts.down_proj",
    "model.layers.19.mlp.shared_experts.gate_proj",
    "model.layers.19.mlp.shared_experts.up_proj",
    "model.layers.19.mlp.shared_experts.down_proj",
    "model.layers.20.mlp.shared_experts.gate_proj",
    "model.layers.20.mlp.shared_experts.up_proj",
    "model.layers.20.mlp.shared_experts.down_proj",
    "model.layers.21.mlp.shared_experts.gate_proj",
    "model.layers.21.mlp.shared_experts.up_proj",
    "model.layers.21.mlp.shared_experts.down_proj",
    "model.layers.22.mlp.shared_experts.gate_proj",
    "model.layers.22.mlp.shared_experts.up_proj",
    "model.layers.22.mlp.shared_experts.down_proj",
    "model.layers.23.mlp.shared_experts.gate_proj",
    "model.layers.23.mlp.shared_experts.up_proj",
    "model.layers.23.mlp.shared_experts.down_proj",
    "model.layers.24.mlp.shared_experts.gate_proj",
    "model.layers.24.mlp.shared_experts.up_proj",
    "model.layers.24.mlp.shared_experts.down_proj",
    "model.layers.25.mlp.shared_experts.gate_proj",
    "model.layers.25.mlp.shared_experts.up_proj",
    "model.layers.25.mlp.shared_experts.down_proj",
    "model.layers.26.mlp.shared_experts.gate_proj",
    "model.layers.26.mlp.shared_experts.up_proj",
    "model.layers.26.mlp.shared_experts.down_proj",
    "model.layers.27.mlp.shared_experts.gate_proj",
    "model.layers.27.mlp.shared_experts.up_proj",
    "model.layers.27.mlp.shared_experts.down_proj",
    "model.layers.28.mlp.shared_experts.gate_proj",
    "model.layers.28.mlp.shared_experts.up_proj",
    "model.layers.28.mlp.shared_experts.down_proj",
    "model.layers.29.mlp.shared_experts.gate_proj",
    "model.layers.29.mlp.shared_experts.up_proj",
    "model.layers.29.mlp.shared_experts.down_proj",
    "model.layers.30.mlp.shared_experts.gate_proj",
    "model.layers.30.mlp.shared_experts.up_proj",
    "model.layers.30.mlp.shared_experts.down_proj",
    "model.layers.31.mlp.shared_experts.gate_proj",
    "model.layers.31.mlp.shared_experts.up_proj",
    "model.layers.31.mlp.shared_experts.down_proj",
    "model.layers.32.mlp.shared_experts.gate_proj",
    "model.layers.32.mlp.shared_experts.up_proj",
    "model.layers.32.mlp.shared_experts.down_proj",
    "model.layers.33.mlp.shared_experts.gate_proj",
    "model.layers.33.mlp.shared_experts.up_proj",
    "model.layers.33.mlp.shared_experts.down_proj",
    "model.layers.34.mlp.shared_experts.gate_proj",
    "model.layers.34.mlp.shared_experts.up_proj",
    "model.layers.34.mlp.shared_experts.down_proj",
    "model.layers.35.mlp.shared_experts.gate_proj",
    "model.layers.35.mlp.shared_experts.up_proj",
    "model.layers.35.mlp.shared_experts.down_proj",
    "model.layers.36.mlp.shared_experts.gate_proj",
    "model.layers.36.mlp.shared_experts.up_proj",
    "model.layers.36.mlp.shared_experts.down_proj",
    "model.layers.37.mlp.shared_experts.gate_proj",
    "model.layers.37.mlp.shared_experts.up_proj",
    "model.layers.37.mlp.shared_experts.down_proj",
    "model.layers.38.mlp.shared_experts.gate_proj",
    "model.layers.38.mlp.shared_experts.up_proj",
    "model.layers.38.mlp.shared_experts.down_proj",
    "model.layers.39.mlp.shared_experts.gate_proj",
    "model.layers.39.mlp.shared_experts.up_proj",
    "model.layers.39.mlp.shared_experts.down_proj",
    "model.layers.40.mlp.shared_experts.gate_proj",
    "model.layers.40.mlp.shared_experts.up_proj",
    "model.layers.40.mlp.shared_experts.down_proj",
    "model.layers.41.mlp.shared_experts.gate_proj",
    "model.layers.41.mlp.shared_experts.up_proj",
    "model.layers.41.mlp.shared_experts.down_proj",
    "model.layers.42.mlp.shared_experts.gate_proj",
    "model.layers.42.mlp.shared_experts.up_proj",
    "model.layers.42.mlp.shared_experts.down_proj",
    "model.layers.43.mlp.shared_experts.gate_proj",
    "model.layers.43.mlp.shared_experts.up_proj",
    "model.layers.43.mlp.shared_experts.down_proj",
    "model.layers.44.mlp.shared_experts.gate_proj",
    "model.layers.44.mlp.shared_experts.up_proj",
    "model.layers.44.mlp.shared_experts.down_proj",
    "model.layers.45.mlp.shared_experts.gate_proj",
    "model.layers.45.mlp.shared_experts.up_proj",
    "model.layers.45.mlp.shared_experts.down_proj",
    "model.layers.46.self_attn.q_proj",
    "model.layers.46.self_attn.k_proj",
    "model.layers.46.self_attn.v_proj",
    "model.layers.46.self_attn.o_proj",
    "model.layers.46.mlp.shared_experts.gate_proj",
    "model.layers.46.mlp.shared_experts.up_proj",
    "model.layers.46.mlp.shared_experts.down_proj",
    "lm_head",
]

recipe = [
    AWQModifier(
        ignore=moe_ignores,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,   # W4A16 (symmetric)
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
    num_workers=4,  # Enable multi-worker processing
#    output_dir=SAVE_DIR,
)

# (Optional redundant save)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

