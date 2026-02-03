from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
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
# Model (Qwen3-Next-80B-A3B-Instruct)
# =========================
MODEL_ID = require_env("SRC_DIR")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data (WikiText)
# =========================
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "wikitext"
DATASET_NAME = "wikitext-2-raw-v1"
DATASET_SPLIT = "validation"

ds = load_dataset(DATASET_ID, DATASET_NAME, split=DATASET_SPLIT)
ds = ds.filter(lambda ex: ex.get("text", "").strip() != "")

n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# Render to chat-style text (batch)
def preprocess_chat(batch):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
        )
        for t in batch["text"]
    ]
    return {"text": rendered}

ds = ds.map(preprocess_chat, batched=True, num_proc=4)

# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 32 (manual setting)
#  - IMPORTANT: skip MoE routers (mlp.gate, mlp.shared_expert_gate), keep quantizing FFN projections
#  - Keep MoE router-related linears and output head unquantized
# =========================
recipe = [
    AWQModifier(
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 32,
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
    calibrate_moe_context=True,
#    output_dir=SAVE_DIR,
)

# (Optional redundant save)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)
