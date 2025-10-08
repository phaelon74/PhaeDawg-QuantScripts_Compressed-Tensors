from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# =========================
# Load ENV Variables
# =========================
from pathlib import Path
import os
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

# =========================
# Model
# =========================
MODEL_ID = require_env("SRC_DIR")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data config
# =========================
NUM_CALIBRATION_SAMPLES = 256      # Adjust as needed
MAX_SEQUENCE_LENGTH = 512

DATASET_ID = "wikitext"
DATASET_NAME = "wikitext-2-raw-v1"  # or "wikitext-103-raw-v1"
DATASET_SPLIT = "validation"        # "train" | "validation" | "test"

# =========================
# Load + sample WikiText
# =========================
ds = load_dataset(DATASET_ID, DATASET_NAME, split=DATASET_SPLIT)

# Remove blank/whitespace-only lines (common in WikiText)
ds = ds.filter(lambda ex: ex.get("text", "").strip() != "")

# Random, reproducible subset of N *raw lines*
n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# =========================
# Preprocess (batch-aware)
# =========================
def preprocess(batch):
    texts = batch["text"]  # list[str]
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
        )
        for t in texts
    ]
    return {"text": rendered}

# Render chat template in batches
ds = ds.map(preprocess, batched=True, num_proc=4)

# =========================
# Tokenize in batches
# =========================
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,  # drop "text" column
    num_proc=4,
)

# =========================
# Quantization recipe
# =========================
weight_args = QuantizationArgs(
    num_bits=8,
    type="int",
    symmetric=True,
    strategy="group",
    group_size=128,
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=["lm_head"],
        targets=["Linear"],
        config_groups={"group_0": quant_scheme},
    ),
]

# =========================
# Run one-shot compression
# =========================
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# =========================
# Quick sanity generation
# =========================
#print("\n\n========== SAMPLE GENERATION ==============")
#dispatch_for_generation(model)
#input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
#output = model.generate(input_ids, max_new_tokens=100)
#print(tokenizer.decode(output[0]))
#print("==========================================\n\n")

# =========================
# Save compressed model
# =========================
SAVE_DIR = require_env("DST_DIR")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
