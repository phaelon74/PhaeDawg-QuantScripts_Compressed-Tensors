from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

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
NUM_CALIBRATION_SAMPLES = 512      # Adjust as needed
MAX_SEQUENCE_LENGTH = 2048

# Dataset configuration with 60/40 split
NEURALMAGIC_DATASET_ID = "neuralmagic/LLM_compression_calibration"
ROMBO_DATASET_ID = "Rombo-Org/Optimized_Reasoning"
DATASET_SPLIT = "train"

# Calculate samples for each dataset (60% neuralmagic, 40% rombo)
NUM_NEURALMAGIC_SAMPLES = int(NUM_CALIBRATION_SAMPLES * 0.6)  # 307 samples
NUM_ROMBO_SAMPLES = NUM_CALIBRATION_SAMPLES - NUM_NEURALMAGIC_SAMPLES  # 205 samples

# =========================
# Load + sample neuralmagic calibration dataset
# =========================
ds_neuralmagic = load_dataset(NEURALMAGIC_DATASET_ID, split=DATASET_SPLIT)

# Random, reproducible subset of N samples
n_nm = min(NUM_NEURALMAGIC_SAMPLES, len(ds_neuralmagic))
ds_neuralmagic = ds_neuralmagic.shuffle(seed=42).select(range(n_nm))

# =========================
# Load + sample Rombo calibration dataset
# =========================
ds_rombo = load_dataset(ROMBO_DATASET_ID, split=DATASET_SPLIT)

# Random, reproducible subset of N samples
n_rombo = min(NUM_ROMBO_SAMPLES, len(ds_rombo))
ds_rombo = ds_rombo.shuffle(seed=43).select(range(n_rombo))


# =========================
# Preprocess (batch-aware)
# =========================
def preprocess_neuralmagic(batch):
    # The neuralmagic dataset has a 'messages' field with pre-formatted conversations
    messages_list = batch["messages"]  # list[list[dict]]
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        for messages in messages_list
    ]
    return {"text": rendered}

def preprocess_rombo(batch):
    # The Rombo dataset has 'instruction', 'input', and 'output' fields
    # Convert to messages format for chat template
    rendered = []
    for i in range(len(batch["instruction"])):
        instruction = batch["instruction"][i]
        inputs = batch["input"][i] if isinstance(batch["input"][i], list) else [batch["input"][i]]
        outputs = batch["output"][i] if isinstance(batch["output"][i], list) else [batch["output"][i]]
        
        # Create messages format: system + user + assistant pairs
        messages = [{"role": "system", "content": instruction}]
        
        # Pair up inputs and outputs (handle cases where counts might differ)
        max_pairs = max(len(inputs), len(outputs))
        for j in range(max_pairs):
            if j < len(inputs):
                messages.append({"role": "user", "content": inputs[j]})
            if j < len(outputs):
                messages.append({"role": "assistant", "content": outputs[j]})
        
        # Apply chat template
        rendered_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        rendered.append(rendered_text)
    
    return {"text": rendered}

# Render chat template in batches for both datasets
ds_neuralmagic = ds_neuralmagic.map(preprocess_neuralmagic, batched=True, num_proc=4)
ds_rombo = ds_rombo.map(preprocess_rombo, batched=True, num_proc=4)

# =========================
# Combine datasets
# =========================
from datasets import concatenate_datasets
ds = concatenate_datasets([ds_neuralmagic, ds_rombo])

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
# Quantization recipe  (W4A16-SYM, Marlin-friendly)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=4,          # 4-bit weights
    type="int",
    symmetric=True,      # SYMMETRIC (Marlin requirement)
    strategy="group",    # group-wise quantization
    group_size=32,      # 32 groupsize for best accuracy, for W4 models (Marlin standard)
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,   # A16 (leave activations in FP16/BF16)
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=["lm_head"],
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
