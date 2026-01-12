import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.utils import dispatch_for_generation

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W4A16 AWQ quantization on Llama model."
)
parser.add_argument(
    "model_path",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "output_path",
    type=str,
    help="Path to the destination directory for saving quantized model."
)

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path

# =========================
# Model
# =========================
MODEL_ID = model_path

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data config
# =========================
NUM_CALIBRATION_SAMPLES = 512      # Adjust as needed
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

# =========================
# Load + sample neuralmagic calibration dataset
# =========================
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)

# Random, reproducible subset of N samples
n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))


# =========================
# Preprocess (batch-aware)
# =========================
def preprocess(batch):
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
# Mixed Precision Quantization Recipe
#   * quantize self_attn layers to FP8_BLOCK (attention: k, q, o, v_proj)
#   * quantize mlp layers to W4A16 with AWQ (MLP: down, gate, up_proj)
# =========================
recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      targets: ["re:.*(k|q|o|v)_proj$"]
      scheme: FP8_BLOCK
    AWQModifier:
      config_groups:
        group_0:
          targets: ["re:.*(down|gate|up)_proj$"]
          weights:
            num_bits: 4
            type: int
            symmetric: true
            group_size: 64
            strategy: group
            dynamic: false
            observer: minmax

      # Layers to exclude from quantization
      ignore:
        - "lm_head"

      # Scaling options
      duo_scaling: true

      mappings:
        - smooth_layer: re:.*post_attention_layernorm$
          balance_layers: ["re:.*gate_proj$", "re:.*up_proj$"]
        - smooth_layer: re:.*up_proj$
          balance_layers: ["re:.*down_proj$"]
"""

# =========================
# Run one-shot compression
# =========================
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
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
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)
