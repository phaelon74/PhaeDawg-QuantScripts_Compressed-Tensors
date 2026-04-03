"""
Gemma 4 W4A16 AWQ (Activation-Aware Weight Quantization)

  - AWQModifier with config_groups for configurable group size (default 32)
  - Custom AWQ mappings for Gemma4ForConditionalGeneration (not in llm-compressor registry)
  - Calibration from a YAML recipe (Recipes/Datasets/*.yaml), same schema as Qwen AWQ scripts
  - Gemma4ForConditionalGeneration + AutoProcessor save for vLLM multimodal paths

  Gemma 4 31B: hybrid local/global attention; global layers omit v_proj (shared KV).
  AWQ skips mappings when a target module is missing.

  Example:
    python Gemma4-W4A16_AWQ.py /path/to/gemma-4-31B/ /path/to/out/ \\
      ../Recipes/Datasets/General_reasoning.yaml --group-size 32

  Requires: pip install -U transformers llmcompressor compressed-tensors accelerate datasets pyyaml
  Note: Gemma 4 requires transformers >= 4.52 (or install from source).
"""
import argparse
import json

import torch.nn as nn
import yaml
from compressed_tensors.offload import dispatch_model
from datasets import Dataset, concatenate_datasets, load_dataset
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
    description="Gemma 4 W4A16 AWQ: recipe YAML calibration + configurable group size."
)
parser.add_argument("model_path", type=str, help="Path to the source model directory.")
parser.add_argument("output_path", type=str, help="Path to save the quantized model.")
parser.add_argument(
    "recipe_yaml",
    type=str,
    help="Path to calibration recipe YAML (calibration_set.datasets, max_seq_length, etc.).",
)
parser.add_argument(
    "--group-size",
    type=int,
    default=32,
    choices=(32, 64, 128),
    help="Quantization group size (default: 32).",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=None,
    help="Override calibration_set.max_seq_length from the recipe when set.",
)
args = parser.parse_args()

MODEL_ID = args.model_path

# =========================
# Recipe YAML (calibration_set)
# =========================
with open(args.recipe_yaml, "r", encoding="utf-8") as f:
    recipe_file = yaml.safe_load(f)

calibration_config = recipe_file.get("calibration_set", {})
MAX_SEQUENCE_LENGTH = calibration_config["max_seq_length"]
if args.max_seq_length is not None:
    MAX_SEQUENCE_LENGTH = args.max_seq_length
SHUFFLE = calibration_config.get("shuffle", True)
SEED = calibration_config.get("seed", 42)
datasets_config = calibration_config.get("datasets", [])

print(f"Loaded recipe: {args.recipe_yaml}")
print(f"  max_seq_length: {MAX_SEQUENCE_LENGTH}  shuffle: {SHUFFLE}  seed: {SEED}")
print(f"  dataset entries in recipe: {len(datasets_config)}")

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
# Dataset formatters (recipe: formatter + columns)
# =========================
def format_sharegpt(example, columns, tokenizer):
    formatted_messages = []
    if len(columns) >= 2 and "system" in columns[0].lower():
        system_prompt = example.get(columns[0], "")
        if system_prompt:
            formatted_messages.append(
                {"role": "system", "content": str(system_prompt)}
            )
        conv_column = columns[1]
    else:
        conv_column = columns[0]

    messages = example.get(conv_column, [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            formatted_messages.append({"role": "user", "content": messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(
                    formatted_messages, tokenize=False
                )
                return {"text": text}
            return {"text": ""}

    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                elif role == "system":
                    role = "system"
                if content:
                    formatted_messages.append(
                        {"role": role, "content": str(content)}
                    )
            elif isinstance(msg, str):
                idx = len([m for m in formatted_messages if m["role"] != "system"])
                role = "user" if idx % 2 == 0 else "assistant"
                formatted_messages.append({"role": role, "content": str(msg)})

    if not formatted_messages:
        return {"text": ""}
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {"text": text}
    except Exception:
        return {"text": ""}


def format_prompt_answer(example, columns, tokenizer):
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    prompt = example.get(prompt_col, "")
    answer = example.get(answer_col, "")
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def format_chat_completion(example, columns, tokenizer):
    for col in columns:
        if col not in example:
            continue
        data = example[col]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                text = tokenizer.apply_chat_template(data, tokenize=False)
                return {"text": text}
            messages = []
            for i, item in enumerate(data):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": str(item)})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": text}
        if isinstance(data, str):
            return {"text": str(data)}
    text = " ".join(str(example.get(col, "")) for col in columns)
    return {"text": text}


def format_raw_text(example, columns, _tokenizer):
    texts = []
    for col in columns:
        if col in example and example[col]:
            texts.append(str(example[col]))
    return {"text": " ".join(texts)}


FORMATTERS = {
    "sharegpt": format_sharegpt,
    "prompt_answer": format_prompt_answer,
    "chat_completion": format_chat_completion,
    "raw_text": format_raw_text,
}


# =========================
# Load HuggingFace datasets listed in the recipe
# =========================
print("\n=== Loading calibration from recipe ===")
all_parts = []
for ds_config in datasets_config:
    dataset_name = ds_config["dataset"]
    split = ds_config.get("split", "train")
    subset = ds_config.get("subset")
    columns = ds_config.get("columns", [])
    formatter_name = ds_config.get("formatter", "raw_text")
    num_samples = ds_config.get("num_samples", 10)
    streaming = ds_config.get("streaming", False)

    load_kw = {}
    if subset:
        load_kw["name"] = subset

    print(
        f"  {dataset_name}  split={split}  subset={subset!r}  "
        f"samples={num_samples}  formatter={formatter_name}"
    )
    try:
        if streaming:
            stream = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                **load_kw,
            )
            rows = list(stream.take(num_samples))
            part = Dataset.from_list(rows)
        else:
            part = load_dataset(dataset_name, split=split, **load_kw)
            n = min(num_samples, len(part))
            part = part.shuffle(seed=SEED).select(range(n))

        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        part = part.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=part.column_names,
            num_proc=1,
        )
        part = part.filter(lambda x: len(x.get("text", "")) > 0)
        all_parts.append(part)
        print(f"    -> {len(part)} rows")
    except Exception as e:
        print(f"    -> WARNING: skipped ({e})")
        continue

if not all_parts:
    raise ValueError("No datasets loaded from recipe; check YAML and Hub access.")

ds = concatenate_datasets(all_parts)
if SHUFFLE:
    ds = ds.shuffle(seed=SEED)

print(f"\n=== Tokenizing {len(ds)} samples (max_length={MAX_SEQUENCE_LENGTH}) ===")


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(
    tokenize_batch,
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

NUM_CALIBRATION_SAMPLES = len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples for oneshot")

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
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
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
