"""
Gemma 4 W4A16 AWQ (Activation-Aware Weight Quantization)

  - AWQModifier with config_groups for configurable group size (default 32)
  - Custom AWQ mappings for Gemma4ForConditionalGeneration (not in llm-compressor registry)
  - Calibration from a YAML recipe (Recipes/Datasets/*.yaml), same schema as Qwen AWQ scripts
  - Gemma4ForConditionalGeneration + AutoProcessor save for vLLM multimodal paths

  Gemma 4 31B: hybrid local/global attention. Attention AWQ mappings are omitted:
  q/k/v_proj are followed by q_norm/k_norm/v_norm, which erase AWQ scaling; MLP uses
  pre_feedforward_layernorm (not post_attention_layernorm) as the smooth layer.

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


def build_gemma4_awq_mappings(model) -> list:
    """
    Two AWQMappings per decoder layer (MLP only). Attention mappings are omitted:
    Gemma4 applies q_norm/k_norm/v_norm immediately after q/k/v_proj, which
    re-normalizes activations and makes input_layernorm->q/k/v (and v_proj->o_proj)
    AWQ harmful rather than helpful (see llm-compressor issues on QK-norm models).

    MLP: pre_feedforward_layernorm feeds gate_proj/up_proj (post_attention_layernorm
    sits before the residual add and does not feed the MLP).

    Module names are model.language_model.layers.* relative to the root
    Gemma4ForConditionalGeneration (same as named_modules() under that root).
    """
    layers = model.model.language_model.layers
    mappings = []
    for i in range(len(layers)):
        p = f"model.language_model.layers.{i}"
        mappings.extend(
            [
                AWQMapping(
                    smooth_layer=f"{p}.pre_feedforward_layernorm",
                    balance_layers=[
                        f"{p}.mlp.gate_proj",
                        f"{p}.mlp.up_proj",
                    ],
                ),
                AWQMapping(
                    smooth_layer=f"{p}.mlp.up_proj",
                    balance_layers=[f"{p}.mlp.down_proj"],
                ),
            ]
        )
    return mappings


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
# Some Gemma checkpoints ship without tokenizer.chat_template; processor may still have it.
if not getattr(tokenizer, "chat_template", None):
    _pt = getattr(processor, "tokenizer", None)
    _tmpl = getattr(_pt, "chat_template", None) if _pt is not None else None
    if _tmpl is not None:
        tokenizer.chat_template = _tmpl
print(f"Loaded model: {MODEL_ID}")

gemma4_awq_mappings = build_gemma4_awq_mappings(model)
_n_layers = len(model.model.language_model.layers)
print(
    f"AWQ mappings: {len(gemma4_awq_mappings)} "
    f"({_n_layers} decoder layers x 2 MLP mappings; no attention mappings — QK/V-norm)"
)

# =========================
# Dataset formatters (recipe: formatter + columns)
# =========================
def messages_to_calibration_text(tokenizer, messages) -> str:
    """Use chat template when present; otherwise plain multi-turn text (AWQ-safe)."""
    if not messages:
        return ""

    norm = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", m.get("from", "user"))
        if role in ("human", "Human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "Assistant", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        content = m.get("content", m.get("value", m.get("text", "")))
        content = str(content).strip() if content is not None else ""
        if content:
            norm.append({"role": role, "content": content})

    if not norm:
        return ""

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                norm,
                tokenize=False,
                add_generation_prompt=False,
            )
        except (ValueError, TypeError, RuntimeError):
            pass
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in norm)


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
                return {
                    "text": messages_to_calibration_text(
                        tokenizer, formatted_messages
                    )
                }
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
    return {"text": messages_to_calibration_text(tokenizer, formatted_messages)}


def format_prompt_answer(example, columns, tokenizer):
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    prompt = example.get(prompt_col, "")
    answer = example.get(answer_col, "")
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]
    return {"text": messages_to_calibration_text(tokenizer, messages)}


def format_chat_completion(example, columns, tokenizer):
    for col in columns:
        if col not in example:
            continue
        data = example[col]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return {"text": messages_to_calibration_text(tokenizer, data)}
            messages = []
            for i, item in enumerate(data):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": str(item)})
            return {"text": messages_to_calibration_text(tokenizer, messages)}
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
            "re:.*multi_modal_projector.*",
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

SAMPLE_PROMPT = "Hello my name is"
messages = [{"role": "user", "content": SAMPLE_PROMPT}]

_tok = getattr(processor, "tokenizer", processor)
has_chat_tmpl = getattr(_tok, "chat_template", None) is not None

if has_chat_tmpl:
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
else:
    prompt_text = SAMPLE_PROMPT

inputs = processor(text=[prompt_text], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
input_len = inputs["input_ids"].shape[-1]

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0][input_len:], skip_special_tokens=True))
print("==========================================\n\n")

# =========================
# Save compressed model + full processor
# =========================
SAVE_DIR = args.output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
