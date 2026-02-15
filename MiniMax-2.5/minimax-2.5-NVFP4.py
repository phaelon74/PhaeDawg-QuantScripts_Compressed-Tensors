"""
MiniMax-M2.5 NVFP4 Quantization Script

Model: https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- 229B params MoE (256 experts, 8 activated per token)
- 62 layers, hidden_size=3072, intermediate_size=1536
- Uses block_sparse_moe with w1, w2, w3 (SwiGLU-style experts)

CRITICAL: All experts are activated during calibration via CalibrationMiniMaxM2SparseMoeBlock.
Only MoE expert layers (w1, w2, w3) are quantized; attention, gate, norms, lm_head are ignored.

NVFP4: FP4 weights + FP4 activations, per-group-16, optimized for Blackwell GPUs.
"""

import argparse
import yaml
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from minimax_m2_moe_calibration import CalibrationMiniMaxM2SparseMoeBlock  # noqa: F401

print("Imported CalibrationMiniMaxM2SparseMoeBlock - all experts activated during calibration")

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run NVFP4 quantization on MiniMax-M2.5 MoE model."
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
parser.add_argument(
    "recipe_yaml",
    type=str,
    help="Path to the dataset recipe YAML file (contains max_seq_length and dataset config)."
)

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path
recipe_yaml_path = args.recipe_yaml

# =========================
# Load Recipe YAML and extract config
# =========================
with open(recipe_yaml_path, 'r') as f:
    recipe_config = yaml.safe_load(f)

# Extract config from calibration_set section
calibration_config = recipe_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']  # Required - fail if missing
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
num_calibration_samples = calibration_config.get('num_samples', None)  # Optional cap
datasets_config = calibration_config.get('datasets', [])

print(f"Loaded recipe from: {recipe_yaml_path}")
print(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  - shuffle: {SHUFFLE}")
print(f"  - seed: {SEED}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# MoE Calibration: Replace MiniMaxM2SparseMoeBlock with calibration version
# =========================
# CRITICAL: Ensures EVERY expert receives calibration data for EVERY sample.
# The original only runs router-selected experts; our calibration module runs ALL tokens
# through ALL experts so NVFP4 can collect proper activation statistics.
_original_moe_modules = {}


def replace_moe_modules_for_calibration(model):
    """Replace all MiniMaxM2SparseMoeBlock with CalibrationMiniMaxM2SparseMoeBlock."""
    global _original_moe_modules
    _original_moe_modules.clear()
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "MiniMaxM2SparseMoeBlock":
            parts = name.split(".")
            attr_name = parts[-1]
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            _original_moe_modules[name] = module
            calibration_module = CalibrationMiniMaxM2SparseMoeBlock(
                original=module,
                config=model.config,
                calibrate_all_experts=True,
            )
            setattr(parent, attr_name, calibration_module)
            replaced_count += 1
    print(f"Replaced {replaced_count} MoE modules with calibration versions (all experts activated)")
    return model


def get_moe_expert_info(model):
    """Extract MoE config for MiniMax-M2.5."""
    config = model.config
    return {
        "num_experts": getattr(config, "num_local_experts", 256),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", 8),
        "num_layers": getattr(config, "num_hidden_layers", 62),
    }


def print_moe_calibration_guidance(model, num_samples, max_seq_len):
    """Print MoE expert coverage analysis."""
    try:
        moe_info = get_moe_expert_info(model)
        print("\n" + "=" * 70)
        print("MoE Expert Coverage Analysis for MiniMax-M2.5")
        print("=" * 70)
        print(f"  Total experts:           {moe_info['num_experts']}")
        print(f"  Experts per token:       {moe_info['num_experts_per_tok']}")
        print(f"  Number of layers:        {moe_info['num_layers']}")
        print("-" * 70)
        print(f"  Calibration setup: {num_samples} samples, max_seq={max_seq_len}")
        estimated_tokens = num_samples * max_seq_len // 2
        expert_activations = estimated_tokens * moe_info["num_experts_per_tok"]
        avg_per_expert = expert_activations / moe_info["num_experts"]
        print(f"  With ALL-experts calibration: every expert receives ~{avg_per_expert:.0f} forward passes")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Note: Could not analyze MoE config: {e}")


def detect_sequential_targets(model):
    """Detect decoder layer type for MiniMax-M2.5 (229B params - memory efficiency)."""
    candidates = ["MiniMaxM2DecoderLayer", "MiniMaxM2Block", "LlamaDecoderLayer"]
    for name, module in model.named_modules():
        t = type(module).__name__
        if t in candidates:
            return [t]
    return None


# =========================
# Model
# =========================
MODEL_ID = model_path

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Model config type: {type(config).__name__}")

# Load in BF16 - model may be stored as FP8 on HF; quantization needs BF16/FP16
# device_map=None for sequential pipeline (loads to CPU, processes layers one-by-one to GPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None,
)

# Ensure no FP8 weights (HF model may have them; torch.amin not implemented for Float8_e4m3fn)
_fp8_dtypes = tuple(
    getattr(torch, d, None) for d in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, d)
)
if _fp8_dtypes:
    fp8_count = 0
    for name, param in model.named_parameters():
        if param.dtype in _fp8_dtypes:
            param.data = param.data.to(torch.bfloat16)
            fp8_count += 1
    if fp8_count:
        print(f"Converted {fp8_count} FP8 parameter(s) to bfloat16 for quantization compatibility")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# FX tracing fix: quantization_config contains QuantizationMethod.FP8 enum which fails to serialize.
if hasattr(model.config, "quantization_config") and model.config.quantization_config is not None:
    _qc = model.config.quantization_config
    _qm = getattr(_qc, "quant_method", "fp8")
    if hasattr(_qm, "value"):
        _qm = _qm.value
    model.config.quantization_config = {
        "quant_method": str(_qm),
        "modules_to_not_convert": list(getattr(_qc, "modules_to_not_convert", ["gate", "e_score_correction_bias", "lm_head"])),
        "activation_scheme": str(getattr(_qc, "activation_scheme", "dynamic")),
        "fmt": str(getattr(_qc, "fmt", "float8_e4m3fn")),
    }
    if hasattr(_qc, "weight_block_size") and _qc.weight_block_size:
        model.config.quantization_config["weight_block_size"] = list(_qc.weight_block_size)
    print("Sanitized quantization_config for FX tracing compatibility")

# Replace MoE modules so ALL experts are activated during calibration
model = replace_moe_modules_for_calibration(model)


# =========================
# Dataset Formatters
# =========================
def format_sharegpt(example, columns, tokenizer):
    """Format ShareGPT-style conversations."""
    formatted_messages = []
    if len(columns) >= 2 and 'system' in columns[0].lower():
        system_prompt = example.get(columns[0], '')
        if system_prompt:
            formatted_messages.append({'role': 'system', 'content': str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    messages = example.get(conv_column, [])
    if isinstance(messages, str):
        try:
            import json
            messages = json.loads(messages)
        except Exception:
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text}
            return {'text': ''}
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                if content:
                    formatted_messages.append({'role': role, 'content': str(content)})
            elif isinstance(msg, str):
                idx = len([m for m in formatted_messages if m['role'] != 'system'])
                role = 'user' if idx % 2 == 0 else 'assistant'
                formatted_messages.append({'role': role, 'content': str(msg)})
    if not formatted_messages:
        return {'text': ''}
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception:
        return {'text': ''}


def format_prompt_answer(example, columns, tokenizer):
    """Format prompt/answer pairs."""
    prompt_col = columns[0]
    answer_col = columns[1] if len(columns) > 1 else columns[0]
    prompt = example.get(prompt_col, '')
    answer = example.get(answer_col, '')
    messages = [
        {'role': 'user', 'content': str(prompt)},
        {'role': 'assistant', 'content': str(answer)}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {'text': text}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {'text': text}
                elif isinstance(data[0], str):
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text}
            elif isinstance(data, str):
                return {'text': str(data)}
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    for col in columns:
        if col in example and example[col]:
            texts.append(str(example[col]))
    return {'text': ' '.join(texts)}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
}


# =========================
# Load datasets from YAML recipe
# =========================
print("\n=== Loading datasets from recipe ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', 10)
    streaming = ds_config.get('streaming', False)
    print(f"  Loading: {dataset_name} (split={split}, samples={num_samples}, formatter={formatter_name})")
    try:
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            ds = ds.take(num_samples)
            ds = list(ds)
            from datasets import Dataset
            ds = Dataset.from_list(ds)
        else:
            ds = load_dataset(dataset_name, split=split)
            n = min(num_samples, len(ds))
            ds = ds.shuffle(seed=SEED).select(range(n))
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,
        )
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        all_datasets.append(ds)
        total_samples += len(ds)
        print(f"    -> Loaded {len(ds)} samples")
    except Exception as e:
        print(f"    -> WARNING: Failed to load {dataset_name}: {e}")
        continue

if not all_datasets:
    raise ValueError("No datasets were successfully loaded from the recipe!")

ds = concatenate_datasets(all_datasets)
print(f"\n=== Total samples loaded: {total_samples} ===")

if SHUFFLE:
    ds = ds.shuffle(seed=SEED)


# =========================
# Tokenize in batches
# =========================
print("\n=== Tokenizing dataset ===")
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

NUM_CALIBRATION_SAMPLES = min(len(ds), num_calibration_samples) if num_calibration_samples else len(ds)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")

print_moe_calibration_guidance(model, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)


# =========================
# NVFP4 recipe for MiniMax-M2.5 MoE
# =========================
# Only quantize MoE expert layers (w1, w2, w3). Ignore gate, attention, norms, lm_head.
# NVFP4: FP4 weights + FP4 activations, per-group-16 (fixed), optimized for Blackwell.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "model.embed_tokens",
        "re:.*block_sparse_moe\\.gate",
        "re:.*e_score_correction_bias",
        "re:.*self_attn.*",
        "re:.*input_layernorm",
        "re:.*\\.norm$",
        "re:.*rotary.*",
        "re:.*mtp.*",
    ],
)

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
print(f"  - MoE: All 256 experts activated per sample (CalibrationMiniMaxM2SparseMoeBlock)")
print(f"  - Quantizing: block_sparse_moe.experts.*.w1, w2, w3 only (NVFP4)")

sequential_targets = detect_sequential_targets(model)
oneshot_kwargs = {
    "model": model,
    "dataset": ds,
    "recipe": recipe,
    "max_seq_length": MAX_SEQUENCE_LENGTH,
    "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
    "tokenizer": tokenizer,
    "moe_calibrate_all_experts": True,
    "tracing_ignore": [
        "_update_causal_mask",
        "create_causal_mask",
        "create_sliding_window_causal_mask",
        "_update_mamba_mask",
        "make_causal_mask",
        "get_causal_mask",
        "mask_interface",
        "mask_function",
        "_prepare_4d_causal_attention_mask",
        "_prepare_fsmt_decoder_inputs",
        "_prepare_4d_causal_attention_mask_with_cache_position",
    ],
}
if sequential_targets:
    oneshot_kwargs["sequential_targets"] = sequential_targets
    print(f"  - Sequential targets: {sequential_targets}")

oneshot(**oneshot_kwargs)

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
