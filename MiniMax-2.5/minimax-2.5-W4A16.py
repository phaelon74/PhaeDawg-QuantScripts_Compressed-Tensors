"""
MiniMax-M2.5 W4A16 AWQ Quantization Script

Model: https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- 229B params MoE (256 experts, 8 activated per token)
- 62 layers, hidden_size=3072, intermediate_size=1536
- Uses block_sparse_moe with w1, w2, w3 (SwiGLU-style experts)

CRITICAL: All experts are activated during calibration via CalibrationMiniMaxM2SparseMoeBlock.
Only MoE expert layers (w1, w2, w3) are quantized; attention, gate, norms, lm_head are ignored.
"""

import argparse
import yaml

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.utils import dispatch_for_generation

from minimax_m2_moe_calibration import CalibrationMiniMaxM2SparseMoeBlock  # noqa: F401

print("Imported CalibrationMiniMaxM2SparseMoeBlock - all experts activated during calibration")

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W4A16 AWQ quantization on MiniMax-M2.5 MoE model."
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
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W4A16 quantization (e.g., 32, 64, 128)."
)

args = parser.parse_args()
model_path = args.model_path
output_path = args.output_path
recipe_yaml_path = args.recipe_yaml
group_size = args.group_size

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
print(f"  - group_size: {group_size}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# MoE Calibration: Replace MiniMaxM2SparseMoeBlock with calibration version
# =========================
# CRITICAL: Ensures EVERY expert receives calibration data for EVERY sample.
# The original only runs router-selected experts; our calibration module runs ALL tokens
# through ALL experts so AWQ can collect proper activation statistics.
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


# =========================
# Model
# =========================
MODEL_ID = model_path

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Model config type: {type(config).__name__}")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# FX tracing fix: quantization_config contains QuantizationMethod.FP8 enum which fails to serialize.
# Replace with a plain dict (no enums) so trace_subgraphs can capture the model graph.
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
    
    # Check if first column is system_prompt (for datasets like Gryphe/Sonnet3.5-Charcard-Roleplay)
    if len(columns) >= 2 and 'system' in columns[0].lower():
        system_prompt = example.get(columns[0], '')
        if system_prompt:
            formatted_messages.append({'role': 'system', 'content': str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    
    # Get conversation data
    messages = example.get(conv_column, [])
    
    # Handle case where messages is a string (some datasets store JSON strings)
    if isinstance(messages, str):
        try:
            import json
            messages = json.loads(messages)
        except:
            # Not JSON, treat as raw text
            formatted_messages.append({'role': 'user', 'content': messages})
            if formatted_messages:
                text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                return {'text': text}
            return {'text': ''}
    
    # Convert to standard format if needed
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('from', 'user'))
                content = msg.get('content', msg.get('value', ''))
                # Normalize role names
                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'bot']:
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                if content:  # Only add if there's content
                    formatted_messages.append({'role': role, 'content': str(content)})
            elif isinstance(msg, str):
                # Alternate user/assistant for string lists
                idx = len([m for m in formatted_messages if m['role'] != 'system'])
                role = 'user' if idx % 2 == 0 else 'assistant'
                formatted_messages.append({'role': role, 'content': str(msg)})
    
    if not formatted_messages:
        return {'text': ''}
    
    try:
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
        return {'text': ''}


def format_prompt_answer(example, columns, tokenizer):
    """Format prompt/answer pairs (e.g., instruction/response)."""
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
    # Try to find messages-like column
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Already in messages format
                    text = tokenizer.apply_chat_template(data, tokenize=False)
                    return {'text': text}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                    return {'text': text}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data)}
    
    # Fallback: concatenate all columns
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
        # Load dataset
        if streaming:
            ds = load_dataset(dataset_name, split=split, streaming=True)
            # Take samples from streaming dataset
            ds = ds.take(num_samples)
            # Convert to regular dataset
            ds = list(ds)
            from datasets import Dataset
            ds = Dataset.from_list(ds)
        else:
            ds = load_dataset(dataset_name, split=split)
            # Sample from dataset
            n = min(num_samples, len(ds))
            ds = ds.shuffle(seed=SEED).select(range(n))
        
        # Get formatter function
        formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
        
        # Apply formatter
        ds = ds.map(
            lambda x: formatter_fn(x, columns, tokenizer),
            remove_columns=ds.column_names,
            num_proc=1,  # Use single proc to avoid tokenizer issues
        )
        
        # Filter out empty texts
        ds = ds.filter(lambda x: len(x.get('text', '')) > 0)
        
        all_datasets.append(ds)
        total_samples += len(ds)
        print(f"    -> Loaded {len(ds)} samples")
        
    except Exception as e:
        print(f"    -> WARNING: Failed to load {dataset_name}: {e}")
        continue

# Concatenate all datasets
if not all_datasets:
    raise ValueError("No datasets were successfully loaded from the recipe!")

ds = concatenate_datasets(all_datasets)
print(f"\n=== Total samples loaded: {total_samples} ===")

# Shuffle combined dataset if requested
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

# MoE expert coverage analysis
print_moe_calibration_guidance(model, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)


# =========================
# AWQ recipe for MiniMax-M2.5 MoE
# =========================
# Only quantize MoE expert layers (w1, w2, w3). Ignore ALL dense/attention/norm layers
# per MiniMax quantization_config.modules_to_not_convert and best practices.
#
# Ignore list (from config + recommendations):
#   - lm_head, embed_tokens: output/input, sensitive
#   - gate, e_score_correction_bias: router, does not quantize well
#   - self_attn: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
#   - input_layernorm, post_attention_layernorm, norm: normalization
#   - MTP layers if present (use_mtp)
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=4,
    type="int",
    symmetric=True,      # Marlin-compatible
    strategy="group",
    group_size=group_size,
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "model.embed_tokens",
            "re:.*block_sparse_moe\\.gate",
            "re:.*e_score_correction_bias",
            "re:.*self_attn.*",        # q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
            "re:.*input_layernorm",
            "re:.*post_attention_layernorm",
            "re:.*\\.norm$",             # model.norm
            "re:.*RMSNorm.*",
            "re:.*rotary.*",
            "re:.*mtp.*",                # Multi-Token Prediction layers (use_mtp)
        ],
        mappings=[
            AWQMapping(
                smooth_layer="re:^model\\.model\\.layers\\.\\d+\\.post_attention_layernorm$",
                balance_layers=[
                    "re:^model\\.model\\.layers\\.\\d+\\.block_sparse_moe\\.experts\\.\\d+\\.w1$",
                    "re:^model\\.model\\.layers\\.\\d+\\.block_sparse_moe\\.experts\\.\\d+\\.w3$",
                ],
            ),
            AWQMapping(
                smooth_layer="re:^model\\.model\\.layers\\.\\d+\\.block_sparse_moe\\.experts\\.\\d+\\.w3$",
                balance_layers=[
                    "re:^model\\.model\\.layers\\.\\d+\\.block_sparse_moe\\.experts\\.\\d+\\.w2$",
                ],
            ),
        ],
        config_groups={"group_0": quant_scheme},
    ),
]

# =========================
# Run one-shot compression
# =========================
print("\n=== Running one-shot compression ===")
print(f"  - MoE: All 256 experts activated per sample (CalibrationMiniMaxM2SparseMoeBlock)")
print(f"  - Quantizing: block_sparse_moe.experts.*.w1, w2, w3 only")

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    moe_calibrate_all_experts=True,  # Ensure all experts receive calibration (with our manual replacement)
    tracing_ignore=[
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

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)
