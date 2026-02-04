import argparse
import yaml
import torch
import gc
import functools

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# =========================
# Comprehensive Fix for FX Tracing Issue with Qwen3-Coder-Next
# =========================
# Qwen3-Coder-Next uses @torch.fx.wrap decorators which create functools.partial
# objects for forward methods. The sequential pipeline's offload code tries to
# access module.forward.__func__ which doesn't exist for functools.partial.
# 
# This comprehensive patch handles functools.partial objects in ALL offload code paths:
# 1. offload_module() - handles individual module offloading
# 2. offload_model() - handles model-level offloading (which calls offload_module)
# 3. Any other offload functions that might access forward methods
#
# CRITICAL: This fix preserves accuracy by:
# - NOT disabling FX wrappers (they're essential for quantization accuracy)
# - NOT modifying model forward methods
# - Only fixing the offload code to handle functools.partial correctly
# =========================
try:
    import compressed_tensors.offload.module as offload_module_mod
    import compressed_tensors.offload.dispatch as offload_dispatch_mod
    
    # Helper function to check if a module has functools.partial forward
    def _has_partial_forward(module):
        """Check if module has a functools.partial forward method."""
        return (hasattr(module, 'forward') and 
                isinstance(module.forward, functools.partial))
    
    # Helper function to safely offload modules with functools.partial forward
    def _safe_offload_module(module, offload_device):
        """Safely offload a module without trying to wrap functools.partial forward."""
        if hasattr(module, 'to'):
            module.to(offload_device)
        # Recursively handle submodules
        for name, child in module.named_children():
            _safe_offload_module(child, offload_device)
    
    # Patch 1: offload_module() - handles individual module offloading
    if hasattr(offload_module_mod, 'offload_module'):
        _original_offload_module = offload_module_mod.offload_module
        
        def patched_offload_module(module, onload_device, offload_device):
            """Patched version that handles functools.partial forward methods."""
            # Check if forward is a functools.partial
            if _has_partial_forward(module):
                # For functools.partial, we can't access __func__ directly
                # Instead, skip the forward function wrapping and just move the module
                # The sequential pipeline will handle forward passes correctly
                _safe_offload_module(module, offload_device)
                return
            
            # For normal forward methods, use original implementation
            try:
                return _original_offload_module(module, onload_device, offload_device)
            except AttributeError as e:
                # If original fails with AttributeError about __func__, fall back to safe offload
                if '__func__' in str(e) or 'functools.partial' in str(e):
                    print(f"⚠ Fallback: Using safe offload for module due to: {e}")
                    _safe_offload_module(module, offload_device)
                else:
                    raise
        
        offload_module_mod.offload_module = patched_offload_module
        print("✓ Patched offload_module() to handle functools.partial forward methods")
    else:
        print("⚠ offload_module not found - skipping patch")
    
    # Patch 2: offload_model() - handles model-level offloading
    if hasattr(offload_dispatch_mod, 'offload_model'):
        _original_offload_model = offload_dispatch_mod.offload_model
        
        def patched_offload_model(model, onload_device, offload_device):
            """Patched version that handles functools.partial forward methods in model."""
            # Check if model or any submodule has functools.partial forward
            has_partial = False
            for name, module in model.named_modules():
                if _has_partial_forward(module):
                    has_partial = True
                    break
            
            if has_partial:
                # Use safe offload for entire model
                _safe_offload_module(model, offload_device)
                return
            
            # For normal models, use original implementation
            try:
                return _original_offload_model(model, onload_device, offload_device)
            except AttributeError as e:
                # If original fails with AttributeError about __func__, fall back to safe offload
                if '__func__' in str(e) or 'functools.partial' in str(e):
                    print(f"⚠ Fallback: Using safe offload for model due to: {e}")
                    _safe_offload_module(model, offload_device)
                else:
                    raise
        
        offload_dispatch_mod.offload_model = patched_offload_model
        print("✓ Patched offload_model() to handle functools.partial forward methods")
    else:
        print("⚠ offload_model not found - skipping patch")
    
    print("✓ Comprehensive offload fix applied - preserves FX wrappers for accuracy")
    
except Exception as e:
    print(f"⚠ Could not apply comprehensive offload fix: {e}")
    import traceback
    traceback.print_exc()
    print("  Sequential processing may still work, but if you see AttributeError,")
    print("  you may need to update compressed_tensors library.")


# =========================
# MoE Expert Coverage Utilities
# =========================
def get_moe_expert_info(model):
    """
    Extract MoE configuration from Qwen3-Coder-Next model.
    
    Returns dict with expert counts and routing info.
    """
    config = model.config
    info = {
        "num_experts": getattr(config, "num_experts", 512),
        "num_experts_per_tok": getattr(config, "num_experts_per_tok", 10),
        "num_shared_experts": getattr(config, "num_shared_experts", 1),
        "num_layers": getattr(config, "num_hidden_layers", 48),
    }
    
    # Calculate minimum samples needed for expert coverage
    # Each token activates num_experts_per_tok experts
    # We want to activate all experts multiple times for good calibration
    # Rule of thumb: (num_experts / num_experts_per_tok) * coverage_factor
    coverage_factor = 3  # Each expert should be activated ~3 times on average
    min_tokens = (info["num_experts"] / info["num_experts_per_tok"]) * coverage_factor
    
    # With typical seq_len of 2048, calculate recommended samples
    typical_seq_len = 2048
    info["recommended_min_samples"] = max(256, int(min_tokens / typical_seq_len * 10))
    info["min_tokens_for_coverage"] = int(min_tokens)
    
    return info


def print_moe_calibration_guidance(model, num_samples, max_seq_len):
    """
    Print guidance for MoE expert coverage during calibration.
    """
    try:
        moe_info = get_moe_expert_info(model)
        
        print("\n" + "="*70)
        print("MoE Expert Coverage Analysis for Qwen3-Coder-Next")
        print("="*70)
        print(f"  Total experts:           {moe_info['num_experts']}")
        print(f"  Experts per token:       {moe_info['num_experts_per_tok']}")
        print(f"  Shared experts:          {moe_info['num_shared_experts']}")
        print(f"  Number of layers:        {moe_info['num_layers']}")
        print("-"*70)
        print(f"  Your calibration setup:")
        print(f"    - Samples:             {num_samples}")
        print(f"    - Max sequence length: {max_seq_len}")
        print(f"    - Estimated tokens:    ~{num_samples * max_seq_len // 2:,} (assuming 50% fill)")
        print("-"*70)
        
        estimated_tokens = num_samples * max_seq_len // 2  # Conservative estimate
        expert_activations_per_token = moe_info['num_experts_per_tok']
        estimated_expert_activations = estimated_tokens * expert_activations_per_token
        avg_activations_per_expert = estimated_expert_activations / moe_info['num_experts']
        
        print(f"  Estimated expert activation coverage:")
        print(f"    - Total expert activations: ~{estimated_expert_activations:,}")
        print(f"    - Avg activations/expert:   ~{avg_activations_per_expert:,.0f}")
        
        # Coverage assessment
        if avg_activations_per_expert >= 100:
            coverage_status = "EXCELLENT - All experts should be well-calibrated"
        elif avg_activations_per_expert >= 50:
            coverage_status = "GOOD - Most experts should have adequate coverage"
        elif avg_activations_per_expert >= 10:
            coverage_status = "FAIR - Some experts may have limited coverage"
        else:
            coverage_status = "WARNING - Increase samples for better expert coverage!"
            
        print(f"    - Coverage assessment:      {coverage_status}")
        
        if avg_activations_per_expert < 50:
            recommended = moe_info['recommended_min_samples']
            print(f"\n  RECOMMENDATION: Consider using {recommended}+ samples")
            print(f"                  with diverse content (code, text, math, etc.)")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Note: Could not analyze MoE config: {e}")


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =========================
# Parse Command-Line Arguments
# =========================
parser = argparse.ArgumentParser(
    description="Run W4A16 AWQ quantization on Qwen3-Next model."
)
parser.add_argument(
    "source_model",
    type=str,
    help="Path to the source model directory."
)
parser.add_argument(
    "output_path",
    type=str,
    help="Path to the destination directory for saving quantized model."
)
parser.add_argument(
    "dataset_config",
    type=str,
    help="Path to the dataset YAML configuration file (contains max_seq_length and dataset config)."
)
parser.add_argument(
    "group_size",
    type=int,
    help="Group size for W4A16 quantization (e.g., 32, 64, 128)."
)

args = parser.parse_args()
source_model_path = args.source_model
output_path = args.output_path
dataset_config_path = args.dataset_config
group_size = args.group_size


# =========================
# Load Dataset Config and extract config
# =========================
with open(dataset_config_path, 'r') as f:
    dataset_config = yaml.safe_load(f)

# Extract config from calibration_set section
calibration_config = dataset_config.get('calibration_set', {})
MAX_SEQUENCE_LENGTH = calibration_config['max_seq_length']  # Required - fail if missing
SHUFFLE = calibration_config.get('shuffle', True)
SEED = calibration_config.get('seed', 42)
num_calibration_samples = calibration_config.get('num_samples', 512)
datasets_config = calibration_config.get('datasets', [])

print(f"Loaded dataset config from: {dataset_config_path}")
print(f"  - max_seq_length: {MAX_SEQUENCE_LENGTH}")
print(f"  - shuffle: {SHUFFLE}")
print(f"  - seed: {SEED}")
print(f"  - num_samples: {num_calibration_samples}")
print(f"  - group_size: {group_size}")
print(f"  - datasets to load: {len(datasets_config)}")

# =========================
# Model
# =========================
MODEL_ID = source_model_path

# =========================
# IMPORTANT: Qwen3-Coder-Next Architecture & Memory
# =========================
# - 80B total params, 3B activated (MoE)
# - 512 experts, 10 activated per token + 1 shared expert
# - Hybrid layout: DeltaNet (linear attention) + Gated Attention
# - 48 layers with repeating pattern
# - Uses @torch.fx.wrap decorators which break FX tracing in sequential pipeline
#
# Memory calculation for 2x RTX PRO 6000 Blackwell (192GB total):
#   - Model weights (BF16):     ~160GB (on CPU during sequential processing)
#   - Per-layer on GPU:        ~3-5GB per Linear layer
#   - Calibration samples:     512 samples × 2048 tokens = processed in batches
#   - Activations per layer:   ~2-5GB per layer during calibration
#   - Peak VRAM per GPU:        ~10-15GB (one layer + activations)
#
# Solution: Sequential processing with model on CPU
#   - Model stays on CPU (device_map=None)
#   - Sequential pipeline loads one Linear layer at a time to GPU
#   - Processes 512 calibration samples through that layer
#   - Quantizes and offloads before next layer
#   - Avoids FX tracing by not tracing the full model forward
#
# NOTE: Sequential processing handles FX-wrapped modules correctly by processing
#       individual Linear layers without tracing the full model graph
# =========================

print("\n" + "="*70)
print("Loading Qwen3-Coder-Next to CPU (device_map=None)")
print("Sequential layer processing will handle one Linear layer at a time")
print("="*70 + "\n")

# Show available GPU memory
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = props.total_memory - torch.cuda.memory_allocated(i)
        print(f"  GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB total, {free_mem / 1e9:.1f}GB free")
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
    print(f"  Total VRAM: {total_vram / 1e9:.1f}GB")
    print(f"\n  NOTE: Model (~160GB) stays on CPU")
    print(f"        Sequential processing loads one Linear layer (~3-5GB) at a time to GPU")
    print(f"        Peak VRAM: ~10-15GB per GPU (layer + calibration activations)")

# Load model to CPU - sequential processing will handle GPU loading layer by layer
# This avoids FX tracing issues by not loading the full model graph
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=None,  # Load to CPU - sequential processing handles GPU layer by layer
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"\nModel loaded to CPU from: {MODEL_ID}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Verify model is on CPU (critical for sequential processing)
model_device = next(model.parameters()).device
if model_device.type == 'cpu':
    print(f"✓ Model device: CPU (sequential processing will load layers to GPU one at a time)")
else:
    print(f"⚠ WARNING: Model is on {model_device} - sequential processing may not work correctly!")
    print(f"  Expected: CPU, Got: {model_device}")


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
                try:
                    text = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
                    return {'text': text}
                except Exception as e:
                    # If chat template fails, return empty
                    print(f"WARNING: Failed to apply chat template: {e}")
                    return {'text': ''}
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
        print(f"WARNING: Failed to apply chat template: {e}")
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
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {'text': text}
    except Exception as e:
        # If chat template fails, return empty
        print(f"WARNING: Failed to apply chat template in prompt_answer format: {e}")
        return {'text': ''}


def format_chat_completion(example, columns, tokenizer):
    """Format chat completion style data."""
    # Try to find messages-like column
    for col in columns:
        if col in example:
            data = example[col]
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Already in messages format
                    try:
                        text = tokenizer.apply_chat_template(data, tokenize=False)
                        return {'text': text}
                    except Exception as e:
                        # If chat template fails, return empty
                        print(f"WARNING: Failed to apply chat template: {e}")
                        return {'text': ''}
                elif isinstance(data[0], str):
                    # List of strings - alternate user/assistant
                    messages = []
                    for i, item in enumerate(data):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        messages.append({'role': role, 'content': str(item)})
                    try:
                        text = tokenizer.apply_chat_template(messages, tokenize=False)
                        return {'text': text}
                    except Exception as e:
                        print(f"WARNING: Failed to apply chat template: {e}")
                        return {'text': ''}
            elif isinstance(data, str):
                # Single text field
                return {'text': str(data)}
    
    # Fallback: concatenate all columns
    text = ' '.join(str(example.get(col, '')) for col in columns)
    return {'text': text}


def format_raw_text(example, columns, tokenizer):
    """Format raw text data."""
    texts = []
    
    # Handle custom parameters from dataset config
    prefix = ""
    if '_formatter_params' in example and isinstance(example['_formatter_params'], dict):
        params = example['_formatter_params']
        if 'prefix' in params:
            prefix = str(params['prefix'])
    
    # Also support prefix in columns[0] for raw_text datasets
    if len(columns) == 1 and columns[0] not in ['text', 'content', 'user', 'prompt', 'problem', 'instruction', 'prompt_input', 'article', 'text_output']:
        prefix += str(columns[0]) + "\n***\n"
    
    for col in columns:
        if col in example and example[col]:
            text_content = str(example[col])
            texts.append(prefix + text_content)
    
    return {'text': ' '.join(texts)}


FORMATTERS = {
    'sharegpt': format_sharegpt,
    'prompt_answer': format_prompt_answer,
    'chat_completion': format_chat_completion,
    'raw_text': format_raw_text,
    'deepmind_code_contests': format_raw_text,
}


# =========================
# Load datasets from YAML config
# =========================
print("\n=== Loading datasets from config ===")
all_datasets = []
total_samples = 0

for ds_config in datasets_config:
    dataset_name = ds_config['dataset']
    split = ds_config.get('split', 'train')
    columns = ds_config.get('columns', [])
    formatter_name = ds_config.get('formatter', 'raw_text')
    num_samples = ds_config.get('num_samples', num_calibration_samples)
    streaming = ds_config.get('streaming', False)
    shuffle = ds_config.get('shuffle', SHUFFLE)
    ds_seed = ds_config.get('seed', SEED)
    
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
            if shuffle:
                ds = ds.shuffle(seed=ds_seed).select(range(n))
            else:
                ds = ds.select(range(0, n))
        
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
    raise ValueError("No datasets were successfully loaded from the config!")

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
    num_proc=1,
)

NUM_CALIBRATION_SAMPLES = min(len(ds), num_calibration_samples)
print(f"Tokenized {NUM_CALIBRATION_SAMPLES} samples (max_seq_length={MAX_SEQUENCE_LENGTH})")

# =========================
# MoE Expert Coverage Analysis
# =========================
# Qwen3-Coder-Next has 512 experts with 10 activated per token.
# We need sufficient diverse samples to activate all experts during calibration.
print_moe_calibration_guidance(model, NUM_CALIBRATION_SAMPLES, MAX_SEQUENCE_LENGTH)


# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - Dynamic group_size from argument
#  - IMPORTANT: skip MoE routers (mlp.gate, mlp.shared_expert_gate), keep quantizing FFN projections
#  - Keep MoE router-related linears and output head unquantized
# =========================
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

weight_args = QuantizationArgs(
    num_bits=4,          # 4-bit weights
    type="int",
    symmetric=True,      # SYMMETRIC (Marlin requirement)
    strategy="group",    # group-wise quantization
    group_size=group_size,  # Dynamic group size from argument
)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,   # A16 (leave activations in FP16/BF16)
    output_activations=None,
)

recipe = [
    AWQModifier(
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        config_groups={"group_0": quant_scheme},
        # NOTE: Do NOT set offload_device - sequential pipeline handles device management
        # Setting offload_device causes AttributeError with Qwen3-Coder-Next's functools.partial forwards
    ),
]

# =========================
# Run one-shot compression
# =========================
print(f"\n=== Running one-shot compression with {NUM_CALIBRATION_SAMPLES} calibration samples ===")
print(f"  - Sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"  - Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print("Note: Using SEQUENTIAL layer processing (model on CPU)")
print("      Each Linear layer loaded to GPU one at a time - avoids OOM")

# Clear memory before quantization
clear_memory()

# Show memory status before quantization
if torch.cuda.is_available():
    print("\nGPU memory before quantization (should be near-empty):")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        free = total - reserved
        print(f"  GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free, {total:.1f}GB total")
    
    # Verify model is still on CPU
    model_device = next(model.parameters()).device
    if model_device.type != 'cpu':
        raise RuntimeError(f"Model moved to {model_device} before quantization! Expected CPU.")
    print(f"\n✓ Verified: Model still on CPU (device={model_device})")

# Run AWQ quantization with sequential layer processing
# Model is on CPU (device_map=None), oneshot() will:
#   1. Load Linear layer N to GPU
#   2. Run 512 calibration samples (2048 tokens each) through layer N
#   3. Collect activation statistics for AWQ scaling
#   4. Compute optimal per-channel scales for layer N
#   5. Quantize layer N weights to INT4
#   6. Move quantized layer back to CPU
#   7. Repeat for Linear layer N+1
#
# sequential_targets=["Linear"] ensures only one Linear layer at a time on GPU
# Peak VRAM usage: ~10-15GB per GPU (one layer + calibration activations)
# This fits comfortably in 96GB RTX PRO 6000 cards
#
# IMPORTANT: With 512 diverse calibration samples, all 512 experts should be
# activated multiple times during calibration, ensuring proper quantization.
#
# NOTE: Sequential processing avoids FX tracing issues by processing individual
#       Linear layers without tracing the full model graph with @torch.fx.wrap decorators
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    tokenizer=tokenizer,
    sequential_targets=["Linear"],  # Process Linear layers one at a time to avoid OOM
)

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)