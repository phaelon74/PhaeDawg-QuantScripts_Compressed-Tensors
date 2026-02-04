import argparse
import yaml
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

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

max_mem = {
    0: "90GiB",
    1: "90GiB",
}

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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",          # <-- shards across GPUs
    max_memory=max_mem,         # <-- forces a 2-GPU split instead of filling GPU0
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Model loaded from: {MODEL_ID}")


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
        offload_device="cuda",   # avoid slow CPU offload
    ),
]

# =========================
# Run one-shot compression
# =========================
print(f"\n=== Running one-shot compression with {NUM_CALIBRATION_SAMPLES} calibration samples ===")
oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        tokenizer=tokenizer,
    )

# =========================
# Save compressed model
# =========================
SAVE_DIR = output_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("\n=== Complete ===")
print("Saved to:", SAVE_DIR)