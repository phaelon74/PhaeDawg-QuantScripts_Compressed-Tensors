"""
GLM-4.7-Flash W8A16 Quantization Script - vLLM Compatible Version 2

FIXES the AWQ smoothing error by NOT ignoring shared_experts.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

from glm4_moe_lite_v2 import CalibrationGlm4MoeLiteMoE  # noqa: F401

print("Imported CalibrationGlm4MoeLiteMoE v2")


_original_moe_modules = {}

def replace_moe_modules_for_calibration(model):
    global _original_moe_modules
    _original_moe_modules.clear()
    
    replaced_count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Glm4MoeLiteMoE":
            parts = name.split(".")
            attr_name = parts[-1]
            
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            _original_moe_modules[name] = module
            
            calibration_module = CalibrationGlm4MoeLiteMoE(
                original=module,
                config=model.config,
                calibrate_all_experts=True
            )
            setattr(parent, attr_name, calibration_module)
            replaced_count += 1
    
    print(f"Replaced {replaced_count} MoE modules with calibration versions (v2)")
    return model


from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name(".env"))

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

SRC_DIR = require_env("SRC_DIR")
DST_DIR = require_env("DST_DIR")

MODEL_ID = SRC_DIR
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = replace_moe_modules_for_calibration(model)

# Calibration data
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

NUM_NEURALMAGIC = int(NUM_CALIBRATION_SAMPLES * 0.6)
NUM_ROMBO = NUM_CALIBRATION_SAMPLES - NUM_NEURALMAGIC

print(f"Loading calibration datasets: {NUM_NEURALMAGIC} from Neural Magic, {NUM_ROMBO} from Rombo")

ds_neuralmagic = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
n_nm = min(NUM_NEURALMAGIC, len(ds_neuralmagic))
ds_neuralmagic = ds_neuralmagic.shuffle(seed=42).select(range(n_nm))

def preprocess_neuralmagic(batch):
    rendered = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        rendered.append(text)
    return {"text": rendered}

ds_neuralmagic = ds_neuralmagic.map(preprocess_neuralmagic, batched=True, num_proc=4)

ds_rombo = load_dataset("Rombo-Org/Optimized_Reasoning", split="train")
n_rombo = min(NUM_ROMBO, len(ds_rombo))
ds_rombo = ds_rombo.shuffle(seed=43).select(range(n_rombo))

def preprocess_rombo(batch):
    rendered = []
    for instruction, inputs, outputs in zip(batch["instruction"], batch["input"], batch["output"]):
        text_parts = [instruction]
        if isinstance(inputs, list) and len(inputs) > 0:
            for inp in inputs:
                if inp and inp.strip():
                    text_parts.append(f"\n\nInput: {inp}")
        if isinstance(outputs, list) and len(outputs) > 0:
            for out in outputs:
                if out and out.strip():
                    text_parts.append(f"\n\nOutput: {out}")
        text = "".join(text_parts)
        rendered.append(text)
    return {"text": rendered}

ds_rombo = ds_rombo.map(preprocess_rombo, batched=True, num_proc=4)

ds = concatenate_datasets([ds_neuralmagic, ds_rombo])
ds = ds.shuffle(seed=44)

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

print(f"Combined calibration dataset: {len(ds)} samples")

# SIMPLIFIED ignore list - DO NOT ignore shared_experts
ignore_list = [
    "lm_head",
    "model.layers.0.self_attn.q_a_proj",
    "model.layers.0.self_attn.q_b_proj",
    "model.layers.0.self_attn.kv_a_proj_with_mqa",
    "model.layers.0.self_attn.kv_b_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "re:.*mlp\\.gate$",
    "re:.*\\.gate\\.weight$",
]

print(f"Ignore list has {len(ignore_list)} entries (shared_experts NOT ignored)")

# AWQ W8A16 recipe
recipe = AWQModifier(
    targets="Linear",
    ignore=ignore_list,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 8,  # W8A16
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
        }
    },
)

SAVE_DIR = DST_DIR

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

for name, module in model.named_modules():
    if type(module).__name__ == "CalibrationGlm4MoeLiteMoE":
        if hasattr(module, '_original_experts'):
            delattr(module, '_original_experts')

linear_count = 0
quantized_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        linear_count += 1
        if hasattr(module, 'quantization_scheme'):
            quantized_count += 1
print(f"Total Linear modules: {linear_count}, Quantized: {quantized_count}")

if hasattr(model, 'generation_config') and model.generation_config is not None:
    if hasattr(model.generation_config, 'temperature') and model.generation_config.temperature is not None:
        if not getattr(model.generation_config, 'do_sample', False):
            model.generation_config.do_sample = True

print(f"\nSaving to: {SAVE_DIR}")
model.save_pretrained(SAVE_DIR, save_compressed=True, max_shard_size="5GB")
tokenizer.save_pretrained(SAVE_DIR)

import json
save_path = Path(SAVE_DIR)
config_path = save_path / "config.json"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    if 'auto_map' in config:
        print("Removing auto_map from config.json")
        del config['auto_map']
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

print("\n" + "=" * 60)
print("QUANTIZATION COMPLETE!")
print("=" * 60)
print(f"\nModel saved to: {SAVE_DIR}")
print(f"\nW8A16: 8-bit integer weights, 16-bit float activations")
print(f"Expected size: ~30 GB")
print(f"\nvLLM command:")
print(f'  vllm serve "{SAVE_DIR}" \\')
print(f'       --tensor-parallel-size 2 \\')
print(f'       --tool-call-parser glm47 \\')
print(f'       --reasoning-parser glm45 \\')
print(f'       --enable-auto-tool-choice \\')
print(f'       --trust-remote-code')
