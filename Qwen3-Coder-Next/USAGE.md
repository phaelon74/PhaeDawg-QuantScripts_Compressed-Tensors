# Qwen3-next-W4A16.py Usage Instructions

## Command-Line Arguments

The updated script now accepts the following command-line arguments:

1. **source_model**: Path to the source Qwen3 model directory
2. **output_path**: Path where to save the quantized model
3. **dataset_config**: Path to YAML configuration file containing dataset settings
4. **group_size**: Integer value for quantization group size (e.g., 32, 64, 128)

## Example Usage

### Basic Usage
```bash
python Qwen3-next-W4A16.py <source_model> <output_path> <dataset_config> <group_size>
```

### Example Command
```bash
python Qwen3-next-W4A16.py "../Qwen/models/Qwen3-Next-80B-A3B-Instruct" "./quantized_output" "dataset_config.yaml" 32
```

## Dataset Configuration File

The dataset configuration YAML file must contain the following structure:

```yaml
calibration_set:
  max_seq_length: 2048
  shuffle: true
  seed: 42
  num_samples: 512
  datasets:
    - dataset: wikitext
      name: wikitext-2-raw-v1
      split: validation
      formatter: raw_text
      columns:
        - text
      num_samples: 512
      streaming: false
```

### Available Dataset Formatters

The script supports multiple dataset formatters:
- **raw_text**: For raw text datasets (like wikitext)
- **chat_completion**: For chat completion style data
- **prompt_answer**: For instruction/response pairs
- **sharegpt**: For ShareGPT-style conversations

## Environment Variables

**None needed** - All configurations are now passed via command-line arguments.

## Key Features

- Supports multiple dataset types via configuration file
- Automatic chat template formatting for Qwen models
- MoE router exclusion (mlp.gate, mlp.shared_expert_gate remain unquantized)
- Returns W4A16 symmetric quantization in Marlin format
- Tokenizes with configurable max sequence length
- Saves with quantization config for vLLM compatibility

## Changes from Original

- Replaced environment variables with command-line arguments
- Added flexible multi-dataset loader
- Removed hardcoded dataset configuration
- Dynamic group size selection
- Consistent with Llama script architecture

## Dependencies

Same as original:
- datasets
- transformers
- llmcompressor
- torch
- compressed_tensors
- yaml