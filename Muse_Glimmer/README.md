---
language:
- en
library_name: vllm
pipeline_tag: image-text-to-text
tags:
- text-generation
- multimodal
- conversational
- compressed-tensors
- PTQ
- w8a16
- quantized
base_model_relation: quantized
quantized_by: TheHouseOfTheDude
license: apache-2.0
base_model:
- meta-models/Muse-Glimmer-30B
---
# Muse-Glimmer-30B-INT8 — PTQ Quantized (W8A16)

## Overview
Post-Training Quantized (PTQ) version of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).

Muse Glimmer-30B is a dense multimodal (vision-language) agentic model (~29.6B total incl. vision) with a Perception Encoder (ViT-G/14, ~1.8B), hybrid sliding-window / full attention (3:1), and 131K context. This checkpoint keeps the vision / perception path in BF16 and quantizes the language-model Linear layers to INT8.

- No calibration dataset
- One-shot quantization (`llmcompressor.oneshot`)
- Uses [llmcompressor](https://github.com/vllm-project/llm-compressor) `QuantizationModifier` with the W8A16 preset (no AWQ, no GPTQ)
- Loaded / saved via `MuseGlimmerForConditionalGeneration` so vLLM sees the correct `language_model` / `vision_*` weight paths
- Compressed with `save_pretrained(..., save_compressed=True)` (compressed-tensors)
- Processor / tokenizer / chat-template sidecars copied from the BF16 source for a complete multimodal package
- Source `generation_config.json` restored so list-valued `eos_token_id` is preserved for multi-stop decoding in vLLM

---

## Quantization

- Scheme: **W8A16**
- Weights: INT8 (per-channel, symmetric)
- Activations: FP16/BF16 (untouched)
- Targets: `Linear` layers only
- Modifier: `QuantizationModifier(targets="Linear", scheme="W8A16", ...)`

### Ignored (left unquantized)

| Pattern | Reason |
| --- | --- |
| `lm_head` | Output projection; quantizing hurts quality |
| `re:.*vision_tower.*` | Perception Encoder (~1.8B ViT-G/14) — keep BF16 |
| `re:.*vision_adapter.*` | Muse multimodal adapter stack |
| `re:.*vision_projection.*` | Muse vision → language projection |
| `re:.*multi_modal_projector.*` | Defensive alias (Kimi parent / renames) |
| `re:.*mm_projector.*` | Defensive alias |
| `re:.*draft.*` | Speculative DFlash / drafter companion (not part of this PTQ) |
| `re:.*dflash.*` | Speculative DFlash / drafter companion (not part of this PTQ) |

Dense model: no MoE router / expert ignores.

### Recipe (from the quant script)

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*vision_adapter.*",
        "re:.*vision_projection.*",
        "re:.*multi_modal_projector.*",
        "re:.*mm_projector.*",
        "re:.*draft.*",
        "re:.*dflash.*",
    ],
)
```

---

## Usage

```bash
vllm serve TheHouseOfTheDude/Muse-Glimmer-30B-INT8 \
  --quantization compressed-tensors
```

Tensor-parallel example:

```bash
vllm serve TheHouseOfTheDude/Muse-Glimmer-30B-INT8 \
  --quantization compressed-tensors \
  -tp <num_gpus>
```

### Recommended sampling (from base model)

- `temperature = 1.0`
- `top_p = 0.95`
- `top_k = 64`

Reasoning strength can be set in the system prompt as `Reasoning strength: <low|medium|high|xhigh>`.

---

## Notes

- Requires **vLLM** with compressed-tensors support
- Not intended for vanilla Transformers inference of the quantized weights
- Vision tower / adapter / projection remain BF16; only LM Linear weights are INT8
- DFlash speculative drafter is **not** included or quantized in this release
- Base model license: Apache 2.0
