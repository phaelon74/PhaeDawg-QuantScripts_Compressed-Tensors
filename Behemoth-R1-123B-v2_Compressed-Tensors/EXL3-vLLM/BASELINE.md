# EXL3 SM86 workstream baseline

This directory is a **separate** workstream from `MixedPrecision/`. Do not modify the compressed-tensors quantization path from here.

## Accuracy authority

Frozen mixed AutoRound GS32 72 GiB checkpoint:

| Field | Value |
| --- | --- |
| Path | `/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/AutoRound_GS32_Mixed_72G` |
| `du -sh` | `72G` (rounded; record exact bytes with `du -sb` on capture) |
| Mean KLD | **0.034004** over 204,700 positions |
| KLD throughput | 605.23 positions/s |
| Backend | compressed-tensors Marlin W4A16 / W8A16 |

KLD scoring stays on the existing harness:

- Recipe: [`../MixedPrecision/recipes/baseline_512.yaml`](../MixedPrecision/recipes/baseline_512.yaml)
- Runner: [`../../../KLD_Scripts/run_batch_kld.sh`](../../../KLD_Scripts/run_batch_kld.sh)
- Reference logits: `~/kld-nightly-vllm/kld-vllm/ref_logits_Behemoth-R1-123B-v2_ctx2048_s512/`

Do not compare Behemoth EXL3 KLD against GLM or any other model.

### EXL3 quality gates

- Minimum: mean KLD ≤ 0.042380 (original AWQ floor) **and** a substantial VRAM saving versus 72G.
- Stretch: approach 0.034004.

## Speed authority

The mixed-Marlin 72G checkpoint on TP4 RTX 3090 is the frozen speed baseline. Capture it with `scripts/capture_marlin_baseline.sh` using [`../../../VLLM-Launch_Scripts/behemoth123b-r1-v2.sh`](../../../VLLM-Launch_Scripts/behemoth123b-r1-v2.sh).

Required metrics at concurrency 1 / 2 / 4:

- TTFT
- prefill tok/s
- decode tok/s
- TPOT / ITL p50 / p95 / p99
- peak VRAM per rank
- 32K BF16-KV capacity

Template: [`manifests/marlin_tp4_baseline.template.json`](manifests/marlin_tp4_baseline.template.json)

## Runtime pins

Capture the complete ABI with `scripts/capture_manifest.sh`. Expected starting point:

| Item | Pin |
| --- | --- |
| vLLM | `0.23.1rc1.dev1114+g7644b1d0a` |
| Torch | `2.11.0+cu130` |
| CUDA toolkit | 13.0 |
| Driver (KLD host) | 580.173.02 |
| Serving GPUs | 4× RTX 3090 SM86, physical 1–4 |
| Conversion GPU | physical 0 only |
| ExLlamaV3 | MIT, pinned in `manifests/stack.json` |

Conversion uses a dedicated ExLlamaV3 environment. The vLLM environment receives only the `exllamav3_ext` extension built against that Torch/CUDA/Python ABI with `TORCH_CUDA_ARCH_LIST=8.6`.
