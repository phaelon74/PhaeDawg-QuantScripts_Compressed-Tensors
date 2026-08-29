# EXL3 SM86 workstream baseline

This directory is a **separate** workstream from `MixedPrecision/`. Do not modify the compressed-tensors quantization path from here.

## Accuracy results

Frozen TP4 eager KLD results (204,700 positions, context 2048, stride 512):

| Checkpoint | `du -sh` | Mean KLD | KLD throughput | Status |
| --- | ---: | ---: | ---: | --- |
| ArtusDev EXL3 3.5-bpw H6 | 51G | 0.045794 twice | 283.75 / 284.52 pos/s | Accepted memory-first |
| ArtusDev EXL3 4.25-bpw H6 | 62G | **0.015800** | 278.61 pos/s | Current quality/size winner |
| Mixed AutoRound GS32 | 72G | 0.034004 | 605.23 pos/s | Frozen Marlin reference |
| Original AWQ GS32 | 66G | 0.042380 | — | Historical minimum gate |

The 4.25-bpw result is 53.5% lower KLD than the 72G AutoRound reference
while using 10G less rounded disk space. KLD throughput was measured on a
different vLLM/Torch stack and does not establish serving performance.
Receipt: [`results/behemoth_exl3_kld.json`](results/behemoth_exl3_kld.json).

KLD scoring stays on the existing harness:

- Recipe: [`../MixedPrecision/recipes/baseline_512.yaml`](../MixedPrecision/recipes/baseline_512.yaml)
- Runner: [`../../../KLD_Scripts/run_batch_kld.sh`](../../../KLD_Scripts/run_batch_kld.sh)
- Reference logits: `/media/netmodels/ref_logits/ref_logits_Behemoth-R1-123B-v2_ctx2048_s512/`

Do not compare Behemoth EXL3 KLD against GLM or any other model.

### EXL3 quality gates

- 3.5-bpw is accepted as a memory-first tier at 0.045794.
- Production candidates should beat 0.034004; 4.25-bpw already passes at
  0.015800.
- The 4.5-bpw H6 target should remain below 70G rounded and preserve or improve
  the 4.25-bpw result. Measure rather than infer its KLD.

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
