# Pinned stacks and GS32 baseline

KLD scoring and quantization use **separate venvs**. Do not install llm-compressor into the vLLM env.

KLD is scored on 4x RTX PRO 6000 Blackwell. The **serving target** remains 4x RTX 3090 (SM86). Candidate checkpoints must stay compressed-tensors W4A16 / W8A16 so Marlin can run on Ampere.

## vLLM / KLD env (`~/kld-nightly-vllm`)

| Package | Version |
| --- | --- |
| vllm | 0.23.1rc1.dev1114+g7644b1d0a |
| compressed_tensors | 0.17.0 |
| transformers | 5.9.0 |
| torch | 2.11.0+cu130 |
| CUDA | 13.0 |
| driver | 580.173.02 |
| GPU | RTX PRO 6000 Blackwell, SM 12.0 x4 |

## llm-compressor env (`~/llmcompressor-nightly`)

| Package | Version |
| --- | --- |
| llmcompressor | 0.13.1.dev50+g91fa8505 |
| compressed_tensors | 0.18.1.a20260824 |
| transformers | 5.16.1 |
| torch | 2.13.0+cu130 |
| CUDA | 13.0 |

## ModelOpt env (`~/modelopt-nightly`)

```
nvidia-modelopt @ git+https://github.com/NVIDIA/Model-Optimizer.git@449a39922b5f5d45b963e43330f62057db4b2209
```

Use ModelOpt only for sensitivity ranking until a TP=4 mixed checkpoint loads in the vLLM env above.

## Frozen checkpoints

| Path | `du -sh` | Mean KLD vs `ref_logits_Behemoth-R1-123B-v2_ctx2048_s512` |
| --- | --- | --- |
| `/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/W4A16_GS32` | 66G | 0.042380 (204700 positions) |
| `/media/fmodels2/TheHouseOfTheDude/Behemoth-R1-123B-v2/W4A16_GS32_AWQMSK` | 66G | 0.042729 (204700 positions; rejected) |
| `/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/Candidate2` | 66G | 0.046951 (204700 positions; GPTQ GS32; rejected) |
| `/media/fmodels/TheHouseOfTheDude/Behemoth-R1-123B-v2/NewQuants/Candidate3_AutoRound_GS32` | 66G | **0.037094** (204700 positions; current winner) |
| GS128 AutoRound mixed candidate | pending exact path/size | 0.043462 (204700 positions; rejected) |

Reference logits: `~/kld-nightly-vllm/kld-vllm/ref_logits_Behemoth-R1-123B-v2_ctx2048_s512/` (25G).

A later candidate only counts if it is ≤70 GiB, lower mean KLD than the reproduced GS32 number, and loadable at TP=4 with ≥32K BF16 KV on the 3090 box.

Candidate3 passes the size, KLD, TP=4, Marlin, 32K-context, memory, and
throughput gates. AutoRound GS32 is the fixed donor configuration for mixed
W4/W8 experiments. The GS128 result is rejected; its W8 capacity did not offset
the accuracy loss from coarser W4 groups. Do not spend another full run on
GS128, AWQ, or pure GPTQ.

Mixed checkpoints must be quantized from the BF16 source in one AutoRound run:
GS32 W4 for the default scheme and GS32 W8 for selected modules. Do not attempt
to promote weights from the already compressed Candidate3 checkpoint.
