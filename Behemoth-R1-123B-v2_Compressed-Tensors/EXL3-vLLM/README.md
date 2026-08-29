# Native EXL3 for vLLM on SM86

Dense-only EXL3 backend and 3.5/4.25/4.5-bpw evaluation for
**Behemoth-R1-123B-v2** on **TP4 RTX 3090**. This workstream does not touch
the compressed-tensors / Marlin path.

```
BF16 --> EXL3 convert (H6, mul1) --> safetensors
     --> vLLM plugin (exl3) --> SM86 decode GEMM / reconstruct+cuBLAS prefill
     --> TP4 gates (KLD, speed, memory, graphs)
```

## Layout

| Path | Role |
| --- | --- |
| `plugin/` | Out-of-tree `vllm.general_plugins` package |
| `tests/` | CPU loader/TP tests plus CUDA parity tests |
| `scripts/` | Manifest, convert, bench, serve, KLD, restart |
| `manifests/` | Frozen pins and receipts |
| `conversion/` | Behemoth inventory and converter pin |

Full install for **d011sd02** (physical GPUs **1–4**, leave 0 and 5 alone): [INSTALL.md](INSTALL.md).

## How to run (Linux GPU host)

1. Capture ABI + Marlin baselines (`scripts/capture_manifest.sh`, `scripts/capture_marlin_baseline.sh`).
2. Build `exllamav3_ext` for SM86 (`scripts/build_exllamav3_ext.sh`).
3. Pass kernel gates (`scripts/kernel_microbench.py`).
4. `pip install -e plugin` into `~/kld-nightly-vllm`.
5. Validate on a small Mistral EXL3 checkpoint.
6. Convert the 4.5-bpw candidate (`scripts/convert_behemoth_exl3_4p5.sh`).
7. Load TP4 `--enforce-eager`, then KLD.
8. Benchmark 1K–32K serving with `scripts/bench_serving_contexts.py`.
9. Tune reconstruct thresholds and decode graphs only after correctness.

## Isolated pins

Keep three pins separate and rebase in this order:

1. Plugin (this tree)
2. ExLlamaV3 extension
3. Upstream vLLM

Carry a vLLM source patch only for a proven missing hook (shard IDs, graph/workspace). Each patch lives in `manifests/patches/` with a regression test.
