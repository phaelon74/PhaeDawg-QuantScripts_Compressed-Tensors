# vllm-exl3-sm86

Out-of-tree dense EXL3 backend for the pinned vLLM runtime. Ported from the dense subset of [local-inference-lab/vLLM PR #139](https://github.com/local-inference-lab/vllm/pull/139). ExLlamaV3 kernels remain MIT-licensed in the separate extension build.

```bash
# inside ~/kld-nightly-vllm
pip install -e Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/plugin
export VLLM_PLUGINS=vllm_exl3_sm86
export VLLM_EXL3_EXT_PATH=/path/to/exllamav3_ext_dir
```

Fail-closed defaults:

- `--enforce-eager` required until `VLLM_EXL3_ALLOW_GRAPHS=1`
- sleep mode and `state_dict()` resave rejected
- missing/conflicting mcg+mul1 markers rejected
- packed QKV / gate-up must receive shard IDs
