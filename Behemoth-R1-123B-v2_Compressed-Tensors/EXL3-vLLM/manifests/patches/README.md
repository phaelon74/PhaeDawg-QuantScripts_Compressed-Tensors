# Minimal vLLM patch queue

Keep this directory empty unless the pinned vLLM runtime
(`0.23.1rc1.dev1114+g7644b1d0a`) cannot:

1. Pass packed-projection shard IDs (`q`/`k`/`v` and `0`/`1`) into the EXL3 weight loader, or
2. Expose a graph/workspace hook required for capture sizes 1/2/4.

If a patch is required:

- Store a `*.patch` plus a one-paragraph `*.md` that names the missing hook.
- Add a regression test under `tests/` that fails without the patch.
- Do not carry SparkInfer, MoE, MLA, MTP, or rank-sliced expert patches.
