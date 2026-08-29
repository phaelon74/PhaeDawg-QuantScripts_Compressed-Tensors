# Behemoth EXL3 conversion

Dedicated environment: `~/exllamav3-convert` (full ExLlamaV3 package).
vLLM environment receives only `exllamav3_ext`.

```bash
export CUDA_VISIBLE_DEVICES=0
./scripts/convert_behemoth_exl3.sh
python scripts/validate_exl3_checkpoint.py "$OUT_DIR"
```

Pins:

- decoder target `3.5` bpw
- `head_bits=6`
- `codebook=mul1`
- calibration `250 x 2048` (pinned bundled corpus limit; 512 overflows wiki articles)
- GPU 0 only
- `--resume` against a work directory large enough for a complete output copy
