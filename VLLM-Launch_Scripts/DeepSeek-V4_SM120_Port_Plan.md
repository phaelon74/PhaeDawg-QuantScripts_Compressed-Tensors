# DeepSeek-V4-Flash on SM120 (RTX 6000 Pro Workstation) — Port Plan

**Status:** DRAFT — needs sign-off before any code starts.
**Target hardware:** 4× NVIDIA RTX 6000 Pro Workstation (Blackwell, SM120, 96 GB).
**Upstream baselines:**
- `vllm-project/vllm` branch `dsv4` (PR #40760).
- `deepseek-ai/DeepGEMM` @ `477618cd51` (the ref vLLM pins in `tools/install_deepgemm.sh`).

---

## 1. Executive summary

DeepSeek-V4-Flash does not run on SM120 today because DeepGEMM ships kernels only for SM90 (Hopper) and SM100 (datacenter Blackwell). SM120 is a different ISA — no UMMA, no TMEM, no WGMMA on BF16, and ~99 KB SMEM vs. 228 KB. Every code path that reaches DeepGEMM hits `DG_HOST_UNREACHABLE("Unsupported architecture")`.

To get V4 running natively, we must write a new family of SM120 kernels for a scoped subset of DeepGEMM, and wire them through the JIT/dispatch machinery that DeepGEMM and vLLM's `dsv4` branch rely on.

**Good news from PR #40760 logs:** V4's MoE experts route through MARLIN (MXFP4), not DeepGEMM. That removes the largest class of kernels (grouped FP8 GEMM) from scope. The work is bounded to: (a) MultiHead HyperConnection (MHC), (b) the Lightning Indexer / sparse MQA logits, and (c) the FP8 block-scaled linear used by MLA projections.

**Rough effort (one experienced CUDA kernel engineer, full-time):**
- Optimistic: 6–8 weeks to a functional prototype that serves V4 on 4×RTX 6000 Pro at any throughput.
- Realistic: 12–16 weeks to performant, numerically validated, merged-into-a-maintained-fork state.
- Pessimistic: 6 months+ if SMEM budgeting forces a kernel rewrite per-layer or numerical issues surface from TF32 tile swaps.

**Non-goals (explicitly out of scope):**
- Matching B200/H200 throughput. The absence of TMEM/UMMA on SM120 is a hard ceiling of roughly 40–60% of SM100 per-SM FP8/TF32 throughput for the kernels we care about.
- Supporting other architectures (SM89/SM86/SM75). We port only for SM120 to keep the fork small.
- Training/backward kernels. Inference only.
- Turning this into a long-term maintained public library. We'll publish our fork, but upstreaming to DeepGEMM proper was explicitly declined by DeepSeek. Plan for an indefinitely-maintained fork.

---

## 2. The hard problem (why SM120 is not SM100 with smaller SMs)

The one-line question to answer: *"What does the reference SM100 kernel do at each cycle, and what's the cheapest SM120 equivalent that still produces bit-equivalent outputs?"*

| Capability | SM90 (H100/H200) | SM100 (B200/GB200) | **SM120 (RTX 6000 Pro / 5090)** |
|---|---|---|---|
| Tensor-core FP8/FP6/FP4 | `wgmma` | `tcgen05` / UMMA into TMEM | **`tcgen05` (narrow precision only, with quantization overhead)** |
| Tensor-core TF32 GEMM | `wgmma.mma_async` | UMMA TF32 into TMEM | **`mma.sync` m16n8k8/m16n8k16 (SM80-style) — no warpgroup, no TMEM** |
| Tensor-core BF16/FP16 GEMM | `wgmma` | UMMA BF16 via TMEM | **`mma.sync` m16n8k16 (SM80 fallback) — no WGMMA, no UMMA** |
| TMEM (tensor memory) | no | yes (~256 KB/SM) | **no — must live entirely in registers + SMEM** |
| TMA (async bulk copy) | yes | yes (expanded) | **yes (full SM90 TMA)** |
| Cluster launch / DSMEM | yes | yes | **yes (partial — fewer features)** |
| Max SMEM per block | 228 KB | 228 KB | **~99 KB** |
| `tcgen05.fence` | n/a | yes | **no** |
| `cp.async.bulk.tensor` | yes | yes | **yes** |
| `stmatrix.aligned` (D.wgmma) | yes | yes | **yes (usable)** |

The practical consequence: we cannot write a kernel that looks like SM100's. We need to design kernels that **look like SM90 minus WGMMA** — i.e., SM80-era MMA fragments scheduled by a producer-consumer pipeline with SM90 TMA for memory. That's a real design, not a config flag.

**Concrete data point from the community (DeepGEMM issue #236):**
> "the sm100 kernel expects 168k smem_size and sm120 only has 99k available… without UMMA its hard to convert native TMEM ops. you WILL need to use sm_80 fallback."

---

## 3. Kernel inventory — what V4 actually calls into DeepGEMM

Derived from:
- The crash trace (we already know MHC hits it).
- PR #40760 startup logs: `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, `fp8_ds_mla`, `Lightning Indexer`, `MARLIN Mxfp4 MoE`.
- DeepGEMM's public API in `csrc/apis/*.hpp`.

### 3.1 In-scope (we MUST port these for V4 to run)

| # | Kernel | DeepGEMM API | vLLM caller | Shapes / dtypes | Called per | Priority |
|---|---|---|---|---|---|---|
| 1 | **MHC pre-norm GEMM** | `hyperconnection::tf32_hc_prenorm_gemm` | `vllm/model_executor/layers/mhc.py:263` via `torch.ops.vllm.mhc_pre` | A: BF16 [m,k], B: FP32 [n,k], D: FP32 [m,n], sqr_sum: FP32 [m]. TF32 math. | Every transformer layer, per token | **P0 — blocker** |
| 2 | **Sparse MQA logits (paged)** | `attention::fp8_paged_mqa_logits` | Lightning Indexer | FP8 Q/K block-scaled, paged KV, BF16 out. | Every prefill + decode step, attention path | **P0 — blocker** |
| 3 | **Paged MQA logits metadata** | `attention::get_paged_mqa_logits_metadata` | Indexer dispatch | index/offset prep | Same | **P0 — cheap, port first** |
| 4 | **FP8 dynamic block-scaled GEMM (NT)** | `gemm::fp8_gemm_nt` | `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` for Fp8LinearMethod | A: FP8e4m3 [m,k] + scales, B: FP8e4m3 [n,k] + scales, D: BF16. 128×128 block scales. | MLA q_proj, kv_proj, o_proj, shared-experts in each layer | **P0 — blocker *unless* we force Marlin path, see §3.3** |
| 5 | FP8 block-scaled GEMM (TN/NN/TT) | `gemm::fp8_gemm_tn` etc. | Same, alternate layouts | Same dtypes, alternate majorness | Possibly yes — depends on layout of weight tensors | P0 if used, P1 otherwise (verify with `nsys profile` or a `grep` of callsites) |

### 3.2 Probably-not-in-scope for V4 (verify by tracing before de-scoping)

| Kernel | Why likely not | How to verify |
|---|---|---|
| `m_grouped_fp8_gemm_*` (grouped MoE) | V4 logs show `MARLIN Mxfp4 MoE backend` — experts use Marlin, not DeepGEMM | `grep m_grouped vllm/model_executor/models/deepseek_v4.py` in the cloned repo |
| `fp8_gemm_nn` / `fp8_gemm_tt` | Non-standard layouts; vLLM typically uses NT | Same |
| BF16 GEMM family | Unlikely — V4 weights are quantized | Same |
| NVFP4 mega-MoE | Marlin handles MXFP4 MoE | Same |

### 3.3 Possible scope reduction via Marlin fallback

From vLLM issue #26211, `VLLM_TEST_FORCE_FP8_MARLIN=1` forces the FP8 linear layers (item #4 above) to go through Marlin on SM120 instead of DeepGEMM. If that flag is honored on the `dsv4` branch, we may be able to **drop #4 and #5 entirely from our kernel work** and only port the MHC + indexer kernels. This would cut the port's scope roughly in half.

**Action item (see §13):** before we start kernel work, confirm whether `VLLM_TEST_FORCE_FP8_MARLIN=1` works on the dsv4 branch for the MLA FP8 linears. This is 30 minutes of verification and saves weeks of kernel work.

### 3.4 Out of scope (confirmed)

- MoE expert GEMMs (MARLIN handles them on SM120).
- FP4 numerics / NVFP4 kernels.
- Communication kernels (NCCL/DeepEP) — those are architecture-agnostic transport.
- Kernels not called during inference (training, backward).

---

## 4. Per-kernel port strategy

### 4.1 MHC `tf32_hc_prenorm_gemm` (the one in your crash trace)

**What it computes:** D = A @ B.T, with `sqr_sum[m] = sum_k A[m,k]^2`. A is BF16, B is FP32. Math is TF32 tensor-core. Output D is FP32.

**Why it exists as a fused kernel:** V4's Multi-Head HyperConnection layer needs the L2 norm of A alongside the matmul; fusing saves a full reload of A.

**Reference implementations we have:**
- `deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh` — **our best starting point.** Uses WGMMA + TMA, fits in ~99 KB SMEM if we keep `BLOCK_M=64, BLOCK_N=?, BLOCK_K=64, kNumStages=2–3`.
- `deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh` — reference for numerical behavior and dispatch patterns; not directly portable (UMMA/TMEM).

**Port strategy (SM120 flavor):**
1. Fork `sm90_tf32_hc_prenorm_gemm.cuh` → `sm120_tf32_hc_prenorm_gemm.cuh`.
2. Replace `WGMMA::wgmma(...)` with `ptx::mma_sync_m16n8k8_tf32(...)` (SM80-style `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`). This is a ~40-line replacement; the fragment layout matches the `wgmma-64n8` A/D layout used by the SM90 kernel, so the swizzled SMEM reader code stays intact.
3. Keep the SM90 TMA producer-consumer pipeline (`tma::copy`, `full_barriers`, `empty_barriers`, `arrive_and_expect_tx`). SM120 supports all of this.
4. Keep the SMEM swizzle code (`get_swizzled_bank_group_idx`) — bank layout is identical on SM120.
5. Keep the FP32 accumulator + `sqr_sum` reduction logic unchanged; it's SM-agnostic.
6. Drop `cudaGridDependencySynchronize()` / PDL calls OR guard them on `__CUDA_ARCH__ >= 900` with a fallback path — some PDL features may not be exposed on SM120 with CUDA 13.0 (verify).
7. Remove `cluster_launch_control` usage if present; use plain grid launch.

**Tile-size targeting for 99 KB SMEM:**
- SMEM layout (SM90 ref): `BLOCK_M*BLOCK_N*4 + kNumStages*(BLOCK_M*BLOCK_K*2 + BLOCK_N*BLOCK_K*4) + barriers`.
- Fit check: with `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, kNumStages=2` → 32KB CD + 2×(8KB A + 32KB B) + barriers ≈ 113 KB. **Doesn't fit.**
- With `BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, kNumStages=3` → 16KB CD + 3×(8KB A + 16KB B) ≈ 88 KB. **Fits.** Uses 3-stage pipeline.
- With `BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, kNumStages=3` → 32KB CD + 3×(4KB A + 16KB B) ≈ 92 KB. **Fits.** Better N-tile.
- Final choice depends on the actual N values V4 passes in (typically hidden-dim related). Bench both.

**Correctness oracle:** the SM100 reference. Run identical random inputs through both and compare D/sqr_sum element-wise with `atol=1e-3, rtol=5e-3` (TF32 gives us ~21 bits of mantissa, so ~1e-3 absolute error is expected).

**Risk:** medium-low. This is the cleanest port of the three because the SM90 impl is already very close structurally to what we need.

### 4.2 Sparse MQA logits `fp8_paged_mqa_logits`

**What it computes:** the Lightning Indexer's attention scores over a paged FP8 KV cache. Used to pick which KV pages to materialize for the full attention pass.

**Complexity:** high. This kernel is a specialized FP8 attention-like op with paged memory access, MQA head dimension reduction, and block-scaled softmax-like output.

**Reference implementation:** DeepGEMM `sm100_fp8_paged_mqa_logits` (file noted in issue #236 as `deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh: No such file or directory` — i.e., the header is explicitly absent for SM120, confirming this is a required port).

**Port strategy:**
1. Start from `sm90_fp8_paged_mqa_logits` if it exists (check the dsv4 vLLM-pinned DeepGEMM commit). If only SM100 exists, port from SM100 but swap UMMA for SM80 `mma.sync` as with the MHC kernel.
2. FP8 tensor-core math on SM120 is `tcgen05`-based with quantization overhead per issue #236 — usable, but we need to decide whether to (a) use `tcgen05` narrow-precision MMA directly, or (b) dequantize A/B to BF16 and use BF16 `mma.sync`. Path (b) is simpler and probably what we should start with for correctness; path (a) is the performance target.
3. The paged memory access pattern is orthogonal to the arch port — stays the same.
4. Output layout (block-scaled) stays the same.

**Risk:** high. We may hit subtleties in FP8 scaling on SM120 that don't match SM100 bit-exactly.

### 4.3 FP8 block-scaled linear `fp8_gemm_nt` (decision pending §3.3)

**If we must port:** near-identical strategy to §4.2 but for a plain GEMM shape without paged memory or MQA reduction. Less complex than the indexer.

**If Marlin fallback works:** de-scope entirely.

---

## 5. Shared infrastructure work (applies to all kernels)

### 5.1 DeepGEMM JIT machinery

DeepGEMM compiles kernels at runtime via NVCC. The dispatch chain:

```
python: deep_gemm.tf32_hc_prenorm_gemm(...)
  ↓
C++ pybind (csrc/apis/hyperconnection.hpp):
  arch_major = get_arch_major()
  dispatch → sm90_* or sm100_*
  ↓
jit_kernels/impls/sm{90,100}_tf32_hc_prenorm_gemm.hpp:
  NVCC-compiles the .cuh template with per-shape constants
  ↓
kernel launch
```

**What we change:**
- `csrc/apis/hyperconnection.hpp` — add `arch_major == 12 → sm120_tf32_hc_prenorm_gemm(...)` branch.
- `csrc/apis/attention.hpp` — similar for indexer.
- `csrc/apis/gemm.hpp` — similar for block-scaled GEMM *if* we port it.
- `deep_gemm/include/deep_gemm/impls/sm120_*.cuh` — new kernel implementations.
- `deep_gemm/jit_kernels/impls/sm120_*.hpp` — JIT template glue (shape/stage heuristics).
- `deep_gemm/jit_kernels/heuristics/sm120_heuristics.py` — block-size / stage-count selection for SM120's smaller SMEM.
- NVCC arch flag: `--gpu-architecture=sm_120a` (the `a` suffix is required for tcgen05 intrinsics).

### 5.2 `device_runtime->get_arch_major()`

Verify this returns 12 on SM120 (should, since it's `cc.major`). If not, patch it.

### 5.3 vLLM side

- `vllm/utils/deep_gemm.py` — the Python wrapper does its own capability checks (e.g., the `_missing` error about "DeepGEMM backend is not available or outdated" from issue #29946). We may need to loosen its SM gate to accept 120.
- `vllm/model_executor/layers/mhc.py` — verify it just calls through `torch.ops.vllm.mhc_pre` without arch gating. If it has its own gate, add SM120.
- Install script (`tools/install_deepgemm.sh`) — point at our fork & pin.

### 5.4 Test harness (must build FIRST)

A dedicated numerical-correctness harness outside vLLM. Required BEFORE writing kernels:

```
scripts/sm120_kernel_tests/
├── test_mhc_prenorm.py        # Compares our SM120 MHC to SM100 reference on 2nd GPU if available, or CPU reference
├── test_paged_mqa_logits.py   # Ditto for indexer
├── test_fp8_gemm.py           # Ditto for FP8 linear (if in scope)
├── cpu_reference.py           # PyTorch-native reference implementations (slow but correct)
└── shape_sweep.py             # Generates shape configs matching V4's actual call sites
```

The CPU reference is non-negotiable — TF32 / FP8 numerics are subtle and we can't debug kernels without a trusted oracle. On SM120-only hardware, CPU FP32 reference is the only option for MHC; FP8 is harder and may need simulated FP8 on CPU (or rent an SM100 box for one afternoon of golden-data generation).

### 5.5 Benchmarking

- `ncu` microbenchmarks per kernel (achieved FLOPS, SMEM pressure, register pressure).
- `nsys` traces of full V4 forward to catch kernel launches we missed.
- End-to-end throughput on the full model on 4×RTX 6000 Pro.

---

## 6. Phase plan

Each phase has a clear gate that says "stop here and reassess if X fails."

### Phase 0 — Scoping & verification (3–5 days)
- [ ] Clone the exact `dsv4` vLLM + pinned DeepGEMM, apply the MHC-only arch-major patch (add `arch_major==12→fall-through-to-SM90` that just reuses the SM90 kernel), build, and run. **Expected:** it compiles but hits `DG_DEVICE_ASSERT(false and "This kernel only support sm_90a")` because the SM90 kernel is guarded by `__CUDA_ARCH__ >= 900` and sm_120a still satisfies that, so actually it *might* run — or it might break on WGMMA.
- [ ] Run `VLLM_TEST_FORCE_FP8_MARLIN=1` launch and confirm it bypasses DeepGEMM for all FP8 linears on dsv4.
- [ ] Inventory all `torch.ops.vllm.*` calls in `deepseek_v4.py` that route through DeepGEMM (grep + `nsys profile` on a dummy forward that bypasses the MHC crash).
- [ ] Produce a final kernel list with confirmed shapes.
- **Gate:** produce a written scope decision. If the final kernel list is only MHC + indexer, we proceed. If it also requires FP8 GEMM and Marlin fallback fails, reassess whether the project is tractable solo or needs more hands.

### Phase 1 — Test harness + SM90 validation (1 week)
- [ ] Build the numerical-correctness harness (§5.4).
- [ ] Validate our harness against the SM90 reference kernel on a rented H100 (few hours, ~$10).
- [ ] Produce golden output files for 50+ shape configs per kernel.
- **Gate:** every kernel in scope must have a reproducible `pytest` that passes on H100 using the reference SM90/SM100 impl. If we can't validate the reference, we can't validate our port.

### Phase 2 — MHC SM120 kernel v0 (2–3 weeks)
- [ ] Fork `sm90_tf32_hc_prenorm_gemm.cuh` → `sm120_tf32_hc_prenorm_gemm.cuh` (§4.1).
- [ ] Replace WGMMA with `mma.sync.m16n8k8.tf32` inline PTX.
- [ ] Wire the JIT glue and dispatch branch.
- [ ] Pass numerical tests within `rtol=5e-3`.
- [ ] V4 model loads and runs dummy forward on SM120 without the MHC crash.
- **Gate:** if V4 forward doesn't complete after Phase 2, freeze and investigate before continuing.

### Phase 3 — Lightning Indexer SM120 kernel v0 (3–5 weeks)
- Same sub-steps as Phase 2 but for `fp8_paged_mqa_logits`. Higher risk: FP8 numerics on SM120.
- **Gate:** V4 end-to-end generates sensible text (log-probability sanity check vs. cloud reference).

### Phase 4 — FP8 block-scaled linear (if Phase 0 said yes) (3–5 weeks)
- Skip if Marlin fallback works.

### Phase 5 — Performance tuning (2–4 weeks)
- Shape-specific heuristic table (`sm120_heuristics.py`).
- Double-buffer tuning, register pressure fixes, `ncu` deltas.
- Target: within 50% of what a naïve BF16 SM120 GEMM of the same shape can achieve (realistic SM120 ceiling).

### Phase 6 — Hardening (1–2 weeks)
- CUDA graph capture compatibility (drop `--enforce-eager`).
- `torch.compile` decomposition path (if we want to drop `--enforce-eager`, we may need to re-address the `auto_functionalized` Inductor issue too — that's a separate bug from the dsv4 branch).
- Documentation, fork README, version-pinning strategy.

---

## 7. Repo layout & fork strategy

We maintain two forks, independent from each other:

```
github.com/<you>/DeepGEMM-sm120   (fork of deepseek-ai/DeepGEMM @ 477618cd51)
  └── branch: sm120
      ├── csrc/apis/*.hpp                       (add arch_major==12 dispatch)
      ├── deep_gemm/include/.../sm120_*.cuh     (new kernels)
      ├── deep_gemm/jit_kernels/impls/sm120_*   (JIT templates)
      ├── deep_gemm/jit_kernels/heuristics/     (SM120 tables)
      └── tests/sm120/                          (correctness harness)

github.com/<you>/vllm-sm120-dsv4  (fork of vllm-project/vllm @ dsv4)
  └── branch: sm120-dsv4
      ├── tools/install_deepgemm.sh             (point at our DeepGEMM fork)
      ├── vllm/utils/deep_gemm.py               (loosen arch gates)
      └── vllm/model_executor/layers/mhc.py     (if arch gate needed)
```

**Version pinning:** each vLLM-sm120-dsv4 commit pins an exact DeepGEMM-sm120 commit. We treat this like a mono-repo even though it's two.

**Upstreaming:** do not attempt. DeepSeek declined SM120 upstream. We publish the fork, document clearly that it's community-maintained, and keep it synced with upstream DeepGEMM on a best-effort basis.

---

## 8. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FP8 numerics diverge non-trivially from SM100 | High | V4 outputs become subtly wrong (hallucinations beyond baseline) | Extensive correctness harness (§5.4); test with actual V4 logits against a cloud SM100 reference before declaring victory |
| 99 KB SMEM forces per-shape kernel variants | High | Heuristic table balloons | Plan for it from the start; design `sm120_heuristics.py` as a first-class artifact |
| `tcgen05` intrinsics on SM120 don't work as documented | Medium | Have to fall back to slower BF16 dequant path | Ship BF16 fallback first (correctness), optimize with tcgen05 later |
| DeepGEMM upstream changes cadence breaks our fork | Medium | Periodic rebases | Pin aggressively; plan 1 rebase cycle per quarter |
| vLLM dsv4 branch gets rewritten before merge | High | Have to re-apply our vLLM changes | Keep vLLM-side changes minimal and localized to clearly-marked files |
| `torch.compile` `auto_functionalized` bug still bites even after kernel port | Medium | Can't drop `--enforce-eager`; perf hit | Pin to eager mode in the launch script; investigate in Phase 6 |
| PCIe-only (no NVLink) on RTX 6000 Pro throttles 4-way EP+DP all-to-all | Low–medium (already 60–80 tok/s reported by others) | Lower throughput than expected | Measure early; consider fewer DP ranks |
| Kernel validates fine but runs <20% of B200 throughput | Medium | Users unhappy | Set expectations in README. SM120 is a workstation part. |
| Solo dev hits a wall on the FP8 indexer kernel | Medium | Project stalls at Phase 3 | Pre-commit to a hard gate: if Phase 3 takes > 6 weeks, cut losses and rent H-series for V4 |

---

## 9. Effort estimate (one kernel engineer, full-time)

| Phase | Optimistic | Realistic | Pessimistic |
|---|---|---|---|
| 0 — scoping | 3 days | 1 week | 2 weeks |
| 1 — harness | 1 week | 1.5 weeks | 3 weeks |
| 2 — MHC kernel | 2 weeks | 3 weeks | 5 weeks |
| 3 — indexer kernel | 3 weeks | 5 weeks | 10 weeks |
| 4 — FP8 GEMM (conditional) | 0 | 0–4 weeks | 6 weeks |
| 5 — perf tuning | 2 weeks | 3 weeks | 5 weeks |
| 6 — hardening | 1 week | 2 weeks | 3 weeks |
| **Total** | **~12 weeks** | **~16–20 weeks** | **~30+ weeks** |

Calendar time will be longer than engineer-weeks due to compile cycles, cross-referencing CUTLASS examples, and inevitable ISA surprises.

**Cost beyond time:**
- ~$200–500 in rented H100/SM100 time for golden-data generation and validation.
- Nothing else beyond your existing hardware.

---

## 10. Required reading / reference material

Bookmark these; we'll hit them repeatedly.

- **NVIDIA PTX ISA 8.5+** — the `mma.sync`, `tcgen05`, TMA sections.
- **CUTLASS `examples/92_blackwell_moe_gemm/`** — official SM100 reference for block-scaled FP8 MoE; good sanity check for our dispatch decisions.
- **CUTLASS `examples/77_blackwell_fmha`** — Blackwell FMHA reference; useful for the indexer kernel.
- **DeepGEMM @ 477618cd51** — our fork base. Read every `sm90_*.cuh` and `sm100_*.cuh` in scope before writing anything.
- **DeepGEMM issues #185, #236** — the maintainer's explicit reasoning for declining SM120.
- **DeepSeek-V4 technical report** (HuggingFace model card) — the MHC section specifically.
- **vLLM PR #40760** — our vLLM baseline. Review `deepseek_v4.py`, `mhc.py`, `deep_gemm.py`, `sparse_attn_indexer.*`.

---

## 11. Open questions / decision gates

Before we start, I need your answers on these:

1. **Solo or team?** The effort estimate assumes one engineer. If you want to parallelize, Phases 2 and 3 can run concurrently with two people; Phase 4 can run concurrently with Phase 2 or 3. Realistic with 2 engineers: 8–12 calendar weeks.

2. **Numerical tolerance policy.** Are we bit-exact (will not match SM100; TF32 rounding order differs) or "close enough" (`rtol=5e-3` on logits, with periodic end-to-end eval spot-checks against a cloud-hosted V4)? I strongly recommend "close enough" with eval gates.

3. **Are you OK renting an H100/B200 for a few days of validation work?** Non-optional for building a trustworthy correctness oracle.

4. **Fallback position.** If Phase 3 (the indexer kernel) stalls past its pessimistic estimate, do we (a) pivot to running on cloud H100s for V4 specifically, (b) switch to a non-DSA DeepSeek model via Marlin, or (c) push harder? I strongly recommend committing to (a) now as a pre-agreed exit.

5. **Publish the fork?** Even if only for ourselves, publishing means other SM120 users can help test. I recommend yes; minimal cost.

6. **Preferred programming cadence.** I can execute this autonomously in larger chunks, or we can review each PR before merge. Kernel code wants review; planning doesn't.

---

## 12. Immediate next actions (if we proceed)

Ordered. Each is a discrete PR/commit:

1. **Create `scripts/sm120_kernel_tests/` scaffolding** — empty pytest files, reference CPU impls for MHC and FP8 GEMM in pure PyTorch. (1–2 days)
2. **Fork DeepGEMM to `DeepGEMM-sm120` at ref `477618cd51`.** Verify it builds with CUDA 13.0. (½ day)
3. **Add the `arch_major == 12 → call_sm90_*(...)` fall-through experiment** — a 10-line patch to see whether the SM90 kernel happens to run on SM120 as-is. If yes (possible but I'd bet 30%), large chunks of Phase 2 evaporate. If no (likely), we have the crash we expect and proceed to write the SM120 kernel. (1 day)
4. **Update `deepseek-v4-flash.sh`** to set `VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_MARLIN_USE_ATOMIC_ADD=1` and re-run to confirm whether FP8 linear is already handled by Marlin. (1 hour)
5. **Phase 0 report** — a single markdown file recording what we found in steps 3–4, and a final kernel list. (½ day)

Only after step 5 do we commit to Phase 2+.

---

## 13. What I recommend you actually do right now

Given the scope, before committing three to six months of work, execute steps 3 and 4 above. They take less than a day combined and will:

- Tell us if the SM90 MHC kernel happens to work on SM120 (best-case outcome — shrinks Phase 2 dramatically).
- Tell us if Marlin handles all the FP8 linear work (very likely outcome — shrinks Phases 4 and 5 to zero).
- Give us a real-measured kernel inventory instead of the inferred one in §3.

If both experiments land favorably, the project becomes "port one kernel (MHC) + port one kernel (indexer)" over ~8 weeks, which is very different from the 16-week realistic estimate above.

If they land unfavorably, at least we'll know *exactly* what we signed up for before signing up.
