#!/usr/bin/env python3
"""Compare N separate exl3_gemm launches vs one exl3_mgemm.

Equal-N pairs (gate/up, k/v) use true widths. Unequal-N groups (q/k/v) pad
the mgemm trellis and C to max(N) because the current plugin ABI has no
size_n_list. That padded path is the cost of fusing without per-matrix
widths, not a serving-ready kernel.

Run on a free GPU after stopping TP4 serve. Physical 0 and 5 stay reserved:
  CUDA_VISIBLE_DEVICES=1 python "$EXL3/scripts/mgemm_microbench.py" --shapes kv
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugin" / "src"))

from vllm_exl3_sm86.constants import (  # noqa: E402
    BEHEMOTH_LAYERS,
    BEHEMOTH_TP4_SHAPES,
    GRAPH_CAPTURE_SIZES,
)

SHAPE_ALIASES = {
    "gate_up": ("gate_proj", "up_proj"),
    "kv": ("k_proj", "v_proj"),
    "qkv": ("q_proj", "k_proj", "v_proj"),
}


def resolve_shapes(spec: str) -> tuple[str, ...]:
    key = spec.strip()
    if key in SHAPE_ALIASES:
        return SHAPE_ALIASES[key]
    names = tuple(part.strip() for part in key.split(",") if part.strip())
    if len(names) < 2:
        raise ValueError(
            "need at least two shapes or an alias (gate_up, kv, qkv); "
            f"got {spec!r}"
        )
    unknown = [name for name in names if name not in BEHEMOTH_TP4_SHAPES]
    if unknown:
        raise ValueError(f"unknown shapes: {unknown}")
    ks = {BEHEMOTH_TP4_SHAPES[name][0] for name in names}
    if len(ks) != 1:
        raise ValueError(f"grouped shapes must share K; got {ks} for {names}")
    return names


def _sync():
    import torch

    torch.cuda.synchronize()


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _mgemm_signature() -> dict[str, object]:
    from vllm_exl3_sm86.ops import _load_exl3_ext

    ext = _load_exl3_ext()
    fn = getattr(ext, "exl3_mgemm", None)
    if fn is None:
        return {"present": False}
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    blob = " ".join(
        [
            " ".join(params),
            str(getattr(fn, "__doc__", "") or ""),
            str(getattr(fn, "__text_signature__", "") or ""),
        ]
    )
    return {
        "present": True,
        "param_names": params,
        "param_count": len(params),
        "has_size_n_list": "size_n_list" in blob,
        "has_c_ptrs": "c_ptrs" in blob,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--shapes",
        default="gate_up",
        help="Alias gate_up/kv/qkv or comma list sharing K, e.g. k_proj,v_proj.",
    )
    parser.add_argument("--bitrate", type=int, default=4)
    parser.add_argument(
        "--m",
        default="",
        help="Comma list of M. Default is GRAPH_CAPTURE_SIZES.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="JSON path. Default results/mgemm_vs_gemm_<shapes>.json",
    )
    args = parser.parse_args()
    names = resolve_shapes(args.shapes)
    k = BEHEMOTH_TP4_SHAPES[names[0]][0]
    widths = [BEHEMOTH_TP4_SHAPES[name][1] for name in names]
    padded = len(set(widths)) != 1
    max_n = max(widths)
    group = len(names)
    ms = (
        tuple(int(part) for part in args.m.split(",") if part.strip())
        if args.m.strip()
        else GRAPH_CAPTURE_SIZES
    )
    out_path = Path(
        args.output
        or (ROOT / "results" / f"mgemm_vs_gemm_{'_'.join(names)}.json")
    )
    import torch

    from vllm_exl3_sm86.ops import (
        _load_exl3_ext,
        call_exl3_gemm,
        call_exl3_mgemm,
        ext_has_mgemm,
    )

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    if not ext_has_mgemm():
        print("exllamav3_ext does not export exl3_mgemm", file=sys.stderr)
        return 2
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    _load_exl3_ext()
    mgemm_sig = _mgemm_signature()
    print(
        f"shapes={list(names)} K={k} Ns={widths} padded={int(padded)} "
        f"bitrate={args.bitrate} 3inst device={args.device} "
        f"size_n_list={mgemm_sig.get('has_size_n_list')}"
    )
    rows = []
    for m in ms:
        x = torch.randn(m, k, dtype=torch.float16, device=device)
        true_trellis = []
        suh = []
        svh = []
        outs = []
        xhads = []
        for n in widths:
            true_trellis.append(
                torch.randint(
                    -1024,
                    1024,
                    (k // 16, n // 16, 16 * args.bitrate),
                    dtype=torch.int16,
                    device=device,
                )
            )
            suh.append(torch.randn(k, dtype=torch.float16, device=device))
            svh.append(torch.randn(n, dtype=torch.float16, device=device))
            outs.append(torch.empty((m, n), dtype=torch.float16, device=device))
            xhads.append(torch.empty((m, k), dtype=torch.float16, device=device))
        if padded:
            mgemm_trellis = [
                torch.randint(
                    -1024,
                    1024,
                    (k // 16, max_n // 16, 16 * args.bitrate),
                    dtype=torch.int16,
                    device=device,
                )
                for _ in names
            ]
            mgemm_svh = [
                torch.randn(max_n, dtype=torch.float16, device=device)
                for _ in names
            ]
        else:
            mgemm_trellis = true_trellis
            mgemm_svh = svh
        gout = torch.empty(
            (group, m, max_n), dtype=torch.float16, device=device
        )
        gxhad = torch.empty((group, m, k), dtype=torch.float16, device=device)
        ptrs_t = torch.tensor(
            [int(t.data_ptr()) for t in mgemm_trellis],
            dtype=torch.int64,
            device=device,
        )
        ptrs_suh = torch.tensor(
            [int(s.data_ptr()) for s in suh],
            dtype=torch.int64,
            device=device,
        )
        ptrs_svh = torch.tensor(
            [int(s.data_ptr()) for s in mgemm_svh],
            dtype=torch.int64,
            device=device,
        )

        def many_gemms(
            trellis=true_trellis,
            suh_list=suh,
            svh_list=svh,
            out_list=outs,
            xhad_list=xhads,
        ):
            for trellis_i, suh_i, svh_i, out_i, xhad_i in zip(
                trellis, suh_list, svh_list, out_list, xhad_list
            ):
                call_exl3_gemm(
                    x, trellis_i, suh_i, svh_i, False, False,
                    out=out_i, x_had=xhad_i,
                )

        def one_mgemm():
            call_exl3_mgemm(
                x.view(1, m, k),
                ptrs_t,
                ptrs_suh,
                ptrs_svh,
                args.bitrate,
                False,
                False,
                gout,
                gxhad,
            )

        many_gemms()
        one_mgemm()
        _sync()
        t_gemm = _time_ms(many_gemms, args.warmup, args.iters)
        t_mgemm = _time_ms(one_mgemm, args.warmup, args.iters)
        layer_ms = {
            "m": m,
            "n_gemm_ms": t_gemm,
            "mgemm_ms": t_mgemm,
            "two_gemm_ms": t_gemm,
            "speedup": t_gemm / t_mgemm if t_mgemm else None,
            "n_gemm_88_ms": t_gemm * BEHEMOTH_LAYERS,
            "mgemm_88_ms": t_mgemm * BEHEMOTH_LAYERS,
            "two_gemm_88_ms": t_gemm * BEHEMOTH_LAYERS,
        }
        rows.append(layer_ms)
        print(
            f"M={m}  {group}_gemm={t_gemm:.3f}ms  mgemm={t_mgemm:.3f}ms  "
            f"speedup={layer_ms['speedup']:.3f}x  "
            f"88x gemm={layer_ms['n_gemm_88_ms']:.1f}ms  "
            f"88x mgemm={layer_ms['mgemm_88_ms']:.1f}ms"
        )
    payload = {
        "shapes": list(names),
        "k": k,
        "widths": widths,
        "padded_to_max_n": padded,
        "max_n": max_n,
        "bitrate": args.bitrate,
        "codebook": "3inst",
        "mgemm_signature": mgemm_sig,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
