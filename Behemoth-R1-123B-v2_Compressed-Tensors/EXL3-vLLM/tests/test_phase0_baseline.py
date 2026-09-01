from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mgemm_shape_aliases():
    mgemm = _load("mgemm_microbench")
    assert mgemm.resolve_shapes("gate_up") == ("gate_proj", "up_proj")
    assert mgemm.resolve_shapes("kv") == ("k_proj", "v_proj")
    assert mgemm.resolve_shapes("qkv") == ("q_proj", "k_proj", "v_proj")
    assert mgemm.resolve_shapes("k_proj,v_proj") == ("k_proj", "v_proj")
    with pytest.raises(ValueError, match="at least two"):
        mgemm.resolve_shapes("gate_proj")
    with pytest.raises(ValueError, match="unknown"):
        mgemm.resolve_shapes("gate_proj,not_a_proj")
    with pytest.raises(ValueError, match="share K"):
        mgemm.resolve_shapes("q_proj,o_proj")


def test_kernel_path_matches_overlay_policy():
    budget = _load("decode_latency_budget")
    assert budget.kernel_path("gate_proj", 4) == "gemv_narrow"
    assert budget.kernel_path("up_proj", 4) == "gemv_narrow"
    assert budget.kernel_path("down_proj", 4) == "gemv_narrow"
    assert budget.kernel_path("o_proj", 4) == "gemv_narrow"
    assert budget.kernel_path("q_proj", 4) == "regular"
    assert budget.kernel_path("k_proj", 4) == "regular"
    assert budget.kernel_path("v_proj", 4) == "regular"
    assert budget.kernel_path("gate_proj", 5) == "regular"
    assert budget.kernel_path("gate_proj", 6) == "regular"
    assert budget.kernel_path("lm_head", 6) == "regular"


def test_mixed_k_inventory_budget(tmp_path, monkeypatch, capsys):
    budget = _load("decode_latency_budget")
    micro = {
        "codebook": "3inst",
        "rows": [
            {
                "name": leaf,
                "bitrate": bitrate,
                "m": 1,
                "exl3_plugin_ms": 0.2 if bitrate == 4 else 0.25,
                "exl3_native_persistent_ms": 0.1 if bitrate == 4 else 0.13,
            }
            for leaf in budget.DECODER_LEAVES
            for bitrate in (4, 5, 6)
        ],
    }
    inventory = {
        "checkpoint": "synthetic",
        "decoder_linears": 616,
        "head": {"bitrate": 6},
        "decoder_leaf_bitrates": {
            "q_proj": {"4": 80, "5": 8},
            "k_proj": {"6": 88},
            "v_proj": {"6": 88},
            "o_proj": {"4": 88},
            "gate_proj": {"4": 70, "5": 18},
            "up_proj": {"4": 70, "5": 18},
            "down_proj": {"4": 88},
        },
    }
    micro_path = tmp_path / "micro.json"
    inv_path = tmp_path / "inv.json"
    out_path = tmp_path / "budget.json"
    micro_path.write_text(json.dumps(micro))
    inv_path.write_text(json.dumps(inventory))
    monkeypatch.setattr(
        "sys.argv",
        [
            "decode_latency_budget.py",
            "--microbench",
            str(micro_path),
            "--inventory",
            str(inv_path),
            "--output",
            str(out_path),
        ],
    )
    rc = budget.main()
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["k56_regular_fallback"]["layers"] == 8 + 88 + 88 + 18 + 18
    assert report["k56_regular_fallback"]["extra_ms"] == pytest.approx(
        (8 + 88 + 88 + 18 + 18) * 0.03
    )
    gate_k5 = next(
        r for r in report["rows"] if r["name"] == "gate_proj" and r["bitrate"] == 5
    )
    assert gate_k5["kernel_path"] == "regular"
    gate_k4 = next(
        r for r in report["rows"] if r["name"] == "gate_proj" and r["bitrate"] == 4
    )
    assert gate_k4["kernel_path"] == "gemv_narrow"
    assert "lm_head" in report["warnings"][0]
    assert report["mgemm_fusion"]["source"] == "marginal_counts"
    assert report["mgemm_fusion"]["kv_fuse_layers_max"] == 88
    capsys.readouterr()


def test_fusion_from_per_layer_inventory():
    budget = _load("decode_latency_budget")
    layers = []
    for i in range(88):
        row = {
            "layer": i,
            "q_proj": 4 if i < 60 else 5,
            "k_proj": 5 if i < 30 else 6,
            "v_proj": 5 if i < 15 else 6,
            "o_proj": 4,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        }
        layers.append(row)
    report = budget.fusion_from_layers(layers)
    assert report["source"] == "per_layer"
    assert report["kv_fuse_layers"] == 15 + 58
    assert report["qkv_fuse_layers"] == 0
    assert report["kv_fuse_by_bitrate"] == {"5": 15, "6": 58}
    assert report["gate_up_fuse_layers"] == 88


def test_mgemm_abi_parses_pinned_decl():
    abi = _load("mgemm_abi")
    text = """
int exl3_mgemm_gr(const at::Tensor& A, int num_tokens = 1);
int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {}
);
"""
    names = abi.parse_exl3_mgemm_params(text)
    assert names[-2:] == ["size_n_list", "c_ptrs"]
    assert "K" in names
    report = abi.describe_abi(text)
    assert report["has_size_n_list"] is True
    assert report["has_c_ptrs"] is True
    assert report["per_matrix_bitrate"] is False
    pinned = abi.describe_abi()
    assert pinned["has_size_n_list"] is True
    assert pinned["bitrate_is_scalar_K"] is True


def test_nsys_kernel_buckets(tmp_path):
    nsys = _load("summarize_nsys")
    csv_path = tmp_path / "kern_sum.csv"
    csv_path.write_text(
        "Time (%),Total Time (ns),Instances,Name\n"
        "50,10000000,88,void exl3_gemm_kernel<4>(half const*, ...)\n"
        "10,2000000,73,void exl3_mgemm_kernel<4>(...)\n"
        "20,4000000,176,ncclDevKernel_AllReduce_Sum_f16\n"
        "15,3000000,88,flash_fwd_kernel\n"
        "5,1000000,88,rms_norm_kernel\n"
    )
    rows = nsys.parse_kern_sum(csv_path)
    report = nsys.summarize(rows, decode_tokens=32)
    assert report["ms_by_bucket"]["exl3_gemm"] == pytest.approx(10.0)
    assert report["ms_by_bucket"]["exl3_mgemm"] == pytest.approx(2.0)
    assert report["ms_by_bucket"]["comm"] == pytest.approx(4.0)
    assert report["ms_by_bucket"]["attention"] == pytest.approx(3.0)
    assert report["ms_per_token_by_bucket"]["exl3_gemm"] == pytest.approx(10.0 / 32)
    log_path = tmp_path / "serve.log"
    log_path.write_text(
        "vllm_exl3_sm86: fused exl3_mgemm decode enabled for K/V pairs\n"
    )
    flags = nsys.parse_serve_log(log_path)
    assert flags["kv_mgemm_enabled"] is True
    assert flags["pair_mgemm_enabled"] is False
