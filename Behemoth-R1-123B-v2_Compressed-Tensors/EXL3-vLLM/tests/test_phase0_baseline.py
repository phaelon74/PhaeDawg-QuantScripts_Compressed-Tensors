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
    capsys.readouterr()
