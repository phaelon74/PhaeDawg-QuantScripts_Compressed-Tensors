from __future__ import annotations

import pytest

from vllm_exl3_sm86.metadata import (
    bitrate_from_trellis_shape,
    codebook_from_suffixes,
    expand_prefix_candidates,
    storage_entry,
    validate_storage_metadata,
)


def _entry(suffixes: list[str], shape=(16, 16, 64), bits=4) -> dict:
    stored = {f"layer.{name}": {"shape": shape if name == "trellis" else [shape[0] * 16]} for name in suffixes}
    stored["layer.trellis"]["shape"] = list(shape)
    return {
        "quant_format": "exl3",
        "bits_per_weight": bits,
        "stored_tensors": stored,
    }


def test_bitrate_from_trellis_shape():
    assert bitrate_from_trellis_shape((768, 192, 48)) == 3
    assert bitrate_from_trellis_shape((768, 192, 64)) == 4
    assert bitrate_from_trellis_shape((768, 512, 96)) == 6
    with pytest.raises(ValueError):
        bitrate_from_trellis_shape((768, 192, 50))
    with pytest.raises(ValueError):
        bitrate_from_trellis_shape((768, 192))


def test_codebook_conflict():
    assert codebook_from_suffixes({"trellis", "suh", "svh", "mul1"}) == "mul1"
    assert codebook_from_suffixes({"trellis", "suh", "svh", "mcg"}) == "mcg"
    with pytest.raises(ValueError):
        codebook_from_suffixes({"mcg", "mul1"})


def test_validate_storage_accepts_dense_mistral_map():
    storage = {
        "model.layers.0.self_attn.q_proj": _entry(["trellis", "suh", "svh", "mul1"], (768, 192, 48), 3),
        "model.layers.0.self_attn.k_proj": _entry(["trellis", "suh", "svh", "mul1"], (768, 16, 48), 3),
        "model.layers.0.self_attn.v_proj": _entry(["trellis", "suh", "svh", "mul1"], (768, 16, 64), 4),
        "model.layers.0.mlp.gate_proj": _entry(["trellis", "suh", "svh", "mul1"]),
        "model.layers.0.mlp.up_proj": _entry(["trellis", "suh", "svh", "mul1"]),
        "lm_head": _entry(["trellis", "suh", "svh", "mul1"], (768, 512, 96), 6),
        "model.embed_tokens": {"quant_format": "unquantized", "stored_tensors": {}},
    }
    assert validate_storage_metadata(storage) == 6


def test_validate_storage_rejects_missing_and_both_markers():
    with pytest.raises(ValueError, match="no EXL3"):
        validate_storage_metadata({"a": {"quant_format": "bf16", "stored_tensors": {}}})
    bad = {
        "layer": {
            "quant_format": "exl3",
            "stored_tensors": {
                "layer.trellis": {"shape": [16, 16, 64]},
                "layer.suh": {"shape": [256]},
                "layer.mcg": {"shape": [1]},
                "layer.mul1": {"shape": [1]},
            },
        }
    }
    with pytest.raises(ValueError, match="both mcg and mul1"):
        validate_storage_metadata(bad)


def test_prefix_candidates_and_lookup():
    storage = {"lm_head": _entry(["trellis", "suh", "svh", "mul1"])}
    assert "model.lm_head" in expand_prefix_candidates("lm_head")
    assert storage_entry(storage, "model.lm_head") is not None
    assert storage_entry(storage, "language_model.lm_head") is not None
