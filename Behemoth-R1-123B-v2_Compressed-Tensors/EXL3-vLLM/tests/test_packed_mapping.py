from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm_exl3_sm86.config import Exl3Config
from vllm_exl3_sm86.linear import Exl3LinearMethod


def _storage_for_layer0() -> dict:
    def rec(shape):
        return {
            "quant_format": "exl3",
            "bits_per_weight": shape[-1] // 16,
            "stored_tensors": {
                "x.trellis": {"shape": shape},
                "x.suh": {"shape": [shape[0] * 16]},
                "x.svh": {"shape": [shape[1] * 16]},
                "x.mul1": {"shape": [1]},
            },
        }

    return {
        "model.layers.0.self_attn.q_proj": rec([768, 192, 48]),
        "model.layers.0.self_attn.k_proj": rec([768, 16, 48]),
        "model.layers.0.self_attn.v_proj": rec([768, 16, 48]),
        "model.layers.0.mlp.gate_proj": rec([768, 448, 64]),
        "model.layers.0.mlp.up_proj": rec([768, 448, 64]),
        "lm_head": rec([768, 512, 96]),
        "model.embed_tokens": {"quant_format": "unquantized", "stored_tensors": {}},
    }


def test_packed_qkv_and_gate_up_source_prefixes():
    cfg = Exl3Config(tensor_storage=_storage_for_layer0())
    method = Exl3LinearMethod(cfg)
    qkv = SimpleNamespace(prefix="model.layers.0.self_attn.qkv_proj")
    assert method._source_prefixes_for_layer(qkv, ["q", "k", "v"]) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    gate = SimpleNamespace(prefix="model.layers.0.mlp.gate_up_proj")
    assert method._source_prefixes_for_layer(gate, [0, 1]) == [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ]


def test_unquantized_embeddings_and_quantized_head():
    cfg = Exl3Config(tensor_storage=_storage_for_layer0())
    assert cfg._linear_prefix_is_exl3("lm_head")
    assert not cfg._linear_prefix_is_exl3("model.embed_tokens")
    assert cfg._linear_prefix_is_exl3("model.layers.0.self_attn.qkv_proj")
    assert cfg.codebook_for_prefix("lm_head") == "mul1"
    assert cfg.bitrate_for_prefix("lm_head") == 6
