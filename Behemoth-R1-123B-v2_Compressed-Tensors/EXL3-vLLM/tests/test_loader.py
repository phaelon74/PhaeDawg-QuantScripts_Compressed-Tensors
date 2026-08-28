from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm_exl3_sm86.config import Exl3Config
from vllm_exl3_sm86.linear import Exl3LinearMethod
from vllm_exl3_sm86.parameter import Exl3Parameter, exl3_weight_loader
from vllm_exl3_sm86.slicing import ShardId


def _make_payload(k: int, n: int, bitrate: int = 3):
    trellis = torch.zeros((k // 16, n // 16, 16 * bitrate), dtype=torch.int16)
    suh = torch.ones(k, dtype=torch.float16)
    svh = torch.ones(n, dtype=torch.float16)
    mul1 = torch.tensor([0x83DCD12D], dtype=torch.int32)
    return trellis, suh, svh, mul1


def test_independent_qkv_not_concatenated():
    trellis = Exl3Parameter(weight_loader=exl3_weight_loader)
    for shard, n in (("q", 768), ("k", 256), ("v", 256)):
        t, _, _, _ = _make_payload(512, n)
        trellis.load_exl3_weight(t, shard)
    assert set(trellis.exl3_tensors) == {"q", "k", "v"}
    assert trellis.exl3_tensors["q"].shape[1] != trellis.exl3_tensors["k"].shape[1]
    with pytest.raises(ValueError, match="twice"):
        trellis.load_exl3_weight(trellis.exl3_tensors["q"], "q")


def test_packed_loader_requires_shard_id():
    param = Exl3Parameter(weight_loader=exl3_weight_loader)
    param.exl3_requires_shard_id = True
    t, _, _, _ = _make_payload(256, 256)
    with pytest.raises(ValueError, match="no shard id"):
        exl3_weight_loader(param, t, None)


def test_state_dict_resave_rejected():
    param = Exl3Parameter(weight_loader=exl3_weight_loader)
    with pytest.raises(RuntimeError, match="state_dict"):
        param._save_to_state_dict({}, "x.", False)


def test_tp_column_and_row_slicing_for_ranks_0_3():
    quant = Exl3Config(
        bits=3.5,
        head_bits=6,
        codebook="mul1",
        tensor_storage={
            "model.layers.0.self_attn.o_proj": {
                "quant_format": "exl3",
                "bits_per_weight": 3,
                "stored_tensors": {
                    "model.layers.0.self_attn.o_proj.trellis": {"shape": [96, 96, 48]},
                    "model.layers.0.self_attn.o_proj.suh": {"shape": [1536]},
                    "model.layers.0.self_attn.o_proj.svh": {"shape": [1536]},
                    "model.layers.0.self_attn.o_proj.mul1": {"shape": [1]},
                },
            }
        },
    )
    method = Exl3LinearMethod(quant)
    for rank in range(4):
        layer = SimpleNamespace(
            prefix="model.layers.0.self_attn.o_proj",
            trellis=Exl3Parameter(weight_loader=exl3_weight_loader),
            suh=Exl3Parameter(weight_loader=exl3_weight_loader),
            svh=Exl3Parameter(weight_loader=exl3_weight_loader),
            mcg=Exl3Parameter(weight_loader=exl3_weight_loader),
            mul1=Exl3Parameter(weight_loader=exl3_weight_loader),
            su=Exl3Parameter(weight_loader=exl3_weight_loader),
            sv=Exl3Parameter(weight_loader=exl3_weight_loader),
            exl3_tp_rank=rank,
            exl3_tp_size=4,
            exl3_parallel_mode="row",
            exl3_input_size=512,
            exl3_input_size_per_partition=128,
            exl3_output_partition_sizes=[512],
            exl3_shard_ids=[None],
            exl3_expected_codebooks={None: "mul1"},
            num_kv_head_replicas=1,
        )
        trellis, suh, svh, mul1 = _make_payload(512, 512, 3)
        layer.trellis.load_exl3_weight(trellis, None)
        layer.suh.load_exl3_weight(suh, None)
        layer.svh.load_exl3_weight(svh, None)
        layer.mul1.load_exl3_weight(mul1, None)
        method._shard_tensors_for_tensor_parallel(layer)
        assert layer.suh.exl3_tensors[None].numel() == 128
        assert layer.trellis.exl3_tensors[None].shape[0] == 8
        assert layer.svh.exl3_tensors[None].numel() == 512
