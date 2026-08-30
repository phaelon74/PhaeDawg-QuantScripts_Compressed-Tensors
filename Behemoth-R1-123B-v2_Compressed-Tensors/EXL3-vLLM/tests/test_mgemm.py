from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_exl3_sm86.grouped import matching_pair_shards, mgemm_disabled
from vllm_exl3_sm86.ops import register_custom_op


def _pair_layer(
    n0: int = 256,
    n1: int = 256,
    bits0: int = 4,
    bits1: int = 4,
    *,
    mcg: tuple[bool, bool] = (False, False),
    mul1: tuple[bool, bool] = (False, False),
    shard_ids: list | None = None,
):
    k = 256
    ids = list(shard_ids) if shard_ids is not None else [0, 1]

    def trellis(n: int, bits: int) -> torch.Tensor:
        return torch.zeros((k // 16, n // 16, 16 * bits), dtype=torch.int16)

    mcg_tensors = {}
    mul1_tensors = {}
    flags_mcg = mcg if len(mcg) == len(ids) else (False,) * len(ids)
    flags_mul1 = mul1 if len(mul1) == len(ids) else (False,) * len(ids)
    trellis_map = {}
    sizes = []
    bits = [bits0, bits1] + [4] * max(0, len(ids) - 2)
    ns = [n0, n1] + [n0] * max(0, len(ids) - 2)
    for i, shard in enumerate(ids):
        trellis_map[shard] = trellis(ns[i], bits[i])
        sizes.append(ns[i])
        if flags_mcg[i]:
            mcg_tensors[shard] = torch.zeros(1, dtype=torch.int32)
        if flags_mul1[i]:
            mul1_tensors[shard] = torch.zeros(1, dtype=torch.int32)
    return SimpleNamespace(
        exl3_shard_ids=ids,
        exl3_output_partition_sizes=sizes,
        trellis=SimpleNamespace(exl3_tensors=trellis_map),
        mcg=SimpleNamespace(exl3_tensors=mcg_tensors),
        mul1=SimpleNamespace(exl3_tensors=mul1_tensors),
    )


def test_gate_up_matching_k4_is_eligible():
    assert matching_pair_shards(_pair_layer()) == [0, 1]


def test_qkv_three_shards_not_eligible():
    layer = _pair_layer(shard_ids=["q", "k", "v"], n0=3072, n1=256)
    layer.exl3_output_partition_sizes = [3072, 256, 256]
    layer.trellis.exl3_tensors["v"] = torch.zeros((16, 16, 64), dtype=torch.int16)
    assert matching_pair_shards(layer) is None


def test_mismatched_bitrate_not_eligible():
    assert matching_pair_shards(_pair_layer(bits0=4, bits1=6)) is None


def test_mismatched_width_not_eligible():
    assert matching_pair_shards(_pair_layer(n0=256, n1=512)) is None


def test_mismatched_codebook_not_eligible():
    assert (
        matching_pair_shards(_pair_layer(mul1=(False, True))) is None
    )


def test_mgemm_disable_env(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_DISABLE_MGEMM", raising=False)
    assert mgemm_disabled() is False
    monkeypatch.setenv("VLLM_EXL3_DISABLE_MGEMM", "1")
    assert mgemm_disabled() is True


def test_custom_op_registers_mgemm_out():
    register_custom_op()
    assert hasattr(torch.ops.vllm, "exl3_mgemm_out")
