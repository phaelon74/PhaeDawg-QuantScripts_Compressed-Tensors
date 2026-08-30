"""Dense EXL3 linear method: independent Q/K/V; fused gate/up decode via mgemm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
)

from .constants import (
    GRAPH_CAPTURE_SIZES,
    HADAMARD_BLOCK,
    MCG_SENTINEL,
    MUL1_SENTINEL,
    TRELLIS_TILE,
)
from .grouped import enable_mgemm_if_eligible
from .nvtx import nvtx_range
from .ops import call_exl3_gemm, call_exl3_mgemm
from .parameter import Exl3Parameter, exl3_weight_loader
from .slicing import (
    ShardId,
    qkv_output_start,
    slice_trellis,
    slice_vector,
    unpack_signs,
)

if TYPE_CHECKING:
    from .config import Exl3Config


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: Exl3Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if layer.__class__.__name__ == "ParallelLMHead":
            org = getattr(layer, "org_vocab_size", None)
            total = getattr(layer, "num_embeddings", None)
            if org is not None and total is not None and org != total:
                raise NotImplementedError(
                    "EXL3 lm_head with added vocabulary is unsupported: the "
                    f"trellis tensor covers the original {org} rows but the "
                    f"layer allocates {total}."
                )
        if isinstance(layer, ReplicatedLinear):
            layer.exl3_tp_rank = 0
            layer.exl3_tp_size = 1
        else:
            layer.exl3_tp_rank = getattr(
                layer, "tp_rank", get_tensor_model_parallel_rank()
            )
            layer.exl3_tp_size = getattr(
                layer, "tp_size", get_tensor_model_parallel_world_size()
            )
        layer.exl3_input_size = input_size
        layer.exl3_input_size_per_partition = input_size_per_partition
        layer.exl3_output_size = output_size
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer.exl3_shard_ids = self._shard_ids_for_layer(layer, output_partition_sizes)
        layer.exl3_parallel_mode = (
            "row" if input_size_per_partition != input_size else "column"
        )
        source_prefixes = self._source_prefixes_for_layer(layer, layer.exl3_shard_ids)
        layer.exl3_expected_codebooks = {
            shard_id: self.quant_config.codebook_for_prefix(source_prefix)
            for shard_id, source_prefix in zip(
                layer.exl3_shard_ids, source_prefixes, strict=True
            )
        }
        requires_shard = len(layer.exl3_shard_ids) > 1
        layer.exl3_sliced_on_load = False
        for name in ("suh", "svh", "su", "sv", "trellis", "mcg", "mul1"):
            param = Exl3Parameter(weight_loader=exl3_weight_loader)
            param.exl3_requires_shard_id = requires_shard
            param.exl3_field = name
            param.exl3_layer = layer
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._materialize_legacy_hadamard(layer)
        missing: list[str] = []
        for attr in ("suh", "svh", "trellis"):
            param = getattr(layer, attr)
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in param.exl3_tensors:
                    missing.append(f"{attr}[{shard_id!r}]")
        for shard_id in layer.exl3_shard_ids:
            expected = layer.exl3_expected_codebooks[shard_id]
            has_mcg = shard_id in layer.mcg.exl3_tensors
            has_mul1 = shard_id in layer.mul1.exl3_tensors
            if has_mcg and has_mul1:
                missing.append(f"codebook[{shard_id!r}]=both mcg and mul1")
            elif expected == "mcg" and not has_mcg:
                missing.append(f"mcg[{shard_id!r}]")
            elif expected == "mul1" and not has_mul1:
                missing.append(f"mul1[{shard_id!r}]")
            elif expected is None and (has_mcg or has_mul1):
                missing.append(f"unexpected codebook[{shard_id!r}]")
        if missing:
            prefix = getattr(layer, "prefix", layer.__class__.__name__)
            raise ValueError(
                f"Missing or inconsistent EXL3 tensors for {prefix}: "
                + ", ".join(missing)
            )

        self._validate_loaded_tensors(layer)
        if not getattr(layer, "exl3_sliced_on_load", False):
            self._shard_tensors_for_tensor_parallel(layer)
            self._validate_loaded_tensors(layer)

        device = layer.trellis.device
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            param = getattr(layer, attr)
            for shard_id, tensor in list(param.exl3_tensors.items()):
                param.exl3_tensors[shard_id] = tensor.to(
                    device=device, non_blocking=True
                ).contiguous()
        self._allocate_decode_workspaces(layer)
        enable_mgemm_if_eligible(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape[:-1]
        original_dtype = x.dtype
        with nvtx_range("exl3.apply"):
            x_2d = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
            mgemm_ws = getattr(layer, "exl3_mgemm_out_ws", None)
            if (
                getattr(layer, "exl3_use_mgemm", False)
                and mgemm_ws is not None
                and x_2d.shape[0] in mgemm_ws
            ):
                output = self._apply_mgemm(layer, x_2d)
            else:
                outputs = [
                    self._apply_one(layer, x_2d, shard_id)
                    for shard_id in layer.exl3_shard_ids
                ]
                output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
            if bias is not None:
                output = output + bias.to(dtype=output.dtype)
            output = output.reshape(*original_shape, output.shape[-1])
            return output if output.dtype == original_dtype else output.to(original_dtype)

    @classmethod
    def _materialize_legacy_hadamard(cls, layer: torch.nn.Module) -> None:
        for packed_name, half_name in (("su", "suh"), ("sv", "svh")):
            packed = getattr(layer, packed_name).exl3_tensors
            half = getattr(layer, half_name).exl3_tensors
            for shard_id in layer.exl3_shard_ids:
                if shard_id not in half and shard_id in packed:
                    half[shard_id] = unpack_signs(packed[shard_id])

    @staticmethod
    def _validate_marker(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"EXL3 {name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != expected:
            raise ValueError(
                f"Invalid EXL3 {name} sentinel 0x{value:08x}; expected 0x{expected:08x}"
            )

    @classmethod
    def _validate_loaded_tensors(cls, layer: torch.nn.Module) -> None:
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            suh = layer.suh.exl3_tensors[shard_id]
            svh = layer.svh.exl3_tensors[shard_id]
            if trellis.dtype != torch.int16 or trellis.ndim != 3:
                raise ValueError("EXL3 trellis must be rank-3 int16")
            bitrate_dim = trellis.shape[2]
            if bitrate_dim % TRELLIS_TILE or not 1 <= bitrate_dim // TRELLIS_TILE <= 8:
                raise ValueError(
                    f"Invalid EXL3 trellis bit width {bitrate_dim} / {TRELLIS_TILE}"
                )
            if suh.dtype != torch.float16 or suh.ndim != 1:
                raise ValueError("EXL3 suh must be rank-1 float16")
            if svh.dtype != torch.float16 or svh.ndim != 1:
                raise ValueError("EXL3 svh must be rank-1 float16")
            k = trellis.shape[0] * TRELLIS_TILE
            n = trellis.shape[1] * TRELLIS_TILE
            if suh.numel() != k or svh.numel() != n:
                raise ValueError(
                    "EXL3 dimensions disagree: "
                    f"trellis={tuple(trellis.shape)}, suh={suh.numel()}, "
                    f"svh={svh.numel()}"
                )
            if k % HADAMARD_BLOCK or n % HADAMARD_BLOCK:
                raise ValueError(
                    f"EXL3 kernel dimensions must be {HADAMARD_BLOCK}-aligned, "
                    f"got K={k}, N={n}"
                )
            if shard_id in layer.mcg.exl3_tensors:
                cls._validate_marker(
                    layer.mcg.exl3_tensors[shard_id], MCG_SENTINEL, "mcg"
                )
            if shard_id in layer.mul1.exl3_tensors:
                cls._validate_marker(
                    layer.mul1.exl3_tensors[shard_id], MUL1_SENTINEL, "mul1"
                )

    @staticmethod
    def _output_shard_size(layer: torch.nn.Module, shard_id: ShardId) -> int:
        if shard_id is None:
            return layer.exl3_output_partition_sizes[0]
        if isinstance(shard_id, str) and shard_id in ("q", "k", "v"):
            return layer.exl3_output_partition_sizes[{"q": 0, "k": 1, "v": 2}[shard_id]]
        if isinstance(shard_id, int):
            return layer.exl3_output_partition_sizes[shard_id]
        return layer.exl3_output_partition_sizes[layer.exl3_shard_ids.index(shard_id)]

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls, layer: torch.nn.Module) -> None:
        if layer.exl3_tp_size == 1:
            return
        if layer.exl3_parallel_mode == "row":
            start = layer.exl3_tp_rank * layer.exl3_input_size_per_partition
            size = layer.exl3_input_size_per_partition
            for shard_id in layer.exl3_shard_ids:
                layer.suh.exl3_tensors[shard_id] = slice_vector(
                    layer.suh.exl3_tensors[shard_id], start, size
                )
                layer.trellis.exl3_tensors[shard_id] = slice_trellis(
                    layer.trellis.exl3_tensors[shard_id],
                    dim=0,
                    start=start,
                    size=size,
                )
            return

        for shard_id in layer.exl3_shard_ids:
            size = cls._output_shard_size(layer, shard_id)
            replicas = int(getattr(layer, "num_kv_head_replicas", 1))
            start = qkv_output_start(
                shard_id, size, layer.exl3_tp_rank, replicas
            )
            layer.svh.exl3_tensors[shard_id] = slice_vector(
                layer.svh.exl3_tensors[shard_id], start, size
            )
            layer.trellis.exl3_tensors[shard_id] = slice_trellis(
                layer.trellis.exl3_tensors[shard_id],
                dim=1,
                start=start,
                size=size,
            )

    @staticmethod
    def _shard_ids_for_layer(
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
    ) -> list[ShardId]:
        if len(output_partition_sizes) == 1:
            return [None]
        if isinstance(layer, QKVParallelLinear) and len(output_partition_sizes) == 3:
            return ["q", "k", "v"]
        return list(range(len(output_partition_sizes)))

    def _source_prefixes_for_layer(
        self, layer: torch.nn.Module, shard_ids: list[ShardId]
    ) -> list[str]:
        prefix = getattr(layer, "prefix", "")
        if len(shard_ids) == 1:
            return [prefix or "lm_head"]
        leaf = prefix.rsplit(".", 1)[-1]
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        sources = getattr(self.quant_config, "packed_modules_mapping", {}).get(leaf)
        if sources and len(sources) == len(shard_ids):
            return [f"{base}.{source}" if base else source for source in sources]
        raise ValueError(
            f"EXL3 does not know the source matrices for packed layer {prefix}; "
            "add it to the model's packed_modules_mapping."
        )

    @classmethod
    def _allocate_decode_workspaces(cls, layer: torch.nn.Module) -> None:
        """Stable per-shard decode buffers so CUDA graphs never capture malloc."""
        max_m = max(GRAPH_CAPTURE_SIZES)
        layer.exl3_decode_max_m = max_m
        layer.exl3_out_ws = {}
        layer.exl3_xhad_ws = {}
        for shard_id in layer.exl3_shard_ids:
            trellis = layer.trellis.exl3_tensors[shard_id]
            k = int(trellis.shape[0] * TRELLIS_TILE)
            n = cls._output_shard_size(layer, shard_id)
            packed_n = int(trellis.shape[1] * TRELLIS_TILE)
            n = max(n, packed_n)
            device = trellis.device
            layer.exl3_out_ws[shard_id] = torch.empty(
                (max_m, n), dtype=torch.float16, device=device
            )
            layer.exl3_xhad_ws[shard_id] = torch.empty(
                (max_m, k), dtype=torch.float16, device=device
            )

    @staticmethod
    def _apply_mgemm(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        m = int(x.shape[0])
        packed_k = int(layer.exl3_mgemm_k)
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            padded = layer.exl3_mgemm_x_ws[m]
            padded.zero_()
            padded[:, : x.shape[-1]].copy_(x)
            x = padded
        out = layer.exl3_mgemm_out_ws[m]
        packed = layer.exl3_mgemm_packed_ws[m]
        call_exl3_mgemm(
            x.view(1, m, packed_k),
            layer.exl3_mgemm_ptrs_trellis,
            layer.exl3_mgemm_ptrs_suh,
            layer.exl3_mgemm_ptrs_svh,
            layer.exl3_mgemm_bitrate,
            layer.exl3_mgemm_mcg,
            layer.exl3_mgemm_mul1,
            out,
            layer.exl3_mgemm_xhad_ws[m],
        )
        n0 = Exl3LinearMethod._output_shard_size(layer, layer.exl3_mgemm_shards[0])
        n1 = Exl3LinearMethod._output_shard_size(layer, layer.exl3_mgemm_shards[1])
        packed[:, :n0].copy_(out[0, :, :n0])
        packed[:, n0 : n0 + n1].copy_(out[1, :, :n1])
        return packed[:, : n0 + n1]

    @staticmethod
    def _apply_one(
        layer: torch.nn.Module, x: torch.Tensor, shard_id: ShardId
    ) -> torch.Tensor:
        trellis = layer.trellis.exl3_tensors[shard_id]
        packed_k = trellis.shape[0] * TRELLIS_TILE
        if x.shape[-1] > packed_k:
            raise ValueError(
                f"EXL3 input width {x.shape[-1]} exceeds packed K={packed_k}"
            )
        if x.shape[-1] < packed_k:
            x = torch.nn.functional.pad(x, (0, packed_k - x.shape[-1]))
        out = None
        x_had = None
        max_m = int(getattr(layer, "exl3_decode_max_m", 0))
        if 0 < x.shape[0] <= max_m:
            out = layer.exl3_out_ws[shard_id][: x.shape[0]]
            x_had = layer.exl3_xhad_ws[shard_id][: x.shape[0]]
            if x_had.shape[-1] != x.shape[-1]:
                x_had = None
                out = None
        output = call_exl3_gemm(
            x,
            trellis,
            layer.suh.exl3_tensors[shard_id],
            layer.svh.exl3_tensors[shard_id],
            shard_id in layer.mcg.exl3_tensors,
            shard_id in layer.mul1.exl3_tensors,
            out=out,
            x_had=x_had,
        )
        logical_n = Exl3LinearMethod._output_shard_size(layer, shard_id)
        if output.shape[-1] < logical_n:
            raise ValueError(
                f"EXL3 packed N={output.shape[-1]} is below logical N={logical_n}"
            )
        return output[..., :logical_n]
