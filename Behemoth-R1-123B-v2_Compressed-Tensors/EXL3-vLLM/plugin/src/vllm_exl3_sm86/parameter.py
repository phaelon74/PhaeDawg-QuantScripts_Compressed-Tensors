"""Independent EXL3 payload parameters. Never concatenate packed trellises."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.parameter import BasevLLMParameter

from .slicing import ShardId, slice_on_load

if TYPE_CHECKING:
    pass


class Exl3Parameter(BasevLLMParameter):
    """Zero-sized parameter holding independently shaped EXL3 components."""

    def __new__(cls, *, weight_loader):
        data = torch.empty(0, dtype=torch.uint8)
        return super().__new__(cls, data=data, weight_loader=weight_loader)

    def __init__(self, *, weight_loader):
        self.exl3_tensors: dict[ShardId, torch.Tensor] = {}
        self.exl3_field: str | None = None
        self.exl3_layer = None
        super().__init__(data=self.data, weight_loader=weight_loader)

    def load_exl3_weight(
        self,
        loaded_weight: torch.Tensor,
        shard_id: ShardId = None,
    ) -> None:
        if shard_id in self.exl3_tensors:
            raise ValueError(
                f"EXL3 payload for shard {shard_id!r} was loaded twice; "
                "refusing to overwrite a packed source matrix."
            )
        self.exl3_tensors[shard_id] = loaded_weight.contiguous()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        raise RuntimeError(
            "EXL3 side-dictionary payloads cannot be saved through "
            "state_dict(); resave/sleep is unsupported."
        )


def exl3_weight_loader(
    param: Exl3Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: ShardId = None,
) -> None:
    if loaded_shard_id is None and getattr(param, "exl3_requires_shard_id", False):
        raise ValueError(
            "EXL3 packed layer loader received no shard id. The pinned vLLM "
            "runtime must pass q/k/v or 0/1 into the weight loader."
        )
    layer = getattr(param, "exl3_layer", None)
    field = getattr(param, "exl3_field", None)
    if layer is not None and field:
        loaded_weight = slice_on_load(layer, field, loaded_weight, loaded_shard_id)
        layer.exl3_sliced_on_load = True
    param.load_exl3_weight(loaded_weight, loaded_shard_id)
