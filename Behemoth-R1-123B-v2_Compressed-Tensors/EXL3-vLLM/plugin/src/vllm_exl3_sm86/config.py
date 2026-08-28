"""Dense EXL3 QuantizationConfig. No MoE / SparkInfer / rank-sliced experts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from .constants import QUANTIZATION_NAME
from .graph import graphs_allowed
from .linear import Exl3LinearMethod
from .metadata import codebook_from_suffixes, storage_entry, stored_suffixes, validate_storage_metadata
from .ops import register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


def _read_quantization_config_json(model_name: str, revision: str | None) -> dict[str, Any]:
    local = Path(model_name) / "quantization_config.json"
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    try:
        from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

        config = get_hf_file_to_dict(
            "quantization_config.json", model_name, revision=revision
        )
        return config or {}
    except Exception:
        return {}


class Exl3Config(QuantizationConfig):
    """Configuration for dense EXL3 trellis checkpoints."""

    def __init__(
        self,
        bits: float | None = None,
        head_bits: float | None = None,
        codebook: str | None = None,
        version: str | None = None,
        tensor_storage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.head_bits = head_bits
        self.codebook = codebook
        self.version = version
        self.tensor_storage = tensor_storage or {}
        self._eager_checked = False
        self.packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        register_custom_op()

    def get_name(self) -> str:
        return QUANTIZATION_NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Exl3Config:
        return cls(
            bits=config.get("bits"),
            head_bits=config.get("head_bits"),
            codebook=config.get("codebook"),
            version=config.get("version"),
            tensor_storage=config.get("tensor_storage"),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: PretrainedConfig | None = None,
    ) -> str | None:
        del hf_config
        if user_quant is not None and user_quant != QUANTIZATION_NAME:
            return None
        if isinstance(hf_quant_cfg, dict) and (
            hf_quant_cfg.get("quant_method") == QUANTIZATION_NAME
            or "tensor_storage" in hf_quant_cfg
        ):
            return QUANTIZATION_NAME
        return None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        if not self.tensor_storage:
            resolved_revision = revision
            if resolved_revision is None and hf_config is not None:
                resolved_revision = getattr(hf_config, "_commit_hash", None)
            config = _read_quantization_config_json(model_name, resolved_revision)
            if not config or not config.get("tensor_storage"):
                raise ValueError(
                    "EXL3 requires quantization_config.json with a non-empty "
                    "tensor_storage map."
                )
            self.bits = config.get("bits", self.bits)
            self.head_bits = config.get("head_bits", self.head_bits)
            self.codebook = config.get("codebook", self.codebook)
            self.version = config.get("version", self.version)
            self.tensor_storage = config["tensor_storage"]

        validate_storage_metadata(self.tensor_storage)
        self._force_independent_lm_head(hf_config)

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        mapped = hf_to_vllm_mapper.apply_dict(self.tensor_storage)
        self.tensor_storage = {**self.tensor_storage, **mapped}

    def _force_independent_lm_head(self, hf_config: PretrainedConfig | None) -> None:
        if hf_config is None or not self.has_quantized_lm_head():
            return
        configs: list[Any] = [hf_config]
        try:
            text_config = hf_config.get_text_config()
        except (AttributeError, TypeError):
            text_config = None
        if text_config is not None and text_config is not hf_config:
            configs.append(text_config)
        changed = False
        for config in configs:
            if getattr(config, "tie_word_embeddings", False):
                config.tie_word_embeddings = False
                changed = True
        if changed:
            logger.warning(
                "EXL3 metadata contains an independently quantized lm_head; "
                "overriding tie_word_embeddings so vLLM instantiates it."
            )

    def _require_fail_closed_runtime(self) -> None:
        if self._eager_checked:
            return
        self._eager_checked = True
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        model_config = vllm_config.model_config
        if getattr(model_config, "enable_sleep_mode", False):
            raise ValueError(
                "EXL3 does not support sleep mode: side-dictionary payloads "
                "are not restored across sleep/wake."
            )
        if getattr(vllm_config, "load_format", "") in {"npcache", "dummy"}:
            raise ValueError("EXL3 does not support dummy/npcache load formats")
        if not model_config.enforce_eager and not graphs_allowed():
            raise ValueError(
                "The EXL3 quantization backend requires eager execution: "
                "pass --enforce-eager (enforce_eager=True). Set "
                "VLLM_EXL3_ALLOW_GRAPHS=1 only after kernel prewarm and "
                "capture/replay tests pass."
            )

    def _hydrate_storage_fallback(self) -> None:
        """Load tensor_storage if this vLLM pin never calls maybe_update_config."""
        candidates: list[str] = []
        env_path = os.environ.get("VLLM_EXL3_QUANT_CONFIG", "").strip()
        if env_path:
            candidates.append(env_path)
        vllm_config = get_current_vllm_config_or_none()
        model = getattr(getattr(vllm_config, "model_config", None), "model", None)
        if model:
            candidates.append(str(Path(model) / "quantization_config.json"))
            candidates.append(str(model))
        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                path = path / "quantization_config.json"
            if not path.is_file():
                continue
            config = json.loads(path.read_text(encoding="utf-8"))
            storage = config.get("tensor_storage")
            if storage:
                self.bits = config.get("bits", self.bits)
                self.head_bits = config.get("head_bits", self.head_bits)
                self.codebook = config.get("codebook", self.codebook)
                self.version = config.get("version", self.version)
                self.tensor_storage = storage
                validate_storage_metadata(self.tensor_storage)
                return

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        self._require_fail_closed_runtime()
        if not self.tensor_storage:
            self._hydrate_storage_fallback()
        if not self.tensor_storage:
            raise ValueError(
                "EXL3 tensor_storage is empty. The pinned vLLM runtime must call "
                "maybe_update_config() or set VLLM_EXL3_QUANT_CONFIG. "
                "See manifests/patches/README.md."
            )
        is_lm_head = layer.__class__.__name__ == "ParallelLMHead"
        if is_lm_head and not prefix:
            prefix = "lm_head"
        if isinstance(layer, LinearBase) or is_lm_head:
            if not self._linear_prefix_is_exl3(prefix):
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self)
        return None

    def _is_exl3_prefix(self, prefix: str) -> bool:
        entry = storage_entry(self.tensor_storage, prefix)
        return entry is not None and entry.get("quant_format") == "exl3"

    def _linear_prefix_is_exl3(self, prefix: str) -> bool:
        if self._is_exl3_prefix(prefix):
            return True
        leaf = prefix.rsplit(".", 1)[-1]
        source_leaves = getattr(self, "packed_modules_mapping", {}).get(leaf)
        if not source_leaves:
            return False
        base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
        return all(
            self._is_exl3_prefix(f"{base}.{source}" if base else source)
            for source in source_leaves
        )

    def codebook_for_prefix(self, prefix: str) -> str | None:
        entry = storage_entry(self.tensor_storage, prefix)
        if entry is None:
            return None
        return codebook_from_suffixes(stored_suffixes(entry))

    def has_quantized_lm_head(self) -> bool:
        return self._is_exl3_prefix("lm_head")

    def bitrate_for_prefix(self, prefix: str) -> int | None:
        entry = storage_entry(self.tensor_storage, prefix)
        if entry is None:
            return None
        if "bits_per_weight" in entry:
            return int(entry["bits_per_weight"])
        stored = entry.get("stored_tensors", {})
        for name, info in stored.items():
            if name.endswith(".trellis") and "shape" in info:
                from .metadata import bitrate_from_trellis_shape

                return bitrate_from_trellis_shape(info["shape"])
        return None
