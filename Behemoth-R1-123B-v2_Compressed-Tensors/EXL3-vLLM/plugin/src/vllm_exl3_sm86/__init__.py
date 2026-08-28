"""vLLM general plugin: register the dense EXL3 quantization config."""

from __future__ import annotations

from .constants import PLUGIN_NAME, QUANTIZATION_NAME
from .ops import register_custom_op
from .version_guard import check_runtime


def register() -> None:
    """Entry point for `vllm.general_plugins`."""
    check_runtime(require_cuda=False)
    from vllm.model_executor.layers.quantization import register_quantization_config

    from .config import Exl3Config

    register_quantization_config(QUANTIZATION_NAME)(Exl3Config)
    register_custom_op()


def register_exl3() -> None:
    register()


__all__ = [
    "PLUGIN_NAME",
    "QUANTIZATION_NAME",
    "register",
    "register_exl3",
]
