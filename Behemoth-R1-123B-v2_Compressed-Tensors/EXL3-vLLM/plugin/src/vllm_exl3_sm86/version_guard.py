"""Fail-closed runtime version guards for the pinned vLLM ABI."""

from __future__ import annotations

import os

from .constants import REQUIRED_CUDA, REQUIRED_TORCH_PREFIX, REQUIRED_VLLM_VERSION


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def check_runtime(*, require_cuda: bool = True) -> None:
    if _truthy("VLLM_EXL3_SKIP_VERSION_GUARD") or _truthy(
        "VLLM_EXL3_ALLOW_VLLM_DRIFT"
    ):
        return
    errors: list[str] = []
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import environment
        raise RuntimeError(f"EXL3 plugin requires torch: {exc}") from exc

    if not str(torch.__version__).startswith(REQUIRED_TORCH_PREFIX):
        errors.append(
            f"torch {torch.__version__} does not start with {REQUIRED_TORCH_PREFIX}"
        )
    cuda = getattr(torch.version, "cuda", None)
    if require_cuda and cuda is not None and not str(cuda).startswith(REQUIRED_CUDA):
        errors.append(f"torch CUDA {cuda} does not start with {REQUIRED_CUDA}")

    try:
        import vllm
    except Exception as exc:
        raise RuntimeError(f"EXL3 plugin requires vllm: {exc}") from exc

    reported = str(getattr(vllm, "__version__", ""))
    if reported != REQUIRED_VLLM_VERSION and not _truthy("VLLM_EXL3_ALLOW_VLLM_DRIFT"):
        errors.append(
            f"vllm {reported!r} != pinned {REQUIRED_VLLM_VERSION!r}. "
            "Set VLLM_EXL3_ALLOW_VLLM_DRIFT=1 only after rebasing the plugin."
        )
    if errors:
        raise RuntimeError(
            "EXL3 SM86 plugin ABI mismatch: "
            + "; ".join(errors)
            + ". Capture a new manifest or restore the pinned stack."
        )
