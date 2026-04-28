"""
MiMo-V2.5 environment / library support probe.

Run on the system that has the model files. No GPU needed, no model load.
Prints versions + whether mimo_v2 is registered in transformers /
llm-compressor / vLLM, plus the list of MoE-calibration classes available
in llm-compressor.modeling.

Usage:
    python check_libs.py
"""
import importlib
import os
import sys


def _ver(pkg):
    try:
        m = importlib.import_module(pkg)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return f"NOT INSTALLED ({type(e).__name__}: {e})"


print("=" * 70)
print("PYTHON / CORE")
print("=" * 70)
print(f"python              : {sys.version.split()[0]}")
print(f"torch               : {_ver('torch')}")
print(f"transformers        : {_ver('transformers')}")
print(f"compressed_tensors  : {_ver('compressed_tensors')}")
print(f"llmcompressor       : {_ver('llmcompressor')}")
print(f"accelerate          : {_ver('accelerate')}")
print(f"safetensors         : {_ver('safetensors')}")
print(f"vllm                : {_ver('vllm')}")
print(f"sglang              : {_ver('sglang')}")

print()
print("=" * 70)
print("TRANSFORMERS — is mimo_v2 a native model_type?")
print("=" * 70)
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    native = "mimo_v2" in CONFIG_MAPPING_NAMES
    print(f"mimo_v2 in CONFIG_MAPPING_NAMES : {native}")
    if native:
        print(f"  -> class: {CONFIG_MAPPING_NAMES['mimo_v2']}")
    else:
        print("  -> NOT native; trust_remote_code=True is REQUIRED")
        print("  -> see https://github.com/huggingface/transformers/pull/45144")
except Exception as e:
    print(f"FAIL: {e}")

print()
print("=" * 70)
print("LLM-COMPRESSOR — registered MoE calibration classes")
print("=" * 70)
try:
    import llmcompressor.modeling as lcm
    files = sorted(
        f for f in os.listdir(os.path.dirname(lcm.__file__)) if f.endswith(".py")
    )
    print(f"modeling/ files ({len(files)}):")
    for f in files:
        print(f"  {f}")
    print()
    cal = sorted(n for n in dir(lcm) if "Calibration" in n or "Calibrate" in n)
    print(f"Calibration* exports ({len(cal)}):")
    for n in cal:
        print(f"  {n}")
    print()
    has_mimo = any("mimo" in f.lower() for f in files) or any(
        "mimo" in n.lower() for n in cal
    )
    print(f"mimo MoE calibration available: {has_mimo}")
    if not has_mimo:
        print("  -> MUST use a calibration-free recipe (RTN W4A16, no AWQ/GPTQ)")
        print("  -> OR write a CalibrationMimoV2MoE wrapper (template avail. in")
        print("     src/llmcompressor/modeling/qwen3_moe.py or minimax_m2 PR #2171)")
except Exception as e:
    print(f"FAIL: {e}")

print()
print("=" * 70)
print("VLLM — is MiMoV2ForCausalLM registered?")
print("=" * 70)
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    mimo_archs = [a for a in archs if "mimo" in a.lower()]
    print(f"mimo-related archs in vLLM: {mimo_archs or 'NONE'}")
except Exception as e:
    print(f"FAIL: {e}")

print()
print("=" * 70)
print("DONE — paste this whole output back to the assistant.")
print("=" * 70)
