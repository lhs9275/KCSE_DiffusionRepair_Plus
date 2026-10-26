"""Fail-closed runtime verification for corrected GPU generation."""

from __future__ import annotations

from typing import Any


REQUIRED_GPU_NAME = "NVIDIA GeForce RTX 5090"
REQUIRED_QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}


def validate_gpu_runtime(torch_module: Any) -> dict[str, Any]:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA is required by corrected_replay_v1")
    gpu_name = str(cuda.get_device_name(0))
    if gpu_name != REQUIRED_GPU_NAME:
        raise RuntimeError(
            f"corrected_replay_v1 requires RTX 5090, got {gpu_name!r}"
        )
    return {
        "gpu_name": gpu_name,
        "cuda_available": True,
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "torch_cuda_version": str(
            getattr(getattr(torch_module, "version", None), "cuda", "unknown")
        ),
    }


def validate_quantized_model(model: Any) -> dict[str, Any]:
    if getattr(model, "is_loaded_in_4bit", False) is not True:
        raise RuntimeError("model is not loaded in required 4-bit mode")
    config = getattr(getattr(model, "config", None), "quantization_config", None)
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        raise RuntimeError("model does not expose its 4-bit quantization config")
    actual = {key: config.get(key) for key in REQUIRED_QUANTIZATION}
    if actual != REQUIRED_QUANTIZATION:
        raise RuntimeError(
            f"4-bit quantization configuration mismatch: {actual!r}"
        )
    return {
        "quantization_verified": True,
        **REQUIRED_QUANTIZATION,
    }
