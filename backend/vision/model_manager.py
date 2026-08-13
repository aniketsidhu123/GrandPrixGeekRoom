"""
Model Manager — Centralized launcher for all AI models.

Handles:
  - Loading/unloading models to GPU/CPU
  - VRAM tracking per model
  - Model lifecycle management
  - Status reporting for the dashboard
"""

import os
import time
import logging
import torch
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ModelManager")


@dataclass
class ModelInfo:
    """Metadata for a loaded model."""
    name: str
    type: str  # "yolo" | "depth" | "vlm"
    path: str
    loaded: bool = False
    device: str = "cpu"
    vram_mb: float = 0.0
    load_time_sec: float = 0.0
    inference_count: int = 0
    last_inference_ms: float = 0.0


class ModelManager:
    """
    Centralized model launcher and VRAM tracker.

    All models in the project are registered and loaded through this manager.
    It tracks GPU memory before/after each model load to report per-model VRAM usage,
    and provides a single status endpoint for the dashboard.
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gpu_name = ""
        self._total_vram_mb = 0.0

        if self._device == "cuda":
            self._gpu_name = torch.cuda.get_device_name(0)
            self._total_vram_mb = round(torch.cuda.get_device_properties(0).total_mem / (1024 ** 2), 1)
            logger.info(f"GPU detected: {self._gpu_name} ({self._total_vram_mb} MB VRAM)")
        else:
            logger.warning("No CUDA GPU detected. All models will run on CPU.")

    @property
    def device(self) -> str:
        return self._device

    def _get_gpu_memory_used_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self._device != "cuda":
            return 0.0
        return round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)

    def _get_gpu_memory_reserved_mb(self) -> float:
        """Get total GPU memory reserved by PyTorch in MB."""
        if self._device != "cuda":
            return 0.0
        return round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)

    def register(self, name: str, model_type: str, path: str) -> ModelInfo:
        """Register a model for tracking (does NOT load it yet)."""
        info = ModelInfo(name=name, type=model_type, path=path, device=self._device)
        self._models[name] = info
        logger.info(f"Registered model: {name} ({model_type}) @ {path}")
        return info

    def load_model(self, name: str, loader_fn) -> bool:
        """
        Load a model using the provided loader function.
        Tracks VRAM before/after to calculate per-model usage.

        Args:
            name: Registered model name
            loader_fn: Callable that performs the actual model loading. Must return True/False.
        """
        if name not in self._models:
            logger.error(f"Model '{name}' is not registered.")
            return False

        info = self._models[name]

        if not os.path.exists(info.path):
            logger.error(f"Weight file not found for '{name}': {info.path}")
            return False

        vram_before = self._get_gpu_memory_used_mb()
        start_time = time.time()

        logger.info(f"Loading '{name}' from {info.path} on {self._device}...")

        try:
            success = loader_fn()
            if not success:
                logger.error(f"Loader returned failure for '{name}'.")
                return False
        except Exception as e:
            logger.error(f"Exception loading '{name}': {e}")
            return False

        load_time = round(time.time() - start_time, 2)
        vram_after = self._get_gpu_memory_used_mb()
        vram_used = round(vram_after - vram_before, 1)

        info.loaded = True
        info.load_time_sec = load_time
        info.vram_mb = max(vram_used, 0.0)  # Can be 0 on CPU

        logger.info(
            f"  ✓ '{name}' loaded in {load_time}s | "
            f"VRAM: +{info.vram_mb} MB | "
            f"Total GPU: {vram_after} MB / {self._total_vram_mb} MB"
        )

        return True

    def unload_model(self, name: str, unloader_fn=None):
        """Unload a model and free VRAM."""
        if name not in self._models:
            return

        info = self._models[name]
        if unloader_fn:
            unloader_fn()

        info.loaded = False
        info.vram_mb = 0.0
        logger.info(f"Unloaded model: {name}")

        if self._device == "cuda":
            torch.cuda.empty_cache()

    def record_inference(self, name: str, latency_ms: float):
        """Record an inference call for a model."""
        if name in self._models:
            self._models[name].inference_count += 1
            self._models[name].last_inference_ms = round(latency_ms, 1)

    def get_status(self) -> dict:
        """Return full status report for the dashboard."""
        models_status = {}
        total_vram_used = 0.0

        for name, info in self._models.items():
            models_status[name] = {
                "loaded": info.loaded,
                "type": info.type,
                "device": info.device,
                "vram_mb": info.vram_mb,
                "load_time_sec": info.load_time_sec,
                "inference_count": info.inference_count,
                "last_inference_ms": info.last_inference_ms,
            }
            if info.loaded:
                total_vram_used += info.vram_mb

        return {
            "gpu_name": self._gpu_name,
            "gpu_available": self._device == "cuda",
            "total_vram_mb": self._total_vram_mb,
            "used_vram_mb": self._get_gpu_memory_used_mb(),
            "reserved_vram_mb": self._get_gpu_memory_reserved_mb(),
            "models_vram_mb": round(total_vram_used, 1),
            "models": models_status,
        }

    def is_loaded(self, name: str) -> bool:
        return self._models.get(name, ModelInfo(name="", type="", path="")).loaded


# ━━━ Singleton instance ━━━
model_manager = ModelManager()
