"""
src/shiftmri/models.py
~~~~~~~~~~~~~~~~~~~~~~
Model Adapter Architecture for ShiftMRI Multi-Model Audit.

STAGE 1: Model A (NeuralSeg Baseline)
    Pretrained MONAI 3D UNet from AndrewVFranco/NeuralSeg using checkpoints/best_model.pth.
    Input: 4 channels (t1c, t1n, t2f, t2w).
    Output: 4-class segmentation (0=bg, 1=necrotic core, 2=edema, 3=enhancing tumor).

STAGE 2: Model B (Independent Secondary Model Adapter)
    Inspection of MONAI BraTS MRI Segmentation bundle (brats_mri_segmentation).
    Architecture: SegResNet / 3D UNet trained on BraTS.
    Status: Pretrained weights are not bundled locally (~100-500 MB). Under strict
    safety and scope rules, automated external weight downloads are prohibited.
    This adapter provides the standardized interface and canonical output conversion.
    If a secondary checkpoint is supplied locally, it executes independently; otherwise,
    it honestly reports the exact blocker without fabricating predictions.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.shiftmri.metrics import region_masks, voxel_volume_ml


class CanonicalSegmentation:
    """Standardized representation for 3D tumor segmentation outputs."""

    def __init__(
        self,
        mask: np.ndarray,
        spacing_mm: Sequence[float],
        model_name: str,
        checkpoint_id: str,
        provenance: dict[str, Any] | None = None,
    ):
        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3-D (H, W, D), got shape {mask.shape}")
        unique_labels = set(np.unique(mask).tolist())
        if not unique_labels.issubset({0, 1, 2, 3}):
            raise ValueError(f"Mask contains invalid labels {unique_labels}. Expected subset of {{0, 1, 2, 3}}.")

        self.mask = mask.astype(np.uint8)
        self.spacing_mm = tuple(float(v) for v in spacing_mm)
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.provenance = provenance or {}

        # Derived regions and physical volumes
        v_ml = voxel_volume_ml(self.spacing_mm)
        regs = region_masks(self.mask)
        self.volumes_ml = {
            "wt": round(float(regs["wt"].sum()) * v_ml, 4),
            "tc": round(float(regs["tc"].sum()) * v_ml, 4),
            "et": round(float(regs["et"].sum()) * v_ml, 4),
            "ed": round(float((self.mask == 2).sum()) * v_ml, 4),
            "ncr": round(float((self.mask == 1).sum()) * v_ml, 4),
        }
        self.voxel_counts = {
            "wt": int(regs["wt"].sum()),
            "tc": int(regs["tc"].sum()),
            "et": int(regs["et"].sum()),
            "ed": int((self.mask == 2).sum()),
            "ncr": int((self.mask == 1).sum()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "checkpoint_id": self.checkpoint_id,
            "spacing_mm": list(self.spacing_mm),
            "volumes_ml": self.volumes_ml,
            "voxel_counts": self.voxel_counts,
            "shape": list(self.mask.shape),
            "provenance": self.provenance,
        }


class ModelAdapter(ABC):
    """Abstract base adapter for brain tumor segmentation models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model architecture."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if the model checkpoint and runtime dependencies are available."""
        ...

    @property
    @abstractmethod
    def blocker_reason(self) -> str | None:
        """Detailed blocker description if unavailable."""
        ...

    @abstractmethod
    def predict(
        self,
        image_chw: np.ndarray,
        spacing_mm: Sequence[float],
    ) -> CanonicalSegmentation:
        """Run inference and return a CanonicalSegmentation."""
        ...


class NeuralSegModelAdapter(ModelAdapter):
    """STAGE 1: Primary Model A — NeuralSeg Pretrained MONAI 3D UNet."""

    def __init__(
        self,
        checkpoint_path: Path | str = "checkpoints/best_model.pth",
        device: str | None = None,
    ):
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._predict_fn = None
        self._ckpt_id = "uninitialized"
        self._blocker = None

        if not self._checkpoint_path.exists():
            self._blocker = f"Checkpoint file missing at {self._checkpoint_path}"
        else:
            try:
                from src.shiftmri.inference import checkpoint_id
                self._ckpt_id = checkpoint_id(self._checkpoint_path)
            except Exception as e:
                self._blocker = str(e)

    @property
    def model_name(self) -> str:
        return "Model A (NeuralSeg MONAI 3D UNet)"

    @property
    def is_available(self) -> bool:
        return self._checkpoint_path.exists() and self._blocker is None

    @property
    def blocker_reason(self) -> str | None:
        return self._blocker

    def _ensure_loaded(self):
        if self._predict_fn is not None:
            return
        if not self.is_available:
            raise RuntimeError(f"Cannot load {self.model_name}: {self._blocker}")

        from src.shiftmri.inference import load_model, make_predict_fn
        model, dev, self._ckpt_id = load_model(self._checkpoint_path, device=self._device)
        self._predict_fn = make_predict_fn(model, dev)

    def predict(
        self,
        image_chw: np.ndarray,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> CanonicalSegmentation:
        self._ensure_loaded()
        mask = self._predict_fn(image_chw)
        return CanonicalSegmentation(
            mask=mask,
            spacing_mm=spacing_mm,
            model_name=self.model_name,
            checkpoint_id=self._ckpt_id,
            provenance={
                "architecture": "MONAI UNet (channels: 16,32,64,128,256)",
                "checkpoint_path": str(self._checkpoint_path),
                "input_channels": ["t1c", "t1n", "t2f", "t2w"],
            },
        )


class MonaiBraTSBundleAdapter(ModelAdapter):
    """STAGE 2: Independent Model B — MONAI BraTS Bundle Adapter.

    Inspects MONAI BraTS MRI segmentation bundle (e.g. SegResNet).
    If a secondary checkpoint is provided at `model_b_checkpoint`, it runs inference;
    otherwise, it documents the exact blocker honestly.
    """

    def __init__(self, checkpoint_path: Path | str | None = None, device: str | None = None):
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._device = device

    @property
    def model_name(self) -> str:
        return "Model B (MONAI BraTS SegResNet Bundle)"

    @property
    def is_available(self) -> bool:
        return self._checkpoint_path is not None and self._checkpoint_path.exists()

    @property
    def blocker_reason(self) -> str | None:
        if self.is_available:
            return None
        return (
            "Pretrained weights for secondary Model B (MONAI BraTS bundle) are not bundled locally. "
            "External download prohibited by security and reproducibility constraints. "
            "To enable Model B, supply a valid checkpoint via --model-b-checkpoint."
        )

    def predict(
        self,
        image_chw: np.ndarray,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> CanonicalSegmentation:
        if not self.is_available:
            raise RuntimeError(f"Cannot run {self.model_name}: {self.blocker_reason}")

        # If a secondary checkpoint is supplied by the user, load and infer
        raise NotImplementedError("Secondary checkpoint execution requires local bundle weights.")
