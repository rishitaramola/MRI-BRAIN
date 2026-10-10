"""
src/shiftmri/perturbations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Controlled MRI perturbations applied in z-score space on PREPROCESSED arrays.

Design notes
------------
- All functions operate on the OUTPUT of preprocess_array(), i.e. arrays already
  N4-corrected and z-score normalised over non-zero voxels.  Applying scale/shift
  BEFORE z-scoring is mathematically cancelled by renormalisation; this module
  therefore applies them AFTER, inside the brain mask, in z-score units.
- Every function is PURE: it returns a new float32 array and never mutates its input.
- Seeds are deterministic (sha256-based, not Python hash()).
- Background (zero) voxels remain exactly 0 after every perturbation.

Ranges used in configs/shiftmri.yaml are engineering judgement, chosen to match the
training augmentation ranges in src/training/transforms.py (RandScaleIntensityd factor
0.1, RandShiftIntensityd offset 0.1, RandGaussianNoised default sigma ~0.1).  They are
NOT calibrated to real inter-scanner or cross-site variation.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # no runtime-only imports needed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_image(image: np.ndarray) -> None:
    """Raise ValueError for any malformed preprocessed image array.

    Expected: ndim == 4, shape (C, H, W, D) with C >= 1, float dtype,
    all finite values, and no zero-length dimension.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Expected np.ndarray, got {type(image).__name__}."
        )
    if image.ndim != 4:
        raise ValueError(
            f"Expected 4-D array (C, H, W, D), got ndim={image.ndim}."
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            f"Expected float dtype, got {image.dtype}. "
            "Run preprocess_array() before calling perturbation functions."
        )
    if any(s == 0 for s in image.shape):
        raise ValueError(
            f"Array has a zero-length dimension: shape={image.shape}."
        )
    if not np.all(np.isfinite(image)):
        n_bad = np.sum(~np.isfinite(image))
        raise ValueError(
            f"Array contains {n_bad} non-finite value(s) (NaN or Inf)."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _seed_to_rng(seed: int | str) -> np.random.Generator:
    """Convert an integer or sha256-derived string seed to a numpy Generator."""
    if isinstance(seed, str):
        digest = hashlib.sha256(seed.encode()).digest()
        # Take first 8 bytes → uint64 → int
        seed = int.from_bytes(digest[:8], byteorder="big")
    return np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)


def _brain_mask(image: np.ndarray) -> np.ndarray:
    """Return a boolean mask of voxels that are non-zero in ANY channel.

    Shape: (H, W, D) — same spatial dims as image[0].
    """
    return np.any(image != 0, axis=0)  # (H, W, D)


# ---------------------------------------------------------------------------
# Public perturbation API
# ---------------------------------------------------------------------------

def gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int | str,
) -> np.ndarray:
    """Add isotropic Gaussian noise inside the brain mask.

    Parameters
    ----------
    image:
        Preprocessed 4-D float32 array of shape (C, H, W, D).
    sigma:
        Standard deviation of noise in z-score units.
    seed:
        Deterministic seed (int or sha256-hex string).

    Returns
    -------
    float32 array with the same shape as *image*.  Background voxels stay 0.
    """
    validate_image(image)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")

    rng = _seed_to_rng(seed)
    mask = _brain_mask(image)  # (H, W, D)

    out = image.astype(np.float32, copy=True)
    noise = rng.standard_normal(image.shape).astype(np.float32) * sigma
    # Apply noise only inside brain mask; background stays 0
    out[:, mask] += noise[:, mask]

    return out


def intensity_scale_shift(
    image: np.ndarray,
    scale: float,
    shift: float,
    seed: int | str,
) -> np.ndarray:
    """Multiplicative scale and additive shift in z-score space, inside brain mask.

    Applied after z-score normalisation, so scale/shift act in z-score units.
    Global scale+shift before z-scoring would be cancelled by the normaliser —
    this is the only meaningful place to apply them.  See tests for proof.

    Parameters
    ----------
    image:
        Preprocessed 4-D float32 array of shape (C, H, W, D).
    scale:
        Multiplicative factor (e.g. 0.9 or 1.1).
    shift:
        Additive offset in z-score units (e.g. ±0.1).
    seed:
        Deterministic seed (currently unused for this deterministic transform,
        accepted for API consistency).

    Returns
    -------
    float32 array with the same shape as *image*.  Background voxels stay 0.
    """
    validate_image(image)

    mask = _brain_mask(image)  # (H, W, D)
    out = image.astype(np.float32, copy=True)
    out[:, mask] = out[:, mask] * scale + shift

    return out


def downsample_restore(
    image: np.ndarray,
    factor: int,
    axis: int,
    seed: int | str,
) -> np.ndarray:
    """Simulate resolution loss: anti-alias → downsample → linear upsample.

    Downsampling is done along a single spatial axis.  The result is
    bilinearly restored to the original shape so the model still receives
    the full-resolution grid.  Background voxels remain 0.

    Parameters
    ----------
    image:
        Preprocessed 4-D float32 array of shape (C, H, W, D).
    factor:
        Integer downsampling factor (2 or 3 are typical; must be >= 2).
    axis:
        Spatial axis to downsample: 0→H, 1→W, 2→D.
        (Channel axis 0 is excluded automatically.)
    seed:
        Deterministic seed (accepted for API consistency).

    Returns
    -------
    float32 array with the same shape as *image*.  Background voxels stay 0.
    """
    validate_image(image)
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}.")
    if axis not in (0, 1, 2):
        raise ValueError(
            f"axis must be 0, 1, or 2 (spatial dims); got {axis}. "
            "Channel dim (0 in the 4-D array) is index 0 of the spatial dims, "
            "i.e. axis 0 → array dim 1, axis 1 → array dim 2, axis 2 → array dim 3."
        )

    from scipy.ndimage import uniform_filter, zoom  # lazy import


    mask = _brain_mask(image)  # (H, W, D)
    out = image.astype(np.float32, copy=True)

    for c in range(out.shape[0]):
        channel = out[c]  # (H, W, D)

        # --- anti-alias: smooth along target axis before downsampling ---
        smoothed = uniform_filter(
            channel, size=[factor if i == axis else 1 for i in range(3)]
        )

        # --- downsample along target axis ---
        zoom_factors = [1.0, 1.0, 1.0]
        zoom_factors[axis] = 1.0 / factor
        downsampled = zoom(smoothed, zoom_factors, order=1, prefilter=False)

        # --- upsample back to original shape ---
        orig_shape = channel.shape
        restore_factors = [o / d for o, d in zip(orig_shape, downsampled.shape)]
        restored = zoom(downsampled, restore_factors, order=1, prefilter=False)

        # Ensure exact original shape (zoom can be off by ±1 voxel)
        if restored.shape != orig_shape:
            slices = tuple(slice(0, s) for s in orig_shape)
            pad_widths = [
                (0, max(0, orig_shape[i] - restored.shape[i]))
                for i in range(3)
            ]
            restored = np.pad(restored, pad_widths, mode="edge")[slices]

        # Zero out background
        restored[~mask] = 0.0
        out[c] = restored.astype(np.float32)

    return out
