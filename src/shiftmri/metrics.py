"""
src/shiftmri/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~
Segmentation and perturbation-stability metrics.

Labels (matching train.py exactly)
-----------------------------------
  0 = background
  1 = necrotic core
  2 = peritumoral edema
  3 = enhancing tumour

Derived regions
---------------
  WT = Whole Tumour  = {1, 2, 3}
  TC = Tumour Core   = {1, 3}
  ET = Enhancing T.  = {3}

NaN conventions
---------------
- Dice is NaN when BOTH the prediction AND the reference are empty.
- HD95 is NaN when EITHER mask is empty (distance undefined).
- Volume change relative is NaN when baseline volume == 0.
- Volume in mL uses voxel spacing: prod(spacing_mm) / 1000.

Cross-check
-----------
HD95 is validated against monai.metrics.HausdorffDistanceMetric in the test
suite when MONAI is importable.

Research prototype — not clinically validated.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

VALID_LABELS: frozenset[int] = frozenset({0, 1, 2, 3})


def _check_labels(arr: np.ndarray, name: str = "mask") -> None:
    unique = np.unique(arr)
    bad = set(unique.tolist()) - {float(v) for v in VALID_LABELS}
    if bad:
        raise ValueError(
            f"{name} contains unexpected label(s) {sorted(bad)}. "
            f"Expected a subset of {{0, 1, 2, 3}}."
        )


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------

def region_masks(seg: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean arrays for each BraTS tumour sub-region.

    Parameters
    ----------
    seg:
        Integer or float array of per-voxel labels (0–3).

    Returns
    -------
    dict with keys ``wt``, ``tc``, ``et``.
    """
    _check_labels(seg, "seg")
    return {
        "wt": (seg >= 1),            # labels 1, 2, 3
        "tc": (seg == 1) | (seg == 3),  # labels 1, 3
        "et": (seg == 3),            # label 3
    }


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def voxel_volume_ml(spacing_mm: Sequence[float]) -> float:
    """Voxel volume in millilitres.

    Parameters
    ----------
    spacing_mm:
        Voxel dimensions in mm, length 3 (x, y, z).
    """
    if len(spacing_mm) != 3:
        raise ValueError(
            f"spacing_mm must have length 3, got {len(spacing_mm)}."
        )
    if any(s <= 0 for s in spacing_mm):
        raise ValueError(
            f"All spacing values must be > 0, got {list(spacing_mm)}."
        )
    return float(np.prod(spacing_mm)) / 1000.0


def region_volumes_ml(seg: np.ndarray, spacing_mm: Sequence[float]) -> dict[str, float]:
    """Volume of each tumour sub-region in mL."""
    vox_ml = voxel_volume_ml(spacing_mm)
    masks = region_masks(seg)
    return {name: float(m.sum()) * vox_ml for name, m in masks.items()}


def volume_change(
    baseline_ml: float,
    perturbed_ml: float,
) -> dict[str, float]:
    """Signed, absolute, and relative volume change.

    Parameters
    ----------
    baseline_ml:
        Volume under the unperturbed prediction (mL).
    perturbed_ml:
        Volume under the perturbed prediction (mL).

    Returns
    -------
    dict with keys ``signed_ml``, ``absolute_ml``, ``relative_pct``.
    ``relative_pct`` is NaN when baseline is 0.
    """
    signed = perturbed_ml - baseline_ml
    absolute = abs(signed)
    if baseline_ml == 0:
        relative = float("nan")
    else:
        relative = 100.0 * signed / baseline_ml
    return {
        "signed_ml": float(signed),
        "absolute_ml": float(absolute),
        "relative_pct": float(relative),
    }


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------

def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice coefficient between two boolean arrays.

    Returns NaN when BOTH masks are empty (no structure to measure).
    Returns 0.0 when only one is empty.
    """
    p = pred.astype(bool)
    r = ref.astype(bool)
    pred_empty = not p.any()
    ref_empty = not r.any()

    if pred_empty and ref_empty:
        return float("nan")
    if pred_empty or ref_empty:
        return 0.0

    intersection = (p & r).sum()
    return float(2 * intersection) / float(p.sum() + r.sum())


# ---------------------------------------------------------------------------
# HD95 in mm
# ---------------------------------------------------------------------------

def _surface_points(mask: np.ndarray, spacing_mm: Sequence[float]) -> np.ndarray:
    """Return coordinates of surface voxels in physical space (mm).

    A voxel is on the surface if it is True and at least one 6-connected
    neighbour is False.

    Returns an (N, 3) float array.  Raises ValueError if mask is empty.
    """
    from scipy.ndimage import binary_erosion  # lazy import

    if not mask.any():
        raise ValueError("Cannot compute surface of an empty mask.")
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    surface = mask & ~eroded
    coords = np.argwhere(surface).astype(np.float64)
    # Scale by spacing
    coords *= np.array(spacing_mm, dtype=np.float64)
    return coords


def hausdorff95(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing_mm: Sequence[float],
) -> float:
    """95th-percentile Hausdorff distance in mm.

    Returns NaN when EITHER mask is empty.
    """
    from scipy.spatial import KDTree  # lazy import

    p = pred.astype(bool)
    r = ref.astype(bool)
    if not p.any() or not r.any():
        return float("nan")

    pts_p = _surface_points(p, spacing_mm)
    pts_r = _surface_points(r, spacing_mm)

    tree_r = KDTree(pts_r)
    tree_p = KDTree(pts_p)

    d_p2r, _ = tree_r.query(pts_p)
    d_r2p, _ = tree_p.query(pts_r)

    hd95 = float(np.percentile(np.concatenate([d_p2r, d_r2p]), 95))
    return hd95


# ---------------------------------------------------------------------------
# Per-region metric bundles
# ---------------------------------------------------------------------------

def segmentation_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing_mm: Sequence[float],
    include_hd95: bool = True,
) -> dict[str, float]:
    """Compute Dice and (optionally) HD95 for WT, TC, ET.

    Parameters
    ----------
    pred / ref:
        Integer label arrays (0–3).
    spacing_mm:
        Voxel spacings in mm.
    include_hd95:
        Set False to skip the expensive Hausdorff computation.

    Returns
    -------
    Flat dict with keys like ``wt_dice``, ``wt_hd95_mm``, etc.
    """
    _check_labels(pred, "pred")
    _check_labels(ref, "ref")

    pred_masks = region_masks(pred)
    ref_masks = region_masks(ref)
    result: dict[str, float] = {}

    for region in ("wt", "tc", "et"):
        result[f"{region}_dice"] = dice(pred_masks[region], ref_masks[region])
        if include_hd95:
            result[f"{region}_hd95_mm"] = hausdorff95(
                pred_masks[region], ref_masks[region], spacing_mm
            )

    return result
