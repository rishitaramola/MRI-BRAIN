"""
src/shiftmri/synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Synthetic MRI-like phantom for smoke tests and demo runs.

The phantom is NOT a real MRI scan.  All outputs are labelled
``data_source=synthetic_phantom``.  It is only used to verify that the
pipeline plumbing works end-to-end without real BraTS data.

The phantom consists of:
- A spherical background of Gaussian noise (zero-mean, small sigma).
- A "brain" region: a large ellipsoid of higher intensity.
- A "tumour" region inside the brain: a smaller sphere with
  labels 1, 2, 3 in concentric shells.
- All arrays are float32, z-score normalised over non-zero voxels.
- Spacing is set to 1.0 mm isotropic.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import numpy as np


PHANTOM_SHAPE = (64, 64, 64)   # small enough to run fast on CPU
PHANTOM_SPACING_MM = (1.0, 1.0, 1.0)


def _sphere_mask(shape, centre, radius):
    cx, cy, cz = centre
    xs, ys, zs = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) <= radius ** 2


def make_phantom(
    shape: tuple[int, int, int] = PHANTOM_SHAPE,
    seed: int = 42,
) -> dict:
    """Build a synthetic 4-channel brain-like phantom.

    Returns
    -------
    dict with keys:
        ``modality_arrays``  – dict {t1c, t1n, t2f, t2w} float32 (H, W, D)
        ``seg``              – uint8 (H, W, D), labels 0–3
        ``spacing_mm``       – (1.0, 1.0, 1.0)
        ``case_name``        – ``"synthetic_phantom"``
        ``mode``             – ``"synthetic"``

    All modality arrays are zero outside the brain mask and z-score
    normalised inside it (matching preprocess_array output convention).
    """
    rng = np.random.default_rng(seed)
    H, W, D = shape
    cx, cy, cz = H // 2, W // 2, D // 2

    brain_radius = min(H, W, D) // 2 - 4
    tumour_radius = brain_radius // 3
    core_radius = tumour_radius // 2

    brain_mask = _sphere_mask(shape, (cx, cy, cz), brain_radius)
    tumour_mask = _sphere_mask(shape, (cx, cy, cz), tumour_radius)
    core_mask = _sphere_mask(shape, (cx, cy, cz), core_radius)

    # Ground-truth labels: 0 bg, 1 necrotic core, 2 edema ring, 3 enhancing
    seg = np.zeros(shape, dtype=np.uint8)
    seg[tumour_mask] = 2          # edema
    seg[core_mask] = 1            # necrotic core
    # Enhancing tumour: thin shell just inside the edema boundary
    enh_radius = tumour_radius - max(1, tumour_radius // 4)
    enh_mask = _sphere_mask(shape, (cx, cy, cz), tumour_radius) & ~_sphere_mask(
        shape, (cx, cy, cz), enh_radius
    )
    seg[enh_mask] = 3

    modality_arrays = {}
    for mod in ("t1c", "t1n", "t2f", "t2w"):
        vol = np.zeros(shape, dtype=np.float32)
        # Brain tissue: moderate intensity + noise
        brain_signal = rng.standard_normal(shape).astype(np.float32) * 0.5 + 2.0
        vol[brain_mask] = brain_signal[brain_mask]
        # Tumour: slightly different contrast per modality
        offset = {"t1c": 1.0, "t1n": -0.5, "t2f": 1.5, "t2w": 0.8}[mod]
        vol[tumour_mask] += offset

        # Z-score over non-zero voxels (matching preprocess_array)
        nz = vol != 0
        if nz.any():
            mu = vol[nz].mean()
            sigma = vol[nz].std()
            if sigma > 0:
                vol = (vol - mu) / sigma
                vol[~nz] = 0.0

        modality_arrays[mod] = vol.astype(np.float32)

    return {
        "modality_arrays": modality_arrays,
        "seg": seg,
        "spacing_mm": PHANTOM_SPACING_MM,
        "case_name": "synthetic_phantom",
        "mode": "synthetic",
    }


def make_phantom_image(shape: tuple[int, int, int] = PHANTOM_SHAPE, seed: int = 42) -> np.ndarray:
    """Return a (4, H, W, D) float32 stack ready for inference.

    Channel order: t1c, t1n, t2f, t2w — matching predict.py.
    """
    phantom = make_phantom(shape, seed)
    ma = phantom["modality_arrays"]
    return np.stack([ma["t1c"], ma["t1n"], ma["t2f"], ma["t2w"]], axis=0).astype(np.float32)
