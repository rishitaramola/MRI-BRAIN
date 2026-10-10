"""
src/shiftmri/dataset_generator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-fidelity 3D Synthetic Multi-Modal Brain MRI Dataset Generator.

Generates realistic 4-channel brain MRI volumes (T1c, T1n, T2-FLAIR, T2w)
along with corresponding 4-class segmentation ground-truth masks:
  0 = Background / healthy brain
  1 = Necrotic tumor core (NCR)
  2 = Peritumoral vasogenic edema (ED)
  3 = Enhancing tumor rim (ET)

Key Features:
-------------
- Anatomical brain structures: cerebral hemispheres, gray/white matter contrast,
  and ventricular CSF spaces.
- Realistic pathological geometry: irregular necrotic core, infiltrative vasogenic
  edema, and hyperintense contrast-enhancing viable margin.
- Multi-parametric MRI physics:
    * T1c: hyperintense enhancing rim (ET), hypointense necrotic core (NCR)
    * T1n: hypointense tumor & edema, standard brain contrast
    * T2-FLAIR: hyperintense edema, attenuated CSF
    * T2w: hyperintense edema and hyperintense CSF
- Cohort generator: creates complete datasets with train/val/test splits,
  saving standard NIfTI (.nii.gz) files following the BraTS GLI convention:
    BraTS-GLI-{ID}-000-{t1c,t1n,t2f,t2w,seg}.nii.gz

Research prototype — not clinically validated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_SHAPE = (128, 128, 128)
DEFAULT_SPACING = (1.0, 1.0, 1.0)


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Return boolean 3D array for an ellipsoid."""
    cx, cy, cz = center
    rx, ry, rz = radii
    xs, ys, zs = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_sq = (
        ((xs - cx) / max(rx, 1e-3)) ** 2
        + ((ys - cy) / max(ry, 1e-3)) ** 2
        + ((zs - cz) / max(rz, 1e-3)) ** 2
    )
    return dist_sq <= 1.0


def generate_synthetic_case(
    case_id: str = "BraTS-GLI-SYN01-000",
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    spacing_mm: tuple[float, float, float] = DEFAULT_SPACING,
    tumor_severity: str = "moderate",  # "mild", "moderate", "severe"
    noise_level: float = 0.05,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a single high-fidelity 3D synthetic MRI case with 4 modalities + segmentation.

    Parameters
    ----------
    case_id : str
        Unique identifier for the synthetic case.
    shape : tuple[int, int, int]
        3D spatial volume dimensions (H, W, D).
    spacing_mm : tuple[float, float, float]
        Voxel spacing in mm.
    tumor_severity : str
        Severity setting influencing tumor volume and edema infiltration.
    noise_level : float
        Gaussian noise standard deviation added to the modalities.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - "case_name": str
        - "spacing_mm": tuple[float, float, float]
        - "modality_arrays": dict[str, np.ndarray] (t1c, t1n, t2f, t2w) float32
        - "seg": np.ndarray (uint8, labels 0..3)
        - "metadata": dict with tumor properties, volume in mL, and seed
    """
    rng = np.random.default_rng(seed)
    H, W, D = shape
    cx, cy, cz = H / 2.0, W / 2.0, D / 2.0

    # 1. Create Brain Anatomy Mask (Ellipsoidal skull/parenchyma)
    brain_rx = (H / 2.0) - 8.0
    brain_ry = (W / 2.0) - 10.0
    brain_rz = (D / 2.0) - 8.0
    brain_mask = _ellipsoid_mask(shape, (cx, cy, cz), (brain_rx, brain_ry, brain_rz))

    # Ventricles (Bilateral elongated CSF cavities near midline)
    ventricle_mask = np.zeros(shape, dtype=bool)
    v_rx, v_ry, v_rz = brain_rx * 0.18, brain_ry * 0.35, brain_rz * 0.12
    ventricle_mask |= _ellipsoid_mask(shape, (cx - brain_rx * 0.15, cy, cz), (v_rx, v_ry, v_rz))
    ventricle_mask |= _ellipsoid_mask(shape, (cx + brain_rx * 0.15, cy, cz), (v_rx, v_ry, v_rz))
    ventricle_mask &= brain_mask

    # 2. Place Tumor Pathology (in left or right cerebral hemisphere, avoiding center)
    side = rng.choice([-1.0, 1.0])
    tx = cx + side * rng.uniform(brain_rx * 0.25, brain_rx * 0.45)
    ty = cy + rng.uniform(-brain_ry * 0.25, brain_ry * 0.25)
    tz = cz + rng.uniform(-brain_rz * 0.20, brain_rz * 0.20)

    # Scale radii according to severity
    severity_mult = {"mild": 0.7, "moderate": 1.0, "severe": 1.35}.get(tumor_severity, 1.0)
    base_r = min(brain_rx, brain_ry, brain_rz) * 0.22 * severity_mult
    tumour_radii = (
        base_r * rng.uniform(0.85, 1.15),
        base_r * rng.uniform(0.80, 1.20),
        base_r * rng.uniform(0.75, 1.10),
    )

    # Edema (vasogenic infiltrative peritumoral halo, larger than core)
    edema_radii = (
        tumour_radii[0] * rng.uniform(1.6, 2.1),
        tumour_radii[1] * rng.uniform(1.5, 2.0),
        tumour_radii[2] * rng.uniform(1.4, 1.9),
    )
    edema_mask = _ellipsoid_mask(shape, (tx, ty, tz), edema_radii) & brain_mask & ~ventricle_mask

    # Enhancing rim (outer margin of tumor core)
    et_mask = _ellipsoid_mask(shape, (tx, ty, tz), tumour_radii) & brain_mask

    # Necrotic core (inside the enhancing rim)
    core_radii = (
        tumour_radii[0] * rng.uniform(0.55, 0.72),
        tumour_radii[1] * rng.uniform(0.52, 0.70),
        tumour_radii[2] * rng.uniform(0.50, 0.68),
    )
    ncr_mask = _ellipsoid_mask(shape, (tx, ty, tz), core_radii) & et_mask

    # Enhancing tumor is the rim between tumor boundary and necrotic core
    et_rim_mask = et_mask & ~ncr_mask

    # Edema surrounds the entire tumor core
    edema_halo_mask = edema_mask & ~et_mask

    # Build Ground Truth Segmentation Mask
    seg = np.zeros(shape, dtype=np.uint8)
    seg[edema_halo_mask] = 2   # Peritumoral edema (ED)
    seg[ncr_mask] = 1          # Necrotic tumor core (NCR)
    seg[et_rim_mask] = 3       # Enhancing tumor (ET)

    # 3. Synthesize 4-Channel MRI Modalities with Physics-Calibrated Intensities
    # Base parenchymal background: White Matter (WM) & Gray Matter (GM)
    base_wm = 1.0
    base_gm = 0.8
    xs, ys, zs = np.ogrid[:shape[0], :shape[1], :shape[2]]
    # Smooth radial gradient for cortical GM vs deep WM
    rad_norm = np.sqrt(((xs - cx) / brain_rx) ** 2 + ((ys - cy) / brain_ry) ** 2 + ((zs - cz) / brain_rz) ** 2)
    parenchyma = np.where(rad_norm > 0.65, base_gm, base_wm).astype(np.float32)

    modality_arrays: dict[str, np.ndarray] = {}

    for mod in ("t1c", "t1n", "t2f", "t2w"):
        vol = np.zeros(shape, dtype=np.float32)

        # Brain parenchyma signal
        vol[brain_mask] = parenchyma[brain_mask]

        # Ventricles (CSF) signal:
        # T1: hypointense CSF (~0.15)
        # T2-FLAIR: attenuated/dark CSF (~0.10)
        # T2w: bright/hyperintense CSF (~2.2)
        if mod in ("t1c", "t1n"):
            vol[ventricle_mask] = 0.15
        elif mod == "t2f":
            vol[ventricle_mask] = 0.10
        elif mod == "t2w":
            vol[ventricle_mask] = 2.20

        # Tumor Sub-Region Specific Physics Signatures:
        if mod == "t1c":
            # T1-contrast: Enhancing rim lights up strongly!
            vol[et_rim_mask] = rng.uniform(2.8, 3.4)
            # Necrotic core is hypointense
            vol[ncr_mask] = rng.uniform(0.3, 0.5)
            # Edema is mildly hypointense
            vol[edema_halo_mask] = rng.uniform(0.5, 0.7)
        elif mod == "t1n":
            # T1-native: No contrast uptake in ET
            vol[et_rim_mask] = rng.uniform(0.7, 0.9)
            vol[ncr_mask] = rng.uniform(0.3, 0.5)
            vol[edema_halo_mask] = rng.uniform(0.5, 0.7)
        elif mod == "t2f":
            # T2-FLAIR: Edema is intensely hyperintense; ET is moderately hyperintense
            vol[edema_halo_mask] = rng.uniform(2.5, 3.1)
            vol[et_rim_mask] = rng.uniform(1.8, 2.3)
            vol[ncr_mask] = rng.uniform(1.3, 1.7)
        elif mod == "t2w":
            # T2w: Both Edema and Necrotic fluid are bright
            vol[edema_halo_mask] = rng.uniform(2.3, 2.8)
            vol[ncr_mask] = rng.uniform(2.1, 2.6)
            vol[et_rim_mask] = rng.uniform(1.6, 2.0)

        # Smooth boundary transitions
        # Add realistic Gaussian noise inside brain mask
        noise = rng.normal(0.0, noise_level, size=shape).astype(np.float32)
        vol[brain_mask] += noise[brain_mask]
        vol[~brain_mask] = 0.0

        # Normalization (Z-score inside non-zero mask, matching preprocessed standard)
        nz = vol > 0.01
        if nz.any():
            mean_val = float(vol[nz].mean())
            std_val = float(vol[nz].std())
            if std_val > 0:
                vol = (vol - mean_val) / std_val
                vol[~nz] = 0.0

        modality_arrays[mod] = vol.astype(np.float32)

    voxel_vol_ml = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    wt_vox = int((seg > 0).sum())
    tc_vox = int(((seg == 1) | (seg == 3)).sum())
    et_vox = int((seg == 3).sum())
    ed_vox = int((seg == 2).sum())
    ncr_vox = int((seg == 1).sum())

    metadata = {
        "case_id": case_id,
        "shape": list(shape),
        "spacing_mm": list(spacing_mm),
        "tumor_severity": tumor_severity,
        "volumes_ml": {
            "wt": round(wt_vox * voxel_vol_ml, 3),
            "tc": round(tc_vox * voxel_vol_ml, 3),
            "et": round(et_vox * voxel_vol_ml, 3),
            "ed": round(ed_vox * voxel_vol_ml, 3),
            "ncr": round(ncr_vox * voxel_vol_ml, 3),
        },
        "voxel_counts": {
            "wt": wt_vox,
            "tc": tc_vox,
            "et": et_vox,
            "ed": ed_vox,
            "ncr": ncr_vox,
        },
        "location": {"side": "left" if side < 0 else "right", "center": [float(tx), float(ty), float(tz)]},
    }

    return {
        "case_name": case_id,
        "spacing_mm": spacing_mm,
        "modality_arrays": modality_arrays,
        "seg": seg,
        "metadata": metadata,
    }


def save_case_nifti(
    case_data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save synthetic case modalities and segmentation as NIfTI (.nii.gz) files.

    Output files follow the official BraTS format:
      {case_name}/{case_name}-t1c.nii.gz
      {case_name}/{case_name}-t1n.nii.gz
      {case_name}/{case_name}-t2f.nii.gz
      {case_name}/{case_name}-t2w.nii.gz
      {case_name}/{case_name}-seg.nii.gz
    """
    import nibabel as nib

    case_name = case_data["case_name"]
    case_dir = Path(output_dir) / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    # Set voxel spacing on diagonal
    for i in range(3):
        affine[i, i] = case_data["spacing_mm"][i]

    saved_paths: dict[str, Path] = {}

    for mod, arr in case_data["modality_arrays"].items():
        img = nib.Nifti1Image(arr.astype(np.float32), affine)
        path = case_dir / f"{case_name}-{mod}.nii.gz"
        nib.save(img, str(path))
        saved_paths[mod] = path

    seg_arr = case_data["seg"]
    seg_img = nib.Nifti1Image(seg_arr.astype(np.uint8), affine)
    seg_path = case_dir / f"{case_name}-seg.nii.gz"
    nib.save(seg_img, str(seg_path))
    saved_paths["seg"] = seg_path

    # Save case metadata JSON
    meta_path = case_dir / f"{case_name}-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(case_data["metadata"], f, indent=2)
    saved_paths["metadata"] = meta_path

    return saved_paths


def generate_synthetic_cohort(
    output_dir: Path,
    num_cases: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    base_seed: int = 100,
) -> dict[str, Any]:
    """Generate an entire synthetic BraTS-style cohort with train/val/test splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    severities = ["mild", "moderate", "severe"]
    case_names: list[str] = []
    cases_meta: list[dict[str, Any]] = []

    for i in range(num_cases):
        case_id = f"BraTS-GLI-SYN{i+1:03d}-000"
        severity = severities[i % len(severities)]
        seed = base_seed + i

        case_data = generate_synthetic_case(
            case_id=case_id,
            shape=shape,
            tumor_severity=severity,
            seed=seed,
        )
        save_case_nifti(case_data, output_dir)
        case_names.append(case_id)
        cases_meta.append(case_data["metadata"])

    # Create dataset splits
    n_train = max(1, int(num_cases * train_ratio))
    n_val = max(1, int(num_cases * val_ratio))
    train_cases = case_names[:n_train]
    val_cases = case_names[n_train : n_train + n_val]
    test_cases = case_names[n_train + n_val :]
    if not test_cases:
        test_cases = val_cases  # fallback for tiny cohorts

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    cohort_manifest = {
        "dataset_name": "Synthetic Brain Tumor MRI Cohort",
        "total_cases": num_cases,
        "shape": list(shape),
        "splits": splits,
        "cases": cases_meta,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cohort_manifest, f, indent=2)

    return cohort_manifest


def main():
    parser = argparse.ArgumentParser(description="Synthetic Brain MRI Dataset Generator")
    parser.add_argument("--num-cases", type=int, default=6, help="Number of cases to generate")
    parser.add_argument("--out-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--shape", type=int, nargs=3, default=[128, 128, 128], help="Volume dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    print(f"Generating {args.num_cases} synthetic MRI cases in {out_path}...")
    manifest = generate_synthetic_cohort(
        output_dir=out_path,
        num_cases=args.num_cases,
        shape=tuple(args.shape),
        base_seed=args.seed,
    )
    print(f"Done! Created {len(manifest['cases'])} cases.")
    print(f"Splits saved at: {out_path / 'splits.json'}")


if __name__ == "__main__":
    main()
