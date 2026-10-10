"""
src/shiftmri/data.py
~~~~~~~~~~~~~~~~~~~~
Case loading and result serialisation.

Two loading modes
-----------------
``processed``: reads {case_name}-{modality}.nii.gz from data/processed/{case_name}/.
    Spacing is taken from the T1c header.  Seg is optional.

``raw``:       reads {case_name}-{modality}.nii.gz from the raw BraTS directory.
    File naming follows BraTS 2023 GLI convention:
      BraTS-GLI-NNNNN-NNN-{t1c,t1n,t2f,t2w,seg}.nii.gz

Missing files are reported together as a single FileNotFoundError listing all
absent paths rather than failing on the first.

Output serialisation
--------------------
write_csv(): appends rows to a CSV (creates header if missing).
write_json(): strict JSON; Python float NaN → JSON null; no NaN in the output.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODALITIES = ("t1c", "t1n", "t2f", "t2w")


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def load_case(
    case_dir: Path,
    mode: str = "processed",
    load_seg: bool = True,
) -> dict[str, Any]:
    """Load one BraTS case from disk.

    Parameters
    ----------
    case_dir:
        Directory containing the NIfTI files.  The directory name is used as
        the case name.
    mode:
        ``"processed"`` or ``"raw"``.  Both use the same filename convention
        so the distinction matters only for documentation / ``data_source`` labelling.
    load_seg:
        Whether to load the segmentation mask.  When False (or the file is absent
        with load_seg=True), ``seg`` is None in the returned dict.

    Returns
    -------
    Dict with keys:
        ``case_name`` (str)
        ``spacing_mm`` (tuple[float, float, float]) from T1c header
        ``modality_arrays`` (dict str→np.ndarray, float32, after loading only — NOT preprocessed)
        ``seg`` (np.ndarray uint8 | None)
        ``mode`` (str)

    Raises
    ------
    FileNotFoundError listing ALL missing required files in one message.
    """
    import nibabel as nib  # lazy import — keeps module importable without nibabel

    case_dir = Path(case_dir)
    case_name = case_dir.name

    required_paths = {
        mod: case_dir / f"{case_name}-{mod}.nii.gz"
        for mod in MODALITIES
    }
    seg_path = case_dir / f"{case_name}-seg.nii.gz"

    # Collect all missing files before raising
    missing = [str(p) for p in required_paths.values() if not p.exists()]
    if load_seg and not seg_path.exists():
        missing.append(str(seg_path))
    if missing:
        raise FileNotFoundError(
            f"Missing files for case '{case_name}':\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    t1c_img = nib.load(required_paths["t1c"])
    spacing_mm = tuple(float(v) for v in t1c_img.header.get_zooms()[:3])

    modality_arrays: dict[str, np.ndarray] = {
        mod: nib.load(p).get_fdata().astype(np.float32)
        for mod, p in required_paths.items()
    }

    seg: np.ndarray | None = None
    if load_seg and seg_path.exists():
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)

    return {
        "case_name": case_name,
        "spacing_mm": spacing_mm,
        "modality_arrays": modality_arrays,
        "seg": seg,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _nan_to_null(obj: Any) -> Any:
    """Recursively replace Python NaN floats with None for JSON serialisation."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_null(v) for v in obj]
    return obj


def write_json(data: dict | list, path: Path) -> None:
    """Write *data* as strict JSON.  NaN and Inf become null.

    Parameters
    ----------
    data:
        Serialisable dict or list (may contain Python float NaN).
    path:
        Output path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _nan_to_null(data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, ensure_ascii=False)


def write_csv(
    rows: Sequence[dict[str, Any]],
    path: Path,
    append: bool = False,
) -> None:
    """Write *rows* to a CSV file.

    Parameters
    ----------
    rows:
        Sequence of dicts with the same keys.
    path:
        Output path.  Parent directories are created.
    append:
        If True and the file exists, rows are appended without re-writing the header.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    write_header = not (append and path.exists())

    mode = "a" if (append and path.exists()) else "w"
    with open(path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            # Replace NaN with empty string in CSV
            clean_row = {
                k: ("" if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                for k, v in row.items()
            }
            writer.writerow(clean_row)
