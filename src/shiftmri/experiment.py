"""
src/shiftmri/experiment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestration: run baseline + perturbations for one case, collect results,
and execute Stage 3 Verification & Quality Control audit.

Multi-Stage Architecture:
-------------------------
- STAGE 1: Primary Model A (NeuralSeg MONAI 3D UNet)
- STAGE 2: Independent Model B (MONAI BraTS bundle adapter; records exact blocker if unbundled)
- STAGE 3: Deterministic Quality Control & Verification Engine (NOT a 3rd neural network)
    Emits auditable verification status:
      - ``consistent_within_tested_limits``
      - ``review_required``
      - ``invalid_input``
      - ``verification_incomplete``

Terminology note:
-----------------
"agreement" metrics (Dice/HD95 of perturbed vs baseline prediction) measure
prediction STABILITY, not accuracy. High agreement does NOT mean either
prediction is clinically correct. Ground-truth metrics are labelled ``gt_*``
and are only computed when an expert reference segmentation is available.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import math
import traceback
from typing import Any, Callable, Sequence

import numpy as np

from src.shiftmri.metrics import (
    region_volumes_ml,
    segmentation_metrics,
    volume_change,
)
from src.shiftmri.perturbations import (
    downsample_restore,
    gaussian_noise,
    intensity_scale_shift,
)
from src.shiftmri.models import CanonicalSegmentation
from src.shiftmri.verifier import QualityControlVerifier


# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------

def _apply_perturbation(
    image: np.ndarray,
    perturb_cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Dispatch to the correct perturbation function from a config dict.

    Config dict keys: ``type`` + perturbation-specific params.
    Supported types: ``gaussian_noise``, ``intensity_scale_shift``, ``downsample_restore``.
    """
    t = perturb_cfg["type"]
    if t == "gaussian_noise":
        return gaussian_noise(image, sigma=perturb_cfg["sigma"], seed=seed)
    elif t == "intensity_scale_shift":
        return intensity_scale_shift(
            image,
            scale=perturb_cfg["scale"],
            shift=perturb_cfg["shift"],
            seed=seed,
        )
    elif t == "downsample_restore":
        return downsample_restore(
            image,
            factor=perturb_cfg["factor"],
            axis=perturb_cfg["axis"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown perturbation type: {t!r}")


# ---------------------------------------------------------------------------
# Single-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(
    case_name: str,
    image: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    perturbations: list[dict[str, Any]],
    spacing_mm: Sequence[float],
    gt_seg: np.ndarray | None,
    seed: int,
    checkpoint_id: str,
    data_source: str,
    include_hd95: bool = True,
    notes: str = "",
    model_b_predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    modality_arrays: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one case: baseline prediction + perturbations + Stage 3 verification.

    Parameters
    ----------
    case_name:
        Identifier string for the case.
    image:
        Preprocessed float32 array (4, H, W, D).
    predict_fn:
        Callable returned by ``make_predict_fn()``.
    perturbations:
        List of perturbation config dicts (from shiftmri.yaml).
    spacing_mm:
        Voxel spacings in mm.
    gt_seg:
        Ground-truth uint8 label array or None.
    seed:
        Per-case seed (sha256-derived int).
    checkpoint_id:
        From ``inference.checkpoint_id()``.
    data_source:
        E.g. ``"brats2023_gli"`` or ``"synthetic_phantom"``.
    include_hd95:
        Whether to compute HD95 (slow on CPU).
    notes:
        Free-text notes attached to every row.
    model_b_predict_fn:
        Optional independent secondary model prediction function (Stage 2).
    modality_arrays:
        Optional raw modality arrays for biophysical cross-checks.

    Returns
    -------
    List of result rows (dicts). Each row always has ``status`` in
    {``"ok"``, ``"error"``, ``"skipped"``}.
    """
    rows: list[dict[str, Any]] = []

    # --- Baseline prediction (Stage 1: Model A) ---
    try:
        baseline_seg = predict_fn(image)
    except Exception:
        tb = traceback.format_exc()
        rows.append({
            "case_name": case_name,
            "perturbation": "baseline",
            "status": "error",
            "error": tb,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "data_source": data_source,
            "notes": notes,
            "verification_status": "invalid_input",
            "verification_warnings": "Baseline inference crashed.",
        })
        # Cannot continue without baseline
        for pcfg in perturbations:
            rows.append({
                "case_name": case_name,
                "perturbation": pcfg.get("name", pcfg["type"]),
                "status": "skipped",
                "error": "Baseline failed; skipped.",
                "seed": seed,
                "checkpoint_id": checkpoint_id,
                "data_source": data_source,
                "notes": notes,
                "verification_status": "skipped",
                "verification_warnings": "",
            })
        return rows

    baseline_vols = region_volumes_ml(baseline_seg, spacing_mm)

    # GT metrics for baseline
    gt_metrics_base: dict[str, float] = {}
    if gt_seg is not None:
        try:
            gt_metrics_base = segmentation_metrics(
                baseline_seg, gt_seg, spacing_mm, include_hd95=include_hd95
            )
        except Exception:
            gt_metrics_base = {}

    # Stage 2: Independent Model B prediction (if available)
    model_b_seg = None
    model_b_status = "not_available_locally (MONAI BraTS bundle unbundled; download prohibited)"
    if model_b_predict_fn is not None:
        try:
            model_b_seg = model_b_predict_fn(image)
            model_b_status = "operational"
        except Exception as e:
            model_b_status = f"failed: {e}"

    # Pre-collect perturbed segmentations for Stage 3 verifier
    perturbed_segs: list[tuple[str, np.ndarray]] = []
    perturbed_errors: list[tuple[str, str]] = []

    for pcfg in perturbations:
        label = pcfg.get("name", pcfg["type"])
        try:
            perturbed_image = _apply_perturbation(image, pcfg, seed)
            p_seg = predict_fn(perturbed_image)
            perturbed_segs.append((label, p_seg))
        except Exception:
            tb = traceback.format_exc()
            perturbed_errors.append((label, tb))

    # --- STAGE 3: Run Deterministic Verification Engine ---
    verifier = QualityControlVerifier()
    canon_a = CanonicalSegmentation(
        mask=baseline_seg,
        spacing_mm=spacing_mm,
        model_name="Model A (NeuralSeg)",
        checkpoint_id=checkpoint_id,
    )
    canon_b = (
        CanonicalSegmentation(
            mask=model_b_seg,
            spacing_mm=spacing_mm,
            model_name="Model B",
            checkpoint_id="model_b_secondary",
        )
        if model_b_seg is not None
        else None
    )
    first_pert_seg = (
        CanonicalSegmentation(
            mask=perturbed_segs[0][1],
            spacing_mm=spacing_mm,
            model_name="Perturbed (Mild)",
            checkpoint_id=f"pert::{perturbed_segs[0][0]}",
        )
        if perturbed_segs
        else None
    )

    verif_res = verifier.verify_segmentations(
        model_a_seg=canon_a,
        model_b_seg=canon_b,
        perturbed_seg=first_pert_seg,
        modality_arrays=modality_arrays,
    )

    verification_status = verif_res.status
    verification_warnings = "; ".join(verif_res.warnings) if verif_res.warnings else "none"

    def _make_row(
        label: str,
        perturbed_seg: np.ndarray,
        status: str = "ok",
        error: str = "",
    ) -> dict[str, Any]:
        perturbed_vols = region_volumes_ml(perturbed_seg, spacing_mm)

        # Agreement = perturbed prediction vs baseline prediction (stability)
        try:
            agreement = segmentation_metrics(
                perturbed_seg, baseline_seg, spacing_mm, include_hd95=include_hd95
            )
        except Exception:
            agreement = {}

        # GT accuracy of perturbed prediction + delta vs baseline
        gt_metrics_perturbed: dict[str, float] = {}
        gt_deltas: dict[str, float] = {}
        if gt_seg is not None:
            try:
                gt_metrics_perturbed = segmentation_metrics(
                    perturbed_seg, gt_seg, spacing_mm, include_hd95=include_hd95
                )
                for k in gt_metrics_base:
                    base_v = gt_metrics_base[k]
                    pert_v = gt_metrics_perturbed.get(k, float("nan"))
                    if math.isnan(base_v) or math.isnan(pert_v):
                        gt_deltas[f"delta_{k}"] = float("nan")
                    else:
                        gt_deltas[f"delta_{k}"] = pert_v - base_v
            except Exception:
                pass

        row: dict[str, Any] = {
            "case_name": case_name,
            "perturbation": label,
            "status": status,
            "error": error,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "data_source": data_source,
            "notes": notes,
            # Stage 2 & Stage 3 fields
            "model_b_status": model_b_status,
            "verification_status": verification_status,
            "verification_warnings": verification_warnings,
        }

        # Baseline volumes
        for reg in ("wt", "tc", "et"):
            row[f"baseline_{reg}_vol_ml"] = baseline_vols.get(reg, float("nan"))

        # Perturbed volumes + changes
        for reg in ("wt", "tc", "et"):
            bv = baseline_vols.get(reg, float("nan"))
            pv = perturbed_vols.get(reg, float("nan"))
            row[f"perturbed_{reg}_vol_ml"] = pv
            vc = volume_change(bv, pv)
            row[f"{reg}_vol_signed_ml"] = vc["signed_ml"]
            row[f"{reg}_vol_abs_ml"] = vc["absolute_ml"]
            row[f"{reg}_vol_rel_pct"] = vc["relative_pct"]

        # Agreement (stability) metrics — labelled as such
        for k, v in agreement.items():
            row[f"agreement_{k}"] = v

        # GT accuracy metrics for baseline
        for k, v in gt_metrics_base.items():
            row[f"baseline_gt_{k}"] = v

        # GT accuracy of perturbed + deltas
        for k, v in gt_metrics_perturbed.items():
            row[f"perturbed_gt_{k}"] = v
        for k, v in gt_deltas.items():
            row[k] = v

        return row

    # Baseline row
    rows.append(_make_row("baseline", baseline_seg))

    # Add successfully executed perturbation rows
    for label, p_seg in perturbed_segs:
        rows.append(_make_row(label, p_seg, status="ok"))

    # Add failed perturbation rows
    for label, err in perturbed_errors:
        rows.append({
            "case_name": case_name,
            "perturbation": label,
            "status": "error",
            "error": err,
            "seed": seed,
            "checkpoint_id": checkpoint_id,
            "data_source": data_source,
            "notes": notes,
            "model_b_status": model_b_status,
            "verification_status": verification_status,
            "verification_warnings": verification_warnings,
        })

    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-perturbation summary statistics over all cases.

    Only rows with status ``"ok"`` are included.
    Spearman correlation of agreement_wt_dice vs perturbed_gt_wt_dice is
    reported only when n >= 5 and neither column is constant.
    """
    try:
        from scipy.stats import spearmanr  # lazy
    except ImportError:
        spearmanr = None

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return {"n_ok": 0, "perturbations": {}}

    by_perturb: dict[str, list[dict]] = {}
    for r in ok_rows:
        by_perturb.setdefault(r["perturbation"], []).append(r)

    summary: dict[str, Any] = {"n_ok": len(ok_rows), "perturbations": {}}

    def _median(vals: list) -> float:
        finite = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
        if not finite:
            return float("nan")
        return float(np.median(finite))

    for perturb, prows in by_perturb.items():
        psum: dict[str, Any] = {"n": len(prows)}

        for metric in (
            "agreement_wt_dice", "agreement_tc_dice", "agreement_et_dice",
            "wt_vol_abs_ml", "tc_vol_abs_ml", "et_vol_abs_ml",
            "wt_vol_rel_pct", "tc_vol_rel_pct", "et_vol_rel_pct",
            "perturbed_gt_wt_dice", "perturbed_gt_tc_dice", "perturbed_gt_et_dice",
        ):
            vals = [r.get(metric, float("nan")) for r in prows]
            psum[f"median_{metric}"] = _median(vals)

        # Spearman: agreement_wt_dice vs perturbed_gt_wt_dice
        if spearmanr is not None:
            agr = [r.get("agreement_wt_dice", float("nan")) for r in prows]
            gt_ = [r.get("perturbed_gt_wt_dice", float("nan")) for r in prows]
            pairs = [
                (a, g) for a, g in zip(agr, gt_)
                if not math.isnan(a) and not math.isnan(g)
            ]
            if len(pairs) >= 5:
                a_vals, g_vals = zip(*pairs)
                if len(set(a_vals)) > 1 and len(set(g_vals)) > 1:
                    rho, pval = spearmanr(a_vals, g_vals)
                    psum["spearman_agreement_vs_gt_rho"] = float(rho)
                    psum["spearman_agreement_vs_gt_pval"] = float(pval)
                    psum["spearman_n"] = len(pairs)
                else:
                    psum["spearman_note"] = "constant column — rho undefined"
            else:
                psum["spearman_note"] = f"n={len(pairs)} < 5 — rho not computed"

        summary["perturbations"][perturb] = psum

    return summary


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    rows: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    """Combine run metadata with summary statistics and Stage 1-3 audit counts."""
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_error = sum(1 for r in rows if r.get("status") == "error")
    n_skipped = sum(1 for r in rows if r.get("status") == "skipped")
    unique_cases = len({r.get("case_name") for r in rows})

    has_gt = any(
        not math.isnan(r.get("baseline_gt_wt_dice", float("nan")))
        for r in rows
    )

    baseline_rows = [r for r in rows if r.get("perturbation") == "baseline"]
    verif_status_counts = {
        "consistent_within_tested_limits": sum(
            1 for r in baseline_rows if r.get("verification_status") == "consistent_within_tested_limits"
        ),
        "review_required": sum(
            1 for r in baseline_rows if r.get("verification_status") == "review_required"
        ),
        "verification_incomplete": sum(
            1 for r in baseline_rows if r.get("verification_status") == "verification_incomplete"
        ),
        "invalid_input": sum(
            1 for r in baseline_rows if r.get("verification_status") == "invalid_input"
        ),
    }

    return {
        "shiftmri_version": "0.1.0",
        "disclaimer": (
            "Research prototype. Not clinically validated. "
            "A change in prediction under perturbation does NOT prove either "
            "prediction is clinically wrong or correct. "
            "Agreement metrics measure stability only."
        ),
        "hypothesis": (
            "Can independent model predictions, controlled perturbation testing, "
            "and explicit disagreement checks identify unstable tumor-volume "
            "measurements more effectively than relying on a single segmentation model?"
        ),
        "stages": {
            "stage_1_primary_model": {
                "model": "Model A (NeuralSeg MONAI 3D UNet)",
                "checkpoint_id": run_meta.get("checkpoint_id"),
                "status": "operational",
            },
            "stage_2_independent_verification": {
                "model": "Model B (MONAI BraTS SegResNet Bundle)",
                "status": "not_available_locally" if not run_meta.get("model_b_available") else "operational",
                "blocker": (
                    "Pretrained weights for Model B not bundled in repository; external download prohibited."
                    if not run_meta.get("model_b_available")
                    else None
                ),
            },
            "stage_3_verification_engine": {
                "type": "Deterministic Verification Engine (not a 3rd neural network)",
                "rules": [
                    "Input shape, dimension, and non-finite value checks",
                    "Valid BraTS label verification {0,1,2,3}",
                    "Perturbation stability agreement threshold (Dice >= 0.85, volume change <= 15%)",
                    "Inter-model disagreement audit (Dice >= 0.80) when Model B available",
                    "Biophysical T1c contrast enhancement consistency",
                ],
                "verification_status_counts": verif_status_counts,
            },
        },
        "run_meta": run_meta,
        "counts": {
            "total_rows": n_total,
            "ok": n_ok,
            "error": n_error,
            "skipped": n_skipped,
            "unique_cases": unique_cases,
        },
        "has_ground_truth": has_gt,
        "summary": summarize(rows),
    }
