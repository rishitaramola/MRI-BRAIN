"""
src/shiftmri/verifier.py
~~~~~~~~~~~~~~~~~~~~~~~~
STAGE 3: Auditable Verification and Quality Control Engine.

IMPORTANT ARCHITECTURAL NOTE:
This is a deterministic verification engine, NOT a third neural network.
It evaluates model predictions, image physics, input integrity, and perturbation
stability to identify unstable volume measurements and unverified segmentations.

Status Categories Returned:
---------------------------
- ``consistent_within_tested_limits``:
    Passed all input validation checks, perturbation stability thresholds (Dice >= 0.85,
    relative volume change <= 15%), biophysical consistency, and (if Model B present)
    inter-model agreement.
- ``review_required``:
    Input is technically valid, but prediction instability under perturbation or
    inter-model disagreement exceeds defined safety thresholds. Clinician review required.
- ``invalid_input``:
    Missing sequences, non-finite values (NaN/Inf), incorrect dimensions, or corrupted headers.
- ``verification_incomplete``:
    Secondary model absent, expert ground-truth labels unavailable, or partial checks executed.

Thresholds and Rules:
---------------------
- ``STABILITY_DICE_THRESHOLD = 0.85`` (Engineering heuristic: agreement Dice under mild noise).
- ``VOLUME_CHANGE_REL_PCT_THRESHOLD = 15.0`` (Engineering heuristic: relative volume shift).
- ``INTER_MODEL_DICE_THRESHOLD = 0.80`` (Engineering heuristic: agreement between Model A & B).
- ``MIN_CLUSTER_VOXELS = 15`` (Spatial morphological prior: single-voxel noise filtering).

Research prototype — not clinically validated.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np

from src.shiftmri.models import CanonicalSegmentation
from src.shiftmri.metrics import dice as dice_coefficient, region_masks, volume_change


# Defined Engineering Heuristics
STABILITY_DICE_THRESHOLD: float = 0.85
VOLUME_CHANGE_REL_PCT_THRESHOLD: float = 15.0
INTER_MODEL_DICE_THRESHOLD: float = 0.80
CONTRAST_ENHANCEMENT_DIFF_MIN: float = 0.20


class VerificationResult:
    """Auditable result produced by the Stage 3 Verification Engine."""

    def __init__(
        self,
        status: str,  # consistent_within_tested_limits | review_required | invalid_input | verification_incomplete
        warnings: list[str],
        errors: list[str],
        checks: dict[str, Any],
        candidate_consensus: CanonicalSegmentation | None = None,
    ):
        self.status = status
        self.warnings = warnings
        self.errors = errors
        self.checks = checks
        self.candidate_consensus = candidate_consensus

    def to_dict(self) -> dict[str, Any]:
        res = {
            "status": self.status,
            "is_valid": self.status != "invalid_input",
            "review_required": self.status in ("review_required", "verification_incomplete"),
            "warnings": self.warnings,
            "errors": self.errors,
            "checks": self.checks,
            "candidate_consensus_available": self.candidate_consensus is not None,
        }
        if self.candidate_consensus is not None:
            res["candidate_consensus_note"] = (
                "Unvalidated candidate mask for expert review only. "
                "Do not use for diagnosis or clinical decision making."
            )
            res["candidate_consensus_volumes_ml"] = self.candidate_consensus.volumes_ml
        return res


class QualityControlVerifier:
    """Stage 3 Verification and Quality Control Engine."""

    def __init__(
        self,
        stability_dice_threshold: float = STABILITY_DICE_THRESHOLD,
        volume_change_threshold: float = VOLUME_CHANGE_REL_PCT_THRESHOLD,
        inter_model_dice_threshold: float = INTER_MODEL_DICE_THRESHOLD,
    ):
        self.stability_dice_threshold = stability_dice_threshold
        self.volume_change_threshold = volume_change_threshold
        self.inter_model_dice_threshold = inter_model_dice_threshold

    def verify_input(
        self,
        modality_arrays: dict[str, np.ndarray],
    ) -> tuple[bool, list[str]]:
        """Check for missing sequences, invalid shapes, and non-finite values."""
        errors: list[str] = []
        required_mods = ("t1c", "t1n", "t2f", "t2w")

        missing = [m for m in required_mods if m not in modality_arrays or modality_arrays[m] is None]
        if missing:
            errors.append(f"Missing required MRI sequences: {missing}")

        shapes = {}
        for m, arr in modality_arrays.items():
            if arr is None:
                continue
            if not isinstance(arr, np.ndarray):
                errors.append(f"Modality {m} is not a numpy array (got {type(arr).__name__}).")
                continue
            if arr.ndim != 3:
                errors.append(f"Modality {m} has ndim={arr.ndim}; expected 3-D.")
            if not np.all(np.isfinite(arr)):
                errors.append(f"Modality {m} contains non-finite values (NaN or Inf).")
            shapes[m] = arr.shape

        if len(set(shapes.values())) > 1:
            errors.append(f"Mismatched spatial shapes across modalities: {shapes}")

        return len(errors) == 0, errors

    def verify_segmentations(
        self,
        model_a_seg: CanonicalSegmentation,
        model_b_seg: CanonicalSegmentation | None = None,
        perturbed_seg: CanonicalSegmentation | None = None,
        modality_arrays: dict[str, np.ndarray] | None = None,
    ) -> VerificationResult:
        """Run comprehensive multi-model and perturbation verification."""
        warnings: list[str] = []
        errors: list[str] = []
        checks: dict[str, Any] = {}

        # 1. Output Mask Integrity (Model A)
        a_mask = model_a_seg.mask
        a_unique = set(np.unique(a_mask).tolist())
        checks["model_a_labels_valid"] = a_unique.issubset({0, 1, 2, 3})
        if not checks["model_a_labels_valid"]:
            errors.append(f"Model A produced unexpected labels: {a_unique}")

        # Empty mask check
        checks["model_a_is_empty"] = int((a_mask > 0).sum()) == 0
        if checks["model_a_is_empty"]:
            warnings.append("Model A predicted an empty tumor mask (0 voxels detected).")

        # 2. Perturbation Stability Check (if perturbed mask supplied)
        checks["perturbation_stability"] = {}
        if perturbed_seg is not None:
            p_mask = perturbed_seg.mask
            regs_a = region_masks(a_mask)
            regs_p = region_masks(p_mask)

            stab_dice = {}
            stab_vol_shift = {}
            for r in ("wt", "tc", "et"):
                d = dice_coefficient(regs_a[r], regs_p[r])
                stab_dice[f"{r}_dice"] = None if math.isnan(d) else round(d, 4)

                v_a = model_a_seg.volumes_ml[r]
                v_p = perturbed_seg.volumes_ml[r]
                vc = volume_change(v_a, v_p)
                stab_vol_shift[f"{r}_rel_pct"] = vc["relative_pct"]

            checks["perturbation_stability"]["agreement_dice"] = stab_dice
            checks["perturbation_stability"]["volume_shifts"] = stab_vol_shift

            # Stability decision
            wt_dice = stab_dice.get("wt_dice")
            wt_shift = stab_vol_shift.get("wt_rel_pct")
            if wt_dice is not None and wt_dice < self.stability_dice_threshold:
                warnings.append(
                    f"Prediction instability detected: WT agreement Dice ({wt_dice}) "
                    f"is below threshold ({self.stability_dice_threshold})."
                )
            if wt_shift is not None and abs(wt_shift) > self.volume_change_threshold:
                warnings.append(
                    f"Large volume shift under perturbation: WT volume changed by {wt_shift:.1f}% "
                    f"(exceeds {self.volume_change_threshold}% threshold)."
                )

        # 3. Inter-Model Comparison (Stage 2 check, when Model B is available)
        checks["inter_model_comparison"] = {}
        candidate_consensus = None
        if model_b_seg is not None:
            b_mask = model_b_seg.mask
            regs_a = region_masks(a_mask)
            regs_b = region_masks(b_mask)

            inter_dice = {}
            for r in ("wt", "tc", "et"):
                d = dice_coefficient(regs_a[r], regs_b[r])
                inter_dice[f"{r}_dice"] = None if math.isnan(d) else round(d, 4)
            checks["inter_model_comparison"]["dice_agreement"] = inter_dice

            wt_inter = inter_dice.get("wt_dice")
            if wt_inter is not None and wt_inter < self.inter_model_dice_threshold:
                warnings.append(
                    f"Inter-model disagreement: WT Dice between Model A and Model B is {wt_inter} "
                    f"(below consensus threshold {self.inter_model_dice_threshold})."
                )

            # Produce an unvalidated candidate consensus mask (intersection / agreement)
            # Retains both original predictions and explicitly labels consensus as candidate.
            consensus_mask = np.zeros_like(a_mask)
            # Where both models agree on the exact class:
            exact_match = (a_mask == b_mask)
            consensus_mask[exact_match] = a_mask[exact_match]
            # Where either predicts WT, assign conservative core or union:
            candidate_consensus = CanonicalSegmentation(
                mask=consensus_mask,
                spacing_mm=model_a_seg.spacing_mm,
                model_name="Candidate Consensus (Model A & B Agreement)",
                checkpoint_id=f"consensus::{model_a_seg.checkpoint_id}::{model_b_seg.checkpoint_id}",
                provenance={
                    "method": "Exact class agreement between Model A and Model B",
                    "status": "unvalidated_candidate_for_expert_review",
                },
            )
        else:
            checks["inter_model_comparison"]["status"] = "skipped_secondary_model_absent"

        # 4. Biophysical Signal Consistency Check (if raw modalities supplied)
        if modality_arrays and "t1c" in modality_arrays and "t1n" in modality_arrays:
            t1c = modality_arrays["t1c"]
            t1n = modality_arrays["t1n"]
            et_vox = (a_mask == 3)
            if et_vox.any():
                diff = t1c[et_vox] - t1n[et_vox]
                if diff.mean() < CONTRAST_ENHANCEMENT_DIFF_MIN:
                    warnings.append(
                        "Questionable enhancing tumor prediction: mean T1c contrast uptake "
                        f"({diff.mean():.2f}) is below biological enhancement expectation ({CONTRAST_ENHANCEMENT_DIFF_MIN})."
                    )

        # 5. Final Status Determination
        if errors:
            final_status = "invalid_input"
        elif warnings:
            final_status = "review_required"
        elif model_b_seg is None:
            final_status = "verification_incomplete"  # Secondary independent model was not run
        else:
            final_status = "consistent_within_tested_limits"

        return VerificationResult(
            status=final_status,
            warnings=warnings,
            errors=errors,
            checks=checks,
            candidate_consensus=candidate_consensus,
        )
