"""
tests/test_shiftmri_verifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for STAGE 2 Model Adapters and STAGE 3 Quality Control Verifier.

Tests:
- Input validation (missing sequences, non-finite values, mismatched shapes).
- Label validation (rejection of invalid labels).
- Perturbation stability evaluation and review_required triggers.
- Multi-model comparison and candidate consensus mask generation.
- Model B blocker reporting when secondary weights are unbundled.
"""

import numpy as np
import pytest

from src.shiftmri.models import (
    CanonicalSegmentation,
    MonaiBraTSBundleAdapter,
    NeuralSegModelAdapter,
)
from src.shiftmri.verifier import (
    INTER_MODEL_DICE_THRESHOLD,
    STABILITY_DICE_THRESHOLD,
    VOLUME_CHANGE_REL_PCT_THRESHOLD,
    QualityControlVerifier,
)


class TestInputVerification:
    def test_valid_input_passes(self):
        verifier = QualityControlVerifier()
        shape = (16, 16, 16)
        mods = {
            "t1c": np.zeros(shape, dtype=np.float32),
            "t1n": np.zeros(shape, dtype=np.float32),
            "t2f": np.zeros(shape, dtype=np.float32),
            "t2w": np.zeros(shape, dtype=np.float32),
        }
        ok, errors = verifier.verify_input(mods)
        assert ok is True
        assert len(errors) == 0

    def test_missing_sequence_fails(self):
        verifier = QualityControlVerifier()
        shape = (16, 16, 16)
        # Missing t2w
        mods = {
            "t1c": np.zeros(shape, dtype=np.float32),
            "t1n": np.zeros(shape, dtype=np.float32),
            "t2f": np.zeros(shape, dtype=np.float32),
        }
        ok, errors = verifier.verify_input(mods)
        assert ok is False
        assert any("Missing required MRI sequences" in e for e in errors)

    def test_nan_values_fail(self):
        verifier = QualityControlVerifier()
        shape = (16, 16, 16)
        bad_t1c = np.zeros(shape, dtype=np.float32)
        bad_t1c[2, 2, 2] = np.nan
        mods = {
            "t1c": bad_t1c,
            "t1n": np.zeros(shape, dtype=np.float32),
            "t2f": np.zeros(shape, dtype=np.float32),
            "t2w": np.zeros(shape, dtype=np.float32),
        }
        ok, errors = verifier.verify_input(mods)
        assert ok is False
        assert any("non-finite" in e for e in errors)

    def test_mismatched_shapes_fail(self):
        verifier = QualityControlVerifier()
        mods = {
            "t1c": np.zeros((16, 16, 16), dtype=np.float32),
            "t1n": np.zeros((16, 16, 16), dtype=np.float32),
            "t2f": np.zeros((16, 16, 16), dtype=np.float32),
            "t2w": np.zeros((16, 16, 20), dtype=np.float32),
        }
        ok, errors = verifier.verify_input(mods)
        assert ok is False
        assert any("Mismatched spatial shapes" in e for e in errors)


class TestModelAdapters:
    def test_model_b_reports_blocker_when_checkpoint_missing(self, tmp_path):
        adapter = MonaiBraTSBundleAdapter(checkpoint_path=tmp_path / "missing_bundle.pt")
        assert adapter.is_available is False
        assert "not bundled locally" in adapter.blocker_reason
        assert "download prohibited" in adapter.blocker_reason

    def test_canonical_segmentation_calculates_physical_volumes(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1  # 27 voxels of necrotic core
        mask[5:8, 5:8, 5:8] = 3  # 27 voxels of enhancing tumor
        # Spacing: 2.0 mm isotropic -> voxel vol = 8.0 mm^3 = 0.008 mL
        canon = CanonicalSegmentation(
            mask=mask,
            spacing_mm=(2.0, 2.0, 2.0),
            model_name="TestModel",
            checkpoint_id="ckpt_123",
        )
        assert canon.voxel_counts["wt"] == 54
        assert canon.voxel_counts["tc"] == 54
        assert canon.voxel_counts["et"] == 27
        assert canon.voxel_counts["ncr"] == 27
        assert abs(canon.volumes_ml["wt"] - (54 * 0.008)) < 1e-4

    def test_canonical_segmentation_rejects_invalid_labels(self):
        bad_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        bad_mask[2, 2, 2] = 5  # invalid label
        with pytest.raises(ValueError, match="invalid labels"):
            CanonicalSegmentation(
                mask=bad_mask,
                spacing_mm=(1.0, 1.0, 1.0),
                model_name="BadModel",
                checkpoint_id="bad",
            )


class TestStage3Verifier:
    def test_verification_incomplete_when_model_b_missing(self):
        verifier = QualityControlVerifier()
        mask_a = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_a[4:10, 4:10, 4:10] = 2  # edema
        canon_a = CanonicalSegmentation(mask_a, (1.0, 1.0, 1.0), "Model A", "ck_a")

        result = verifier.verify_segmentations(model_a_seg=canon_a, model_b_seg=None)
        assert result.status == "verification_incomplete"
        assert result.candidate_consensus is None

    def test_review_required_on_inter_model_disagreement(self):
        verifier = QualityControlVerifier()
        # Model A predicts lesion in left quadrant
        mask_a = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_a[2:6, 2:6, 2:6] = 3

        # Model B predicts lesion in right quadrant (complete disagreement)
        mask_b = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_b[10:14, 10:14, 10:14] = 3

        canon_a = CanonicalSegmentation(mask_a, (1.0, 1.0, 1.0), "Model A", "ck_a")
        canon_b = CanonicalSegmentation(mask_b, (1.0, 1.0, 1.0), "Model B", "ck_b")

        result = verifier.verify_segmentations(model_a_seg=canon_a, model_b_seg=canon_b)
        assert result.status == "review_required"
        assert any("Inter-model disagreement" in w for w in result.warnings)

    def test_consistent_within_tested_limits_on_high_agreement(self):
        verifier = QualityControlVerifier()
        # Both models agree almost entirely
        mask_a = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_a[4:12, 4:12, 4:12] = 2

        mask_b = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_b[4:12, 4:12, 4:12] = 2

        canon_a = CanonicalSegmentation(mask_a, (1.0, 1.0, 1.0), "Model A", "ck_a")
        canon_b = CanonicalSegmentation(mask_b, (1.0, 1.0, 1.0), "Model B", "ck_b")

        result = verifier.verify_segmentations(model_a_seg=canon_a, model_b_seg=canon_b)
        assert result.status == "consistent_within_tested_limits"
        assert result.candidate_consensus is not None
        assert result.candidate_consensus.provenance["status"] == "unvalidated_candidate_for_expert_review"

    def test_perturbation_instability_triggers_review_required(self):
        verifier = QualityControlVerifier()
        # Baseline mask has tumor
        mask_a = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_a[4:12, 4:12, 4:12] = 3

        # Perturbed mask collapsed (fragile prediction)
        mask_p = np.zeros((16, 16, 16), dtype=np.uint8)
        mask_p[4:6, 4:6, 4:6] = 3  # tiny remnant

        canon_a = CanonicalSegmentation(mask_a, (1.0, 1.0, 1.0), "Model A", "ck_a")
        canon_p = CanonicalSegmentation(mask_p, (1.0, 1.0, 1.0), "Perturbed", "ck_p")

        result = verifier.verify_segmentations(model_a_seg=canon_a, perturbed_seg=canon_p)
        assert result.status == "review_required"
        assert any("instability detected" in w.lower() or "large volume shift" in w.lower() for w in result.warnings)
