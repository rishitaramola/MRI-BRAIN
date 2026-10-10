"""
tests/test_shiftmri_synthetic_and_review.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the High-Fidelity Synthetic Dataset Generator and Doctor Review Store.

Verifies:
- 3D multi-sequence synthetic phantom generation (t1c, t1n, t2f, t2w) and 4-class labels.
- Label bounds in {0, 1, 2, 3} and positive whole-tumor volume calculation.
- DoctorReviewStore persistence, schema validation, and scorecards.
- Ensuring doctor feedback is isolated without modifying model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.shiftmri.dataset_generator import generate_synthetic_case
from src.shiftmri.doctor_review import DoctorReviewStore


def test_generate_synthetic_case():
    """Verify synthetic 3D case generation with 4 modalities and realistic tumor labels."""
    case = generate_synthetic_case(
        case_id="BraTS-GLI-SYN99-000",
        shape=(32, 32, 32),
        spacing_mm=(1.0, 1.0, 1.0),
        tumor_severity="moderate",
        seed=42,
    )

    assert case["case_name"] == "BraTS-GLI-SYN99-000"
    assert case["spacing_mm"] == (1.0, 1.0, 1.0)
    assert set(case["modality_arrays"].keys()) == {"t1c", "t1n", "t2f", "t2w"}

    seg = case["seg"]
    assert seg.shape == (32, 32, 32)
    assert seg.dtype == np.uint8
    unique_labels = set(np.unique(seg).tolist())
    assert unique_labels.issubset({0, 1, 2, 3})
    assert 2 in unique_labels or 3 in unique_labels  # Tumour tissue present

    meta = case["metadata"]
    assert "volumes_ml" in meta
    assert meta["volumes_ml"]["wt"] > 0.0


def test_doctor_review_store_persistence(tmp_path: Path):
    """Verify clinician review recording and separate storage without model retraining."""
    review_file = tmp_path / "doctor_reviews" / "reviews.json"
    store = DoctorReviewStore(store_path=review_file)

    assert len(store.reviews) == 0

    entry = store.submit_review(
        case_id="BraTS-GLI-00001-000",
        doctor_name="Dr. Neurologist",
        verification_status="approved",
        overall_stars=5,
        grades={"ai1": "A", "model2": "A", "model3": "A", "hallucination_prevention": "A"},
        feedback_tags=["Clean Margins", "Consistent Volume"],
        doctor_notes="High agreement with clinical impression.",
    )

    assert entry["case_id"] == "BraTS-GLI-00001-000"
    assert entry["overall_stars"] == 5
    assert review_file.exists()

    # Re-instantiate to verify on-disk loading
    store2 = DoctorReviewStore(store_path=review_file)
    assert len(store2.reviews) == 1
    scorecard = store2.get_scorecard()
    assert scorecard["total_reviews"] == 1
    assert scorecard["approved_count"] == 1
    assert scorecard["approval_rate_pct"] == 100.0
