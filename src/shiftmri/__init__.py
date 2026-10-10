"""
src/shiftmri/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~
ShiftMRI — Multi-Model Brain Tumor Segmentation Audit.

Research prototype. Not clinically validated. Not a diagnostic tool.
"""

from src.shiftmri.models import (
    CanonicalSegmentation,
    MonaiBraTSBundleAdapter,
    NeuralSegModelAdapter,
)
from src.shiftmri.verifier import (
    QualityControlVerifier,
    VerificationResult,
)
from src.shiftmri.doctor_review import DoctorReviewStore
from src.shiftmri.dataset_generator import (
    generate_synthetic_case,
    generate_synthetic_cohort,
)
from src.shiftmri.experiment import (
    build_report,
    evaluate_case,
)

__all__ = [
    "CanonicalSegmentation",
    "NeuralSegModelAdapter",
    "MonaiBraTSBundleAdapter",
    "QualityControlVerifier",
    "VerificationResult",
    "DoctorReviewStore",
    "generate_synthetic_case",
    "generate_synthetic_cohort",
    "build_report",
    "evaluate_case",
]
