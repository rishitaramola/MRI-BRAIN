"""
tests/test_shiftmri_perturbations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src.shiftmri.perturbations.

All tests use synthetic numpy arrays — no checkpoint needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shiftmri.perturbations import (
    downsample_restore,
    gaussian_noise,
    intensity_scale_shift,
    validate_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_image(shape=(4, 16, 16, 16), seed=0) -> np.ndarray:
    """Return a float32 array that looks like a preprocessed 4-channel volume.

    Background (outside a centre sphere) is exactly 0.
    """
    rng = np.random.default_rng(seed)
    img = rng.standard_normal(shape).astype(np.float32)
    # Mask out background (corners)
    H, W, D = shape[1], shape[2], shape[3]
    cx, cy, cz = H // 2, W // 2, D // 2
    xs, ys, zs = np.mgrid[0:H, 0:W, 0:D]
    brain = ((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) <= (H // 2 - 1) ** 2
    img[:, ~brain] = 0.0
    return img


@pytest.fixture
def image():
    return _make_image()


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:
    def test_valid_passes(self, image):
        validate_image(image)  # should not raise

    def test_wrong_ndim(self, image):
        with pytest.raises(ValueError, match="4-D"):
            validate_image(image[0])  # 3-D

    def test_integer_dtype(self, image):
        with pytest.raises(ValueError, match="float"):
            validate_image(image.astype(np.int16))

    def test_nan_raises(self, image):
        bad = image.copy()
        bad[0, 1, 1, 1] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            validate_image(bad)

    def test_inf_raises(self, image):
        bad = image.copy()
        bad[0, 1, 1, 1] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            validate_image(bad)

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            validate_image(np.zeros((4, 0, 16, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def test_determinism(self, image):
        a = gaussian_noise(image, sigma=0.1, seed=1)
        b = gaussian_noise(image, sigma=0.1, seed=1)
        np.testing.assert_array_equal(a, b)

    def test_seed_sensitivity(self, image):
        a = gaussian_noise(image, sigma=0.1, seed=1)
        b = gaussian_noise(image, sigma=0.1, seed=2)
        assert not np.array_equal(a, b), "Different seeds should yield different noise"

    def test_input_not_mutated(self, image):
        original = image.copy()
        gaussian_noise(image, sigma=0.1, seed=7)
        np.testing.assert_array_equal(image, original)

    def test_output_shape_dtype(self, image):
        out = gaussian_noise(image, sigma=0.1, seed=0)
        assert out.shape == image.shape
        assert out.dtype == np.float32

    def test_background_preserved(self, image):
        out = gaussian_noise(image, sigma=1.0, seed=0)
        bg = np.all(image == 0, axis=0)
        assert np.all(out[:, bg] == 0.0), "Background voxels must remain 0"

    def test_zero_sigma_identity(self, image):
        out = gaussian_noise(image, sigma=0.0, seed=0)
        np.testing.assert_allclose(out, image, atol=1e-6)

    def test_negative_sigma_raises(self, image):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_noise(image, sigma=-0.1, seed=0)

    def test_string_seed(self, image):
        a = gaussian_noise(image, sigma=0.1, seed="case_001")
        b = gaussian_noise(image, sigma=0.1, seed="case_001")
        np.testing.assert_array_equal(a, b)

    def test_string_seed_different_cases(self, image):
        a = gaussian_noise(image, sigma=0.1, seed="case_001")
        b = gaussian_noise(image, sigma=0.1, seed="case_002")
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Intensity scale/shift
# ---------------------------------------------------------------------------

class TestIntensityScaleShift:
    def test_determinism(self, image):
        a = intensity_scale_shift(image, scale=1.1, shift=0.0, seed=0)
        b = intensity_scale_shift(image, scale=1.1, shift=0.0, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_input_not_mutated(self, image):
        original = image.copy()
        intensity_scale_shift(image, scale=0.9, shift=0.1, seed=0)
        np.testing.assert_array_equal(image, original)

    def test_output_shape_dtype(self, image):
        out = intensity_scale_shift(image, scale=1.0, shift=0.0, seed=0)
        assert out.shape == image.shape
        assert out.dtype == np.float32

    def test_background_preserved(self, image):
        out = intensity_scale_shift(image, scale=2.0, shift=5.0, seed=0)
        bg = np.all(image == 0, axis=0)
        assert np.all(out[:, bg] == 0.0), "Background voxels must remain 0"

    def test_identity_transform(self, image):
        out = intensity_scale_shift(image, scale=1.0, shift=0.0, seed=0)
        np.testing.assert_allclose(out, image, atol=1e-6)

    def test_scale_effect(self, image):
        out = intensity_scale_shift(image, scale=2.0, shift=0.0, seed=0)
        mask = np.any(image != 0, axis=0)
        np.testing.assert_allclose(
            out[:, mask], image[:, mask] * 2.0, atol=1e-5
        )

    def test_shift_effect(self, image):
        out = intensity_scale_shift(image, scale=1.0, shift=1.0, seed=0)
        mask = np.any(image != 0, axis=0)
        np.testing.assert_allclose(
            out[:, mask], image[:, mask] + 1.0, atol=1e-5
        )

    def test_prescale_cancelled_by_zscore(self):
        """
        Proof that applying scale/shift BEFORE z-scoring is cancelled by normalisation.

        This is why ShiftMRI applies perturbations AFTER preprocessing
        (in z-score space), not before.  If the perturbation were applied
        pre-preprocessing, it would be invisible to the model.

        We show: z_score(scale * x + shift) == z_score(x) for any finite scale>0.
        """
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100,)).astype(np.float32) + 2.0  # simulate brain signal
        x[x < 0] = 0.0  # background
        nz = x != 0

        def zscore(arr):
            out = arr.copy()
            mu, sigma = arr[nz].mean(), arr[nz].std()
            out[nz] = (arr[nz] - mu) / sigma
            return out

        z_original = zscore(x)
        z_scaled = zscore(x * 3.0 + 1.0)
        # Pre-z-score scaling produces the SAME normalised array
        np.testing.assert_allclose(z_original[nz], z_scaled[nz], atol=1e-5)


# ---------------------------------------------------------------------------
# Downsample restore
# ---------------------------------------------------------------------------

class TestDownsampleRestore:
    def test_determinism(self, image):
        a = downsample_restore(image, factor=2, axis=2, seed=0)
        b = downsample_restore(image, factor=2, axis=2, seed=0)
        np.testing.assert_array_equal(a, b)

    def test_input_not_mutated(self, image):
        original = image.copy()
        downsample_restore(image, factor=2, axis=2, seed=0)
        np.testing.assert_array_equal(image, original)

    def test_output_shape_dtype(self, image):
        out = downsample_restore(image, factor=2, axis=2, seed=0)
        assert out.shape == image.shape
        assert out.dtype == np.float32

    def test_background_preserved(self, image):
        out = downsample_restore(image, factor=2, axis=2, seed=0)
        bg = np.all(image == 0, axis=0)
        assert np.all(out[:, bg] == 0.0), "Background voxels must remain 0"

    def test_all_axes(self, image):
        for axis in (0, 1, 2):
            out = downsample_restore(image, factor=2, axis=axis, seed=0)
            assert out.shape == image.shape

    def test_factor_3(self, image):
        out = downsample_restore(image, factor=3, axis=2, seed=0)
        assert out.shape == image.shape

    def test_bad_factor_raises(self, image):
        with pytest.raises(ValueError, match="factor"):
            downsample_restore(image, factor=1, axis=2, seed=0)

    def test_bad_axis_raises(self, image):
        with pytest.raises(ValueError, match="axis"):
            downsample_restore(image, factor=2, axis=3, seed=0)

    def test_output_differs_from_input(self, image):
        out = downsample_restore(image, factor=2, axis=2, seed=0)
        # Downsampling should change at least some brain voxels
        mask = np.any(image != 0, axis=0)
        assert not np.allclose(out[:, mask], image[:, mask]), \
            "Downsampling should alter brain voxels"
