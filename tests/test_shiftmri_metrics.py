"""
tests/test_shiftmri_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src.shiftmri.metrics.

No checkpoint needed — pure numpy/scipy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shiftmri.metrics import (
    dice,
    hausdorff95,
    region_masks,
    region_volumes_ml,
    segmentation_metrics,
    volume_change,
    voxel_volume_ml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_spacing():
    return (1.0, 1.0, 1.0)


@pytest.fixture
def aniso_spacing():
    return (0.9375, 0.9375, 1.0)   # typical BraTS in-plane vs slice


def _sphere_seg(shape=(20, 20, 20), centre=None, radius=5, label=1) -> np.ndarray:
    if centre is None:
        centre = tuple(s // 2 for s in shape)
    cx, cy, cz = centre
    xs, ys, zs = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    seg = np.zeros(shape, dtype=np.uint8)
    mask = ((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) <= radius ** 2
    seg[mask] = label
    return seg


# ---------------------------------------------------------------------------
# voxel_volume_ml
# ---------------------------------------------------------------------------

class TestVoxelVolumeMl:
    def test_isotropic_1mm(self, unit_spacing):
        assert voxel_volume_ml(unit_spacing) == pytest.approx(0.001)

    def test_anisotropic(self, aniso_spacing):
        expected = (0.9375 * 0.9375 * 1.0) / 1000.0
        assert voxel_volume_ml(aniso_spacing) == pytest.approx(expected)

    def test_zero_spacing_raises(self):
        with pytest.raises(ValueError):
            voxel_volume_ml((0.0, 1.0, 1.0))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            voxel_volume_ml((1.0, 1.0))


# ---------------------------------------------------------------------------
# region_masks
# ---------------------------------------------------------------------------

class TestRegionMasks:
    def test_all_zero_seg(self):
        seg = np.zeros((10, 10, 10), dtype=np.uint8)
        masks = region_masks(seg)
        assert not masks["wt"].any()
        assert not masks["tc"].any()
        assert not masks["et"].any()

    def test_label_membership(self):
        seg = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        masks = region_masks(seg)
        # wt: labels 1,2,3
        np.testing.assert_array_equal(masks["wt"][0, 0], [False, True, True, True])
        # tc: labels 1,3
        np.testing.assert_array_equal(masks["tc"][0, 0], [False, True, False, True])
        # et: label 3 only
        np.testing.assert_array_equal(masks["et"][0, 0], [False, False, False, True])

    def test_invalid_label_raises(self):
        seg = np.array([[[5]]], dtype=np.uint8)
        with pytest.raises(ValueError, match="unexpected label"):
            region_masks(seg)


# ---------------------------------------------------------------------------
# volume_change
# ---------------------------------------------------------------------------

class TestVolumeChange:
    def test_no_change(self):
        vc = volume_change(10.0, 10.0)
        assert vc["signed_ml"] == pytest.approx(0.0)
        assert vc["absolute_ml"] == pytest.approx(0.0)
        assert vc["relative_pct"] == pytest.approx(0.0)

    def test_increase(self):
        vc = volume_change(10.0, 12.0)
        assert vc["signed_ml"] == pytest.approx(2.0)
        assert vc["absolute_ml"] == pytest.approx(2.0)
        assert vc["relative_pct"] == pytest.approx(20.0)

    def test_decrease(self):
        vc = volume_change(10.0, 8.0)
        assert vc["signed_ml"] == pytest.approx(-2.0)
        assert vc["absolute_ml"] == pytest.approx(2.0)
        assert vc["relative_pct"] == pytest.approx(-20.0)

    def test_zero_baseline_gives_nan(self):
        vc = volume_change(0.0, 5.0)
        assert math.isnan(vc["relative_pct"])
        assert vc["absolute_ml"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# region_volumes_ml
# ---------------------------------------------------------------------------

class TestRegionVolumesMl:
    def test_anisotropic_spacing(self, aniso_spacing):
        seg = _sphere_seg(label=3)  # ET only → also WT and TC
        vols = region_volumes_ml(seg, aniso_spacing)
        n_et = (seg == 3).sum()
        expected_et_ml = n_et * voxel_volume_ml(aniso_spacing)
        assert vols["et"] == pytest.approx(expected_et_ml)
        # WT must equal ET when only label 3 present
        assert vols["wt"] == pytest.approx(vols["et"])


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

class TestDice:
    def test_perfect_overlap(self):
        a = np.ones((10, 10, 10), dtype=bool)
        assert dice(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[0, 0, 0] = True
        b[9, 9, 9] = True
        assert dice(a, b) == pytest.approx(0.0)

    def test_both_empty_is_nan(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        b = np.zeros((5, 5, 5), dtype=bool)
        assert math.isnan(dice(a, b))

    def test_one_empty_is_zero(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        b = np.ones((5, 5, 5), dtype=bool)
        assert dice(a, b) == pytest.approx(0.0)

    def test_known_value(self):
        # Two 3×3×3 cubes with 50% overlap
        a = np.zeros((10, 10, 10), dtype=bool)
        b = np.zeros((10, 10, 10), dtype=bool)
        a[0:3, 0:3, 0:3] = True  # 27 voxels
        b[0:3, 0:3, 1:4] = True  # 27 voxels, overlap 0:3,0:3,1:3 = 18
        # Dice = 2*18 / (27+27) = 36/54 = 2/3
        assert dice(a, b) == pytest.approx(2 / 3, rel=1e-4)

    def test_zero_volume_mask(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        b = np.ones((5, 5, 5), dtype=bool)
        assert dice(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HD95
# ---------------------------------------------------------------------------

class TestHD95:
    def test_identical_masks(self, unit_spacing):
        seg = _sphere_seg()
        mask = seg == 1
        h = hausdorff95(mask, mask, unit_spacing)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_empty_mask_is_nan(self, unit_spacing):
        empty = np.zeros((10, 10, 10), dtype=bool)
        nonempty = np.ones((10, 10, 10), dtype=bool)
        assert math.isnan(hausdorff95(empty, nonempty, unit_spacing))
        assert math.isnan(hausdorff95(nonempty, empty, unit_spacing))

    def test_spacing_scales_result(self):
        seg = _sphere_seg(radius=3)
        mask = (seg == 1)
        h1 = hausdorff95(mask, np.roll(mask, 2, axis=0), (1.0, 1.0, 1.0))
        h2 = hausdorff95(mask, np.roll(mask, 2, axis=0), (2.0, 1.0, 1.0))
        # Doubling x-spacing should double the x-contribution
        assert h2 > h1

    def test_cross_check_monai(self, unit_spacing):
        """Compare against monai.metrics.HausdorffDistanceMetric when available."""
        pytest.importorskip("monai")
        import torch
        from monai.metrics import HausdorffDistanceMetric

        seg1 = _sphere_seg(radius=4, label=1)
        seg2 = _sphere_seg(radius=4, centre=(12, 10, 10), label=1)

        our_h95 = hausdorff95((seg1 == 1), (seg2 == 1), unit_spacing)

        # MONAI expects (B, C, H, W, D), binary
        pred_t = torch.from_numpy((seg1 == 1).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        ref_t = torch.from_numpy((seg2 == 1).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        metric = HausdorffDistanceMetric(percentile=95, include_background=True)
        monai_h95 = metric(pred_t, ref_t).item()

        # Should agree within 2 mm (surface approximation differences)
        assert abs(our_h95 - monai_h95) < 2.0, (
            f"HD95 mismatch: ours={our_h95:.3f}, monai={monai_h95:.3f}"
        )


# ---------------------------------------------------------------------------
# segmentation_metrics
# ---------------------------------------------------------------------------

class TestSegmentationMetrics:
    def test_keys_present(self, unit_spacing):
        pred = _sphere_seg(label=1)
        ref = _sphere_seg(label=1)
        m = segmentation_metrics(pred, ref, unit_spacing, include_hd95=False)
        assert "wt_dice" in m
        assert "tc_dice" in m
        assert "et_dice" in m
        assert "wt_hd95_mm" not in m

    def test_keys_with_hd95(self, unit_spacing):
        pred = _sphere_seg(label=1)
        ref = _sphere_seg(label=1)
        m = segmentation_metrics(pred, ref, unit_spacing, include_hd95=True)
        assert "wt_hd95_mm" in m

    def test_invalid_label_raises(self, unit_spacing):
        pred = np.full((5, 5, 5), 9, dtype=np.uint8)
        ref = np.zeros((5, 5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            segmentation_metrics(pred, ref, unit_spacing)
