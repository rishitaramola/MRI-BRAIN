"""
tests/test_shiftmri_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src.shiftmri.inference (checkpoint_id, load_model, make_predict_fn).

No real checkpoint required for unit tests.
The optional smoke test at the bottom requires the real checkpoint and is skipped
if torch or the weights are absent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shiftmri.inference import LFS_POINTER_MARKER, checkpoint_id


# ---------------------------------------------------------------------------
# checkpoint_id
# ---------------------------------------------------------------------------

class TestCheckpointId:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        p = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError, match="not found"):
            checkpoint_id(p)

    def test_lfs_pointer_raises_value_error(self, tmp_path):
        p = tmp_path / "lfs.pth"
        p.write_bytes(LFS_POINTER_MARKER + b"\nversion 1\noid sha256:abc\nsize 100\n")
        with pytest.raises(ValueError, match="LFS"):
            checkpoint_id(p)

    def test_valid_file_returns_name_and_hash(self, tmp_path):
        p = tmp_path / "best_model.pth"
        p.write_bytes(b"\x80\x02" + b"x" * 100)  # fake binary (not LFS)
        cid = checkpoint_id(p)
        assert cid.startswith("best_model.pth::")
        hex_part = cid.split("::")[1]
        assert len(hex_part) == 12
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_determinism(self, tmp_path):
        p = tmp_path / "w.pth"
        p.write_bytes(b"someweights")
        assert checkpoint_id(p) == checkpoint_id(p)

    def test_different_content_different_id(self, tmp_path):
        p1 = tmp_path / "a.pth"
        p2 = tmp_path / "b.pth"
        p1.write_bytes(b"content_a")
        p2.write_bytes(b"content_b")
        assert checkpoint_id(p1) != checkpoint_id(p2)


# ---------------------------------------------------------------------------
# load_model (error paths, no real torch needed for some)
# ---------------------------------------------------------------------------

class TestLoadModelErrors:
    def test_missing_checkpoint_exits(self, tmp_path):
        pytest.importorskip("torch")
        from src.shiftmri.inference import load_model
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "missing.pth")

    def test_lfs_pointer_raises(self, tmp_path):
        pytest.importorskip("torch")
        from src.shiftmri.inference import load_model
        p = tmp_path / "lfs.pth"
        p.write_bytes(LFS_POINTER_MARKER + b"\nsize 57000000\n")
        with pytest.raises(ValueError, match="LFS"):
            load_model(p)

    def test_bad_state_dict_raises(self, tmp_path):
        pytest.importorskip("torch")
        import torch
        from src.shiftmri.inference import load_model
        # Save a checkpoint with wrong architecture keys
        bad_ckpt = {"model": {"wrong_key": torch.zeros(1)}}
        p = tmp_path / "bad.pth"
        torch.save(bad_ckpt, p)
        with pytest.raises(RuntimeError, match="mismatch|unexpected"):
            load_model(p, device="cpu")

    def test_missing_model_key_raises(self, tmp_path):
        pytest.importorskip("torch")
        import torch
        from src.shiftmri.inference import load_model
        # Save checkpoint without "model" key
        p = tmp_path / "nokey.pth"
        torch.save({"epoch": 1}, p)
        with pytest.raises(ValueError, match="model"):
            load_model(p, device="cpu")


# ---------------------------------------------------------------------------
# Optional smoke test — real checkpoint required
# ---------------------------------------------------------------------------

REAL_CHECKPOINT = Path("checkpoints/best_model.pth")


@pytest.mark.skipif(
    not REAL_CHECKPOINT.exists(),
    reason="Real checkpoint not available; skipping smoke test.",
)
def test_smoke_real_checkpoint_phantom_inference():
    """
    End-to-end smoke test: load real checkpoint, run inference on a small
    synthetic phantom (64³ volume), check output shape and dtype.

    This test is SKIPPED if torch or the checkpoint are unavailable.
    It uses a small volume to keep CPU runtime under ~60 s.
    """
    pytest.importorskip("torch")
    pytest.importorskip("monai")

    from src.shiftmri.inference import load_model, make_predict_fn
    from src.shiftmri.synthetic import make_phantom_image

    model, device, ckpt_id = load_model(REAL_CHECKPOINT, device="cpu")
    predict = make_predict_fn(model, device, roi_size=(64, 64, 64), sw_batch_size=1)

    image = make_phantom_image(shape=(64, 64, 64))
    seg = predict(image)

    assert seg.shape == (64, 64, 64), f"Expected (64,64,64), got {seg.shape}"
    assert seg.dtype == np.uint8, f"Expected uint8, got {seg.dtype}"
    assert set(np.unique(seg).tolist()).issubset({0, 1, 2, 3}), \
        f"Unexpected labels: {np.unique(seg)}"

    print(f"\n[smoke] ckpt={ckpt_id}  device={device}  "
          f"unique_labels={np.unique(seg).tolist()}")
