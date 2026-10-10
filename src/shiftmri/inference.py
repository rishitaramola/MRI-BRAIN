"""
src/shiftmri/inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin, reusable inference wrapper that mirrors the logic in
src/inference/predict.py WITHOUT modifying that file.

Key design choices
------------------
- load_model() gives actionable error messages for: file missing, Git LFS pointer
  (the file is tiny ASCII), wrong state-dict keys.
- make_predict_fn() returns a callable that accepts a preprocessed 4-D float32
  numpy array (C, H, W, D) and returns a 3-D uint8 numpy label array.
- Orientationd/Spacingd in get_inference_transforms() are MONAI MetaTensor-aware.
  When input has no affine metadata (plain numpy → tensor) the resampling is a
  no-op because there is no header to reorient against.  This matches the
  behaviour in predict.py and is intentional — see audit notes.
- torch and monai are imported lazily so the pure-numpy modules load without them.

Research prototype — not clinically validated.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------

LFS_POINTER_MARKER = b"version https://git-lfs.github.com"


def checkpoint_id(path: Path) -> str:
    """Return ``filename::sha256[:12]`` for traceability.

    Raises FileNotFoundError if missing, ValueError if an LFS pointer.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run training or place best_model.pth in the checkpoints/ directory."
        )
    with open(path, "rb") as fh:
        header = fh.read(64)
    if header.startswith(LFS_POINTER_MARKER):
        raise ValueError(
            f"Checkpoint at {path} is a Git LFS pointer ({path.stat().st_size} bytes), "
            "not real weights.  Run `git lfs pull` or download the weights manually."
        )
    # Full sha256
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    return f"{path.name}::{sha.hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    device: str | None = None,
):
    """Load get_model() with pretrained weights.

    Parameters
    ----------
    checkpoint_path:
        Path to best_model.pth.
    device:
        ``'cuda'``, ``'cpu'``, ``'mps'``, or None (auto-select).

    Returns
    -------
    torch.nn.Module in eval mode.

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError  (with clear messages).
    """
    import torch  # lazy

    from src.training.model import get_model

    checkpoint_path = Path(checkpoint_path)

    # Gate against missing file and LFS pointers
    ckpt_id = checkpoint_id(checkpoint_path)  # raises on failure

    if device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    else:
        _device = torch.device(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to torch.load {checkpoint_path}: {exc}\n"
            "The file may be corrupted.  Check file size (~57 MB expected)."
        ) from exc

    if "model" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain key 'model'. "
            f"Found keys: {list(ckpt.keys())}. "
            "This may not be a NeuralSeg/ShiftMRI checkpoint."
        )

    model = get_model().to(_device)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"State-dict mismatch loading {checkpoint_path}:\n{exc}\n"
            "The checkpoint may have been saved from a different model architecture."
        ) from exc

    model.eval()
    return model, _device, ckpt_id


# ---------------------------------------------------------------------------
# Predict function factory
# ---------------------------------------------------------------------------

def make_predict_fn(
    model,
    device,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 1,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict function mirroring the /segment endpoint in predict.py.

    The returned function takes a preprocessed float32 numpy array of shape
    (4, H, W, D) and returns a uint8 numpy array of shape (H, W, D).

    Channel order must be: t1c, t1n, t2f, t2w  (matching predict.py line 78-83).
    """
    import torch  # lazy
    from monai.inferers import sliding_window_inference  # lazy
    from src.training.transforms import get_inference_transforms  # lazy

    def predict(image_chw: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image_chw:
            float32 array, shape (4, H, W, D), preprocessed (z-scored, BBox-cropped).

        Returns
        -------
        uint8 array, shape (H, W, D) with labels 0–3.
        """
        tensors = {"image": image_chw}
        tensors = get_inference_transforms()(tensors)
        image_t = tensors["image"]

        image_t = image_t.float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = sliding_window_inference(
                image_t,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
            )

        seg = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return seg

    return predict
