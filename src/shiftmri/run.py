"""
src/shiftmri/run.py
~~~~~~~~~~~~~~~~~~~
Command-line entry point for ShiftMRI perturbation experiments.

Usage examples
--------------
# Synthetic-only demo (no real data or checkpoint needed beyond numpy/scipy):
python -m src.shiftmri.run --synthetic-demo --output-dir outputs/shiftmri/demo

# Real data + checkpoint:
python -m src.shiftmri.run \\
    --data-dir data/processed \\
    --input-mode processed \\
    --split test \\
    --checkpoint checkpoints/best_model.pth \\
    --config configs/shiftmri.yaml \\
    --output-dir outputs/shiftmri/run1 \\
    --seed 42

# Subset of cases by name:
python -m src.shiftmri.run \\
    --data-dir data/processed \\
    --cases BraTS-GLI-00000-000 BraTS-GLI-00001-000 \\
    --checkpoint checkpoints/best_model.pth \\
    --output-dir outputs/shiftmri/run2

Research prototype — not clinically validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("shiftmri")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load shiftmri.yaml. Falls back to built-in defaults if path is None."""
    import yaml  # lazy

    if path is None or not Path(path).exists():
        log.warning("Config not found; using built-in defaults.")
        return _default_config()

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _default_config() -> dict[str, Any]:
    """Built-in perturbation config matching configs/shiftmri.yaml."""
    return {
        "perturbations": [
            {"name": "noise_sigma0.1", "type": "gaussian_noise", "sigma": 0.1},
            {"name": "noise_sigma0.2", "type": "gaussian_noise", "sigma": 0.2},
            {"name": "scale0.9_shift0",   "type": "intensity_scale_shift", "scale": 0.9,  "shift": 0.0},
            {"name": "scale1.1_shift0",   "type": "intensity_scale_shift", "scale": 1.1,  "shift": 0.0},
            {"name": "scale1.0_shift+0.1","type": "intensity_scale_shift", "scale": 1.0,  "shift": 0.1},
            {"name": "scale1.0_shift-0.1","type": "intensity_scale_shift", "scale": 1.0,  "shift": -0.1},
            {"name": "scale0.8_shift+0.2","type": "intensity_scale_shift", "scale": 0.8,  "shift": 0.2},
            {"name": "downsample_2x_ax2", "type": "downsample_restore", "factor": 2, "axis": 2},
            {"name": "downsample_3x_ax2", "type": "downsample_restore", "factor": 3, "axis": 2},
        ]
    }


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _discover_cases(data_dir: Path, split: str | None, cases: list[str] | None) -> list[Path]:
    """Return list of case directories to process."""
    if cases:
        return [data_dir / c for c in cases]

    if split:
        splits_path = Path("data") / "splits" / "splits.json"
        if not splits_path.exists():
            log.error("splits.json not found at %s", splits_path)
            sys.exit(1)
        with open(splits_path) as fh:
            splits = json.load(fh)
        if split not in splits:
            log.error("Split %r not found. Available: %s", split, list(splits.keys()))
            sys.exit(1)
        return [data_dir / c for c in splits[split]]

    # All subdirectories
    return sorted(p for p in data_dir.iterdir() if p.is_dir())


def _case_seed(case_name: str, global_seed: int) -> int:
    """Deterministic per-case seed from sha256, not Python hash()."""
    raw = f"{global_seed}:{case_name}".encode()
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], byteorder="big") & 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ShiftMRI perturbation audit (research prototype).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to data/processed (or raw) directory.")
    parser.add_argument("--input-mode", choices=["processed", "raw"], default="processed")
    parser.add_argument("--split", default=None,
                        help="Split name from splits.json (e.g. 'test').")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Explicit case names (overrides --split).")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Cap the number of cases (useful for quick tests).")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/shiftmri.yaml"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/shiftmri/run1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="torch device string, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--no-hd95", action="store_true",
                        help="Skip HD95 computation (much faster on CPU).")
    parser.add_argument("--synthetic-demo", action="store_true",
                        help="Run on synthetic phantom only (no real data/checkpoint needed "
                             "for the plumbing check; inference skipped without checkpoint).")
    parser.add_argument("--model-b-checkpoint", type=Path, default=None,
                        help="Optional path to independent Model B checkpoint (e.g. MONAI BraTS bundle).")
    args = parser.parse_args(argv)

    from src.shiftmri.data import write_csv, write_json
    from src.shiftmri.experiment import build_report, evaluate_case

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    perturbations = cfg.get("perturbations", _default_config()["perturbations"])

    # ------------------------------------------------------------------ #
    # Synthetic demo path
    # ------------------------------------------------------------------ #
    if args.synthetic_demo:
        log.info("=== SYNTHETIC DEMO MODE ===")
        log.info("Running phantom plumbing check (no real data or checkpoint required).")
        from src.shiftmri.synthetic import PHANTOM_SPACING_MM, make_phantom, make_phantom_image

        phantom = make_phantom()
        image = make_phantom_image()
        gt_seg = phantom["seg"]
        spacing_mm = PHANTOM_SPACING_MM

        # Try to load model if checkpoint exists; otherwise use dummy predict
        predict_fn = None
        ckpt_id = "no_checkpoint"
        if args.checkpoint.exists():
            try:
                from src.shiftmri.inference import load_model, make_predict_fn
                model, device, ckpt_id = load_model(args.checkpoint, device=args.device)
                predict_fn = make_predict_fn(model, device)
                log.info("Checkpoint loaded: %s", ckpt_id)
            except Exception as exc:
                log.warning("Could not load checkpoint (%s); using zero-mask stub.", exc)
        else:
            log.warning(
                "Checkpoint not found at %s. Using zero-mask stub for plumbing check.",
                args.checkpoint,
            )

        if predict_fn is None:
            # Stub: return a zero mask (background only) — plumbing only
            def predict_fn(img: np.ndarray) -> np.ndarray:  # noqa: F811
                return np.zeros(img.shape[1:], dtype=np.uint8)
            ckpt_id = "stub_zero_mask"

        seed = _case_seed("synthetic_phantom", args.seed)
        rows = evaluate_case(
            case_name="synthetic_phantom",
            image=image,
            predict_fn=predict_fn,
            perturbations=perturbations,
            spacing_mm=spacing_mm,
            gt_seg=gt_seg,
            seed=seed,
            checkpoint_id=ckpt_id,
            data_source="synthetic_phantom",
            include_hd95=not args.no_hd95,
            notes="synthetic-demo mode",
        )

        run_meta = {
            "mode": "synthetic_demo",
            "checkpoint_id": ckpt_id,
            "seed": args.seed,
            "include_hd95": not args.no_hd95,
            "config": str(args.config),
            "model_b_available": args.model_b_checkpoint is not None and args.model_b_checkpoint.exists(),
        }
        report = build_report(rows, run_meta)

        write_csv(rows, args.output_dir / "results.csv")
        write_json(rows, args.output_dir / "results.json")
        write_json(report, args.output_dir / "report.json")
        log.info("Outputs written to %s", args.output_dir)
        log.info("Rows: %d total, %d ok, %d errors",
                 len(rows),
                 sum(1 for r in rows if r.get("status") == "ok"),
                 sum(1 for r in rows if r.get("status") == "error"))
        return 0

    # ------------------------------------------------------------------ #
    # Real data path
    # ------------------------------------------------------------------ #
    if args.data_dir is None:
        log.error("--data-dir is required unless --synthetic-demo is set.")
        return 1

    if not args.data_dir.exists():
        log.error("data-dir does not exist: %s", args.data_dir)
        return 1

    # Validate checkpoint before discovering cases
    from src.shiftmri.inference import checkpoint_id as get_ckpt_id
    from src.shiftmri.inference import load_model, make_predict_fn

    try:
        ckpt_id = get_ckpt_id(args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    try:
        model, device, ckpt_id = load_model(args.checkpoint, device=args.device)
        predict_fn = make_predict_fn(model, device)
        log.info("Model loaded on %s | %s", device, ckpt_id)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        return 1

    from src.shiftmri.data import load_case

    case_dirs = _discover_cases(args.data_dir, args.split, args.cases)
    if args.max_cases:
        case_dirs = case_dirs[: args.max_cases]

    if not case_dirs:
        log.error("No cases found.")
        return 1

    log.info("Processing %d case(s).", len(case_dirs))

    from src.preprocessing.preprocess import preprocess_array

    all_rows: list[dict] = []
    t_start = time.time()

    for case_dir in case_dirs:
        case_name = case_dir.name
        log.info("  Case: %s", case_name)

        try:
            case_data = load_case(case_dir, mode=args.input_mode, load_seg=True)
        except FileNotFoundError as exc:
            log.warning("  Skipping %s: %s", case_name, exc)
            all_rows.append({
                "case_name": case_name,
                "perturbation": "baseline",
                "status": "skipped",
                "error": str(exc),
                "seed": _case_seed(case_name, args.seed),
                "checkpoint_id": ckpt_id,
                "data_source": args.input_mode,
                "notes": "",
            })
            continue

        spacing_mm = case_data["spacing_mm"]
        gt_seg = case_data.get("seg")

        # Build modality_arrays dict and run preprocess_array
        modality_arrays_raw = case_data["modality_arrays"]
        if gt_seg is not None:
            modality_arrays_raw["seg"] = gt_seg.astype(float)

        try:
            processed = preprocess_array(modality_arrays_raw)
        except Exception as exc:
            log.warning("  Preprocessing failed for %s: %s", case_name, exc)
            all_rows.append({
                "case_name": case_name,
                "perturbation": "baseline",
                "status": "error",
                "error": str(exc),
                "seed": _case_seed(case_name, args.seed),
                "checkpoint_id": ckpt_id,
                "data_source": args.input_mode,
                "notes": "",
            })
            continue

        # Stack into (4, H, W, D) channel order matching predict.py
        image = np.stack([
            processed["t1c"],
            processed["t1n"],
            processed["t2f"],
            processed["t2w"],
        ], axis=0).astype(np.float32)

        gt_seg_processed = processed.get("seg")
        if gt_seg_processed is not None:
            gt_seg_processed = gt_seg_processed.astype(np.uint8)

        seed = _case_seed(case_name, args.seed)
        rows = evaluate_case(
            case_name=case_name,
            image=image,
            predict_fn=predict_fn,
            perturbations=perturbations,
            spacing_mm=spacing_mm,
            gt_seg=gt_seg_processed,
            seed=seed,
            checkpoint_id=ckpt_id,
            data_source=args.input_mode,
            include_hd95=not args.no_hd95,
        )
        all_rows.extend(rows)

    elapsed = time.time() - t_start
    log.info("Finished in %.1f s. Rows: %d total.", elapsed, len(all_rows))

    run_meta = {
        "data_dir": str(args.data_dir),
        "input_mode": args.input_mode,
        "split": args.split,
        "checkpoint_id": ckpt_id,
        "seed": args.seed,
        "device": str(device) if "device" in dir() else args.device,
        "include_hd95": not args.no_hd95,
        "config": str(args.config),
        "elapsed_s": round(elapsed, 1),
        "model_b_available": args.model_b_checkpoint is not None and args.model_b_checkpoint.exists(),
    }

    report = build_report(all_rows, run_meta)

    from src.shiftmri.data import write_csv, write_json

    write_csv(all_rows, args.output_dir / "results.csv")
    write_json(all_rows, args.output_dir / "results.json")
    write_json(report, args.output_dir / "report.json")
    log.info("Outputs written to %s", args.output_dir)

    n_errors = sum(1 for r in all_rows if r.get("status") == "error")
    return 1 if n_errors > 0 and len(all_rows) == n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
