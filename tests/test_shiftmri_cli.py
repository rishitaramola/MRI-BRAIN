"""
tests/test_shiftmri_cli.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the src.shiftmri.run CLI.

Covers: --synthetic-demo success path, missing checkpoint exits non-zero.
No real MRI data required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(args: list[str]) -> int:
    """Run the CLI main() function and return its exit code."""
    from src.shiftmri.run import main
    return main(args)


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

class TestSyntheticDemo:
    def test_runs_and_produces_outputs(self, tmp_path):
        """--synthetic-demo should complete and write results.csv, results.json, report.json."""
        code = run_cli([
            "--synthetic-demo",
            "--output-dir", str(tmp_path / "out"),
            "--no-hd95",          # faster
            "--seed", "7",
        ])
        assert code == 0, "CLI should exit 0 on successful synthetic demo"

        out = tmp_path / "out"
        assert (out / "results.csv").exists(), "results.csv missing"
        assert (out / "results.json").exists(), "results.json missing"
        assert (out / "report.json").exists(), "report.json missing"

    def test_report_is_valid_json(self, tmp_path):
        run_cli([
            "--synthetic-demo",
            "--output-dir", str(tmp_path / "out"),
            "--no-hd95",
        ])
        report_path = tmp_path / "out" / "report.json"
        data = json.loads(report_path.read_text())
        assert "disclaimer" in data
        assert "counts" in data
        assert "summary" in data

    def test_results_no_nan_in_json(self, tmp_path):
        """Strict JSON: the output file must not contain the string 'NaN'."""
        run_cli([
            "--synthetic-demo",
            "--output-dir", str(tmp_path / "out"),
            "--no-hd95",
        ])
        content = (tmp_path / "out" / "results.json").read_text()
        assert "NaN" not in content, "JSON output must not contain NaN literal"

    def test_all_perturbations_present(self, tmp_path):
        run_cli([
            "--synthetic-demo",
            "--output-dir", str(tmp_path / "out"),
            "--no-hd95",
        ])
        results = json.loads((tmp_path / "out" / "results.json").read_text())
        labels = {r["perturbation"] for r in results}
        assert "baseline" in labels
        # At least some perturbations ran
        assert len(labels) > 1


# ---------------------------------------------------------------------------
# Missing checkpoint — real data path should exit non-zero
# ---------------------------------------------------------------------------

class TestMissingCheckpoint:
    def test_exits_nonzero_without_checkpoint(self, tmp_path):
        # Point at a real data dir (even empty) and a nonexistent checkpoint
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "case_001").mkdir()  # one (incomplete) case dir

        code = run_cli([
            "--data-dir", str(data_dir),
            "--checkpoint", str(tmp_path / "nonexistent.pth"),
            "--output-dir", str(tmp_path / "out"),
            "--no-hd95",
        ])
        assert code != 0, "Should exit non-zero when checkpoint is missing"
