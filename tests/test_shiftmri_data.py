"""
tests/test_shiftmri_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for src.shiftmri.data (load_case, write_json, write_csv).

No real MRI data needed.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shiftmri.data import _nan_to_null, write_csv, write_json


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestWriteJson:
    def test_nan_becomes_null(self, tmp_path):
        data = {"a": float("nan"), "b": 1.0, "c": [float("nan"), 2.0]}
        out = tmp_path / "test.json"
        write_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["a"] is None
        assert loaded["b"] == pytest.approx(1.0)
        assert loaded["c"][0] is None
        assert loaded["c"][1] == pytest.approx(2.0)

    def test_inf_becomes_null(self, tmp_path):
        data = {"x": float("inf"), "y": float("-inf")}
        out = tmp_path / "out.json"
        write_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded["x"] is None
        assert loaded["y"] is None

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c.json"
        write_json({"ok": True}, out)
        assert out.exists()

    def test_round_trip(self, tmp_path):
        data = {"int": 1, "str": "hello", "list": [1, 2, 3], "nested": {"x": 0.5}}
        out = tmp_path / "rt.json"
        write_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_nan_to_null_recursive(self):
        obj = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("nan")}}
        clean = _nan_to_null(obj)
        assert clean["a"] is None
        assert clean["b"][0] is None
        assert clean["b"][1] == pytest.approx(1.0)
        assert clean["c"]["d"] is None


# ---------------------------------------------------------------------------
# CSV serialisation
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_creates_file(self, tmp_path):
        rows = [{"a": 1, "b": 2.0}, {"a": 3, "b": 4.0}]
        out = tmp_path / "test.csv"
        write_csv(rows, out)
        assert out.exists()
        content = out.read_text()
        assert "a,b" in content
        assert "1,2.0" in content or "1," in content

    def test_nan_becomes_empty(self, tmp_path):
        rows = [{"a": float("nan"), "b": 1.0}]
        out = tmp_path / "nan.csv"
        write_csv(rows, out)
        content = out.read_text()
        # NaN row should have empty first field
        assert ",1.0" in content

    def test_empty_rows_noop(self, tmp_path):
        out = tmp_path / "empty.csv"
        write_csv([], out)
        assert not out.exists()

    def test_append_mode(self, tmp_path):
        out = tmp_path / "app.csv"
        rows1 = [{"x": 1}]
        rows2 = [{"x": 2}]
        write_csv(rows1, out, append=False)
        write_csv(rows2, out, append=True)
        lines = out.read_text().strip().split("\n")
        # Header + 2 data rows
        assert len(lines) == 3

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "dir" / "out.csv"
        write_csv([{"k": "v"}], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# load_case (without real nibabel volumes — use a stub)
# ---------------------------------------------------------------------------

class TestLoadCase:
    def test_missing_files_raises_all_at_once(self, tmp_path):
        case_dir = tmp_path / "BraTS-GLI-00000-000"
        case_dir.mkdir()
        # Only t1c present — others missing
        (case_dir / "BraTS-GLI-00000-000-t1c.nii.gz").write_bytes(b"fake")

        from src.shiftmri.data import load_case
        with pytest.raises(FileNotFoundError) as exc_info:
            load_case(case_dir, mode="processed", load_seg=False)
        err = str(exc_info.value)
        # All three missing modalities should be listed in the same error
        assert "t1n" in err
        assert "t2f" in err
        assert "t2w" in err

    def test_missing_seg_included_in_error(self, tmp_path):
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        from src.shiftmri.data import MODALITIES, load_case
        for mod in MODALITIES:
            (case_dir / f"case-{mod}.nii.gz").write_bytes(b"fake")
        # seg is missing; load_seg=True should report it
        with pytest.raises(FileNotFoundError) as exc_info:
            load_case(case_dir, mode="processed", load_seg=True)
        assert "seg" in str(exc_info.value)
