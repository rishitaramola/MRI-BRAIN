import pytest
torch = pytest.importorskip("torch")
from pathlib import Path
from src.training.dataset import BraTSDataset

@pytest.mark.skipif(not Path("data/processed").exists(), reason="No data available")
def test_dataset_output():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "processed"

    case_names = sorted(case.name for case in DATA_DIR.iterdir())
    first_case_name = case_names[:1]
    test_dataset = BraTSDataset(DATA_DIR, first_case_name)

    sample = test_dataset[0]
    assert "image" in sample, "Missing image key"
    assert "label" in sample, "Missing label key"
    assert sample["image"].shape[0] == 4, "Image data incorrect shape"
    assert sample["image"].dtype == torch.float32, "image should be float32"
    assert sample["label"].shape[0] == 1, "Label data incorrect shape"
    assert sample["label"].dtype == torch.long, "image should be long datatype"

