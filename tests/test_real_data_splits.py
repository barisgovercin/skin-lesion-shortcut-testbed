"""Real-data split checks: run automatically whenever a dataset is present.

For each dataset found under $SHORTCUT_DATA_ROOT (default data/raw/<dataset>), the
splits rebuilt from that download must be identical to the shipped manifests in
splits/, match the train/val/test sizes logged by the paper's runs, and keep every
lesion/patient in a single split. Only the metadata CSVs are needed. Datasets
that are not downloaded are skipped.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.data import splits

DATA_ROOT = Path(os.environ.get("SHORTCUT_DATA_ROOT", ROOT / "data" / "raw"))


@pytest.mark.parametrize("dataset", list(splits.PREPARE))
def test_shipped_manifests_match_real_data(dataset):
    data_dir = DATA_ROOT / dataset
    if not any(data_dir.glob("*.csv")):
        pytest.skip(f"{dataset} metadata not found in {data_dir}")
    if not splits.manifest_path(ROOT / "splits", dataset, 42).exists():
        pytest.skip("no shipped manifests in splits/")

    for seed, r in splits.verify_manifests(dataset, data_dir, ROOT / "splits").items():
        assert r["identical"], f"seed {seed}: {r['n_differences']} rows differ from splits/"
        assert r["paper_sizes"]["match"], f"seed {seed}: {r['paper_sizes']}"
        assert r["leakage"]["groups_in_multiple_splits"] == 0
        assert r["leakage"]["images_in_multiple_splits"] == 0
