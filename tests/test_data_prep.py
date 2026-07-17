"""Offline tests for ISIC 2020 data preparation.

These validate the split/encoding logic WITHOUT needing the real dataset, a GPU,
or any image files (prepare_isic2020 only reads the CSV). Run from the repo root:

    python tests/test_data_prep.py     # plain asserts, no pytest needed
    pytest tests/test_data_prep.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.dataset import (
    prepare_isic2020,
    stratified_group_split,
    encode_metadata_isic2020,
    compute_class_weights,
)


SITES = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital"]


def _make_synthetic_isic(n_patients=120, pos_patient_frac=0.35, n_orphan_malignant=15, seed=0):
    """Build an ISIC-2020-like dataframe: multiple images per patient, plus some
    orphan (no patient_id) malignant images like the external-malignant set."""
    rng = np.random.default_rng(seed)
    rows = []
    img_counter = 0
    n_pos = int(n_patients * pos_patient_frac)
    pos_patients = set(rng.choice(n_patients, size=n_pos, replace=False).tolist())

    for pid in range(n_patients):
        n_img = int(rng.integers(1, 4))  # 1-3 images per patient
        is_pos_patient = pid in pos_patients
        for j in range(n_img):
            # A positive patient has at least one malignant image (the first).
            target = 1 if (is_pos_patient and j == 0) else 0
            rows.append({
                "image_name": f"ISIC_{img_counter:07d}",
                "patient_id": f"IP_{pid:04d}",
                "sex": rng.choice(["male", "female", None]),
                "age_approx": rng.choice([20, 35, 50, 65, 80, np.nan]),
                "anatom_site_general_challenge": rng.choice(SITES + [None]),
                "target": target,
            })
            img_counter += 1

    # Orphan malignant images with no patient_id.
    for _ in range(n_orphan_malignant):
        rows.append({
            "image_name": f"ISIC_{img_counter:07d}",
            "patient_id": np.nan,
            "sex": rng.choice(["male", "female"]),
            "age_approx": rng.choice([30, 45, 60]),
            "anatom_site_general_challenge": rng.choice(SITES),
            "target": 1,
        })
        img_counter += 1

    return pd.DataFrame(rows)


def _write_csv(df, tmpdir):
    path = Path(tmpdir) / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_no_patient_leakage():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        train, val, test, img_dirs, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)

    tr, va, te = set(train["patient_id"]), set(val["patient_id"]), set(test["patient_id"])
    assert tr.isdisjoint(va), "train/val share a patient"
    assert tr.isdisjoint(te), "train/test share a patient"
    assert va.isdisjoint(te), "val/test share a patient"
    # Every row is accounted for exactly once.
    assert len(train) + len(val) + len(test) == len(df)


def test_split_proportions_and_classes():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        train, val, test, _, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)

    total = len(df)
    # Per-image proportions should be roughly 60/20/20 (loose bounds — groups vary in size).
    assert 0.50 <= len(train) / total <= 0.70
    assert 0.12 <= len(val) / total <= 0.28
    assert 0.12 <= len(test) / total <= 0.28
    # Both classes present in every split (critical for AUROC computation).
    for name, split in [("train", train), ("val", val), ("test", test)]:
        assert split["label"].nunique() == 2, f"{name} is single-class"


def test_metadata_encoding():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        train, val, test, _, metadata_cols = prepare_isic2020(tmp, seed=42, include_metadata=True)

    assert len(metadata_cols) == 8, f"expected 8 metadata features, got {len(metadata_cols)}"
    for col in metadata_cols:
        assert col in train.columns
        assert not train[col].isna().any(), f"NaN in metadata column {col}"
    # age_norm must be in [0, 1]; one-hot site/sex columns must be 0/1.
    assert train["age_norm"].between(0, 1).all()
    for col in metadata_cols:
        if col != "age_norm":
            assert set(train[col].unique()).issubset({0.0, 1.0})


def test_determinism():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        _, _, test_a, _, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)
        _, _, test_b, _, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)
        _, _, test_c, _, _ = prepare_isic2020(tmp, seed=123, include_metadata=False)

    assert set(test_a["image_id"]) == set(test_b["image_id"]), "same seed -> different split"
    assert set(test_a["image_id"]) != set(test_c["image_id"]), "different seed -> identical split"


def test_label_from_benign_malignant_fallback():
    # No 'target' column: label must be derived from benign_malignant.
    df = _make_synthetic_isic()
    df["benign_malignant"] = np.where(df["target"] == 1, "malignant", "benign")
    df = df.drop(columns=["target"])
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        train, val, test, _, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)
    assert set(train["label"].unique()).issubset({0, 1})
    assert train["label"].sum() > 0


def test_class_weights_balanced():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        _write_csv(df, tmp)
        train, _, _, _, _ = prepare_isic2020(tmp, seed=42, include_metadata=False)
    w = compute_class_weights(train)
    # Minority (malignant) class must get the larger weight.
    assert w[1] > w[0]


def test_prepare_isic_engineered_and_train_concat():
    df = _make_synthetic_isic()
    with tempfile.TemporaryDirectory() as tmp:
        # File named train_concat.csv (the real on-disk name) must be discovered.
        path = Path(tmp) / "train_concat.csv"
        df.to_csv(path, index=False)
        train, val, test, _, cols = prepare_isic2020(
            tmp, seed=42, include_metadata=True,
            engineered=True, patient_aggregates=True,
        )
    assert len(cols) == 29, f"expected 29 engineered+agg cols, got {len(cols)}"
    for col in cols:
        assert not train[col].isna().any(), f"NaN in {col}"


if __name__ == "__main__":
    tests = [
        test_no_patient_leakage,
        test_split_proportions_and_classes,
        test_metadata_encoding,
        test_determinism,
        test_label_from_benign_malignant_fallback,
        test_class_weights_balanced,
        test_prepare_isic_engineered_and_train_concat,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
