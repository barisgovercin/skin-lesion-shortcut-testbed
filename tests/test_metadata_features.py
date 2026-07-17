"""Offline unit tests for metadata feature-engineering helpers and encoders.

No CSV, GPU, or image files needed. Run from repo root:
    py -3.11 tests/test_metadata_features.py
    pytest tests/test_metadata_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.metadata_features import (
    UV_GROUPS, AGE_BINS, HAM_UV_MAP, ISIC_UV_MAP, uv_group, age_bin,
)
from src.data.dataset import encode_metadata_ham10000, encode_metadata_isic2020

# Every raw value seen in the real data must map to a known UV group.
HAM_LOCS = ["back", "lower extremity", "trunk", "upper extremity", "abdomen",
            "face", "chest", "foot", "unknown", "neck", "scalp", "hand",
            "ear", "genital", "acral"]
ISIC_SITES = ["torso", "lower extremity", "upper extremity", "head/neck",
              "anterior torso", "posterior torso", "palms/soles",
              "oral/genital", "lateral torso"]


def test_uv_group_maps_all_known_values():
    for v in HAM_LOCS:
        assert uv_group(v, HAM_UV_MAP) in UV_GROUPS
    for v in ISIC_SITES:
        assert uv_group(v, ISIC_UV_MAP) in UV_GROUPS
    # Previously-dropped HAM categories now carry signal (not 'unknown').
    assert uv_group("ear", HAM_UV_MAP) == "chronic"
    assert uv_group("hand", HAM_UV_MAP) == "chronic"
    assert uv_group("acral", HAM_UV_MAP) == "acral_rare"
    # ISIC torso variants collapse to one intermittent group.
    assert uv_group("anterior torso", ISIC_UV_MAP) == "intermittent"
    assert uv_group("posterior torso", ISIC_UV_MAP) == "intermittent"


def test_uv_group_missing_and_unknown():
    assert uv_group(np.nan, ISIC_UV_MAP) == "unknown"
    assert uv_group(None, HAM_UV_MAP) == "unknown"
    assert uv_group("unknown", HAM_UV_MAP) == "unknown"
    assert uv_group("not a real site", HAM_UV_MAP) == "unknown"


def test_age_bin_boundaries():
    assert age_bin(10) == "lt30"
    assert age_bin(29.9) == "lt30"
    assert age_bin(30) == "30_50"
    assert age_bin(50) == "50_65"
    assert age_bin(64) == "50_65"
    assert age_bin(65) == "65_75"
    assert age_bin(80) == "75plus"
    assert age_bin(np.nan) is None
    assert age_bin(None) is None
    assert AGE_BINS == ["lt30", "30_50", "50_65", "65_75", "75plus"]


def _ham_frame():
    return pd.DataFrame({
        "age": [25, 55, 80, np.nan, 70],
        "sex": ["male", "female", "unknown", "male", np.nan],
        "localization": ["ear", "back", "acral", "hand", "unknown"],
        "dx": ["nv", "mel", "akiec", "bcc", "nv"],
    })


def test_ham_raw_unchanged():
    _, cols = encode_metadata_ham10000(_ham_frame(), engineered=False)
    assert len(cols) == 12, f"raw HAM must stay 12 cols, got {len(cols)}"
    assert cols[:2] == ["age_norm", "sex_male"]


def test_ham_engineered_columns():
    df, cols = encode_metadata_ham10000(_ham_frame(), engineered=True)
    assert len(cols) == 31, f"engineered HAM must be 31 cols, got {len(cols)}"
    for c in cols:
        assert c in df.columns
        assert not df[c].isna().any(), f"NaN in {c}"
    # Previously-dropped categories now land in exactly one UV group per row.
    uv_cols = [c for c in cols if c.startswith("uv_")]
    assert (df[uv_cols].sum(axis=1) == 1).all()
    # The 'hand'/'ear' rows are chronic; the 'acral' row is acral_rare.
    assert df.loc[0, "uv_chronic"] == 1.0  # ear
    assert df.loc[2, "uv_acral_rare"] == 1.0  # acral
    # age_missing flags the NaN-age row; sex_unknown flags 'unknown'/NaN sex.
    assert df.loc[3, "age_missing"] == 1.0
    assert df.loc[2, "sex_unknown"] == 1.0
    assert df.loc[4, "sex_unknown"] == 1.0


def _isic_frame():
    return pd.DataFrame({
        "age_approx": [20, 50, 90, np.nan, 60, 45],
        "sex": ["male", "female", np.nan, "male", "female", "male"],
        "anatom_site_general_challenge": [
            "head/neck", "anterior torso", "palms/soles",
            "lower extremity", np.nan, "posterior torso",
        ],
        "patient_id": ["P1", "P1", "P2", "P2", "P2", "P3"],
        "target": [0, 1, 0, 0, 1, 0],
    })


def test_isic_engineered_columns():
    df, cols = encode_metadata_isic2020(_isic_frame(), engineered=True)
    assert len(cols) == 27, f"engineered ISIC must be 27 cols, got {len(cols)}"
    uv_cols = [c for c in cols if c.startswith("uv_")]
    assert (df[uv_cols].sum(axis=1) == 1).all()
    # NaN site -> unknown group; torso variants -> intermittent.
    assert df.loc[4, "uv_unknown"] == 1.0
    assert df.loc[1, "uv_intermittent"] == 1.0
    assert df.loc[5, "uv_intermittent"] == 1.0
    for c in cols:
        assert not df[c].isna().any(), f"NaN in {c}"


def test_isic_patient_aggregates():
    df, cols = encode_metadata_isic2020(
        _isic_frame(), engineered=True, patient_aggregates=True
    )
    assert len(cols) == 29, f"engineered+agg ISIC must be 29 cols, got {len(cols)}"
    assert "images_per_patient" in cols and "log_images_per_patient" in cols
    # P1 has 2 images, P2 has 3, P3 has 1.
    assert df.loc[0, "images_per_patient"] == 2.0
    assert df.loc[2, "images_per_patient"] == 3.0
    assert df.loc[5, "images_per_patient"] == 1.0


def test_spec_at_95_sens_perfect_separation():
    from scripts.diagnose_metadata import spec_at_95_sens
    y_val = np.array([0, 0, 1, 1])
    p_val = np.array([0.1, 0.2, 0.8, 0.9])
    y_test = np.array([0, 0, 1, 1])
    p_test = np.array([0.15, 0.25, 0.75, 0.95])
    # Perfectly separable -> specificity 1.0 at 95% sensitivity.
    assert spec_at_95_sens(y_val, p_val, y_test, p_test) == 1.0


if __name__ == "__main__":
    tests = [
        test_uv_group_maps_all_known_values,
        test_uv_group_missing_and_unknown,
        test_age_bin_boundaries,
        test_ham_raw_unchanged,
        test_ham_engineered_columns,
        test_isic_engineered_columns,
        test_isic_patient_aggregates,
        test_spec_at_95_sens_perfect_separation,
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
