"""Shared, dataset-agnostic metadata feature-engineering helpers.

Pure pandas/numpy (no torch) so offline data tests run fast. Used by
encode_metadata_ham10000 / encode_metadata_isic2020 when engineered=True.

The UV-exposure grouping encodes the clinical fact that skin-cancer type tracks
chronic vs intermittent vs acral UV exposure of the body site. The same four
groups are defined for both datasets so engineered features are comparable across
them (supports the planned train-ISIC -> test-HAM transfer experiment).
"""
import numpy as np
import pandas as pd

UV_GROUPS = ["chronic", "intermittent", "acral_rare", "unknown"]
AGE_BINS = ["lt30", "30_50", "50_65", "65_75", "75plus"]

HAM_UV_MAP = {
    "face": "chronic", "neck": "chronic", "scalp": "chronic",
    "ear": "chronic", "hand": "chronic",
    "back": "intermittent", "trunk": "intermittent", "chest": "intermittent",
    "abdomen": "intermittent", "upper extremity": "intermittent",
    "lower extremity": "intermittent",
    "foot": "acral_rare", "acral": "acral_rare", "genital": "acral_rare",
    "unknown": "unknown",
}
ISIC_UV_MAP = {
    "head/neck": "chronic",
    "torso": "intermittent", "anterior torso": "intermittent",
    "posterior torso": "intermittent", "lateral torso": "intermittent",
    "upper extremity": "intermittent", "lower extremity": "intermittent",
    "palms/soles": "acral_rare", "oral/genital": "acral_rare",
}


def uv_group(value, mapping):
    """Map a raw localization/site string to one of UV_GROUPS. NaN/unknown -> 'unknown'."""
    if pd.isna(value):
        return "unknown"
    return mapping.get(str(value).strip().lower(), "unknown")


def age_bin(age):
    """Return the age-bin label, or None when age is missing."""
    if pd.isna(age):
        return None
    a = float(age)
    if a < 30:
        return "lt30"
    if a < 50:
        return "30_50"
    if a < 65:
        return "50_65"
    if a < 75:
        return "65_75"
    return "75plus"


def add_engineered(result, loc_col, age_col, mapping):
    """Append engineered columns to `result` in place; return the added names.

    Requires `result` to already contain numeric `age_norm` and `sex_male`
    columns (the interactions use them) and a raw `sex` column. Adds, in order:
    4 UV-group one-hots, 5 age-bin one-hots, `age_missing`, `sex_unknown`,
    4 `age_x_uv_*`, 4 `sex_x_uv_*` (19 columns total).
    """
    added = []

    groups = result[loc_col].apply(lambda v: uv_group(v, mapping))
    for g in UV_GROUPS:
        col = f"uv_{g}"
        result[col] = (groups == g).astype(float)
        added.append(col)

    bins = result[age_col].apply(age_bin)
    for b in AGE_BINS:
        col = f"age_bin_{b}"
        result[col] = (bins == b).astype(float)
        added.append(col)
    result["age_missing"] = result[age_col].isna().astype(float)
    added.append("age_missing")

    sex = result["sex"].astype(str).str.lower()
    result["sex_unknown"] = (~sex.isin(["male", "female"])).astype(float)
    added.append("sex_unknown")

    for g in UV_GROUPS:
        col = f"age_x_uv_{g}"
        result[col] = result["age_norm"] * result[f"uv_{g}"]
        added.append(col)
    for g in UV_GROUPS:
        col = f"sex_x_uv_{g}"
        result[col] = result["sex_male"] * result[f"uv_{g}"]
        added.append(col)

    return added
