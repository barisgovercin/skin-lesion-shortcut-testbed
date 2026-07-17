"""Offline tests for synthetic clinical text generation (no GPU/data/LLM needed).

Validates that label-leaking content is removed at exactly the right filter level,
matching Watson et al. (2026) Table 3. Run from the repo root:

    python tests/test_synthetic_text.py
    pytest tests/test_synthetic_text.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.preprocessing.synthetic_text import (
    generate_note, generate_all_levels, add_synthetic_text,
    LEVELS, ORIG, CFILT, DFILT, FFILT,
)

MAL_ROW = {"image_id": "ISIC_mal", "dx": "mel", "label": 1, "age": 55, "sex": "male", "localization": "back"}
BEN_ROW = {"image_id": "ISIC_ben", "dx": "nv", "label": 0, "age": 40, "sex": "female", "localization": "trunk"}


def test_all_levels_nonempty():
    out = generate_all_levels(MAL_ROW, seed=0)
    assert set(out) == set(LEVELS)
    for lvl in LEVELS:
        assert isinstance(out[lvl], str) and out[lvl].strip()


def test_condition_name_removed_at_cfilt():
    out = generate_all_levels(MAL_ROW, seed=0)
    assert "melanoma" in out[ORIG].lower()
    for lvl in (CFILT, DFILT, FFILT):
        assert "melanoma" not in out[lvl].lower(), f"condition name leaked into {lvl}"


def test_diagnosis_words_removed_at_dfilt():
    out = generate_all_levels(MAL_ROW, seed=0)
    assert "malignant" in out[ORIG].lower()
    assert "malignant" in out[CFILT].lower()  # CFilt keeps the verdict word
    for lvl in (DFILT, FFILT):
        assert "malignant" not in out[lvl].lower(), f"'malignant' leaked into {lvl}"


def test_leading_removed_only_at_ffilt():
    out = generate_all_levels(MAL_ROW, seed=0)
    markers = ("referral", "refer", "excision", "concerning")
    for lvl in (ORIG, CFILT, DFILT):
        assert any(m in out[lvl].lower() for m in markers), f"leading phrasing missing from {lvl}"
    assert not any(m in out[FFILT].lower() for m in markers), "leading phrasing leaked into FFilt"


def test_facts_survive_all_levels():
    out = generate_all_levels(MAL_ROW, seed=0)
    for lvl in LEVELS:
        text = out[lvl].lower()
        assert "55-year-old" in text and "back" in text, f"patient facts missing from {lvl}"


def test_benign_leakage_pattern():
    out = generate_all_levels(BEN_ROW, seed=0)
    assert "naevus" in out[ORIG].lower()
    assert all("naevus" not in out[lvl].lower() for lvl in (CFILT, DFILT, FFILT))
    assert "benign" in out[ORIG].lower() and "benign" in out[CFILT].lower()
    assert all("benign" not in out[lvl].lower() for lvl in (DFILT, FFILT))


def test_ffilt_has_no_leak_tokens():
    leak_tokens = [
        "malignant", "benign", "melanoma", "carcinoma", "keratosis", "naevus",
        "dermatofibroma", "vascular", "referral", "refer", "excision",
        "monitoring", "reassurance", "concerning",
    ]
    for row in (MAL_ROW, BEN_ROW):
        ffilt = generate_all_levels(row, seed=0)[FFILT].lower()
        for tok in leak_tokens:
            assert tok not in ffilt, f"FFilt leaked '{tok}' for dx={row['dx']}"


def test_length_monotonic():
    # Stripping content can only shorten the note: Orig >= CFilt >= DFilt >= FFilt.
    out = generate_all_levels(MAL_ROW, seed=0)
    lengths = [len(out[lvl]) for lvl in LEVELS]
    assert lengths == sorted(lengths, reverse=True), f"non-monotonic lengths: {lengths}"


def test_determinism_and_variety():
    a = generate_all_levels(MAL_ROW, seed=7)
    b = generate_all_levels(MAL_ROW, seed=7)
    assert a == b, "same seed must give identical notes"
    # Phrasing should vary across seeds (banks are small, so check over a range).
    variants = {generate_note(MAL_ROW, ORIG, seed=s) for s in range(20)}
    assert len(variants) > 1, "phrasing should vary across seeds"


def test_label_derived_from_dx_when_no_label():
    row = {"image_id": "x", "dx": "bcc", "age": 60, "sex": "male", "localization": "face"}
    orig = generate_note(row, ORIG, seed=0).lower()
    assert "malignant" in orig and "basal cell carcinoma" in orig


def test_missing_metadata_is_safe():
    row = {"image_id": "y", "dx": "nv", "label": 0, "age": None, "sex": None, "localization": None}
    out = generate_all_levels(row, seed=0)
    assert "adult" in out[FFILT].lower() and "unspecified site" in out[FFILT].lower()


def test_add_synthetic_text_column():
    df = pd.DataFrame([MAL_ROW, BEN_ROW])
    out = add_synthetic_text(df, level=DFILT, seed=0)
    assert "text" in out.columns and len(out) == 2
    assert all(isinstance(t, str) and t for t in out["text"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
