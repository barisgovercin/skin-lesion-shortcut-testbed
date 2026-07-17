"""Offline tests for controllable-leakage synthetic text (no GPU/data needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.preprocessing import synthetic_text as st


def _row(label=1, dx="mel"):
    return pd.Series({"image_id": "X1", "dx": dx, "label": label,
                      "age": 60, "sex": "female", "localization": "back"})


def _frame(n=400, mal_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < mal_frac).astype(int)
    return pd.DataFrame({
        "image_id": [f"I{i}" for i in range(n)],
        "dx": ["mel" if l else "nv" for l in labels],
        "label": labels, "age": 55, "sex": "male", "localization": "back",
    })


# ---- Task 1: pair + leakage_p API ----------------------------------------

def test_backward_compat_p1_matches_default():
    r = _row()
    assert st.generate_note(r, level=st.ORIG, seed=0, leakage_p=1.0) == \
           st.generate_note(r, level=st.ORIG, seed=0)


def test_generate_pair_facts_identical_diagnosis_differs():
    r = _row(label=1)
    concordant, flipped = st.generate_pair(r, level=st.ORIG, seed=0)
    assert concordant != flipped
    assert st.facts_only(r, seed=0) == st.facts_only(r, seed=0)
    # The facts substring (present in both) must be byte-identical.
    facts = st.facts_only(r, seed=0)
    assert facts and facts in concordant and facts in flipped
    assert ("malignant" in concordant.lower()) and ("malignant" not in flipped.lower())


# ---- Task 2: leakage_p statistics / determinism / columns ----------------

def test_leakage_p_realised_concordance():
    df = _frame()
    for p in (0.5, 0.7, 1.0):
        # CFilt keeps the benign/malignant verdict word, so concordance is detectable.
        out = st.add_text_columns(df, level=st.CFILT, seed=1, leakage_p=p)
        says_mal = out["text"].str.contains("malignant", case=False)
        concord = (says_mal == out["label"].astype(bool)).mean()
        assert abs(concord - p) < 0.08, f"p={p} got {concord:.2f}"


def test_determinism_same_inputs():
    df = _frame(n=50)
    a = st.add_text_columns(df, level=st.ORIG, seed=3, leakage_p=0.8)["text"].tolist()
    b = st.add_text_columns(df, level=st.ORIG, seed=3, leakage_p=0.8)["text"].tolist()
    assert a == b


def test_add_text_columns_pair():
    df = _frame(n=20)
    out = st.add_text_columns(df, level=st.ORIG, seed=0, pair=True)
    assert {"text_concordant", "text_flipped"}.issubset(out.columns)
    assert (out["text_concordant"] != out["text_flipped"]).all()


# ---- Task 3: ISIC adapter -------------------------------------------------

def test_isic_shaped_row():
    r = pd.Series({"image_name": "ISIC_1", "target": 1, "age_approx": 70,
                   "sex": "male", "anatom_site_general_challenge": "torso"})
    note = st.generate_note(r, level=st.ORIG, seed=0, leakage_p=1.0)
    assert "70-year-old" in note and "malignant" in note.lower()
    c, f = st.generate_pair(r, level=st.ORIG, seed=0)
    assert "malignant" in c.lower() and "malignant" not in f.lower()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
