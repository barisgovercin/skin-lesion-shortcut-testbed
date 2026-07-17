"""Offline checks for the LLM note generator via its deterministic mock backend.

No API key, network, GPU, or dataset needed. Verifies that the shortcut-testbed
invariants (identical facts across a pair, leakage_p concordance, filter-level
constraints) hold for src/preprocessing/llm_text.py exactly as they do for the
template engine.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.preprocessing.synthetic_text import _norm, ORIG, CFILT, DFILT, FFILT
from src.preprocessing.llm_text import add_llm_text_columns, _facts_text


def _df():
    return pd.DataFrame([
        {"image_id": "MEL1", "dx": "mel", "age": 55, "sex": "male",
         "localization": "back", "label": 1},
        {"image_id": "NV1", "dx": "nv", "age": 40, "sex": "female",
         "localization": "trunk", "label": 0},
    ])


def _tmp_cache(name):
    return Path(tempfile.gettempdir()) / f"llm_test_{name}.json"


def test_ffilt_is_facts_only():
    df = _df()
    out = add_llm_text_columns(df, level=FFILT, seed=7, leakage_p=1.0, provider="mock",
                               cache_path=_tmp_cache("ffilt"))
    for i, r in df.iterrows():
        expected = _facts_text(_norm(r), 7)
        assert out.loc[i, "text"] == expected, "FFilt note must be facts-only"


def test_pair_shares_facts_but_differs_in_clause():
    df = _df()
    out = add_llm_text_columns(df, level=DFILT, seed=7, leakage_p=1.0, pair=True,
                               provider="mock", cache_path=_tmp_cache("pair"))
    for i, r in df.iterrows():
        facts = _facts_text(_norm(r), 7)
        assert out.loc[i, "text_concordant"].startswith(facts)
        assert out.loc[i, "text_flipped"].startswith(facts)
        # Same facts, opposite impression -> the notes must differ.
        assert out.loc[i, "text_concordant"] != out.loc[i, "text_flipped"]


def test_leakage_p_extremes():
    df = _df()
    out = add_llm_text_columns(df, level=DFILT, seed=7, leakage_p=1.0, pair=True,
                               provider="mock", cache_path=_tmp_cache("p1"))
    assert (out["text"] == out["text_concordant"]).all(), "p=1.0 -> concordant"
    out0 = add_llm_text_columns(df, level=DFILT, seed=7, leakage_p=0.0, pair=True,
                                provider="mock", cache_path=_tmp_cache("p0"))
    assert (out0["text"] == out0["text_flipped"]).all(), "p=0.0 -> flipped"


def test_dfilt_hides_verdict_and_condition():
    df = _df()
    out = add_llm_text_columns(df, level=DFILT, seed=7, leakage_p=1.0, provider="mock",
                               cache_path=_tmp_cache("dfilt"))
    joined = " ".join(out["text"]).lower()
    for forbidden in ("malignant", "benign", "melanoma", "naevus", "nevus"):
        assert forbidden not in joined, f"DFilt must not reveal '{forbidden}'"


def test_orig_may_name_condition():
    df = _df()
    out = add_llm_text_columns(df, level=ORIG, seed=7, leakage_p=1.0, provider="mock",
                               cache_path=_tmp_cache("orig"))
    # The malignant melanoma row should name the condition at Orig level.
    assert "melanoma" in out.loc[0, "text"].lower()


def test_cache_written():
    df = _df()
    cp = _tmp_cache("cachecheck")
    if cp.exists():
        cp.unlink()
    add_llm_text_columns(df, level=CFILT, seed=1, leakage_p=1.0, provider="mock", cache_path=cp)
    assert cp.exists() and cp.stat().st_size > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all llm_text mock tests passed")
