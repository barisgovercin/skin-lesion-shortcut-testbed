"""Offline checks for the figure-helper pure functions (no GPU/data/model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.plots import _ece, _mean_ci
from src.evaluation.attribution import _merge_wordpieces, text_fraction


def test_ece_calibrated_scores_are_low():
    # Labels drawn with probability equal to the score -> near-perfect calibration.
    rng = np.random.default_rng(0)
    y_score = rng.uniform(0, 1, 5000)
    y_true = (rng.uniform(0, 1, 5000) < y_score).astype(int)
    assert _ece(y_true, y_score, n_bins=10) < 0.03


def test_ece_confident_but_wrong_is_one():
    y_true = np.zeros(100)
    y_score = np.ones(100)
    assert abs(_ece(y_true, y_score, n_bins=10) - 1.0) < 1e-9


def test_mean_ci_basic():
    m, ci = _mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and ci == 0.0
    m, ci = _mean_ci([0.9, 1.0, 1.1])
    assert abs(m - 1.0) < 1e-9 and ci > 0


def test_merge_wordpieces_sums_and_drops_specials():
    tokens = ["[CLS]", "mel", "##ano", "##ma", "refer", "[SEP]", "[PAD]"]
    attn = [1, 1, 1, 1, 1, 1, 0]
    scores = [9.0, 1.0, 2.0, 3.0, 5.0, 9.0, 9.0]
    words, ws = _merge_wordpieces(tokens, attn, scores)
    assert words == ["melanoma", "refer"]
    assert abs(ws[0] - 6.0) < 1e-9 and abs(ws[1] - 5.0) < 1e-9


def test_text_fraction_bounds():
    assert text_fraction(0, 0) == 0.0
    assert abs(text_fraction(1.0, 3.0) - 0.25) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all figure-helper tests passed")
