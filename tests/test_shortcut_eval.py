"""Offline tests for shortcut-reliance metrics + attribution helper."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.shortcut import srs_from_probs, ffilt_gap_from_auroc
from src.evaluation.attribution import text_fraction


def test_srs_zero_when_unchanged():
    p = np.array([0.2, 0.8, 0.5])
    out = srs_from_probs(p, p)
    assert out["srs"] == 0.0 and out["flip_rate"] == 0.0


def test_srs_mean_abs_and_flip_rate():
    p_orig = np.array([0.9, 0.1, 0.6])
    p_flip = np.array([0.2, 0.4, 0.55])
    out = srs_from_probs(p_orig, p_flip)
    assert abs(out["srs"] - np.mean(np.abs(p_orig - p_flip))) < 1e-9
    # class flips at 0.5: 0.9->0.2 (yes); 0.1->0.4 (no); 0.6->0.55 (no) -> 1/3
    assert abs(out["flip_rate"] - (1 / 3)) < 1e-9


def test_ffilt_gap():
    assert abs(ffilt_gap_from_auroc(0.98, 0.91) - 0.07) < 1e-9


def test_text_fraction_normalises():
    assert abs(text_fraction(3.0, 1.0) - 0.75) < 1e-9
    assert text_fraction(0.0, 0.0) == 0.0


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
