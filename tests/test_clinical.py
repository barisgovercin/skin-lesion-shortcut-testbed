"""Offline checks for the clinical-evaluation helpers (no GPU/data/model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.clinical import (fit_temperature, apply_temperature, ece,
                                     net_benefit, net_benefit_treat_all,
                                     decision_curve, _sigmoid,
                                     fit_platt, apply_platt, fit_isotonic, apply_isotonic)


def test_platt_fixes_bias():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.0, 4000)
    y = (rng.uniform(size=4000) < _sigmoid(z)).astype(float)
    biased = z + 2.0                                   # systematic positive shift
    a, b = fit_platt(biased, y)
    assert ece(y, apply_platt(biased, a, b)) < ece(y, _sigmoid(biased))


def test_isotonic_improves_or_matches():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1.0, 4000)
    y = (rng.uniform(size=4000) < _sigmoid(z)).astype(float)
    over = _sigmoid(z * 3.0)                            # overconfident probabilities
    iso = fit_isotonic(over, y)
    assert ece(y, apply_isotonic(iso, over)) <= ece(y, over) + 1e-6


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.0, 4000)                       # well-calibrated logits
    y = (rng.uniform(size=4000) < _sigmoid(z)).astype(float)
    over = z * 3.0                                     # overconfident logits
    T = fit_temperature(over, y)
    assert T > 1.5                                     # should shrink them back (~3)
    assert ece(y, apply_temperature(over, T)) < ece(y, _sigmoid(over))


def test_temperature_near_one_for_calibrated():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.5, 5000)
    y = (rng.uniform(size=5000) < _sigmoid(z)).astype(float)
    assert 0.7 < fit_temperature(z, y) < 1.4


def test_net_benefit_all_correct():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.8, 0.1, 0.2])
    # pt=0.5 -> pred [1,1,0,0], tp=2, fp=0, n=4 -> 0.5
    assert abs(net_benefit(y, p, 0.5) - 0.5) < 1e-9


def test_net_benefit_treat_all_zero_at_prevalence():
    y = np.array([1, 1, 0, 0])                         # prevalence 0.5
    assert abs(net_benefit_treat_all(y, 0.5)) < 1e-9   # 0.5 - 0.5*(0.5/0.5) = 0


def test_decision_curve_shapes():
    y = np.array([1, 0, 1, 0, 1]); p = np.array([0.8, 0.2, 0.6, 0.3, 0.7])
    th, m, a = decision_curve(y, p)
    assert len(th) == len(m) == len(a) and len(th) > 10


def test_ece_confident_wrong_is_one():
    y = np.zeros(100); p = np.ones(100)
    assert abs(ece(y, p) - 1.0) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all clinical helper tests passed")
