"""Tests for the review-round statistics (src/evaluation/review.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.evaluation import review as rv
from src.evaluation.shortcut import srs_from_probs


def test_holm_matches_a_hand_computed_example():
    # sorted p: 0.01 (x3), 0.02 (x2), 0.04 (x1) -> 0.03, 0.04, 0.04; input order kept
    assert np.allclose(rv.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert rv.holm([0.6, 0.7])[1] == 1.0


def test_threshold_reaches_the_target_sensitivity_on_the_fitting_set():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.normal(size=500) + 1.5 * y
    t = rv.threshold_at_sensitivity(y, s, 0.95)
    op = rv.operating_point(y, s, t)
    assert op["sensitivity"] >= 0.95
    # one notch higher would fall below the target
    assert rv.operating_point(y, s, np.nextafter(t, np.inf))["sensitivity"] < 0.95


def test_srs_variants_prob_scale_equals_the_paper_metric():
    rng = np.random.default_rng(2)
    zc, zf = rng.normal(size=300), rng.normal(size=300)
    v = rv.srs_variants(zc, zf)
    assert v["prob"] == pytest.approx(srs_from_probs(rv.sigmoid(zc), rv.sigmoid(zf))["srs"])
    assert v["logit"] == pytest.approx(np.mean(np.abs(zc - zf)))


def test_rank_srs_ignores_monotone_recalibration_and_is_zero_without_reliance():
    rng = np.random.default_rng(3)
    zc = rng.normal(size=400)
    zf = zc + rng.normal(scale=0.5, size=400)
    warp = lambda z: np.tanh(z) * 3 + z ** 3     # strictly increasing
    assert rv.srs_variants(warp(zc), warp(zf))["rank"] == pytest.approx(rv.srs_variants(zc, zf)["rank"])
    assert rv.srs_variants(zc, zc) == {"prob": 0.0, "logit": 0.0, "rank": 0.0}


def test_cluster_bootstrap_keeps_groups_together_and_brackets_the_mean():
    values = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    groups = np.array(["a", "a", "b", "b", "c", "d"])
    lo, hi = rv.cluster_bootstrap_ci(values, groups, n_boot=500, seed=0)
    assert lo <= values.mean() <= hi
    same = np.array(["a"] * 6)
    assert rv.cluster_bootstrap_ci(values, same, n_boot=50) == [0.5, 0.5]


def test_early_warning_finds_the_level_where_a_metric_starts_to_move():
    levels = [0.5, 0.6, 0.7, 0.8]
    rise = {0.5: [0.01, 0.02, 0.015, 0.012, 0.018], 0.6: [0.011, 0.019, 0.016, 0.013, 0.017],
            0.7: [0.20, 0.25, 0.22, 0.21, 0.24], 0.8: [0.50, 0.55, 0.52, 0.51, 0.54]}
    out = rv.early_warning(levels, rise, 0.5, direction=+1)
    assert out["first_significant_level"] == 0.7
    assert out["rows"][-1]["share_of_full_change"] == pytest.approx(1.0)
    flat = {lv: [0.9, 0.91, 0.89, 0.9, 0.9] for lv in levels}
    assert rv.early_warning(levels, flat, 0.5, direction=-1)["first_significant_level"] is None


def _fake_runs(tmp_path):
    """Synthetic prediction files: image_only ignores the note, naive follows it,
    counterfactual_aug ignores it, ffilt_only in between."""
    from src.evaluation.predictions import save_predictions
    for seed in (42, 123):
        rng = np.random.default_rng(seed)
        nv, nt = 300, 400
        vy, ty = rng.integers(0, 2, nv), rng.integers(0, 2, nt)
        img_v, img_t = rng.normal(size=nv) + 1.5 * vy, rng.normal(size=nt) + 1.5 * ty
        ids = np.array([f"ISIC_{seed}_{i}" for i in range(nt)])
        groups = np.array([f"L{i // 2}" for i in range(nt)])
        for method, w in (("image_only", 0.0), ("naive", 6.0), ("counterfactual_aug", 0.0),
                          ("ffilt_only", 0.0)):
            lead_t = (2 * ty - 1) * w
            save_predictions(
                tmp_path / f"shortcut_ham10000_{method}_p1.0_seed{seed}.npz",
                {"dataset": "ham10000", "method": method, "seed": seed, "text_source": "template"},
                val_image_id=np.array([f"V{i}" for i in range(nv)]), val_group_id=np.arange(nv).astype(str),
                val_label=vy, val_logit=img_v + (2 * vy - 1) * w, val_logit_ffilt=img_v,
                test_image_id=ids, test_group_id=groups, test_label=ty,
                test_logit=img_t + lead_t, test_logit_flip=img_t - lead_t,
                test_logit_visible=img_t + lead_t, test_logit_ffilt=img_t)


def test_prediction_report_runs_end_to_end(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ra", Path(__file__).parent.parent / "scripts" / "review_analyses.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    _fake_runs(tmp_path)
    out = ra.predictions(tmp_path, tmp_path / "out", n_boot=50)
    runs = out["ham10000"]["per_run"]
    assert runs["image_only_seed42"]["srs"]["prob"] == 0.0
    assert runs["naive_seed42"]["srs"]["prob"] > 0.9
    assert runs["naive_seed42"]["honest"]["sens_valfixed"] > 0.85
    dca = out["ham10000"]["decision_curves"]
    assert dca["seeds"] == [42, 123] and dca["n"] == 800
    # identical image predictions -> identical net benefit for image-only and counterfactual
    assert dca["thresholds"]["0.1"]["cf_minus_image"] == pytest.approx(0.0)
    assert (tmp_path / "out" / "figures" / "dca_ham10000.png").exists()
    assert (tmp_path / "out" / "prediction_analyses.md").read_text().count("\n| Image only |") == 3
