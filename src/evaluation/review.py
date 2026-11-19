"""Statistics added for the Scientific Reports review round (November 2026).

Pure numpy/scipy functions, used by scripts/review_analyses.py:

- holm: Holm step-down adjustment for a family of p-values.
- threshold_at_sensitivity / operating_point: a 95%-sensitivity threshold fixed on
  the validation set and the sensitivity and specificity it gives on the test set.
- lesion_shift and srs_variants: the per-lesion quantity behind SRS, and SRS on the
  probability, logit and rank scales.
- cluster_bootstrap_ci: percentile CI that resamples lesions or patients.
- early_warning: at which leakage level a metric first departs from its baseline.
"""
import numpy as np
from scipy import stats


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def holm(pvals):
    """Holm-adjusted p-values, in the input order (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def threshold_at_sensitivity(y, score, target=0.95):
    """Largest threshold t such that predicting positive for score >= t reaches the
    target sensitivity on (y, score). Fit it on validation, apply it to test."""
    y = np.asarray(y).astype(bool)
    pos = np.sort(np.asarray(score, dtype=float)[y])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(k, 1) - 1])


def operating_point(y, score, t):
    """Sensitivity and specificity of the rule score >= t."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(score, dtype=float) >= t
    return {"sensitivity": float(pred[y].mean()), "specificity": float((~pred[~y]).mean())}


def lesion_shift(z_conc, z_flip):
    """Per-lesion shift |p(concordant) - p(flipped)|; SRS is its mean over lesions."""
    return np.abs(sigmoid(z_conc) - sigmoid(z_flip))


def srs_variants(z_conc, z_flip):
    """SRS on three scales from the logits of the concordant and the flipped note.

    prob  : mean |p_c - p_f| (the paper's SRS, bounded in [0, 1]).
    logit : mean |z_c - z_f| (unbounded; large for confident models).
    rank  : mean |F(z_c) - F(z_f)|, where F is the empirical CDF of the model's scores
            on the concordant test notes. It is unchanged by any monotone recalibration
            of the scores, so it cannot depend on calibration.
    """
    z_c = np.asarray(z_conc, dtype=np.float64)
    z_f = np.asarray(z_flip, dtype=np.float64)
    ref = np.sort(z_c)
    ecdf = lambda z: np.searchsorted(ref, z, side="right") / len(ref)
    return {"prob": float(np.mean(lesion_shift(z_c, z_f))),
            "logit": float(np.mean(np.abs(z_c - z_f))),
            "rank": float(np.mean(np.abs(ecdf(z_c) - ecdf(z_f))))}


def cluster_bootstrap_ci(values, groups, n_boot=2000, seed=0, level=0.95):
    """Percentile CI of the mean of `values`, resampling whole groups (lesions or
    patients) with replacement so that images of one lesion stay together."""
    values = np.asarray(values, dtype=float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    g = len(sums)
    draws = rng.integers(0, g, size=(n_boot, g))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def early_warning(levels, by_level, baseline, direction, alpha=0.05):
    """When does a metric first move away from its value at the baseline level?

    by_level maps each leakage level to per-seed values in the same seed order.
    For every level above the baseline, a one-sided paired t-test checks a change in
    `direction` (+1 rise, -1 fall); the p-values are Holm-adjusted across levels. Also
    returns the share of the full baseline-to-maximum change reached at each level.
    """
    base = np.asarray(by_level[baseline], dtype=float)
    top = max(levels)
    full = float(np.mean(by_level[top]) - base.mean())
    rows, pvals = [], []
    for lv in sorted(l for l in levels if l != baseline):
        v = np.asarray(by_level[lv], dtype=float)
        d = direction * (v - base)
        if np.allclose(d, d[0]):
            p = 0.0 if d[0] > 0 else 1.0
        else:
            p = float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue)
        pvals.append(p)
        rows.append({"level": lv, "mean": float(v.mean()),
                     "change": float(v.mean() - base.mean()),
                     "share_of_full_change": float((v.mean() - base.mean()) / full) if full else float("nan"),
                     "p": p})
    for r, pa in zip(rows, holm(pvals)):
        r["p_holm"] = float(pa)
    first = next((r["level"] for r in rows if r["p_holm"] < alpha), None)
    return {"rows": rows, "first_significant_level": first}
