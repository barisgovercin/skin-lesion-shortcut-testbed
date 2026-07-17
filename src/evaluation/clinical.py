"""Clinical-utility evaluation: probability calibration and decision-curve analysis.

Pure numerical helpers (no torch, no I/O) so they are unit-testable offline. The
driver in scripts/clinical_eval.py feeds these per-sample logits/labels obtained
from the saved vision checkpoints.

- Temperature scaling: a single scalar T (fit on the validation split by
  minimising NLL) that divides the logits before the sigmoid; the standard,
  leak-free post-hoc calibrator (Guo et al., 2017).
- Decision-curve analysis: net benefit of using the model at a decision
  threshold p_t, versus the "treat all" and "treat none" defaults
  (Vickers & Elkin, 2006). Net benefit = TP/N - (FP/N) * p_t/(1-p_t).
"""
import numpy as np
from scipy.optimize import minimize_scalar


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits, labels):
    """Return the temperature T>0 that minimises validation NLL of sigmoid(logit/T)."""
    logits = np.asarray(logits, float)
    labels = np.asarray(labels, float)

    def nll(temp):
        p = np.clip(_sigmoid(logits / temp), 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(logits, temp):
    """Calibrated probabilities sigmoid(logit / T)."""
    return _sigmoid(np.asarray(logits, float) / float(temp))


def ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (equal-width bins on the predicted probability)."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += (m.sum() / n) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(out)


def net_benefit(y_true, y_prob, pt):
    """Net benefit of treating everyone with predicted probability >= pt."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    if n == 0 or pt >= 1.0:
        return 0.0
    pred = y_prob >= pt
    tp = float(np.sum(pred & (y_true == 1)))
    fp = float(np.sum(pred & (y_true == 0)))
    return tp / n - (fp / n) * (pt / (1.0 - pt))


def net_benefit_treat_all(y_true, pt):
    """Net benefit of the 'treat everyone' default at threshold pt."""
    y_true = np.asarray(y_true, float)
    if pt >= 1.0:
        return 0.0
    prev = float(np.mean(y_true))
    return prev - (1.0 - prev) * (pt / (1.0 - pt))


def decision_curve(y_true, y_prob, thresholds=None):
    """Return (thresholds, model NB, treat-all NB). Treat-none NB is 0 by definition."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    thresholds = np.asarray(thresholds, float)
    model = np.array([net_benefit(y_true, y_prob, t) for t in thresholds])
    treat_all = np.array([net_benefit_treat_all(y_true, t) for t in thresholds])
    return thresholds, model, treat_all


def fit_platt(logits, labels):
    """Platt scaling: fit (a, b) so sigmoid(a*logit + b) is calibrated. Unlike
    temperature, the bias b corrects a systematic shift, not only sharpness."""
    from sklearn.linear_model import LogisticRegression
    logits = np.asarray(logits, float).reshape(-1, 1)
    labels = np.asarray(labels, int)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits, labels)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(logits, a, b):
    return _sigmoid(a * np.asarray(logits, float) + b)


def fit_isotonic(probs, labels):
    """Non-parametric isotonic calibration fit on (probability, label) pairs.
    Returns the fitted regressor; use apply_isotonic to map new probabilities."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    return iso.fit(np.asarray(probs, float), np.asarray(labels, float))


def apply_isotonic(iso, probs):
    return np.asarray(iso.predict(np.asarray(probs, float)), float)
