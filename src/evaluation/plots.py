"""Plots for the shortcut-learning testbed."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _agg(rows, key, by):
    out = {}
    for r in rows:
        out.setdefault(r[by], []).append(r[key])
    xs = sorted(out)
    return xs, [float(np.mean(out[x])) for x in xs]


def plot_dose_response(results_json, out_png):
    rows = json.load(open(results_json))
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, lbl in [("srs", "SRS"), ("ffilt_gap", "FFilt-gap")]:
        xs, ys = _agg(rows, key, "leakage_p")
        ax.plot(xs, ys, marker="o", label=lbl)
    ax.set_xlabel("leakage strength p")
    ax.set_ylabel("shortcut reliance")
    ax.set_title("Dose-response: shortcut reliance vs leakage strength")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_bakeoff(results_json, out_png):
    rows = json.load(open(results_json))
    xs, srs = _agg(rows, "srs", "method")
    _, au = _agg(rows, "auroc_ffilt", "method")
    x = np.arange(len(xs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, srs, w, label="SRS (lower = better)")
    ax.bar(x + w / 2, au, w, label="honest AUROC (FFilt)")
    ax.set_xticks(x); ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_title("Mitigation bake-off")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


# --- Headline-model diagnostics (ROC/PR + calibration) -----------------------
# Consume a per-sample prediction cache written by scripts/make_figures.py:
#   {"dataset":..., "backbone":..., "seeds": {"42": {"y_true": [...], "y_score": [...]}, ...}}
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)
from scipy import stats


def _mean_ci(vals):
    v = np.asarray(vals, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, 0.0
    return m, float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1))


def _ece(y_true, y_score, n_bins=10):
    """Expected Calibration Error (equal-width bins on the predicted probability)."""
    y_true = np.asarray(y_true, float); y_score = np.asarray(y_score, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y_true[m].mean() - y_score[m].mean())
    return float(ece)


def _load_preds(pred_json):
    data = json.load(open(pred_json))
    seeds = data["seeds"]
    items = [(s, np.asarray(d["y_true"], float), np.asarray(d["y_score"], float))
             for s, d in sorted(seeds.items(), key=lambda kv: int(kv[0]))]
    return data, items


def plot_roc_pr(pred_json, out_png, title=None):
    """ROC and Precision-Recall, one thin curve per seed plus the mean; legend
    reports mean AUROC / AP with a 95% CI across seeds."""
    data, items = _load_preds(pred_json)
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(12, 5))

    grid = np.linspace(0, 1, 200)
    tprs, aurocs = [], []
    for _, yt, ys in items:
        fpr, tpr, _ = roc_curve(yt, ys)
        aurocs.append(roc_auc_score(yt, ys))
        axr.plot(fpr, tpr, color="tab:blue", alpha=0.25, lw=1)
        it = np.interp(grid, fpr, tpr); it[0] = 0.0
        tprs.append(it)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    am, ac = _mean_ci(aurocs)
    axr.plot(grid, mean_tpr, color="tab:blue", lw=2.5,
             label=f"mean ROC (AUROC {am:.3f} $\\pm$ {ac:.3f})")
    axr.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    axr.set_xlabel("False positive rate (1 - specificity)")
    axr.set_ylabel("True positive rate (sensitivity)")
    axr.set_title("ROC"); axr.legend(loc="lower right"); axr.grid(alpha=0.3)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1.02)

    precs, aps = [], []
    pos_frac = np.concatenate([yt for _, yt, _ in items]).mean()
    for _, yt, ys in items:
        prec, rec, _ = precision_recall_curve(yt, ys)
        aps.append(average_precision_score(yt, ys))
        axp.plot(rec, prec, color="tab:orange", alpha=0.25, lw=1)
        order = np.argsort(rec)
        precs.append(np.interp(grid, rec[order], prec[order]))
    mean_prec = np.mean(precs, axis=0)
    apm, apc = _mean_ci(aps)
    axp.plot(grid, mean_prec, color="tab:orange", lw=2.5,
             label=f"mean PR (AP {apm:.3f} $\\pm$ {apc:.3f})")
    axp.axhline(pos_frac, ls="--", color="grey", lw=1,
                label=f"chance ({pos_frac:.2f})")
    axp.set_xlabel("Recall (sensitivity)"); axp.set_ylabel("Precision (PPV)")
    axp.set_title("Precision-Recall"); axp.legend(loc="upper right"); axp.grid(alpha=0.3)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)

    if title:
        fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_calibration(pred_json, out_png, title=None, n_bins=10):
    """Reliability diagram (pooled over seeds) + confidence histogram; legend
    reports mean ECE with a 95% CI across seeds."""
    data, items = _load_preds(pred_json)
    yt_all = np.concatenate([yt for _, yt, _ in items])
    ys_all = np.concatenate([ys for _, _, ys in items])
    eces = [_ece(yt, ys, n_bins) for _, yt, ys in items]
    em, ec = _mean_ci(eces)

    # Pooled reliability points (only non-empty bins).
    bins = np.linspace(0, 1, n_bins + 1)
    centers, obs = [], []
    idx = np.clip(np.digitize(ys_all, bins) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            centers.append(ys_all[m].mean()); obs.append(yt_all[m].mean())

    fig, (axc, axh) = plt.subplots(1, 2, figsize=(12, 5))
    axc.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    axc.plot(centers, obs, marker="o", color="tab:green",
             label=f"model (ECE {em:.3f} $\\pm$ {ec:.3f})")
    axc.set_xlabel("Predicted probability of malignancy")
    axc.set_ylabel("Observed malignant frequency")
    axc.set_title("Reliability diagram"); axc.legend(loc="upper left"); axc.grid(alpha=0.3)
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)

    axh.hist(ys_all, bins=20, color="tab:green", alpha=0.7)
    axh.set_xlabel("Predicted probability of malignancy"); axh.set_ylabel("Count")
    axh.set_title("Confidence histogram"); axh.grid(alpha=0.3); axh.set_xlim(0, 1)

    if title:
        fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_text_frac_bar(bakeoff_json, out_png, title=None):
    """Per-method text-attribution fraction (mean IG |attr| mass on text) with 95% CI,
    read straight from a shortcut bake-off JSON. Log-y because values span ~15x."""
    rows = json.load(open(bakeoff_json))
    from collections import defaultdict
    d = defaultdict(list)
    for r in rows:
        if r.get("text_frac") is not None:
            d[r["method"]].append(r["text_frac"])
    order = ["naive", "adversarial", "ffilt_only", "consistency", "counterfactual_aug"]
    methods = [m for m in order if m in d] + [m for m in d if m not in order]
    means = [float(np.mean(d[m])) for m in methods]
    cis = [_mean_ci(d[m])[1] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(methods)), means, yerr=cis, capsize=4, color="tab:purple", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", "\n") for m in methods])
    ax.set_ylabel("text-attribution fraction (IG, log scale)")
    ax.set_title(title or "How much the model relies on text (IG modality attribution)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
