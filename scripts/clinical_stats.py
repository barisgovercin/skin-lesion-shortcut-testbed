"""Statistical robustness for the clinical evaluation (GPU-free, from cached logits).

Complements scripts/clinical_eval.py with the analyses a reviewer of a calibration
+ decision-curve paper will ask for, all computed from the cached per-seed
validation/test logits (results/clinical_logits_*.json), so no retraining:

  - Brier score (a proper scoring rule) per calibrator, mean +/- 95% CI over 5 seeds.
  - Adaptive-bin (equal-frequency) ECE, to show the ECE finding is not an artefact
    of fixed-width bins.
  - Paired significance (t-test over 5 seeds; Wilcoxon as a check) that Platt and
    isotonic beat temperature (and uncalibrated) on ECE.
  - Bootstrap 95% CIs for the net benefit of the isotonic-calibrated model and for
    its margin over treat-all, at representative referral thresholds.

  python scripts/clinical_stats.py
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats

from src.evaluation.clinical import (fit_temperature, apply_temperature, ece,
                                      fit_platt, apply_platt, fit_isotonic, apply_isotonic,
                                      _sigmoid, net_benefit, net_benefit_treat_all)

SEEDS = [42, 123, 456, 789, 1024]
CALIBRATORS = ["uncalibrated", "temperature", "platt", "isotonic"]
PTS = [0.05, 0.10, 0.20, 0.30]


def _mci(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    return (m, 0.0) if len(v) < 2 else (m, float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece_adaptive(y, p, n_bins=10):
    """Equal-frequency (quantile) binned ECE."""
    y = np.asarray(y, float); p = np.asarray(p, float); n = len(y)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(out)


def _load(ds, backbone):
    d = json.load(open(f"results/clinical_logits_{ds}_{backbone}.json"))
    return {int(k): {kk: np.array(vv, float) for kk, vv in v.items()} for k, v in d.items()}


def _per_seed(logits):
    M = {c: {"ece_fixed": [], "ece_adapt": [], "brier": []} for c in CALIBRATORS}
    pooled_y, pooled_iso = [], []
    for _, d in sorted(logits.items()):
        vL, vY, tL, tY = d["vL"], d["vY"], d["tL"], d["tY"]
        T = fit_temperature(vL, vY)
        a, b = fit_platt(vL, vY)
        iso = fit_isotonic(_sigmoid(vL), vY)
        preds = {"uncalibrated": _sigmoid(tL), "temperature": apply_temperature(tL, T),
                 "platt": apply_platt(tL, a, b), "isotonic": apply_isotonic(iso, _sigmoid(tL))}
        for c, p in preds.items():
            M[c]["ece_fixed"].append(ece(tY, p))
            M[c]["ece_adapt"].append(ece_adaptive(tY, p))
            M[c]["brier"].append(brier(tY, p))
        pooled_y.append(tY); pooled_iso.append(preds["isotonic"])
    return M, np.concatenate(pooled_y), np.concatenate(pooled_iso)


def _paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    tp = float(stats.ttest_rel(a, b).pvalue)
    try:
        wp = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        wp = float("nan")
    return tp, wp


def _boot_nb(y, p, pts, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y); y = np.asarray(y, float); p = np.asarray(p, float)
    model = {pt: [] for pt in pts}; ta = {pt: [] for pt in pts}; diff = {pt: [] for pt in pts}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        for pt in pts:
            m = net_benefit(yy, pp, pt); a = net_benefit_treat_all(yy, pt)
            model[pt].append(m); ta[pt].append(a); diff[pt].append(m - a)
    def ci(d):
        arr = np.asarray(d, float)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return {pt: {"model_pt": net_benefit(y, p, pt), "model_ci": ci(model[pt]),
                 "treat_all_pt": net_benefit_treat_all(y, pt), "treat_all_ci": ci(ta[pt]),
                 "diff_ci": ci(diff[pt])} for pt in pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham10000", "isic2020", "isic_dicm_17k"])
    ap.add_argument("--backbone", default="convnext_base")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    md = ["# Clinical evaluation: statistical robustness\n",
          "GPU-free, from cached per-seed logits. Brier and adaptive-bin (equal-frequency) "
          "ECE complement the fixed-bin ECE; paired t-tests over 5 seeds test the calibrator "
          "differences (Wilcoxon shown as a check, with a p>=0.0625 floor at n=5); net-benefit "
          "95% CIs are a sample-level bootstrap (B={}) on the pooled isotonic-calibrated test "
          "predictions.\n".format(args.bootstrap)]
    out_json = {}

    for ds in args.datasets:
        try:
            logits = _load(ds, args.backbone)
        except FileNotFoundError:
            print(f"skip {ds}: no cached logits"); continue
        M, pooled_y, pooled_iso = _per_seed(logits)

        md += [f"\n## {ds}\n",
               "**Calibration metrics (mean +/- 95% CI over 5 seeds; lower is better):**\n",
               "| calibrator | ECE (fixed) | ECE (adaptive) | Brier |",
               "|---|---|---|---|"]
        cell = {}
        for c in CALIBRATORS:
            ef, ea, br = _mci(M[c]["ece_fixed"]), _mci(M[c]["ece_adapt"]), _mci(M[c]["brier"])
            cell[c] = {"ece_fixed": ef, "ece_adapt": ea, "brier": br}
            md.append(f"| {c} | {ef[0]:.3f} +/- {ef[1]:.3f} | {ea[0]:.3f} +/- {ea[1]:.3f} | "
                      f"{br[0]:.4f} +/- {br[1]:.4f} |")

        # paired significance on fixed-bin ECE
        comps = [("platt", "temperature"), ("isotonic", "temperature"),
                 ("platt", "uncalibrated"), ("isotonic", "uncalibrated"),
                 ("temperature", "uncalibrated")]
        md += ["\n**Paired significance on ECE (fixed), 5 seeds:**\n",
               "| comparison | mean dECE | t-test p | Wilcoxon p |", "|---|---|---|---|"]
        sig = {}
        for x, yb in comps:
            d = np.array(M[x]["ece_fixed"]) - np.array(M[yb]["ece_fixed"])
            tp, wp = _paired(M[x]["ece_fixed"], M[yb]["ece_fixed"])
            sig[f"{x}_vs_{yb}"] = {"mean_delta": float(d.mean()), "t_p": tp, "wilcoxon_p": wp}
            md.append(f"| {x} vs {yb} | {d.mean():+.3f} | {tp:.4g} | {wp:.4g} |")

        nb = _boot_nb(pooled_y, pooled_iso, PTS, B=args.bootstrap)
        md += ["\n**Net benefit (isotonic-calibrated), bootstrap 95% CI:**\n",
               "| threshold | model NB [95% CI] | treat-all NB | margin over treat-all [95% CI] |",
               "|---|---|---|---|"]
        for pt in PTS:
            r = nb[pt]
            md.append(f"| {pt:.2f} | {r['model_pt']:.3f} [{r['model_ci'][0]:.3f}, {r['model_ci'][1]:.3f}] "
                      f"| {r['treat_all_pt']:.3f} | {r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f} |")

        out_json[ds] = {"calibration": cell, "significance": sig,
                        "net_benefit": {str(pt): nb[pt] for pt in PTS}}
        print(f"{ds}: ECE fixed temp {cell['temperature']['ece_fixed'][0]:.3f} / "
              f"platt {cell['platt']['ece_fixed'][0]:.3f} (t-p {sig['platt_vs_temperature']['t_p']:.2g}); "
              f"Brier temp {cell['temperature']['brier'][0]:.4f} / platt {cell['platt']['brier'][0]:.4f}; "
              f"NB@0.10 margin CI {nb[0.10]['diff_ci']}")

    Path("results/reports").mkdir(parents=True, exist_ok=True)
    Path("results/reports/clinical_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    json.dump(out_json, open("results/clinical_stats.json", "w"), indent=2)
    print("Saved results/reports/clinical_stats.md and results/clinical_stats.json")


if __name__ == "__main__":
    main()
