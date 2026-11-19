"""Analyses added for the Scientific Reports review round (November 2026).

  python scripts/review_analyses.py logged
      From the logged per-seed results (no GPU, no predictions needed):
      - Holm-adjusted p-values for the bake-off comparisons (Reviewer 2, point 5).
      - Early-warning analysis of the dose-response: the first leakage level at which
        SRS and honest AUROC move significantly away from p = 0.5 (Reviewer 2, point 2).

  python scripts/review_analyses.py predictions --preds DIR
      From the per-lesion predictions saved by run_shortcut.py --save-preds:
      operating points at a validation-fixed threshold, SRS variants with
      lesion-clustered CIs, the ablation table and decision curves on shared test cases.

Writes results/review/*.json and results/review/*.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import review as rv  # noqa: E402
from src.evaluation.clinical import (apply_isotonic, fit_isotonic, net_benefit,  # noqa: E402
                                     net_benefit_treat_all)
from src.evaluation.predictions import load_predictions  # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "review"
BEST = "counterfactual_aug"
OTHERS = ["consistency", "ffilt_only", "adversarial", "naive"]
METRICS = {"auroc_ffilt": "honest AUROC", "srs": "SRS"}
BAKEOFFS = {"HAM10000": "shortcut_ham10000_bakeoff.json",
            "ISIC 2020": "shortcut_isic2020_bakeoff.json",
            "ISIC-DICM-17K": "shortcut_isic_dicm_17k_bakeoff.json",
            "HAM10000, LLM notes": "shortcut_ham10000_llm_bakeoff.json"}
DOSES = {k: v.replace("_bakeoff", "_dose") for k, v in BAKEOFFS.items()}


def _by_seed(rows, method, metric):
    rows = sorted((r for r in rows if r["method"] == method), key=lambda r: r["seed"])
    return [r["seed"] for r in rows], np.array([r[metric] for r in rows], float)


def holm_bakeoff():
    """Paired t-tests of counterfactual augmentation against every other method on
    honest AUROC and SRS, Holm-adjusted within each dataset (8 tests) and across all
    datasets together (32 tests)."""
    tests = []
    for ds, fname in BAKEOFFS.items():
        rows = json.load(open(RESULTS / fname))
        for metric in METRICS:
            seeds_b, best = _by_seed(rows, BEST, metric)
            for other in OTHERS:
                seeds_o, vals = _by_seed(rows, other, metric)
                assert seeds_b == seeds_o, (ds, other)
                p = float(stats.ttest_rel(best, vals).pvalue)
                tests.append({"dataset": ds, "metric": metric, "comparison": f"{BEST} vs {other}",
                              "mean_diff": float((best - vals).mean()), "p": p})
    for ds in BAKEOFFS:
        fam = [t for t in tests if t["dataset"] == ds]
        for t, pa in zip(fam, rv.holm([t["p"] for t in fam])):
            t["p_holm_dataset"] = float(pa)
    for t, pa in zip(tests, rv.holm([t["p"] for t in tests])):
        t["p_holm_all"] = float(pa)
    return tests


def early_warning_dose():
    out = {}
    for ds, fname in DOSES.items():
        path = RESULTS / fname
        if not path.exists():
            continue
        rows = [r for r in json.load(open(path)) if r["method"] == "naive"]
        levels = sorted({r["leakage_p"] for r in rows})
        res = {}
        for metric, direction in (("srs", +1), ("auroc_ffilt", -1)):
            by_level = {}
            for lv in levels:
                seeds, vals = _by_seed([r for r in rows if r["leakage_p"] == lv], "naive", metric)
                by_level[lv] = vals
            res[metric] = rv.early_warning(levels, by_level, levels[0], direction)
        out[ds] = res
    return out


def _fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def logged():
    OUT.mkdir(parents=True, exist_ok=True)
    tests = holm_bakeoff()
    ew = early_warning_dose()
    json.dump({"holm_bakeoff": tests, "early_warning": ew},
              open(OUT / "logged_analyses.json", "w"), indent=2)

    md = ["# Review analyses from the logged per-seed results\n",
          "## Bake-off comparisons with Holm correction\n",
          "Paired t-tests over the five shared seeds, counterfactual augmentation minus the "
          "other method. Holm adjustment within each dataset (8 tests) and across all 32.\n",
          "| dataset | metric | comparison | mean diff | p | Holm (dataset) | Holm (all) |",
          "|---|---|---|---|---|---|---|"]
    for t in tests:
        md.append(f"| {t['dataset']} | {METRICS[t['metric']]} | vs {t['comparison'].split(' vs ')[1]} | "
                  f"{t['mean_diff']:+.4f} | {_fmt_p(t['p'])} | {_fmt_p(t['p_holm_dataset'])} | "
                  f"{_fmt_p(t['p_holm_all'])} |")
    lost = [t for t in tests if t["p"] < 0.05 <= t["p_holm_dataset"]]
    md.append(f"\nSignificant before but not after the within-dataset Holm correction: "
              f"{len(lost)} of {sum(t['p'] < 0.05 for t in tests)}.")
    for t in lost:
        md.append(f"- {t['dataset']}, {METRICS[t['metric']]}, {t['comparison']}: "
                  f"p = {_fmt_p(t['p'])} -> {_fmt_p(t['p_holm_dataset'])}")

    md += ["\n## Early warning in the dose-response (naive model)\n",
           "One-sided paired t-test of each leakage level against p = 0.5 (SRS rising, honest "
           "AUROC falling), Holm-adjusted across levels. 'Share' is the fraction of the full "
           "p = 0.5 to 1.0 change already reached.\n"]
    for ds, res in ew.items():
        md += [f"\n**{ds}.** First significant level: SRS p = "
               f"{res['srs']['first_significant_level']}, honest AUROC p = "
               f"{res['auroc_ffilt']['first_significant_level']}.\n",
               "| p | SRS | share | Holm p | honest AUROC | share | Holm p |",
               "|---|---|---|---|---|---|---|"]
        for s, a in zip(res["srs"]["rows"], res["auroc_ffilt"]["rows"]):
            md.append(f"| {s['level']} | {s['mean']:.3f} | {s['share_of_full_change']:.2f} | "
                      f"{_fmt_p(s['p_holm'])} | {a['mean']:.3f} | {a['share_of_full_change']:.2f} | "
                      f"{_fmt_p(a['p_holm'])} |")
    (OUT / "logged_analyses.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


# ---------------------------------------------------------------- predictions part

PTS = [0.05, 0.10, 0.20, 0.30]
ABLATION = [("image_only", "Image only"), ("ffilt_only", "Facts-only note (filtering)"),
            ("naive", "Original note (naive)"), ("counterfactual_aug", "Augmented note (counterfactual)")]
LABEL = dict(ABLATION)


def _mci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1))


def _cell(v, nd=3):
    m, h = _mci(v)
    return f"{m:.{nd}f} ± {h:.{nd}f}" if len(v) > 1 else f"{m:.{nd}f}"


def load_runs(pred_dir):
    runs = []
    for f in sorted(Path(pred_dir).glob("*.npz")):
        d = load_predictions(f)
        m = d["meta"]
        ds = m["dataset"] + ("_llm" if m.get("text_source") == "llm" else "")
        runs.append({"dataset": ds, "method": m["method"], "seed": int(m["seed"]), "d": d})
    return runs


def run_metrics(d, n_boot=1000):
    """Operating points, AUROC/AP and SRS variants for one trained model."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, vy = d["test_label"], d["val_label"]
    out = {}
    for kind, vz, tz in (("honest", d["val_logit_ffilt"], d["test_logit_ffilt"]),
                         ("visible", d["val_logit"], d["test_logit_visible"])):
        op = rv.operating_point(y, tz, rv.threshold_at_sensitivity(vy, vz, 0.95))
        read = rv.operating_point(y, tz, rv.threshold_at_sensitivity(y, tz, 0.95))
        out[kind] = {"auroc": float(roc_auc_score(y, tz)), "ap": float(average_precision_score(y, tz)),
                     "sens_valfixed": op["sensitivity"], "spec_valfixed": op["specificity"],
                     "spec95_test_read": read["specificity"]}
    srs = rv.srs_variants(d["test_logit"], d["test_logit_flip"])
    iso = fit_isotonic(rv.sigmoid(d["val_logit"]), vy)
    pc = apply_isotonic(iso, rv.sigmoid(d["test_logit"]))
    pf = apply_isotonic(iso, rv.sigmoid(d["test_logit_flip"]))
    srs["prob_isotonic"] = float(np.mean(np.abs(pc - pf)))
    shift = rv.lesion_shift(d["test_logit"], d["test_logit_flip"])
    srs["prob_ci"] = rv.cluster_bootstrap_ci(shift, d["test_group_id"], n_boot=n_boot)
    srs["median"], srs["q90"], srs["q99"] = (float(np.quantile(shift, q)) for q in (0.5, 0.9, 0.99))
    srs["max"] = float(shift.max())
    srs["share_over_0.1"] = float(np.mean(shift > 0.1))
    srs["share_over_0.5"] = float(np.mean(shift > 0.5))
    out["srs"] = srs
    return out


def _calibrated_honest(d):
    """Isotonic calibration fitted on the validation facts-only notes, applied to the test
    facts-only notes (for the image-only model both are its image predictions)."""
    iso = fit_isotonic(rv.sigmoid(d["val_logit_ffilt"]), d["val_label"])
    return apply_isotonic(iso, rv.sigmoid(d["test_logit_ffilt"]))


def decision_curves(runs, dataset, models=("image_only", "counterfactual_aug", "naive"),
                    pts=PTS, n_boot=2000, seed=0):
    """Net benefit of each model on the same pooled test cases (seeds both models share),
    with lesion-level paired bootstrap CIs for the margin over refer-all and for the
    difference counterfactual augmentation minus image only."""
    have = {m: {r["seed"]: r["d"] for r in runs if r["dataset"] == dataset and r["method"] == m}
            for m in models}
    models = [m for m in models if have[m]]
    seeds = sorted(set.intersection(*(set(have[m]) for m in models))) if models else []
    if not seeds:
        return None
    y, groups, probs = [], [], {m: [] for m in models}
    for s in seeds:
        ref = have[models[0]][s]
        for m in models:
            assert list(have[m][s]["test_image_id"]) == list(ref["test_image_id"]), (m, s)
            probs[m].append(_calibrated_honest(have[m][s]))
        y.append(ref["test_label"])
        groups.append(np.char.add(f"{s}:", ref["test_group_id"].astype(str)))
    y = np.concatenate(y).astype(float)
    groups = np.concatenate(groups)
    probs = {m: np.concatenate(v) for m, v in probs.items()}
    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        boots.append(idx)
    res = {"seeds": seeds, "n": int(len(y)), "prevalence": float(y.mean()), "thresholds": {}}
    for pt in pts:
        row = {"refer_all": net_benefit_treat_all(y, pt)}
        for m in models:
            row[m] = net_benefit(y, probs[m], pt)
            d = [net_benefit(y[i], probs[m][i], pt) - net_benefit_treat_all(y[i], pt) for i in boots]
            row[f"{m}_minus_refer_all_ci"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        if {"image_only", "counterfactual_aug"} <= set(models):
            d = [net_benefit(y[i], probs["counterfactual_aug"][i], pt) - net_benefit(y[i], probs["image_only"][i], pt)
                 for i in boots]
            row["cf_minus_image"] = row["counterfactual_aug"] - row["image_only"]
            row["cf_minus_image_ci"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        res["thresholds"][str(pt)] = row
    res["curve"] = {"pt": [float(t) for t in np.round(np.arange(0.01, 0.61, 0.01), 2)]}
    res["curve"]["refer_all"] = [net_benefit_treat_all(y, t) for t in res["curve"]["pt"]]
    for m in models:
        res["curve"][m] = [net_benefit(y, probs[m], t) for t in res["curve"]["pt"]]
    return res


def _figures(per_run, dca, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ds, res in dca.items():
        if not res:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.6))
        c = res["curve"]
        ax.plot(c["pt"], c["refer_all"], color="0.5", ls="--", label="Refer all")
        ax.axhline(0, color="0.7", lw=0.8, label="Refer none")
        for m in ("image_only", "counterfactual_aug", "naive"):
            if m in c:
                ax.plot(c["pt"], c[m], label=LABEL.get(m, m))
        top = max(max(c[m]) for m in c if m != "pt")
        ax.set_ylim(-0.05, top * 1.1 + 0.01)
        ax.set_xlabel("Referral threshold"); ax.set_ylabel("Net benefit"); ax.set_title(ds)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / f"dca_{ds}.png", dpi=200); plt.close(fig)
    for ds in sorted({k[0] for k in per_run}):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for m, label in ABLATION:
            shifts = [v["shift"] for (d2, m2, _), v in per_run.items() if d2 == ds and m2 == m]
            if shifts and m != "image_only":
                ax.hist(np.concatenate(shifts), bins=np.linspace(0, 1, 51), histtype="step", label=label)
        ax.set_yscale("log"); ax.set_xlabel("Lesion-level shift |p(concordant) - p(flipped)|")
        ax.set_ylabel("Lesions (all seeds)"); ax.set_title(ds); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / f"srs_distribution_{ds}.png", dpi=200); plt.close(fig)


def predictions(pred_dir, out_dir=OUT, n_boot=2000):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(pred_dir)
    per_run = {}
    for r in runs:
        m = run_metrics(r["d"], n_boot=min(n_boot, 1000))
        m["shift"] = rv.lesion_shift(r["d"]["test_logit"], r["d"]["test_logit_flip"])
        per_run[(r["dataset"], r["method"], r["seed"])] = m
    datasets = sorted({k[0] for k in per_run})
    dca = {ds: decision_curves(runs, ds, n_boot=n_boot) for ds in datasets}

    md = ["# Review analyses from the per-lesion predictions\n",
          f"{len(runs)} trained models from `{pred_dir}`. Mean ± 95% CI over seeds.\n"]
    summary = {}
    for ds in datasets:
        methods = [m for m, _ in ABLATION if any(k[:2] == (ds, m) for k in per_run)]
        get = lambda m, f: [f(v) for k, v in per_run.items() if k[:2] == (ds, m)]
        seeds = {m: sorted(k[2] for k in per_run if k[:2] == (ds, m)) for m in methods}
        md += [f"\n## {ds}\n", "Seeds: " + "; ".join(f"{m} {seeds[m]}" for m in methods) + "\n",
               "### Ablation (Reviewer 2, point 4)\n",
               "| model | AUROC, original notes | AUROC, facts-only notes | SRS |", "|---|---|---|---|"]
        for m in methods:
            md.append(f"| {LABEL[m]} | {_cell(get(m, lambda v: v['visible']['auroc']))} | "
                      f"{_cell(get(m, lambda v: v['honest']['auroc']))} | "
                      f"{_cell(get(m, lambda v: v['srs']['prob']))} |")
        md += ["\n### Operating point on facts-only notes (Reviewer 1, minor 4; Reviewer 2, point 5)\n",
               "Threshold fixed on the validation set at 95% sensitivity, then applied to the test set. "
               "'Read on test' is the specificity at the test set's own 95%-sensitivity threshold "
               "(the earlier method).\n",
               "| model | test sensitivity | test specificity | specificity, read on test | AP |",
               "|---|---|---|---|---|"]
        for m in methods:
            md.append(f"| {LABEL[m]} | {_cell(get(m, lambda v: v['honest']['sens_valfixed']))} | "
                      f"{_cell(get(m, lambda v: v['honest']['spec_valfixed']))} | "
                      f"{_cell(get(m, lambda v: v['honest']['spec95_test_read']))} | "
                      f"{_cell(get(m, lambda v: v['honest']['ap']))} |")
        md += ["\n### SRS on other scales (Reviewer 2, point 2)\n",
               "prob = the paper's SRS; isotonic = prob after isotonic calibration fitted on validation; "
               "logit = mean |logit difference|; rank = mean difference in score percentile "
               "(unchanged by any monotone recalibration).\n",
               "| model | prob | prob, isotonic | logit | rank | median shift | 99th pct | share > 0.1 |",
               "|---|---|---|---|---|---|---|---|"]
        for m in methods:
            md.append(f"| {LABEL[m]} | {_cell(get(m, lambda v: v['srs']['prob']), 4)} | "
                      f"{_cell(get(m, lambda v: v['srs']['prob_isotonic']), 4)} | "
                      f"{_cell(get(m, lambda v: v['srs']['logit']), 3)} | "
                      f"{_cell(get(m, lambda v: v['srs']['rank']), 4)} | "
                      f"{_cell(get(m, lambda v: v['srs']['median']), 4)} | "
                      f"{_cell(get(m, lambda v: v['srs']['q99']), 4)} | "
                      f"{_cell(get(m, lambda v: v['srs']['share_over_0.1']), 4)} |")
        res = dca[ds]
        if res:
            md += ["\n### Decision curves on the same test cases (Reviewer 1, point 3)\n",
                   f"Seeds {res['seeds']}, {res['n']} pooled test cases, prevalence used for refer-all "
                   f"{res['prevalence']:.3f}. Isotonic calibration fitted on each seed's validation set "
                   "(facts-only notes) and applied to its test set. CIs: lesion-level paired bootstrap.\n",
                   "| threshold | refer all | " + " | ".join(LABEL[m] for m in res["thresholds"][str(PTS[0])]
                                                      if m in LABEL) + " | cf minus image [95% CI] |",
                   "|---|---|" + "---|" * sum(m in LABEL for m in res["thresholds"][str(PTS[0])]) + "---|"]
            for pt in PTS:
                row = res["thresholds"][str(pt)]
                cells = [f"{row[m]:.3f} [{row[f'{m}_minus_refer_all_ci'][0]:+.3f}, "
                         f"{row[f'{m}_minus_refer_all_ci'][1]:+.3f}]" for m in row if m in LABEL]
                diff = (f"{row['cf_minus_image']:+.4f} [{row['cf_minus_image_ci'][0]:+.4f}, "
                        f"{row['cf_minus_image_ci'][1]:+.4f}]") if "cf_minus_image" in row else "n/a"
                md.append(f"| {pt} | {row['refer_all']:.3f} | " + " | ".join(cells) + f" | {diff} |")
            md.append("\nModel cells: net benefit [95% CI of its margin over refer-all].")
        summary[ds] = {"per_run": {f"{m}_seed{s}": {k: v for k, v in val.items() if k != "shift"}
                                   for (d2, m, s), val in per_run.items() if d2 == ds},
                       "decision_curves": res}
    json.dump(summary, open(out_dir / "prediction_analyses.json", "w"), indent=2)
    (out_dir / "prediction_analyses.md").write_text("\n".join(md) + "\n")
    _figures(per_run, dca, out_dir / "figures")
    print("\n".join(md))
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="part", required=True)
    sub.add_parser("logged")
    pr = sub.add_parser("predictions")
    pr.add_argument("--preds", required=True, help="folder with the .npz files")
    pr.add_argument("--out", default=str(OUT))
    pr.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if args.part == "logged":
        logged()
    else:
        predictions(args.preds, args.out, args.bootstrap)


if __name__ == "__main__":
    main()
