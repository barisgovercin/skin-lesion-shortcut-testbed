"""Metadata-only diagnostic: signal ceiling + raw vs engineered encoding.

CPU-only, no images/torch needed. For each seed it builds the SAME leakage-free
split as the training pipeline, fits metadata-only LogisticRegression and
HistGradientBoosting on raw vs engineered encodings, and reports AUROC / AP /
Spec@95%Sens (threshold tuned on val, applied to test) as mean +/- 95% CI.
Prints feature importances and writes results/metadata_diagnostic_{dataset}.json.

Run:
    py -3.11 scripts/diagnose_metadata.py --dataset ham10000
    py -3.11 scripts/diagnose_metadata.py --dataset isic2020 --patient-aggregates
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.inspection import permutation_importance

from src.data.dataset import prepare_ham10000, prepare_isic2020

SEEDS = [42, 123, 456, 789, 1024]
PREPARE = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020}


def mean_ci(vals, confidence=0.95):
    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, 0.0
    se = stats.sem(vals)
    return mean, float(se * stats.t.ppf((1 + confidence) / 2, len(vals) - 1))


def spec_at_95_sens(y_val, p_val, y_test, p_test):
    """Threshold tuned for >=95% sensitivity on val; specificity measured on test."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    idx = np.where(tpr >= 0.95)[0]
    threshold = thr[idx[0]] if len(idx) else 0.0
    pred = (p_test >= threshold).astype(int)
    neg = int((y_test == 0).sum())
    if neg == 0:
        return 0.0
    tn = int(((pred == 0) & (y_test == 0)).sum())
    return tn / neg


def make_models():
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "histgb": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=0
        ),
    }


def evaluate_encoding(dataset, engineered, patient_agg):
    prep = PREPARE[dataset]
    metric_names = ["auroc", "ap", "spec95"]
    rows = {m: {k: [] for k in metric_names} for m in make_models()}
    importances = {}
    for seed in SEEDS:
        kwargs = dict(seed=seed, include_metadata=True, engineered=engineered)
        if dataset == "isic2020":
            kwargs["patient_aggregates"] = patient_agg
        train_df, val_df, test_df, _, cols = prep(**kwargs)
        Xtr, ytr = train_df[cols].to_numpy(float), train_df["label"].to_numpy(int)
        Xva, yva = val_df[cols].to_numpy(float), val_df["label"].to_numpy(int)
        Xte, yte = test_df[cols].to_numpy(float), test_df["label"].to_numpy(int)
        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            pva = model.predict_proba(Xva)[:, 1]
            pte = model.predict_proba(Xte)[:, 1]
            rows[name]["auroc"].append(roc_auc_score(yte, pte))
            rows[name]["ap"].append(average_precision_score(yte, pte))
            rows[name]["spec95"].append(spec_at_95_sens(yva, pva, yte, pte))
            if seed == SEEDS[0] and name == "histgb":
                r = permutation_importance(
                    model, Xte, yte, n_repeats=10, random_state=0, scoring="roc_auc"
                )
                order = np.argsort(r.importances_mean)[::-1]
                importances = {cols[i]: float(r.importances_mean[i]) for i in order}
    summary = {
        m: {k: mean_ci(v) for k, v in metrics.items()} for m, metrics in rows.items()
    }
    return summary, importances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PREPARE), default="ham10000")
    ap.add_argument("--patient-aggregates", action="store_true")
    args = ap.parse_args()

    out = {}
    for enc in ("raw", "engineered"):
        summary, imp = evaluate_encoding(
            args.dataset,
            engineered=(enc == "engineered"),
            patient_agg=args.patient_aggregates,
        )
        out[enc] = {"metrics": summary, "importances": imp}
        print(f"\n===== {args.dataset} | {enc} encoding =====")
        for model, metrics in summary.items():
            line = "  ".join(
                f"{k}={mc[0]:.4f}+/-{mc[1]:.4f}" for k, mc in metrics.items()
            )
            print(f"  [{model}] {line}")
        if imp:
            print("  top features (histgb permutation importance):")
            for f, v in list(imp.items())[:10]:
                print(f"    {v:+.4f}  {f}")

    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"metadata_diagnostic_{args.dataset}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
