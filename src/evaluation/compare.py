"""Aggregate experiment result JSONs into thesis-ready comparison tables.

The runner (scripts/run_experiments.py) writes one JSON per
dataset+backbone(+text filter level):

    results/experiment_results_{dataset}_{backbone}[_{level}].json

Each file holds a list of per-seed records:
    {"mode", "seed", "backbone", "best_val_auroc",
     "test_metrics": {"auroc", "ap", "specificity_at_95_sens", ...}}

This module reads them all, recovers dataset/backbone/level from the filename,
and reports mean +/- 95% CI (Student-t) per (dataset, backbone, mode, level) -
matching the reporting protocol of Watson et al. (2026). Pure pandas/numpy/scipy
so it runs anywhere and is unit-tested offline with synthetic JSONs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("ham10000", "isic2020")
LEVELS = ("Orig", "CFilt", "DFilt", "FFilt")
METRICS = ("auroc", "ap", "specificity_at_95_sens")
PREFIX = "experiment_results_"

# Stable display order: vision -> +metadata -> +text -> +metadata+text,
# and within a mode, the leading-language levels from least to most filtered.
_MODE_ORDER = {"vision": 0, "vision_metadata": 1, "vision_text": 2, "vision_metadata_text": 3}
_LEVEL_ORDER = {None: -1, **{lvl: i for i, lvl in enumerate(LEVELS)}}


def parse_result_filename(name):
    """Recover (dataset, backbone, level) from an experiment_results_* filename.

    Backbone names contain underscores (e.g. convnext_base), so we anchor on the
    known dataset prefix and the optional trailing filter level instead of a
    naive split. Returns (dataset, backbone, level|None) or None if not a match.
    """
    if not name.startswith(PREFIX) or not name.endswith(".json"):
        return None
    stem = name[len(PREFIX):-len(".json")]
    tokens = stem.split("_")
    if not tokens or tokens[0] not in DATASETS:
        return None
    dataset = tokens[0]
    rest = tokens[1:]
    level = None
    if rest and rest[-1] in LEVELS:
        level = rest[-1]
        rest = rest[:-1]
    backbone = "_".join(rest) if rest else ""
    return dataset, backbone, level


def load_results(results_dir="results"):
    """Load every experiment_results_*.json into a tidy per-seed DataFrame."""
    rows = []
    for path in sorted(Path(results_dir).glob(f"{PREFIX}*.json")):
        parsed = parse_result_filename(path.name)
        if parsed is None:
            continue
        dataset, backbone, level = parsed
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            tm = rec.get("test_metrics", {})
            rows.append({
                "dataset": dataset,
                "backbone": rec.get("backbone", backbone),
                "level": level,
                "mode": rec.get("mode"),
                "seed": rec.get("seed"),
                **{m: tm.get(m) for m in METRICS},
            })
    return pd.DataFrame(rows)


def _mean_ci(values, confidence=0.95):
    """Mean and half-width of the Student-t CI (matches the runner's compute_ci)."""
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    if n < 2:
        return mean, 0.0
    from scipy import stats

    se = stats.sem(vals)
    ci = float(se * stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, ci


def aggregate(df, confidence=0.95):
    """Mean +/- 95% CI per (dataset, backbone, mode, level), sorted for display."""
    if df.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "backbone", "mode", "level"]
    out = []
    for keys, g in df.groupby([df[c] if c != "level" else df[c].fillna("-") for c in group_cols], sort=False):
        dataset, backbone, mode, level = keys
        level = None if level == "-" else level
        row = {"dataset": dataset, "backbone": backbone, "mode": mode,
               "level": level, "n_seeds": int(g["seed"].nunique())}
        for m in METRICS:
            mean, ci = _mean_ci(g[m].tolist(), confidence)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci"] = ci
        out.append(row)
    agg = pd.DataFrame(out)
    agg["_m"] = agg["mode"].map(lambda x: _MODE_ORDER.get(x, 99))
    agg["_l"] = agg["level"].map(lambda x: _LEVEL_ORDER.get(_none_if_nan(x), 99))
    agg = agg.sort_values(["dataset", "backbone", "_m", "_l"]).drop(columns=["_m", "_l"])
    # Note: pandas represents the "no filter level" case as NaN in the `level`
    # column (non-text modes). Use _none_if_nan / pd.isna at the boundary.
    return agg.reset_index(drop=True)


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and v != v) else v


def _fmt(mean, ci):
    if mean != mean:  # NaN
        return "-"
    return f"{mean:.4f} +/- {ci:.4f}"


def format_markdown(agg, title="Experiment comparison"):
    """Render the aggregated table as GitHub-flavoured markdown."""
    if agg.empty:
        return f"### {title}\n\n_No results found._\n"
    lines = [f"### {title}", "",
             "| Dataset | Backbone | Mode | Level | Seeds | AUROC | AP | Spec@95%Sens |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['backbone']} | {r['mode']} | {_none_if_nan(r['level']) or '-'} "
            f"| {r['n_seeds']} | {_fmt(r['auroc_mean'], r['auroc_ci'])} "
            f"| {_fmt(r['ap_mean'], r['ap_ci'])} "
            f"| {_fmt(r['specificity_at_95_sens_mean'], r['specificity_at_95_sens_ci'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results into a comparison table.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default=None, help="Optional path to write markdown (also writes a .csv alongside).")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    agg = aggregate(df)
    md = format_markdown(agg)
    print(md)

    if args.out:
        out = Path(args.out)
        out.write_text(md)
        agg.to_csv(out.with_suffix(".csv"), index=False)
        print(f"Wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
