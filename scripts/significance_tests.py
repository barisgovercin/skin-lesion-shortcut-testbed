"""
Statistical-significance analysis for the shortcut-learning testbed.

Adds paired significance testing on top of the existing 5-seed 95% CIs, as
requested for the manuscript. Two analyses:

1. Mitigation bake-off (p=1.0): pairwise PAIRED tests (same 5 seeds) comparing
   the best method (counterfactual_aug) against every other method, for both
   the honest image-based AUROC (`auroc_ffilt`) and the Shortcut Reliance Score
   (`srs`). Reports mean difference, 95% CI, paired t-test p, Wilcoxon
   signed-rank p, and paired effect size (Cohen's d_z). Run for HAM and ISIC.

2. Dose-response: trend test (Spearman rho) between leakage_p and honest AUROC
   / SRS, per seed then averaged, plus a pooled test.

Note: with n=5 seeds the two-sided Wilcoxon signed-rank p cannot go below
0.0625 (= 2 / 2^5); the paired t-test is the primary test and Wilcoxon is
reported as a non-parametric sanity check.

Usage:
    python scripts/significance_tests.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_MD = RESULTS / "reports" / "significance_tests.md"

BEST = "counterfactual_aug"
OTHERS = ["consistency", "ffilt_only", "adversarial", "naive"]
METRICS = {"auroc_ffilt": "Honest AUROC (higher better)", "srs": "SRS (lower better)"}


def load(path: Path) -> list[dict]:
    return json.load(open(path))


def by_seed(records: list[dict], method: str, metric: str) -> np.ndarray:
    """Return metric values ordered by seed for one method (paired alignment)."""
    rows = sorted((r for r in records if r["method"] == method), key=lambda r: r["seed"])
    return np.array([r[metric] for r in rows], dtype=float)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison a - b (a = best method, b = comparison method)."""
    diff = a - b
    n = len(diff)
    mean_d = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    t_p = stats.ttest_rel(a, b).pvalue if sd > 0 else (0.0 if mean_d != 0 else 1.0)
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:  # all-zero differences
        w_p = 1.0
    dz = mean_d / sd if sd > 0 else float("inf") * np.sign(mean_d)
    return {"mean_diff": mean_d, "ci": ci, "t_p": float(t_p), "w_p": float(w_p), "dz": float(dz)}


def bakeoff_block(name: str, path: Path, lines: list[str]) -> None:
    recs = load(path)
    lines.append(f"\n### {name} bake-off (p=1.0, n=5 seeds)\n")
    lines.append(f"Paired comparison: **{BEST}** vs each other method.\n")
    for metric, desc in METRICS.items():
        lines.append(f"\n**{desc}**: mean {BEST} = {by_seed(recs, BEST, metric).mean():.4f}\n")
        lines.append("| vs method | mean Δ (best−other) | 95% CI | paired t p | Wilcoxon p | Cohen's d_z |")
        lines.append("|---|---|---|---|---|---|")
        best_vals = by_seed(recs, BEST, metric)
        for other in OTHERS:
            ov = by_seed(recs, other, metric)
            if len(ov) != len(best_vals):
                continue
            r = paired_test(best_vals, ov)
            sig = "**" if r["t_p"] < 0.05 else ""
            lines.append(
                f"| {other} | {sig}{r['mean_diff']:+.4f}{sig} | "
                f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['t_p']:.4f} | "
                f"{r['w_p']:.4f} | {r['dz']:+.2f} |"
            )


def dose_block(name: str, path: Path, lines: list[str]) -> None:
    recs = load(path)
    seeds = sorted(set(r["seed"] for r in recs))
    lines.append(f"\n### {name} dose-response trend (Spearman rho: leakage_p vs metric)\n")
    lines.append("| metric | mean rho (per-seed) | pooled rho | pooled p |")
    lines.append("|---|---|---|---|")
    for metric, desc in METRICS.items():
        per_seed_rho = []
        for s in seeds:
            rows = sorted((r for r in recs if r["seed"] == s), key=lambda r: r["leakage_p"])
            ps = [r["leakage_p"] for r in rows]
            ys = [r[metric] for r in rows]
            per_seed_rho.append(stats.spearmanr(ps, ys).statistic)
        all_p = [r["leakage_p"] for r in recs]
        all_y = [r[metric] for r in recs]
        pooled = stats.spearmanr(all_p, all_y)
        lines.append(
            f"| {desc} | {np.mean(per_seed_rho):+.3f} | {pooled.statistic:+.3f} | {pooled.pvalue:.2e} |"
        )


def main() -> None:
    lines: list[str] = ["# Significance tests: shortcut-learning testbed\n"]
    lines.append(
        "Paired tests over the 5 shared seeds (42, 123, 456, 789, 1024). "
        "Primary test: paired t-test. Wilcoxon signed-rank is a non-parametric "
        "sanity check whose two-sided p floors at 0.0625 for n=5. "
        "Δ>0 for Honest AUROC and Δ<0 for SRS both favour counterfactual_aug.\n"
    )
    datasets = [
        ("HAM10000", "shortcut_ham10000"),
        ("ISIC 2020", "shortcut_isic2020"),
        ("ISIC-DICM-17K", "shortcut_isic_dicm_17k"),
        ("HAM10000 (LLM-realistic text)", "shortcut_ham10000_llm"),
    ]
    lines.append("## Mitigation bake-off")
    for name, stem in datasets:
        path = RESULTS / f"{stem}_bakeoff.json"
        if path.exists():
            bakeoff_block(name, path, lines)
    lines.append("\n## Dose-response")
    for name, stem in datasets:
        path = RESULTS / f"{stem}_dose.json"
        if path.exists():
            dose_block(name, path, lines)

    text = "\n".join(lines) + "\n"
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[WROTE] {OUT_MD}")


if __name__ == "__main__":
    main()
