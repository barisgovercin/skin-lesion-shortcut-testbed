"""Regenerate the manuscript's shortcut figures (Figs 2 to 4) from results/.

Reads the shortcut result JSONs (5 seeds each) and writes vector PDF and 300 dpi
PNG files named as in the manuscript. No titles (the captions carry them); points
are means with 95% t-intervals over seeds.

    python scripts/make_paper_figures.py            # -> results/figures/paper/
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DPI = 300

HAM_C, ISIC_C, DICM_C = "tab:blue", "tab:orange", "tab:green"
TMPL_C, LLM_C = "0.55", "tab:purple"
METHODS = ["naive", "ffilt_only", "counterfactual_aug", "consistency", "adversarial"]
MLABELS = ["naive", "ffilt\nonly", "counterfactual\naug", "consistency", "adversarial"]


def _agg(rows, key, by):
    """Mean and 95% CI (t-based) of `key`, grouped by `by`."""
    d = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            d[r[by]].append(float(r[key]))
    xs = sorted(d)
    means = [float(np.mean(d[x])) for x in xs]
    cis = [float(stats.sem(d[x]) * stats.t.ppf(0.975, len(d[x]) - 1)) if len(d[x]) > 1 else 0.0
           for x in xs]
    return xs, means, cis


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf/.png")


def dose_fig(series, out_dir, name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lbl, col, rows in series:
        xs, am, ac = _agg(rows, "auroc_ffilt", "leakage_p")
        _, sm, sc = _agg(rows, "srs", "leakage_p")
        a1.errorbar(xs, am, yerr=ac, marker="o", color=col, capsize=3, label=lbl)
        a2.errorbar(xs, sm, yerr=sc, marker="o", color=col, capsize=3, label=lbl)
    a1.set_xlabel("leakage strength $p$")
    a1.set_ylabel("honest AUROC (FFilt test)")
    a2.set_xlabel("leakage strength $p$")
    a2.set_ylabel("SRS (shortcut reliance)")
    a2.set_ylim(0, 1.02)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(loc="best")
    fig.tight_layout()
    _save(fig, out_dir, name)


def bakeoff_fig(groups, out_dir, name):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    x = np.arange(len(METHODS))
    w = 0.8 / len(groups)
    for i, (lbl, col, rows) in enumerate(groups):
        honest = {m: (mu, c) for m, mu, c in zip(*_agg(rows, "auroc_ffilt", "method"))}
        srs = {m: (mu, c) for m, mu, c in zip(*_agg(rows, "srs", "method"))}
        off = (i - (len(groups) - 1) / 2) * w
        a1.bar(x + off, [honest[m][0] for m in METHODS], w,
               yerr=[honest[m][1] for m in METHODS], capsize=3, color=col, label=lbl)
        a2.bar(x + off, [srs[m][0] for m in METHODS], w,
               yerr=[srs[m][1] for m in METHODS], capsize=3, color=col, label=lbl)
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels(MLABELS, fontsize=8.5)
        a.grid(axis="y", alpha=0.3)
        a.legend(loc="best")
    a1.set_ylabel("honest AUROC (FFilt)")
    a1.set_ylim(0.6, 1.0)
    a2.set_ylabel("SRS (lower is better)")
    a2.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, out_dir, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "paper"))
    args = ap.parse_args(argv)
    res, out = Path(args.results_dir), Path(args.out)

    def load(name):
        return json.loads((res / f"{name}.json").read_text())

    ham_d, ham_b = load("shortcut_ham10000_dose"), load("shortcut_ham10000_bakeoff")
    isic_d, isic_b = load("shortcut_isic2020_dose"), load("shortcut_isic2020_bakeoff")
    dicm_d, dicm_b = load("shortcut_isic_dicm_17k_dose"), load("shortcut_isic_dicm_17k_bakeoff")
    llm_d, llm_b = load("shortcut_ham10000_llm_dose"), load("shortcut_ham10000_llm_bakeoff")

    three = [("HAM10000", HAM_C), ("ISIC 2020", ISIC_C), ("ISIC-DICM-17K", DICM_C)]
    dose_fig([(l, c, r) for (l, c), r in zip(three, (ham_d, isic_d, dicm_d))], out,
             "Fig2shortcut_ham_vs_isic_dose")
    bakeoff_fig([(l, c, r) for (l, c), r in zip(three, (ham_b, isic_b, dicm_b))], out,
                "Fig3shortcut_ham_vs_isic_bakeoff")
    dose_fig([("template text", TMPL_C, ham_d), ("LLM-realistic text", LLM_C, llm_d)], out,
             "Fig4a_shortcut_ham10000_llm_vs_template_dose")
    bakeoff_fig([("template text", TMPL_C, ham_b), ("LLM-realistic text", LLM_C, llm_b)], out,
                "Fig4b_shortcut_ham10000_llm_vs_template_bakeoff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
