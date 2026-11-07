"""Reproduce the manuscript's tables and check every reported number against results/.

No GPU, no data download: everything is recomputed from the per-seed result JSONs
in results/ (written by the training runs). Each number quoted in the manuscript is
listed in CLAIMS with its location and printed value, recomputed, and compared at
the printed precision.

    python scripts/reproduce_tables.py

Prints the regenerated tables and a pass/fail line per claim, writes
results/reports/manuscript_tables.md and results/reports/manuscript_check.md, and
exits with code 1 if any claim does not match.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.significance_tests import by_seed, paired_test  # noqa: E402
from src.evaluation.compare import _mean_ci  # noqa: E402

DATASETS = {"ham10000": "HAM10000", "isic2020": "ISIC 2020", "isic_dicm_17k": "ISIC-DICM-17K"}
METHODS = ["naive", "ffilt_only", "counterfactual_aug", "consistency", "adversarial"]
OTHERS = ["naive", "ffilt_only", "consistency", "adversarial"]
NB_THRESHOLDS = ["0.05", "0.1", "0.2", "0.3"]


# ---- Recompute ----------------------------------------------------------------------

def _load(results_dir, name):
    return json.loads((Path(results_dir) / name).read_text())


def _shortcut_facts(prefix, dose, bake, f):
    d, b = pd.DataFrame(dose), pd.DataFrame(bake)
    g = d.groupby("leakage_p")[["auroc", "auroc_ffilt", "ffilt_gap", "srs"]].mean()
    for p, r in g.iterrows():
        f[f"{prefix}.dose.honest.{p:.1f}"] = float(r.auroc_ffilt)
        f[f"{prefix}.dose.visible.{p:.1f}"] = float(r.auroc)
        f[f"{prefix}.dose.gap.{p:.1f}"] = float(r.ffilt_gap)
        f[f"{prefix}.dose.srs.{p:.1f}"] = float(r.srs)
    f[f"{prefix}.dose.honest_before_collapse"] = [float(g.loc[p, "auroc_ffilt"]) for p in g.index if p < 1.0]
    for metric, col in (("honest", "auroc_ffilt"), ("srs", "srs")):
        s = stats.spearmanr(d["leakage_p"], d[col])
        f[f"{prefix}.dose.spearman_rho.{metric}"] = float(s.statistic)
        f[f"{prefix}.dose.spearman_p.{metric}"] = float(s.pvalue)

    means = b.groupby("method")[["auroc_ffilt", "srs", "flip_rate"]].mean()
    for m, r in means.iterrows():
        f[f"{prefix}.bake.{m}.honest"] = float(r.auroc_ffilt)
        f[f"{prefix}.bake.{m}.srs"] = float(r.srs)
        f[f"{prefix}.bake.{m}.flip"] = float(r.flip_rate)
    others = [m for m in means.index if m != "counterfactual_aug"]
    f[f"{prefix}.best.honest"] = (float(means.loc["counterfactual_aug", "auroc_ffilt"]),
                                  [float(means.loc[m, "auroc_ffilt"]) for m in others])
    f[f"{prefix}.best.srs"] = (float(means.loc["counterfactual_aug", "srs"]),
                               [float(means.loc[m, "srs"]) for m in others])
    f[f"{prefix}.consistency_vs_cf.srs"] = (float(means.loc["consistency", "srs"]),
                                            [float(means.loc["counterfactual_aug", "srs"])])
    f[f"{prefix}.ranking.honest"] = {m: float(means.loc[m, "auroc_ffilt"]) for m in means.index}
    for metric, col in (("honest", "auroc_ffilt"), ("srs", "srs")):
        for other in OTHERS:
            t = paired_test(by_seed(bake, "counterfactual_aug", col), by_seed(bake, other, col))
            key = f"{prefix}.paired.{metric}.{other}"
            f[f"{key}.delta"], f[f"{key}.p"], f[f"{key}.dz"] = t["mean_diff"], t["t_p"], t["dz"]
            f[f"{key}.ci_lo"], f[f"{key}.ci_hi"] = (float(x) for x in t["ci"])


def compute(results_dir):
    """Every quantity the manuscript reports, recomputed from the result JSONs."""
    f = {}
    for ds in DATASETS:
        runs = _load(results_dir, f"experiment_results_{ds}_convnext_base.json")
        for mode in ("vision", "vision_metadata"):
            f[f"{mode}.{ds}"] = _mean_ci([r["test_metrics"]["auroc"] for r in runs
                                          if r["mode"] == mode])
        dose = _load(results_dir, f"shortcut_{ds}_dose.json")
        _shortcut_facts(ds, dose, _load(results_dir, f"shortcut_{ds}_bakeoff.json"), f)
    # Vision + synthetic text: HAM10000 from the Orig-level text run, ISIC sets from
    # the naive model at full leakage (p = 1.0).
    orig = _load(results_dir, "experiment_results_ham10000_convnext_base_Orig.json")
    f["text_leak.ham10000"] = float(np.mean([r["test_metrics"]["auroc"] for r in orig]))
    for ds in ("isic2020", "isic_dicm_17k"):
        f[f"text_leak.{ds}"] = f[f"{ds}.dose.visible.1.0"]

    _shortcut_facts("llm", _load(results_dir, "shortcut_ham10000_llm_dose.json"),
                    _load(results_dir, "shortcut_ham10000_llm_bakeoff.json"), f)

    # Training-budget control (HAM10000): filtering with every example twice per epoch.
    ctrl = _load(results_dir, "training_budget_control/summary.json")["rows"]
    bake = _load(results_dir, "shortcut_ham10000_bakeoff.json")
    for metric, col in (("honest", "auroc_ffilt"), ("srs", "srs")):
        f[f"budget.ffilt_only_2x.{metric}"] = float(by_seed(ctrl, "ffilt_only_2x", col).mean())
        t = paired_test(by_seed(bake, "counterfactual_aug", col), by_seed(ctrl, "ffilt_only_2x", col))
        key = f"budget.paired.{metric}"
        f[f"{key}.delta"], f[f"{key}.p"] = t["mean_diff"], t["t_p"]
        f[f"{key}.ci_lo"], f[f"{key}.ci_hi"] = (float(x) for x in t["ci"])

    clinical = _load(results_dir, "clinical_stats.json")
    for ds in DATASETS:
        nb = clinical[ds]["net_benefit"]
        for pt in NB_THRESHOLDS:
            f[f"nb.{ds}.{pt}.model"] = float(nb[pt]["model_pt"])
            f[f"nb.{ds}.{pt}.all"] = float(nb[pt]["treat_all_pt"])
            f[f"nb.{ds}.{pt}.diff_lo"] = float(nb[pt]["diff_ci"][0])
        f[f"nb.{ds}.diff_lo_min"] = min(f[f"nb.{ds}.{pt}.diff_lo"] for pt in NB_THRESHOLDS)
        f[f"nb.{ds}.model_min"] = min(f[f"nb.{ds}.{pt}.model"] for pt in NB_THRESHOLDS)

    return f


# ---- Compare ------------------------------------------------------------------------

def _fmt(value, printed):
    digits = len(printed.split(".")[1]) if "." in printed else 0
    return f"{value:+.{digits}f}" if printed.startswith("+") else f"{value:.{digits}f}"


def check(rule, printed, value):
    """True if `value` agrees with the manuscript's `printed` text under `rule`."""
    if rule == "round":
        return _fmt(value, printed) == printed
    if rule == "pm":
        mean, ci = printed.split("±")
        return _fmt(value[0], mean) == mean and _fmt(value[1], ci) == ci
    if rule == "lt":
        return bool(value < float(printed.lstrip("<")))
    if rule == "gt":
        return bool(value > float(printed.lstrip(">")))
    if rule == "significant":
        return bool(value < 0.05)
    if rule == "not_significant":
        return bool(value >= 0.05)
    if rule == "range":
        lo, hi = (float(x) for x in printed.split("-"))
        return all(lo <= v <= hi for v in value)
    if rule in ("lowest", "highest"):
        digits = int(printed)
        x, others = round(value[0], digits), [round(o, digits) for o in value[1]]
        return x <= min(others) if rule == "lowest" else x >= max(others)
    if rule == "order":
        names = re.split(r">=|>", printed)
        ops = re.findall(r">=|>", printed)
        for a, op, b in zip(names, ops, names[1:]):
            va, vb = value[a], value[b]
            if not (round(va, 3) >= round(vb, 3) if op == ">=" else va > vb):
                return False
        return True
    raise ValueError(f"unknown rule {rule}")


def run_checks(facts, claims):
    out = []
    for c in claims:
        value = facts[c["key"]]
        out.append({**c, "value": value, "ok": check(c["rule"], c["printed"], value)})
    return out


# ---- What the manuscript says ---------------------------------------------------------

def _c(where, rule, printed, key):
    return {"where": where, "rule": rule, "printed": printed, "key": key}


def _claims():
    c = []
    name = DATASETS
    # Table 1 (replication) and the matching sentences in Results and the abstract.
    for ds, v, vm in (("ham10000", "0.906±0.008", "0.907±0.008"),
                      ("isic2020", "0.982±0.005", "0.983±0.005"),
                      ("isic_dicm_17k", "0.863±0.005", "0.862±0.008")):
        c.append(_c(f"Table 1 · Vision only · {name[ds]}", "pm", v, f"vision.{ds}"))
        c.append(_c(f"Table 1 · Vision + metadata · {name[ds]}", "pm", vm, f"vision_metadata.{ds}"))
        c.append(_c(f"Table 1 · Vision + synthetic text · {name[ds]}", "round", "1.000",
                    f"text_leak.{ds}"))
    # Dose-response (Results text, Figure 2 caption, Table 3).
    for ds, lo, hi, gap, srs in (("ham10000", "0.917", "0.711", "0.289", "0.998"),
                                 ("isic2020", "0.985", "0.897", "0.103", "0.999"),
                                 ("isic_dicm_17k", "0.879", "0.679", "0.321", "0.999")):
        c.append(_c(f"Dose-response · honest AUROC at p=0.5 · {name[ds]}", "round", lo,
                    f"{ds}.dose.honest.0.5"))
        c.append(_c(f"Dose-response · honest AUROC at p=1.0 · {name[ds]}", "round", hi,
                    f"{ds}.dose.honest.1.0"))
        c.append(_c(f"Dose-response · FFilt-gap at p=1.0 · {name[ds]}", "round", gap,
                    f"{ds}.dose.gap.1.0"))
        c.append(_c(f"Table 3 · SRS at p=1.0 · {name[ds]}", "round", srs, f"{ds}.dose.srs.1.0"))
        for metric in ("honest", "srs"):
            c.append(_c(f"Dose-response · Spearman trend ({metric}) p<0.001 · {name[ds]}",
                        "lt", "<0.001", f"{ds}.dose.spearman_p.{metric}"))
    # Table 2 (bake-off at p = 1.0).
    table2 = {"ham10000": {"naive": ("0.711", "0.998"), "ffilt_only": ("0.918", "0.007"),
                           "counterfactual_aug": ("0.928", "0.004"),
                           "consistency": ("0.924", "0.002"), "adversarial": ("0.898", "0.102")},
              "isic2020": {"naive": ("0.897", "0.999"), "ffilt_only": ("0.985", "0.002"),
                           "counterfactual_aug": ("0.987", "0.001"),
                           "consistency": ("0.986", "0.001"), "adversarial": ("0.971", "0.124")},
              "isic_dicm_17k": {"naive": ("0.679", "0.999"), "ffilt_only": ("0.880", "0.006"),
                                "counterfactual_aug": ("0.889", "0.003"),
                                "consistency": ("0.887", "0.003"),
                                "adversarial": ("0.784", "0.679")}}
    for ds, rows in table2.items():
        for m, (honest, srs) in rows.items():
            c.append(_c(f"Table 2 · {m} honest · {name[ds]}", "round", honest, f"{ds}.bake.{m}.honest"))
            c.append(_c(f"Table 2 · {m} SRS · {name[ds]}", "round", srs, f"{ds}.bake.{m}.srs"))
        c.append(_c(f"Table 2 caption · counterfactual_aug has the highest honest AUROC · {name[ds]}",
                    "highest", "3", f"{ds}.best.honest"))
        c.append(_c(f"Table 2 caption · counterfactual_aug SRS is near zero (<0.01) · {name[ds]}",
                    "lt", "<0.01", f"{ds}.bake.counterfactual_aug.srs"))
        c.append(_c(f"Table 2 caption · consistency SRS similar or slightly lower than cf · {name[ds]}",
                    "lowest", "3", f"{ds}.consistency_vs_cf.srs"))
        c.append(_c(f"Results · method ranking on honest AUROC · {name[ds]}", "order",
                    "counterfactual_aug>=consistency>ffilt_only>adversarial>naive",
                    f"{ds}.ranking.honest"))
    # Bake-off text: Flip Rate.
    for ds in DATASETS:
        c.append(_c(f"Results · naive Flip Rate · {name[ds]}", "round", "1.000", f"{ds}.bake.naive.flip"))
    for ds, fr in (("ham10000", "0.004"), ("isic2020", "0.001"), ("isic_dicm_17k", "0.004")):
        c.append(_c(f"Results · counterfactual_aug Flip Rate · {name[ds]}", "round", fr,
                    f"{ds}.bake.counterfactual_aug.flip"))
    c.append(_c("Results · adversarial Flip Rate · ISIC-DICM-17K", "round", "0.876",
                "isic_dicm_17k.bake.adversarial.flip"))
    c.append(_c("Results · adversarial SRS ~0.68 · ISIC-DICM-17K", "round", "0.68",
                "isic_dicm_17k.bake.adversarial.srs"))
    # Bake-off text: paired tests, counterfactual_aug vs filtering.
    for ds, p, d, lo, hi, dz in (("ham10000", "0.006", "+0.010", "+0.005", "+0.015", "2.4"),
                                 ("isic2020", "0.001", "+0.002", "+0.002", "+0.003", "4.1"),
                                 ("isic_dicm_17k", "0.037", "+0.009", "+0.001", "+0.017", "1.4")):
        k = f"{ds}.paired.honest.ffilt_only"
        c += [_c(f"Results · cf vs filtering, p · {name[ds]}", "round", p, f"{k}.p"),
              _c(f"Results · cf vs filtering, mean gain · {name[ds]}", "round", d, f"{k}.delta"),
              _c(f"Results · cf vs filtering, 95% CI low · {name[ds]}", "round", lo, f"{k}.ci_lo"),
              _c(f"Results · cf vs filtering, 95% CI high · {name[ds]}", "round", hi, f"{k}.ci_hi"),
              _c(f"Results · cf vs filtering, Cohen's d · {name[ds]}", "round", dz, f"{k}.dz"),
              _c(f"Contribution 4 · cf beats filtering on SRS (p<0.05) · {name[ds]}",
                 "significant", "p<0.05", f"{ds}.paired.srs.ffilt_only.p")]
    for ds, rule, p, gain in (("ham10000", "lt", "<0.001", "+0.216"),
                              ("isic2020", "round", "0.007", "+0.091"),
                              ("isic_dicm_17k", "lt", "<0.001", "+0.209")):
        k = f"{ds}.paired.honest.naive"
        c += [_c(f"Results · cf vs naive p · {name[ds]}", rule, p, f"{k}.p"),
              _c(f"Results · cf vs naive, mean gain · {name[ds]}", "round", gain, f"{k}.delta"),
              _c(f"Results · cf vs naive, d>2 · {name[ds]}", "gt", ">2", f"{k}.dz")]
    for ds, p in (("isic2020", "0.18"), ("isic_dicm_17k", "0.43")):
        k = f"{ds}.paired.honest.consistency"
        c += [_c(f"Results · consistency indistinguishable from cf · {name[ds]}",
                 "not_significant", "p>=0.05", f"{k}.p"),
              _c(f"Results · cf vs consistency, p · {name[ds]}", "round", p, f"{k}.p")]
    c += [_c("Results · cf vs consistency, p · HAM10000", "round", "0.04",
             "ham10000.paired.honest.consistency.p"),
          _c("Results · cf vs consistency, mean difference · HAM10000", "round", "0.003",
             "ham10000.paired.honest.consistency.delta")]
    # Training-budget control (Results, HAM10000).
    c += [_c("Control · budget-matched filtering honest AUROC", "round", "0.926",
             "budget.ffilt_only_2x.honest"),
          _c("Control · counterfactual_aug honest AUROC", "round", "0.928",
             "ham10000.bake.counterfactual_aug.honest"),
          _c("Control · cf minus budget-matched filtering, honest AUROC", "round", "+0.002",
             "budget.paired.honest.delta"),
          _c("Control · 95% CI low", "round", "-0.002", "budget.paired.honest.ci_lo"),
          _c("Control · 95% CI high", "round", "+0.005", "budget.paired.honest.ci_hi"),
          _c("Control · p", "round", "0.30", "budget.paired.honest.p"),
          _c("Control · cf matches budget-matched filtering (not significant)", "not_significant",
             "p>=0.05", "budget.paired.honest.p"),
          _c("Control · counterfactual_aug SRS", "round", "0.004",
             "ham10000.bake.counterfactual_aug.srs"),
          _c("Control · budget-matched filtering SRS", "round", "0.006", "budget.ffilt_only_2x.srs"),
          _c("Control · cf keeps a lower SRS", "lt", "<0", "budget.paired.srs.delta"),
          _c("Control · lower SRS, p", "round", "0.015", "budget.paired.srs.p"),
          _c("Control · lower SRS is significant", "significant", "p<0.05", "budget.paired.srs.p")]
    # LLM-generated notes (HAM10000).
    c += [_c("LLM · honest AUROC holds near 0.91 before collapsing", "range", "0.90-0.92",
             "llm.dose.honest_before_collapse"),
          _c("LLM · Spearman rho for SRS vs p", "round", "0.99", "llm.dose.spearman_rho.srs"),
          _c("LLM · Spearman p<0.001 for SRS vs p", "lt", "<0.001", "llm.dose.spearman_p.srs"),
          _c("LLM · counterfactual_aug honest AUROC", "round", "0.928", "llm.bake.counterfactual_aug.honest"),
          _c("LLM · counterfactual_aug SRS", "round", "0.006", "llm.bake.counterfactual_aug.srs"),
          _c("LLM · naive honest AUROC", "round", "0.656", "llm.bake.naive.honest"),
          _c("LLM · cf vs naive p<0.001", "lt", "<0.001", "llm.paired.honest.naive.p"),
          _c("LLM · cf vs filtering p", "round", "0.003", "llm.paired.honest.ffilt_only.p"),
          _c("LLM · cf vs adversarial p", "round", "0.019", "llm.paired.honest.adversarial.p"),
          _c("LLM · cf vs consistency p", "round", "0.048", "llm.paired.honest.consistency.p")]
    # Table 4 (net benefit) and the DCA sentences.
    table4 = {"ham10000": {"0.1": ("0.153", "0.106"), "0.3": ("0.101", "-0.149")},
              "isic2020": {"0.1": ("0.118", "0.036"), "0.3": ("0.110", "-0.239")},
              "isic_dicm_17k": {"0.1": ("0.463", "0.453"), "0.3": ("0.379", "0.296")}}
    for ds, rows in table4.items():
        for pt, (model, all_) in rows.items():
            c.append(_c(f"Table 4 · model NB at {pt} · {name[ds]}", "round", model, f"nb.{ds}.{pt}.model"))
            c.append(_c(f"Table 4 · refer-all NB at {pt} · {name[ds]}", "round", all_, f"nb.{ds}.{pt}.all"))
        c.append(_c(f"DCA · margin over refer-all excludes zero at every threshold · {name[ds]}",
                    "gt", ">0", f"nb.{ds}.diff_lo_min"))
        c.append(_c(f"DCA · model net benefit positive at every threshold · {name[ds]}",
                    "gt", ">0", f"nb.{ds}.model_min"))
    for ds in ("ham10000", "isic2020"):
        c.append(_c(f"DCA · refer-all harmful at 0.30 · {name[ds]}", "lt", "<0", f"nb.{ds}.0.3.all"))
    return c


CLAIMS = _claims()


# ---- Tables -----------------------------------------------------------------------------

def tables_markdown(f):
    L = ["# Manuscript tables, regenerated from results/", "",
         "## Table 1. Replication (test AUROC, mean ± 95% CI over 5 seeds)", "",
         "| Configuration | " + " | ".join(DATASETS.values()) + " |", "|---|---|---|---|"]
    for label, key in (("Vision only (ConvNeXt-Base)", "vision"), ("Vision + metadata", "vision_metadata")):
        L.append(f"| {label} | " + " | ".join(f"{f[f'{key}.{d}'][0]:.3f} ± {f[f'{key}.{d}'][1]:.3f}"
                                               for d in DATASETS) + " |")
    L.append("| Vision + synthetic text | " + " | ".join(f"{f[f'text_leak.{d}']:.3f}" for d in DATASETS) + " |")
    L += ["", "## Table 2. Mitigation bake-off at p = 1.0 (honest AUROC / SRS, mean of 5 seeds)", "",
          "| Method | " + " | ".join(DATASETS.values()) + " |", "|---|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{f[f'{d}.bake.{m}.honest']:.3f} / {f[f'{d}.bake.{m}.srs']:.3f}"
                                          for d in DATASETS) + " |")
    L += ["", "## Table 3. Generalisation scorecard", "",
          "| Finding | " + " | ".join(DATASETS.values()) + " |", "|---|---|---|---|",
          "| Vision baseline (AUROC) | " + " | ".join(f"{f[f'vision.{d}'][0]:.3f}" for d in DATASETS) + " |",
          "| Vision + text leaks | " + " | ".join(f"{f[f'text_leak.{d}']:.3f}" for d in DATASETS) + " |",
          "| Honest AUROC at p = 1.0 | " + " | ".join(f"{f[f'{d}.dose.honest.1.0']:.3f}" for d in DATASETS) + " |",
          "| SRS at p = 1.0 | " + " | ".join(f"{f[f'{d}.dose.srs.1.0']:.3f}" for d in DATASETS) + " |",
          "| counterfactual_aug honest / SRS | " + " | ".join(
              f"{f[f'{d}.bake.counterfactual_aug.honest']:.3f} / {f[f'{d}.bake.counterfactual_aug.srs']:.3f}"
              for d in DATASETS) + " |",
          "", "## Table 4. Net benefit of the calibrated image model", "",
          "| Dataset | p_t | Model NB | Refer-all NB | Refer-none NB |", "|---|---|---|---|---|"]
    for d, label in DATASETS.items():
        for pt in ("0.1", "0.3"):
            L.append(f"| {label} | {float(pt):.2f} | {f[f'nb.{d}.{pt}.model']:.3f} | "
                     f"{f[f'nb.{d}.{pt}.all']:.3f} | 0 |")
    L += ["", "## LLM-generated notes (HAM10000), bake-off at p = 1.0", "",
          "| Method | honest AUROC | SRS |", "|---|---|---|"]
    for m in METHODS:
        L.append(f"| {m} | {f[f'llm.bake.{m}.honest']:.3f} | {f[f'llm.bake.{m}.srs']:.3f} |")
    return "\n".join(L) + "\n"


def _show(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list):
        return f"{value[0]:.4f} vs others {', '.join(f'{v:.4f}' for v in value[1])}"
    if isinstance(value, tuple):
        return f"{value[0]:.4f} ± {value[1]:.4f}"
    if isinstance(value, dict):
        return ", ".join(f"{k} {v:.4f}" for k, v in sorted(value.items(), key=lambda kv: -kv[1]))
    if isinstance(value, list):
        return ", ".join(f"{v:.4f}" for v in value)
    return f"{value:.4g}" if abs(value) < 1e-3 and value != 0 else f"{value:.4f}"


def check_markdown(results):
    bad = [r for r in results if not r["ok"]]
    L = ["# Manuscript number check", "",
         f"{len(results) - len(bad)} of {len(results)} reported numbers and statements match "
         "the values recomputed from results/.", ""]
    if bad:
        L += ["## Mismatches", "", "| Where | Manuscript | Recomputed |", "|---|---|---|"]
        L += [f"| {r['where']} | {r['printed']} | {_show(r['value'])} |" for r in bad]
        L.append("")
    L += ["## All checks", "", "| | Where | Manuscript | Recomputed |", "|---|---|---|---|"]
    L += [f"| {'OK' if r['ok'] else 'MISMATCH'} | {r['where']} | {r['printed']} | {_show(r['value'])} |"
          for r in results]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "reports"))
    args = ap.parse_args(argv)

    facts = compute(args.results_dir)
    results = run_checks(facts, CLAIMS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = tables_markdown(facts)
    (out / "manuscript_tables.md").write_text(tables)
    (out / "manuscript_check.md").write_text(check_markdown(results))
    print(tables)
    bad = [r for r in results if not r["ok"]]
    print(f"{len(results) - len(bad)} of {len(results)} manuscript numbers match results/.")
    for r in bad:
        print(f"  MISMATCH  {r['where']}: manuscript {r['printed']}, recomputed {_show(r['value'])}")
    print(f"Wrote {out / 'manuscript_tables.md'} and {out / 'manuscript_check.md'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
