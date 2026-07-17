"""Clinical-utility evaluation: probability calibration + decision-curve analysis.

Runs on the saved vision-only checkpoints (no retraining). For each dataset and
seed it extracts validation and test logits (cached to disk so re-runs are
instant), fits three calibrators on the validation split (temperature, Platt,
isotonic), applies them to test, and reports ECE for each plus a decision curve
on the best-calibrated (isotonic) probabilities. Writes per-dataset figures +
JSON and a combined markdown summary.

  python scripts/clinical_eval.py
  python scripts/clinical_eval.py --datasets ham10000 isic2020 isic_dicm_17k --backbone convnext_base
"""
import sys
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy import stats

from src.data.dataset import (prepare_ham10000, prepare_isic2020, prepare_isic_dicm_17k,
                              get_transforms, SkinLesionDataset)
from src.models.vision import VisionClassifier
from src.training.trainer import _run_batch
from src.evaluation.clinical import (fit_temperature, apply_temperature, ece, decision_curve,
                                     fit_platt, apply_platt, fit_isotonic, apply_isotonic,
                                     _sigmoid)

SEEDS = [42, 123, 456, 789, 1024]
PREP = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020,
        "isic_dicm_17k": prepare_isic_dicm_17k}
DATA_DIR = {"ham10000": "data/raw/ham10000", "isic2020": "data/raw/isic2020",
            "isic_dicm_17k": "data/raw/isic_dicm_17k"}
FIG = Path("results/figures")


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mci(v):
    v = np.asarray(v, float); m = float(v.mean())
    return (m, 0.0) if len(v) < 2 else (m, float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)))


@torch.no_grad()
def _logits_labels(model, df, img_dirs, device, img_size=224, bs=64):
    ds = SkinLesionDataset(df, img_dirs, get_transforms("test", img_size))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    logits, labels = [], []
    for batch in loader:
        out, y = _run_batch(model, batch, "vision", device)   # raw logits (pre-sigmoid)
        logits.extend(out.float().cpu().numpy().tolist())
        labels.extend(y.float().cpu().numpy().tolist())
    return np.array(logits), np.array(labels)


def _get_logits(dataset, backbone, device):
    """Per-seed validation/test logits+labels, cached to disk."""
    cache = Path(f"results/clinical_logits_{dataset}_{backbone}.json")
    if cache.exists():
        raw = json.load(open(cache))
        return {int(k): {kk: np.array(vv, float) for kk, vv in v.items()} for k, v in raw.items()}
    prep = PREP[dataset]
    out = {}
    for seed in SEEDS:
        ck = Path(f"models/checkpoints/{dataset}_vision_{backbone}_seed{seed}/best_model.pt")
        if not ck.exists():
            print(f"  skip {dataset} seed{seed}: no checkpoint")
            continue
        _set_seed(seed)
        _, val_df, test_df, img_dirs, _ = prep(data_dir=DATA_DIR[dataset], seed=seed,
                                               include_metadata=False, engineered=False)
        model = VisionClassifier(backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model_state_dict"])
        model.eval()
        vL, vY = _logits_labels(model, val_df, img_dirs, device)
        tL, tY = _logits_labels(model, test_df, img_dirs, device)
        out[seed] = {"vL": vL, "vY": vY, "tL": tL, "tY": tY}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {dataset} seed{seed}: n_val={len(vY)} n_test={len(tY)}")
    if out:
        json.dump({str(k): {kk: vv.tolist() for kk, vv in v.items()} for k, v in out.items()},
                  open(cache, "w"))
    return out


def eval_dataset(dataset, backbone, device):
    logits = _get_logits(dataset, backbone, device)
    if not logits:
        return None
    methods = ["uncalibrated", "temperature", "platt", "isotonic"]
    per_seed = {m: [] for m in methods}
    Ts = []
    pooled_y, pooled_uncal, pooled_temp, pooled_platt, pooled_iso = [], [], [], [], []
    for seed, d in sorted(logits.items()):
        vL, vY, tL, tY = d["vL"], d["vY"], d["tL"], d["tY"]
        T = fit_temperature(vL, vY)
        a, b = fit_platt(vL, vY)
        iso = fit_isotonic(_sigmoid(vL), vY)
        p_un = _sigmoid(tL)
        p_t = apply_temperature(tL, T)
        p_p = apply_platt(tL, a, b)
        p_i = apply_isotonic(iso, _sigmoid(tL))
        per_seed["uncalibrated"].append(ece(tY, p_un))
        per_seed["temperature"].append(ece(tY, p_t))
        per_seed["platt"].append(ece(tY, p_p))
        per_seed["isotonic"].append(ece(tY, p_i))
        Ts.append(T)
        pooled_y.append(tY); pooled_uncal.append(p_un); pooled_temp.append(p_t)
        pooled_platt.append(p_p); pooled_iso.append(p_i)
        print(f"  {dataset} seed{seed}: T={T:.2f}  ECE un {ece(tY, p_un):.3f} / temp "
              f"{ece(tY, p_t):.3f} / platt {ece(tY, p_p):.3f} / iso {ece(tY, p_i):.3f}")
    return {"dataset": dataset, "per_seed": per_seed, "T": Ts,
            "pooled": {"y": np.concatenate(pooled_y),
                       "uncal": np.concatenate(pooled_uncal),
                       "temp": np.concatenate(pooled_temp),
                       "platt": np.concatenate(pooled_platt),
                       "iso": np.concatenate(pooled_iso)}}


def _reliability(ax, y, p, color, label, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    cx, cy = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            cx.append(p[m].mean()); cy.append(y[m].mean())
    ax.plot(cx, cy, marker="o", color=color, label=label)


def make_figure(res, out_png):
    d = res["dataset"]
    Y = res["pooled"]["y"]
    UN, TEMP, PLATT, ISO = (res["pooled"]["uncal"], res["pooled"]["temp"],
                            res["pooled"]["platt"], res["pooled"]["iso"])
    eu = _mci(res["per_seed"]["uncalibrated"])
    et = _mci(res["per_seed"]["temperature"])
    ep = _mci(res["per_seed"]["platt"])
    ei = _mci(res["per_seed"]["isotonic"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    _reliability(a1, Y, UN, "tab:red", f"uncalibrated (ECE {eu[0]:.3f})")
    _reliability(a1, Y, TEMP, "tab:orange", f"temperature (ECE {et[0]:.3f})")
    _reliability(a1, Y, PLATT, "#1b7837", f"Platt (ECE {ep[0]:.3f})")
    _reliability(a1, Y, ISO, "#66bd63", f"isotonic (ECE {ei[0]:.3f})")
    a1.set_xlabel("Predicted probability of malignancy")
    a1.set_ylabel("Observed malignant frequency")
    a1.set_title(f"Calibration: {d}"); a1.legend(loc="upper left"); a1.grid(alpha=0.3)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)

    th, model_nb, all_nb = decision_curve(Y, ISO)
    a2.plot(th, model_nb, color="tab:blue", label="model (isotonic-calibrated)")
    a2.plot(th, all_nb, ls="--", color="tab:gray", label="treat all")
    a2.axhline(0, color="black", lw=1, label="treat none")
    a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit")
    a2.set_title(f"Decision curve: {d}"); a2.legend(loc="upper right"); a2.grid(alpha=0.3)
    a2.set_xlim(0, 0.6); a2.set_ylim(min(-0.02, float(np.min(model_nb))), max(0.05, float(np.max(model_nb)) * 1.1))

    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ham10000", "isic2020", "isic_dicm_17k"])
    ap.add_argument("--backbone", default="convnext_base")
    args = ap.parse_args()
    device = _dev(); print(f"Device: {device}")
    FIG.mkdir(parents=True, exist_ok=True)

    md = ["# Clinical evaluation: calibration + decision-curve analysis\n",
          "Vision-only model. Three calibrators are fit on the validation split and applied to "
          "test (leak-free); ECE is mean +/- 95% CI over 5 seeds. Temperature scaling corrects "
          "only sharpness; Platt and isotonic can also correct a systematic bias. The decision "
          "curve uses pooled isotonic-calibrated test predictions.\n",
          "| dataset | ECE uncalibrated | ECE temperature | ECE Platt | ECE isotonic | mean T |",
          "|---|---|---|---|---|---|"]
    for ds in args.datasets:
        res = eval_dataset(ds, args.backbone, device)
        if res is None:
            print(f"no checkpoints for {ds}")
            continue
        cells = {m: _mci(res["per_seed"][m]) for m in ["uncalibrated", "temperature", "platt", "isotonic"]}
        T = _mci(res["T"])
        md.append(f"| {ds} | {cells['uncalibrated'][0]:.3f} +/- {cells['uncalibrated'][1]:.3f} | "
                  f"{cells['temperature'][0]:.3f} +/- {cells['temperature'][1]:.3f} | "
                  f"{cells['platt'][0]:.3f} +/- {cells['platt'][1]:.3f} | "
                  f"{cells['isotonic'][0]:.3f} +/- {cells['isotonic'][1]:.3f} | {T[0]:.2f} |")
        make_figure(res, FIG / f"calibration_dca_{ds}.png")
        json.dump({"dataset": ds, "per_seed": res["per_seed"], "T": res["T"],
                   "ece": {m: cells[m] for m in cells}},
                  open(f"results/clinical_eval_{ds}.json", "w"), indent=2)
        print(f"Saved figure + json for {ds}")

    out = Path("results/reports/clinical_eval.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
