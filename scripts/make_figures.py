"""Generate publication figures from existing checkpoints (no retraining).

Sub-commands (``--what``):
  vision-curves : ROC/PR + reliability(calibration) for the headline vision-only
                  models. Per-sample scores are recomputed from the saved
                  checkpoints and cached to results/predictions_*.json.
  ig            : token-level Integrated-Gradients heatmap, naive vs
                  counterfactual_aug, on a leaking DFilt test note (the referral
                  phrase is the shortcut the naive model latches onto).
  text-frac     : per-method text-attribution fraction bar, from the bake-off JSON.
  all           : run all of the above for HAM10000.

Examples:
  python scripts/make_figures.py --what vision-curves --dataset ham10000 --backbone convnext_base
  python scripts/make_figures.py --what ig
  python scripts/make_figures.py --what all
"""
import sys
import json
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.dataset import (prepare_ham10000, prepare_isic2020, prepare_isic_dicm_17k,
                              get_transforms, SkinLesionDataset, MultimodalTextDataset)
from src.models.vision import VisionClassifier
from src.models.multimodal import MultimodalClassifier
from src.models.text import DEFAULT_TEXT_MODEL, build_tokenizer
from src.preprocessing.synthetic_text import add_text_columns
from src.training.trainer import _run_batch
from src.evaluation.plots import plot_roc_pr, plot_calibration, plot_text_frac_bar, _mean_ci
from src.evaluation.attribution import ig_token_attributions, render_token_heatmap

SEEDS = [42, 123, 456, 789, 1024]
PREP = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020,
        "isic_dicm_17k": prepare_isic_dicm_17k}
DATA_DIR = {"ham10000": "data/raw/ham10000", "isic2020": "data/raw/isic2020",
            "isic_dicm_17k": "data/raw/isic_dicm_17k"}
FIG_DIR = Path("results/figures")


def _set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1b/1c: ROC/PR + calibration for the vision-only headline models ---------
@torch.no_grad()
def dump_vision_predictions(dataset, backbone, seeds, img_size=224, batch_size=64):
    device = _device()
    prep = PREP[dataset]
    out = {"dataset": dataset, "mode": "vision", "backbone": backbone, "seeds": {}}
    for seed in seeds:
        ckpt = Path(f"models/checkpoints/{dataset}_vision_{backbone}_seed{seed}/best_model.pt")
        if not ckpt.exists():
            print(f"  skip seed {seed}: no checkpoint ({ckpt})")
            continue
        _set_seed(seed)
        _, _, test_df, img_dirs, _ = prep(data_dir=DATA_DIR[dataset], seed=seed,
                                          include_metadata=False, engineered=False)
        ds = SkinLesionDataset(test_df, img_dirs, get_transforms("test", img_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        model = VisionClassifier(backbone=backbone, pretrained=False).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        ys, yt = [], []
        for batch in loader:
            outputs, labels = _run_batch(model, batch, "vision", device)
            ys.extend(torch.sigmoid(outputs).float().cpu().numpy().tolist())
            yt.extend(labels.float().cpu().numpy().tolist())
        out["seeds"][str(seed)] = {"y_true": yt, "y_score": ys}
        print(f"  seed {seed}: n={len(yt)} pos={int(np.sum(yt))} "
              f"mean_score={np.mean(ys):.3f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def vision_curves(dataset, backbone):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = dump_vision_predictions(dataset, backbone, SEEDS)
    if not preds["seeds"]:
        print(f"No checkpoints for {dataset}/{backbone}; skipping curves.")
        return
    cache = Path(f"results/predictions_{dataset}_vision_{backbone}.json")
    json.dump(preds, open(cache, "w"))
    plot_roc_pr(cache, FIG_DIR / f"roc_pr_{dataset}_vision_{backbone}.png",
                title=f"{dataset} vision-only ({backbone})")
    plot_calibration(cache, FIG_DIR / f"calibration_{dataset}_vision_{backbone}.png",
                     title=f"{dataset} vision-only ({backbone})")
    print(f"Saved ROC/PR + calibration for {dataset}/{backbone} -> {FIG_DIR}")


# --- 1a: token-level IG heatmap, naive vs counterfactual_aug -----------------
def _load_mm_model(ckpt_path, cfg, device):
    model = MultimodalClassifier(
        backbone=cfg["model"]["backbone"], pretrained=False,
        use_metadata=False, use_text=True,
        text_model_name=cfg["model"].get("text_model", DEFAULT_TEXT_MODEL),
        text_pretrained=False, text_freeze=cfg["model"].get("text_freeze", False)).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model


def ig_heatmap(cfg_path="configs/shortcut_ham10000.yaml", seed=42, n_steps=32):
    cfg = yaml.safe_load(open(cfg_path))
    device = _device()
    ds_name = cfg["data"]["dataset"]
    level = cfg["data"]["text_filter_level"]
    _set_seed(seed)
    _, _, te, img_dirs, _ = PREP[ds_name](data_dir=cfg["data"]["data_dir"], seed=seed)
    te_t = add_text_columns(te, level=level, seed=seed, leakage_p=1.0, pair=True)

    mal = te_t[te_t["label"] == 1].reset_index(drop=True)
    if len(mal) == 0:
        print("No malignant test example found; aborting IG."); return
    example = mal.iloc[[0]].reset_index(drop=True)
    note = str(example.iloc[0]["text"])
    tok = build_tokenizer(cfg["model"].get("text_model", DEFAULT_TEXT_MODEL))
    ds = MultimodalTextDataset(example, img_dirs, tok, get_transforms("test", cfg["data"]["img_size"]),
                               text_col="text", max_length=cfg["data"]["text_max_length"])
    image, input_ids, attn, _ = ds[0]
    image = image.unsqueeze(0); input_ids = input_ids.unsqueeze(0); attn = attn.unsqueeze(0)
    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())

    ckpts = {
        "naive (no mitigation)":
            f"models/checkpoints/shortcut_{ds_name}_naive_p1.0_seed{seed}/best_model.pt",
        "counterfactual augmentation (fixed)":
            f"models/checkpoints/shortcut_{ds_name}_counterfactual_aug_p1.0_seed{seed}/best_model.pt",
    }
    scores = {}
    for label_name, cp in ckpts.items():
        if not Path(cp).exists():
            print(f"  missing checkpoint {cp}; skipping"); continue
        model = _load_mm_model(cp, cfg, device)
        scores[label_name] = ig_token_attributions(model, image, input_ids, attn, device, n_steps)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not scores:
        print("No IG scores computed (checkpoints missing)."); return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"ig_tokens_{ds_name}.png"
    render_token_heatmap(
        tokens, attn[0].tolist(), scores, out,
        title=(f"Integrated-Gradients token attribution on a leaking note "
               f"({level}, p=1.0). Darker = higher attribution."))
    print(f"Saved IG heatmap -> {out}")
    print(f"Example note: {note}")


def text_frac_bar(dataset="ham10000"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bakeoff = Path(f"results/shortcut_{dataset}_bakeoff.json")
    if not bakeoff.exists():
        print(f"No bake-off JSON at {bakeoff}; skipping text-frac bar."); return
    out = FIG_DIR / f"text_frac_bar_{dataset}.png"
    plot_text_frac_bar(bakeoff, out,
                       title=f"Text reliance by mitigation ({dataset}, IG modality attribution)")
    print(f"Saved text-frac bar -> {out}")


# --- Cross-dataset transfer matrix (train-on-row, test-on-column) ------------
@torch.no_grad()
def _predict_vision(model, df, img_dirs, device, img_size=224, batch_size=64):
    ds = SkinLesionDataset(df, img_dirs, get_transforms("test", img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    ys, yt = [], []
    for batch in loader:
        outputs, labels = _run_batch(model, batch, "vision", device)
        ys.extend(torch.sigmoid(outputs).float().cpu().numpy().tolist())
        yt.extend(labels.float().cpu().numpy().tolist())
    return np.array(yt), np.array(ys)


def _plot_transfer_matrix(matrix, datasets, out_png, backbone):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(datasets)
    M = np.array([[matrix[s][t]["mean"] for t in datasets] for s in datasets])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(datasets)
    ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
    for i in range(n):
        for j in range(n):
            cell = matrix[datasets[i]][datasets[j]]
            ax.text(j, i, f"{cell['mean']:.3f}\n$\\pm$ {cell['ci']:.3f}", ha="center", va="center",
                    color="white" if cell["mean"] < 0.80 else "black", fontsize=10)
    ax.set_title(f"Cross-dataset transfer AUROC (vision-only, {backbone})\n"
                 f"diagonal = in-domain test; off-diagonal = fully external")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def transfer_matrix(backbone="convnext_base",
                    datasets=("ham10000", "isic2020", "isic_dicm_17k")):
    """Train-on-row, test-on-column AUROC. Each cell averages, over the 5 seeds,
    the src-seed model evaluated on the tgt-seed test split. Diagonal reproduces
    the in-domain numbers; off-diagonal is fully external transfer.

    Label semantics differ (HAM malignant = mel+bcc+akiec; ISIC 2020 = melanoma;
    ISIC-DICM-17K = melanoma vs non-melanoma), so transfer measures how well the
    malignancy/melanoma ranking carries across datasets."""
    device = _device()
    datasets = list(datasets)
    tgt_cache = {}

    def tgt_test(tgt, seed):
        key = (tgt, seed)
        if key not in tgt_cache:
            _set_seed(seed)
            _, _, test_df, img_dirs, _ = PREP[tgt](data_dir=DATA_DIR[tgt], seed=seed,
                                                   include_metadata=False, engineered=False)
            tgt_cache[key] = (test_df, img_dirs)
        return tgt_cache[key]

    matrix = {}
    for src in datasets:
        matrix[src] = {}
        per_tgt = {t: [] for t in datasets}
        for seed in SEEDS:
            ckpt = Path(f"models/checkpoints/{src}_vision_{backbone}_seed{seed}/best_model.pt")
            if not ckpt.exists():
                print(f"  missing {ckpt}; skip"); continue
            model = VisionClassifier(backbone=backbone, pretrained=False).to(device)
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state_dict"]); model.eval()
            for tgt in datasets:
                test_df, img_dirs = tgt_test(tgt, seed)
                yt, ys = _predict_vision(model, test_df, img_dirs, device)
                per_tgt[tgt].append(float(roc_auc_score(yt, ys)))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for tgt in datasets:
            vals = per_tgt[tgt]
            m, ci = _mean_ci(vals) if vals else (float("nan"), 0.0)
            matrix[src][tgt] = {"mean": m, "ci": ci, "per_seed": vals}
            tag = "in-domain" if src == tgt else "TRANSFER"
            print(f"  train {src:>14} -> test {tgt:>14} [{tag:9}]: AUROC {m:.3f} +/- {ci:.3f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(f"results/experiment_results_transfer_matrix_{backbone}.json")
    json.dump(matrix, open(out, "w"), indent=2)
    _plot_transfer_matrix(matrix, datasets, FIG_DIR / f"transfer_matrix_{backbone}.png", backbone)
    print(f"Saved transfer matrix -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["vision-curves", "ig", "text-frac", "transfer", "all"],
                    default="all")
    ap.add_argument("--dataset", default="ham10000")
    ap.add_argument("--backbone", default="convnext_base")
    args = ap.parse_args()

    print(f"Device: {_device()}")
    if args.what in ("vision-curves", "all"):
        vision_curves(args.dataset, args.backbone)
    if args.what in ("ig", "all"):
        ig_heatmap()
    if args.what in ("text-frac", "all"):
        text_frac_bar(args.dataset)
    if args.what == "transfer":
        transfer_matrix(args.backbone)


if __name__ == "__main__":
    main()
