"""Shortcut-learning testbed sweeps.

  --mode dose     : vary leakage_p (dose-response of shortcut reliance), method=naive
  --mode bakeoff  : at p=1.0, compare mitigation methods
                    {naive, ffilt_only, counterfactual_aug, consistency, adversarial}

Reuses the existing multimodal vision+text pipeline; writes results/shortcut_*.json
with AUROC / AP / Spec@95 / honest-AUROC(FFilt) / FFilt-gap / SRS / flip-rate.
"""
import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy import stats

from src.data.dataset import (prepare_ham10000, prepare_isic2020, prepare_isic_dicm_17k,
                              get_transforms, compute_class_weights,
                              MultimodalTextDataset)
from src.data.text_datasets import PairedTextDataset, build_counterfactual_df
from src.models.multimodal import MultimodalClassifier
from src.models.text import DEFAULT_TEXT_MODEL, build_tokenizer
from src.preprocessing.synthetic_text import add_text_columns, FFILT
from src.training.trainer import train, evaluate
from src.training.debias import AdversarialTextHead
from src.training.debias_trainer import consistency_train_step, adversarial_train_step
from src.evaluation.shortcut import srs_from_loader, ffilt_gap_from_auroc
from src.evaluation.attribution import ig_text_fraction

SEEDS = [42, 123, 456, 789, 1024]
PREP = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020,
        "isic_dicm_17k": prepare_isic_dicm_17k}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def mean_ci(v):
    v = np.asarray(v, float); m = float(v.mean())
    return (m, 0.0) if len(v) < 2 else (m, float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)))


def _loader(df, img_dirs, tok, split, cfg, shuffle=None):
    ds = MultimodalTextDataset(df, img_dirs, tok, get_transforms(split, cfg["data"]["img_size"]),
                               text_col="text", max_length=cfg["data"]["text_max_length"])
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], pin_memory=True)


def _paired_loader(df, img_dirs, tok, split, cfg, shuffle):
    ds = PairedTextDataset(df, img_dirs, tok, get_transforms(split, cfg["data"]["img_size"]),
                           max_length=cfg["data"]["text_max_length"])
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["num_workers"], pin_memory=True)


def _build_model(cfg, device):
    return MultimodalClassifier(
        backbone=cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"],
        use_metadata=False, use_text=True,
        text_model_name=cfg["model"].get("text_model", DEFAULT_TEXT_MODEL),
        text_freeze=cfg["model"].get("text_freeze", False)).to(device)


def run_seed(cfg, seed, leakage_p, method, device):
    set_seed(seed)
    dataset = cfg["data"]["dataset"]
    prep = PREP[dataset]
    tr, va, te, img_dirs, _ = prep(data_dir=cfg["data"]["data_dir"], seed=seed)
    level = cfg["data"]["text_filter_level"]
    tok = build_tokenizer(cfg["model"].get("text_model", DEFAULT_TEXT_MODEL))

    # Pick the text engine + the honest FFilt probe engine.
    #  - data.text_source: llm: LLM-generated realistic notes with identical
    #    leakage semantics, but the FFilt probe stays template facts-only (no API
    #    cost).
    #  - unset -> template, so existing configs behave exactly as before.
    text_source = cfg["data"].get("text_source", "template")
    if text_source == "llm":
        from src.preprocessing.llm_text import add_llm_text_columns
        _prov = cfg["data"].get("llm_provider", "anthropic")
        _lmodel = cfg["data"].get("llm_model", "")

        def mk(df, level, seed, leakage_p=1.0, pair=False):
            return add_llm_text_columns(df, level=level, seed=seed, leakage_p=leakage_p,
                                        pair=pair, provider=_prov, model=_lmodel)
        ffilt_mk = add_text_columns
    else:
        mk = add_text_columns
        ffilt_mk = add_text_columns

    if method == "ffilt_only":
        tr_t = mk(tr, level=FFILT, seed=seed, leakage_p=leakage_p)
    else:
        tr_t = mk(tr, level=level, seed=seed, leakage_p=leakage_p, pair=True)
    va_t = mk(va, level=level, seed=seed, leakage_p=leakage_p)
    te_t = mk(te, level=level, seed=seed, leakage_p=leakage_p, pair=True)
    te_ffilt = ffilt_mk(te, level=FFILT, seed=seed)

    cw = compute_class_weights(tr_t)
    pos_w = float(cw[1] / cw[0])
    model = _build_model(cfg, device)
    _ts_tag = "_llm" if text_source == "llm" else ""
    save_dir = Path(f"models/checkpoints/shortcut_{cfg['data']['dataset']}{_ts_tag}_{method}_p{leakage_p}_seed{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if method in ("consistency", "adversarial"):
        paired_train = _paired_loader(tr_t, img_dirs, tok, "train", cfg, shuffle=True)
        params = list(model.parameters())
        adv_head = None
        if method == "adversarial":
            adv_head = AdversarialTextHead(model.text_encoder.feature_dim).to(device)
            params += list(adv_head.parameters())
        opt = torch.optim.AdamW(params, lr=cfg["training"]["lr"],
                                weight_decay=cfg["training"]["weight_decay"])
        lam_c = cfg["shortcut"].get("consistency_lambda", 1.0)
        lam_a = cfg["shortcut"].get("adversarial_lambda", 1.0)
        for _ in range(cfg["training"]["epochs"]):
            for batch in paired_train:
                if method == "consistency":
                    consistency_train_step(model, batch, opt, device, pos_w, lam_c)
                else:
                    adversarial_train_step(model, adv_head, batch, opt, device, pos_w, lam_a)
        torch.save({"model_state_dict": model.state_dict()}, save_dir / "best_model.pt")
    else:
        if method == "counterfactual_aug":
            tr_t = build_counterfactual_df(tr_t)
        train_loader = _loader(tr_t, img_dirs, tok, "train", cfg)
        val_loader = _loader(va_t, img_dirs, tok, "val", cfg)
        tcfg = {**cfg["training"], "pos_weight": pos_w, "save_dir": str(save_dir)}
        train(model, train_loader, val_loader, tcfg, device, "vision_text")

    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))
    m_leaky = evaluate(model, _loader(te_t, img_dirs, tok, "test", cfg, shuffle=False),
                       crit, device, "vision_text")
    m_ffilt = evaluate(model, _loader(te_ffilt, img_dirs, tok, "test", cfg, shuffle=False),
                       crit, device, "vision_text")
    srs = srs_from_loader(model, _paired_loader(te_t, img_dirs, tok, "test", cfg, shuffle=False), device)

    # IG modality attribution on one test batch (best-effort; None if captum errors).
    try:
        ig_batch = next(iter(_loader(te_t, img_dirs, tok, "test", cfg, shuffle=False)))
        text_frac = ig_text_fraction(model, ig_batch, device)
    except Exception as exc:
        print(f"  IG attribution skipped: {type(exc).__name__}: {exc}")
        text_frac = None

    # Metrics are extracted; the ~2GB checkpoint is throwaway. Drop it so a long
    # sweep does not accumulate hundreds of GB and fill the disk.
    shutil.rmtree(save_dir, ignore_errors=True)

    return {"seed": seed, "leakage_p": leakage_p, "method": method,
            "auroc": m_leaky["auroc"], "ap": m_leaky["ap"],
            "spec95": m_leaky["specificity_at_95_sens"],
            "auroc_ffilt": m_ffilt["auroc"],
            "ffilt_gap": ffilt_gap_from_auroc(m_leaky["auroc"], m_ffilt["auroc"]),
            "srs": srs["srs"], "flip_rate": srs["flip_rate"], "text_frac": text_frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["dose", "bakeoff"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dataset {cfg['data']['dataset']} | mode {args.mode}")

    if args.mode == "dose":
        jobs = [(p, "naive") for p in cfg["shortcut"]["leakage_ps"]]
    else:
        jobs = [(1.0, m) for m in cfg["shortcut"]["methods"]]

    results = []
    for p, mth in jobs:
        rows = [run_seed(cfg, s, p, mth, device) for s in args.seeds]
        results.extend(rows)
        au = mean_ci([r["auroc"] for r in rows]); srs = mean_ci([r["srs"] for r in rows])
        gap = mean_ci([r["ffilt_gap"] for r in rows]); hon = mean_ci([r["auroc_ffilt"] for r in rows])
        print(f"[{args.mode}] p={p} method={mth} | AUROC {au[0]:.4f}+/-{au[1]:.4f} "
              f"honestAUROC {hon[0]:.4f} SRS {srs[0]:.4f}+/-{srs[1]:.4f} "
              f"FFiltGap {gap[0]:.4f}+/-{gap[1]:.4f}")

    Path("results").mkdir(exist_ok=True)
    _out_tag = "_llm" if cfg["data"].get("text_source") == "llm" else ""
    out = Path("results") / f"shortcut_{cfg['data']['dataset']}{_out_tag}_{args.mode}.json"
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
