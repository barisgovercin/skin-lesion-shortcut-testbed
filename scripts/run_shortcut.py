"""Shortcut-learning testbed sweeps.

  --mode dose     : vary leakage_p (dose-response of shortcut reliance), method=naive
  --mode bakeoff  : at p=1.0, compare mitigation methods
                    {naive, ffilt_only, counterfactual_aug, consistency, adversarial},
                    plus image_only (the same pipeline with the text branch removed)

--save-preds DIR also writes each run's per-lesion validation and test logits
(results/predictions/*.npz) for the reviewer analyses in scripts/review_analyses.py.

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
from src.evaluation.predictions import collect_logits, collect_paired_logits, save_predictions
from sklearn.metrics import roc_auc_score

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


def _build_model(cfg, device, use_text=True):
    return MultimodalClassifier(
        backbone=cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"],
        use_metadata=False, use_text=use_text,
        text_model_name=cfg["model"].get("text_model", DEFAULT_TEXT_MODEL),
        text_freeze=cfg["model"].get("text_freeze", False)).to(device)


def _template_maker(df, level, seed, leakage_p=1.0, pair=False, part=None):
    """Template notes; `part` is accepted for a common signature and ignored."""
    return add_text_columns(df, level=level, seed=seed, leakage_p=leakage_p, pair=pair)


def _llm_maker(cfg, split):
    """LLM notes. With split=True, part="train" / "test" draws the diagnosis clause only
    from the training or the held-out clauses of the pool (held-out wording test)."""
    from src.preprocessing.llm_text import add_llm_text_columns
    provider = cfg["data"].get("llm_provider", "anthropic")
    llm_model = cfg["data"].get("llm_model", "")

    def mk(df, level, seed, leakage_p=1.0, pair=False, part=None):
        return add_llm_text_columns(df, level=level, seed=seed, leakage_p=leakage_p,
                                    pair=pair, provider=provider, model=llm_model,
                                    pool_part=part if split else None)
    return mk


def _note_makers(cfg):
    """(engine for the configured notes, engine for the honest FFilt probe).

    data.text_source: llm uses LLM-generated notes with identical leakage semantics;
    the FFilt probe stays template facts-only (no API cost). Unset -> template.
    data.llm_pool_split: true keeps the held-out clauses of the LLM pool for the test
    notes only (training and validation notes use the remaining clauses).
    """
    if cfg["data"].get("text_source", "template") != "llm":
        return _template_maker, add_text_columns
    return _llm_maker(cfg, cfg["data"].get("llm_pool_split", False)), add_text_columns


def _text_tag(cfg):
    if cfg["data"].get("text_source") != "llm":
        return ""
    return "_llm_heldout" if cfg["data"].get("llm_pool_split") else "_llm"


def cross_test_notes(cfg, te, seed, leakage_p):
    """Test notes written by the other note engine, with wording the model never saw:
    held-out LLM clauses for a template-trained model, template notes for an
    LLM-trained one. Returns (source name, notes with flip pairs)."""
    level = cfg["data"]["text_filter_level"]
    if cfg["data"].get("text_source", "template") == "llm":
        return "template", _template_maker(te, level, seed, leakage_p, pair=True)
    return "llm_heldout", _llm_maker(cfg, True)(te, level, seed, leakage_p, pair=True, part="test")


def training_notes(cfg, tr, seed, leakage_p, method):
    """Training notes for one mitigation method.

    ffilt_only_2x is the training-budget control: filtered notes with every row
    twice per epoch, i.e. the same number of rows and steps as counterfactual_aug.
    """
    mk, _ = _note_makers(cfg)
    if method in ("ffilt_only", "ffilt_only_2x"):
        tr_t = mk(tr, level=FFILT, seed=seed, leakage_p=leakage_p, part="train")
        if method == "ffilt_only_2x":
            tr_t = tr_t.loc[tr_t.index.repeat(2)].reset_index(drop=True)
        return tr_t
    tr_t = mk(tr, level=cfg["data"]["text_filter_level"], seed=seed, leakage_p=leakage_p, pair=True,
              part="train")
    return build_counterfactual_df(tr_t) if method == "counterfactual_aug" else tr_t


def test_notes(cfg, te, seed, leakage_p):
    """Test notes at the configured level (with flip pairs) and the FFilt probe."""
    mk, ffilt_mk = _note_makers(cfg)
    te_t = mk(te, level=cfg["data"]["text_filter_level"], seed=seed, leakage_p=leakage_p, pair=True,
              part="test")
    return te_t, ffilt_mk(te, level=FFILT, seed=seed)


def evaluate_model(model, cfg, te_t, te_ffilt, img_dirs, tok, device, pos_w):
    """Visible AUROC on the leaky notes, honest AUROC on FFilt notes, SRS and FlipRate."""
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))
    m_leaky = evaluate(model, _loader(te_t, img_dirs, tok, "test", cfg, shuffle=False),
                       crit, device, "vision_text")
    m_ffilt = evaluate(model, _loader(te_ffilt, img_dirs, tok, "test", cfg, shuffle=False),
                       crit, device, "vision_text")
    srs = srs_from_loader(model, _paired_loader(te_t, img_dirs, tok, "test", cfg, shuffle=False), device)
    return {"auroc": m_leaky["auroc"], "ap": m_leaky["ap"],
            "spec95": m_leaky["specificity_at_95_sens"],
            "auroc_ffilt": m_ffilt["auroc"],
            "ffilt_gap": ffilt_gap_from_auroc(m_leaky["auroc"], m_ffilt["auroc"]),
            "srs": srs["srs"], "flip_rate": srs["flip_rate"]}


def _group_ids(df):
    """Lesion (HAM10000, ISIC-DICM-17K) or patient (ISIC 2020) identifier per row."""
    for col in ("lesion_id", "patient_id"):
        if col in df.columns:
            return df[col].astype(str).to_numpy()
    return df["image_id"].astype(str).to_numpy()


def export_predictions(path, model, cfg, va_t, va_ffilt, te_t, te_ffilt, img_dirs, tok,
                       device, meta, metrics, cross=None):
    """Save per-lesion logits on the validation and test notes of one trained model.

    test_logit / test_logit_flip are the concordant and flipped notes of each lesion
    (the SRS pair), test_logit_visible the notes behind the visible AUROC and
    test_logit_ffilt the facts-only notes behind the honest AUROC; val_logit and
    val_logit_ffilt are the same two kinds of note on the validation set, for
    thresholds and calibration fitted on validation only. The saved arrays are
    checked against the metrics the run logged. cross=(source, notes) adds the
    concordant and flipped logits on notes from the other note engine
    (test_logit_cross, test_logit_cross_flip).
    """
    for a, b in ((va_t, va_ffilt), (te_t, te_ffilt)):
        assert list(a["image_id"]) == list(b["image_id"]), "note frames are not row-aligned"
    ld = lambda df, split: _loader(df, img_dirs, tok, split, cfg, shuffle=False)
    v_logit, v_y = collect_logits(model, ld(va_t, "val"), device)
    vf_logit, _ = collect_logits(model, ld(va_ffilt, "val"), device)
    t_vis, t_y = collect_logits(model, ld(te_t, "test"), device)
    tf_logit, _ = collect_logits(model, ld(te_ffilt, "test"), device)
    t_c, t_f, _ = collect_paired_logits(
        model, _paired_loader(te_t, img_dirs, tok, "test", cfg, shuffle=False), device)

    extra = {}
    if cross is not None:
        source, te_x = cross
        assert list(te_x["image_id"]) == list(te_t["image_id"]), "cross notes are not row-aligned"
        x_c, x_f, _ = collect_paired_logits(
            model, _paired_loader(te_x, img_dirs, tok, "test", cfg, shuffle=False), device)
        extra = {"test_logit_cross": x_c, "test_logit_cross_flip": x_f}
        meta = {**meta, "cross_source": source}

    sig = lambda z: 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    diffs = {"auroc": abs(roc_auc_score(t_y, t_vis) - metrics["auroc"]),
             "auroc_ffilt": abs(roc_auc_score(t_y, tf_logit) - metrics["auroc_ffilt"]),
             "srs": abs(float(np.mean(np.abs(sig(t_c) - sig(t_f)))) - metrics["srs"])}
    check = {"max_abs_diff": float(max(diffs.values())), **{k: float(v) for k, v in diffs.items()}}
    save_predictions(
        path, {**meta, "metrics": metrics, "check": check},
        val_image_id=va_t["image_id"].astype(str).to_numpy(), val_group_id=_group_ids(va_t),
        val_label=v_y.astype(np.int8), val_logit=v_logit, val_logit_ffilt=vf_logit,
        test_image_id=te_t["image_id"].astype(str).to_numpy(), test_group_id=_group_ids(te_t),
        test_label=t_y.astype(np.int8), test_logit=t_c, test_logit_flip=t_f,
        test_logit_visible=t_vis, test_logit_ffilt=tf_logit, **extra)
    print(f"  saved predictions {Path(path).name} (max difference to logged metrics "
          f"{check['max_abs_diff']:.2e})")
    return check


def save_checkpoint(path, model, meta):
    """Model weights plus run metadata (no optimizer state), for release."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("meta", {})


def run_seed(cfg, seed, leakage_p, method, device, keep_dir=None, preds_dir=None):
    set_seed(seed)
    dataset = cfg["data"]["dataset"]
    prep = PREP[dataset]
    tr, va, te, img_dirs, _ = prep(data_dir=cfg["data"]["data_dir"], seed=seed)
    level = cfg["data"]["text_filter_level"]
    tok = build_tokenizer(cfg["model"].get("text_model", DEFAULT_TEXT_MODEL))
    mk, _ = _note_makers(cfg)

    tr_t = training_notes(cfg, tr, seed, leakage_p, method)
    va_t = mk(va, level=level, seed=seed, leakage_p=leakage_p, part="train")
    te_t, te_ffilt = test_notes(cfg, te, seed, leakage_p)

    cw = compute_class_weights(tr_t)
    pos_w = float(cw[1] / cw[0])
    model = _build_model(cfg, device, use_text=(method != "image_only"))
    text_source = cfg["data"].get("text_source", "template")
    run_name = f"shortcut_{dataset}{_text_tag(cfg)}_{method}_p{leakage_p}_seed{seed}"
    save_dir = Path(f"models/checkpoints/{run_name}")
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
        train_loader = _loader(tr_t, img_dirs, tok, "train", cfg)
        val_loader = _loader(va_t, img_dirs, tok, "val", cfg)
        tcfg = {**cfg["training"], "pos_weight": pos_w, "save_dir": str(save_dir)}
        train(model, train_loader, val_loader, tcfg, device, "vision_text")

    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = evaluate_model(model, cfg, te_t, te_ffilt, img_dirs, tok, device, pos_w)

    # IG modality attribution on one test batch (best-effort; None if captum errors).
    try:
        ig_batch = next(iter(_loader(te_t, img_dirs, tok, "test", cfg, shuffle=False)))
        text_frac = ig_text_fraction(model, ig_batch, device)
    except Exception as exc:
        print(f"  IG attribution skipped: {type(exc).__name__}: {exc}")
        text_frac = None

    if preds_dir:
        _, ffilt_mk = _note_makers(cfg)
        Path(preds_dir).mkdir(parents=True, exist_ok=True)
        export_predictions(Path(preds_dir) / f"{run_name}.npz", model, cfg, va_t,
                           ffilt_mk(va, level=FFILT, seed=seed), te_t, te_ffilt, img_dirs, tok,
                           device, {"dataset": dataset, "method": method, "seed": seed,
                                    "leakage_p": leakage_p, "text_source": text_source,
                                    "llm_pool_split": bool(cfg["data"].get("llm_pool_split")),
                                    "run_name": run_name}, metrics,
                           cross=(cross_test_notes(cfg, te, seed, leakage_p)
                                  if cfg["data"].get("cross_text_test") else None))

    if keep_dir:
        save_checkpoint(Path(keep_dir) / f"{run_name}.pt", model,
                        {"dataset": dataset, "method": method, "seed": seed,
                         "leakage_p": leakage_p, "text_source": text_source,
                         "config": cfg, "metrics": metrics})
    # The full training checkpoint (~2GB with optimizer state) is throwaway. Drop it
    # so a long sweep does not accumulate hundreds of GB and fill the disk.
    shutil.rmtree(save_dir, ignore_errors=True)

    return {"seed": seed, "leakage_p": leakage_p, "method": method, **metrics,
            "text_frac": text_frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["dose", "bakeoff"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--keep-checkpoints", metavar="DIR",
                    help="also save each trained model (weights only) to DIR")
    ap.add_argument("--save-preds", metavar="DIR",
                    help="also save each run's per-lesion validation and test logits to DIR")
    ap.add_argument("--out", help="results JSON (default results/shortcut_<dataset>_<mode>.json)")
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
        rows = [run_seed(cfg, s, p, mth, device, args.keep_checkpoints, args.save_preds)
                for s in args.seeds]
        results.extend(rows)
        au = mean_ci([r["auroc"] for r in rows]); srs = mean_ci([r["srs"] for r in rows])
        gap = mean_ci([r["ffilt_gap"] for r in rows]); hon = mean_ci([r["auroc_ffilt"] for r in rows])
        print(f"[{args.mode}] p={p} method={mth} | AUROC {au[0]:.4f}+/-{au[1]:.4f} "
              f"honestAUROC {hon[0]:.4f} SRS {srs[0]:.4f}+/-{srs[1]:.4f} "
              f"FFiltGap {gap[0]:.4f}+/-{gap[1]:.4f}")

    Path("results").mkdir(exist_ok=True)
    _out_tag = _text_tag(cfg)
    out = Path(args.out) if args.out else (
        Path("results") / f"shortcut_{cfg['data']['dataset']}{_out_tag}_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
