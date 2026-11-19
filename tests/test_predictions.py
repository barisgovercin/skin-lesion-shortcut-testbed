"""Offline tests for the per-lesion prediction export used by the reviewer analyses
(scripts/run_shortcut.py --save-preds). No model download."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.evaluation.predictions import load_predictions, save_predictions


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load("run_shortcut")


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.lower().split()]
        ids = (ids + [0] * max_length)[:max_length]
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": (ids > 0).long()}


class TextReader(nn.Module):
    """Image colour plus a term that depends on the note, so concordant and flipped
    notes give different logits (SRS > 0)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([4.0, -4.0, 0.0]))

    def forward(self, img, text=None):
        ids, mask = text
        return img.mean(dim=(2, 3)) @ self.w + 0.01 * (ids * mask).float().sum(dim=1) / 50.0


CFG = {"data": {"img_size": 16, "batch_size": 8, "num_workers": 0, "text_max_length": 32,
                "text_filter_level": "DFilt"}}


def _frame(tmp_path, n, prefix):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        label = int(i % 3 == 0)
        colour = (200, 40, 40) if label else (40, 200, 40)
        noisy = np.clip(np.array(colour) + rng.integers(-60, 60, 3), 0, 255).astype(np.uint8)
        Image.new("RGB", (20, 20), tuple(int(c) for c in noisy)).save(tmp_path / f"{prefix}{i}.png")
        rows.append({"image_id": f"{prefix}{i}", "lesion_id": f"L{prefix}{i // 2}", "dx": "mel" if label else "nv",
                     "label": label, "age": 50, "sex": "male", "localization": "back"})
    return pd.DataFrame(rows)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "run.npz"
    ids = np.array(["ISIC_1", "ISIC_22"])
    save_predictions(path, {"method": "naive", "seed": 42}, test_image_id=ids,
                     test_logit=np.array([0.5, -1.25], np.float32))
    d = load_predictions(path)
    assert d["meta"] == {"method": "naive", "seed": 42}
    assert list(d["test_image_id"]) == ["ISIC_1", "ISIC_22"]
    assert np.allclose(d["test_logit"], [0.5, -1.25])


def test_export_matches_the_logged_metrics(tmp_path):
    va = _frame(tmp_path, 24, "V")
    te = _frame(tmp_path, 40, "T")
    va_t = rs.add_text_columns(va, level="DFilt", seed=42, leakage_p=1.0)
    va_ffilt = rs.add_text_columns(va, level=rs.FFILT, seed=42)
    te_t, te_ffilt = rs.test_notes(CFG, te, seed=42, leakage_p=1.0)
    model, tok, dev = TextReader(), WordTokenizer(), torch.device("cpu")

    metrics = rs.evaluate_model(model, CFG, te_t, te_ffilt, [tmp_path], tok, dev, pos_w=2.0)
    path = tmp_path / "preds.npz"
    check = rs.export_predictions(path, model, CFG, va_t, va_ffilt, te_t, te_ffilt,
                                  [tmp_path], tok, dev, {"method": "naive"}, metrics)
    d = load_predictions(path)

    assert len(d["val_logit"]) == len(d["val_logit_ffilt"]) == 24
    assert len(d["test_logit"]) == len(d["test_logit_flip"]) == len(d["test_logit_ffilt"]) == 40
    assert list(d["test_image_id"]) == list(te["image_id"])
    assert list(d["test_group_id"]) == list(te["lesion_id"])
    assert np.array_equal(d["test_label"], te["label"].to_numpy())
    # The saved arrays reproduce the metrics computed by the training run.
    assert roc_auc_score(d["test_label"], d["test_logit_ffilt"]) == pytest.approx(metrics["auroc_ffilt"])
    assert roc_auc_score(d["test_label"], d["test_logit_visible"]) == pytest.approx(metrics["auroc"])
    p_c, p_f = 1 / (1 + np.exp(-d["test_logit"])), 1 / (1 + np.exp(-d["test_logit_flip"]))
    assert np.mean(np.abs(p_c - p_f)) == pytest.approx(metrics["srs"], abs=1e-6)
    assert metrics["srs"] > 0
    assert check["max_abs_diff"] < 1e-6 and d["meta"]["method"] == "naive"


def test_image_only_model_has_no_text_branch():
    cfg = {"model": {"backbone": "convnext_base", "pretrained": False}}
    model = rs._build_model(cfg, torch.device("cpu"), use_text=False)
    assert not model.use_text and not hasattr(model, "text_encoder")


def test_pool_split_keeps_test_wording_out_of_training(tmp_path):
    from src.preprocessing import llm_text as lt
    df = pd.DataFrame([{"image_id": f"I{i}", "dx": "mel" if i % 2 else "nv", "label": i % 2,
                        "age": 40 + i % 30, "sex": "female", "localization": "back"} for i in range(200)])
    import json
    cache = tmp_path / "pools.json"
    cache.write_text(json.dumps({f"mock||DFilt|{m}|": [f"Impression {m} variant {i}." for i in range(30)]
                                 for m in (0, 1)}))
    kw = dict(level="DFilt", seed=7, pair=True, provider="mock", cache_path=cache, pool_size=30)
    train = lt.add_llm_text_columns(df, pool_part="train", **kw)
    test = lt.add_llm_text_columns(df, pool_part="test", **kw)
    whole = lt.add_llm_text_columns(df, **kw)
    facts = [lt._facts_text(lt._norm(r), 7) for _, r in df.iterrows()]
    clauses = lambda notes: {n[len(f):].strip() for n, f in zip(notes, facts)}
    tr_c = clauses(train["text_concordant"]) | clauses(train["text_flipped"])
    te_c = clauses(test["text_concordant"]) | clauses(test["text_flipped"])
    assert tr_c and te_c and not (tr_c & te_c)
    assert all(lt.heldout_clause(c) for c in te_c) and not any(lt.heldout_clause(c) for c in tr_c)
    # facts are untouched, and pool_part=None reproduces the original behaviour
    assert list(test["text_concordant"].str[:20]) == list(train["text_concordant"].str[:20])
    assert list(whole["text"]) == list(lt.add_llm_text_columns(df, **kw)["text"])


def test_export_adds_notes_from_the_other_engine(tmp_path):
    va = _frame(tmp_path, 12, "V")
    te = _frame(tmp_path, 20, "T")
    cfg = {**CFG, "data": {**CFG["data"], "cross_text_test": True}}
    va_t = rs.add_text_columns(va, level="DFilt", seed=42, leakage_p=1.0)
    va_ffilt = rs.add_text_columns(va, level=rs.FFILT, seed=42)
    te_t, te_ffilt = rs.test_notes(cfg, te, seed=42, leakage_p=1.0)
    source, te_x = rs.cross_test_notes(cfg, te, seed=42, leakage_p=1.0)
    assert source == "llm_heldout" and list(te_x["image_id"]) == list(te["image_id"])
    assert set(te_x["text_concordant"]).isdisjoint(set(te_t["text_concordant"]))
    model, tok, dev = TextReader(), WordTokenizer(), torch.device("cpu")
    metrics = rs.evaluate_model(model, cfg, te_t, te_ffilt, [tmp_path], tok, dev, pos_w=2.0)
    rs.export_predictions(tmp_path / "x.npz", model, cfg, va_t, va_ffilt, te_t, te_ffilt,
                          [tmp_path], tok, dev, {"method": "naive"}, metrics, cross=(source, te_x))
    d = load_predictions(tmp_path / "x.npz")
    assert d["meta"]["cross_source"] == "llm_heldout"
    assert len(d["test_logit_cross"]) == len(d["test_logit_cross_flip"]) == 20
    assert not np.allclose(d["test_logit_cross"], d["test_logit"])


def test_pool_split_refuses_an_empty_part():
    from src.preprocessing import llm_text as lt
    pool = [c for c in (f"clause {i}" for i in range(50)) if not lt.heldout_clause(c)][:3]
    with pytest.raises(ValueError):
        lt._pool_part(pool, "test")
    assert lt._pool_part(pool, None) == pool

