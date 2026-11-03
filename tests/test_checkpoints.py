"""Offline tests for checkpoint saving/loading and the shared evaluation path used by
scripts/run_shortcut.py and scripts/evaluate_checkpoint.py (no model download)."""
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load("run_shortcut")


class WordTokenizer:
    """Tiny stand-in for a HuggingFace tokenizer: one id per known word."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.lower().split()]
        ids = (ids + [0] * max_length)[:max_length]
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": (ids > 0).long()}


class ImageOnly(nn.Module):
    """Ignores the note entirely: a known-answer model with SRS = 0."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([4.0, -4.0, 0.0]))

    def forward(self, img, text=None):
        return img.mean(dim=(2, 3)) @ self.w


CFG = {"data": {"img_size": 16, "batch_size": 8, "num_workers": 0, "text_max_length": 32,
                "text_filter_level": "DFilt"}}


@pytest.fixture
def test_set(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        label = int(i % 3 == 0)
        colour = (200, 40, 40) if label else (40, 200, 40)
        noisy = np.clip(np.array(colour) + rng.integers(-60, 60, 3), 0, 255).astype(np.uint8)
        Image.new("RGB", (20, 20), tuple(int(c) for c in noisy)).save(tmp_path / f"IMG_{i}.png")
        rows.append({"image_id": f"IMG_{i}", "dx": "mel" if label else "nv", "label": label,
                     "age": 50, "sex": "male", "localization": "back"})
    return pd.DataFrame(rows), [tmp_path]


def test_evaluate_model_gives_zero_srs_for_an_image_only_model(test_set):
    df, img_dirs = test_set
    te_t, te_ffilt = rs.test_notes(CFG, df, seed=42, leakage_p=1.0)
    out = rs.evaluate_model(ImageOnly(), CFG, te_t, te_ffilt, img_dirs, WordTokenizer(),
                            torch.device("cpu"), pos_w=2.0)
    assert out["srs"] == 0.0 and out["flip_rate"] == 0.0
    assert out["auroc"] == out["auroc_ffilt"] and out["ffilt_gap"] == 0.0
    assert set(out) >= {"auroc", "ap", "spec95", "auroc_ffilt", "ffilt_gap", "srs", "flip_rate"}


def test_checkpoint_keeps_weights_and_metadata_but_not_the_optimizer(tmp_path):
    model = ImageOnly()
    path = tmp_path / "ckpt.pt"
    rs.save_checkpoint(path, model, {"method": "naive", "seed": 42, "leakage_p": 1.0})
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in saved
    assert saved["meta"] == {"method": "naive", "seed": 42, "leakage_p": 1.0}

    fresh = ImageOnly()
    with torch.no_grad():
        fresh.w.zero_()
    meta = rs.load_checkpoint(path, fresh)
    assert torch.equal(fresh.w, model.w) and meta["seed"] == 42


@pytest.mark.parametrize("method, rows_per_lesion, filtered", [
    ("naive", 1, False), ("ffilt_only", 1, True),
    ("ffilt_only_2x", 2, True), ("counterfactual_aug", 2, False)])
def test_training_notes_per_method(test_set, method, rows_per_lesion, filtered):
    df, _ = test_set
    tr_t = rs.training_notes(CFG, df, seed=42, leakage_p=1.0, method=method)
    assert len(tr_t) == rows_per_lesion * len(df)
    assert sorted(tr_t["image_id"].value_counts().unique()) == [rows_per_lesion]
    leading = ("refer", "excision", "urgent", "routine", "reassurance", "no urgent")
    has_leading = tr_t["text"].str.lower().str.contains("|".join(leading))
    assert (not has_leading.any()) if filtered else has_leading.all()


def test_paper_reference_picks_the_row_for_method_and_seed(tmp_path):
    import json
    ev = _load("evaluate_checkpoint")
    rows = [{"method": "naive", "seed": 42, "auroc_ffilt": 0.70, "srs": 0.99},
            {"method": "counterfactual_aug", "seed": 123, "auroc_ffilt": 0.91, "srs": 0.004},
            {"method": "counterfactual_aug", "seed": 42, "auroc_ffilt": 0.92, "srs": 0.003}]
    path = tmp_path / "bakeoff.json"
    path.write_text(json.dumps(rows))
    assert ev.paper_reference(path, "counterfactual_aug", 42)["auroc_ffilt"] == 0.92
    assert ev.paper_reference(path, "adversarial", 42) is None
    assert ev.paper_reference(tmp_path / "missing.json", "naive", 42) is None
