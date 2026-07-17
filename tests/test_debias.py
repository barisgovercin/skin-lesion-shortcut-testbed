"""Offline tests for debiasing components (no GPU/data needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from src.data.text_datasets import build_counterfactual_df
from src.training.debias import GradientReversal, AdversarialTextHead
from src.training.debias_trainer import consistency_train_step


def _df():
    return pd.DataFrame({
        "image_id": ["A", "B"], "label": [1, 0],
        "text_concordant": ["malignant note A", "benign note B"],
        "text_flipped": ["benign note A", "malignant note B"],
    })


def test_build_counterfactual_df_duplicates_with_true_label():
    out = build_counterfactual_df(_df())
    assert len(out) == 4
    assert list(out["label"]) == [1, 1, 0, 0]      # label preserved on both copies
    assert {"malignant note A", "benign note A"}.issubset(set(out["text"]))


def test_gradient_reversal_flips_sign():
    x = torch.tensor([2.0], requires_grad=True)
    y = GradientReversal.apply(x, 1.0)
    y.backward()
    assert x.grad.item() < 0       # forward identity; backward flips +1 -> -1


def test_adversarial_head_shapes():
    head = AdversarialTextHead(in_dim=8, hidden=4)
    out = head(torch.randn(3, 8), lambd=1.0)
    assert out.shape == (3,)


def test_consistency_step_runs_and_returns_floats():
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Linear(4, 1)
        def forward(self, image, text=None):
            ids, _ = text
            return self.w(ids.float()).squeeze(-1)
    m = Stub(); opt = torch.optim.SGD(m.parameters(), lr=0.1)
    img = torch.zeros(2, 3)
    batch = (img, torch.randn(2, 4), torch.ones(2, 4),
             torch.randn(2, 4), torch.ones(2, 4), torch.tensor([1.0, 0.0]))
    out = consistency_train_step(m, batch, opt, torch.device("cpu"),
                                 pos_weight=1.0, lambd=1.0)
    assert set(out) == {"loss", "bce", "consistency"}
    assert all(isinstance(v, float) for v in out.values())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
