"""Offline wiring tests for the text modality and late-fusion model.

These exercise the fusion shapes and the trainer's per-mode batch unpacking
WITHOUT downloading BioClinicalBERT or ImageNet weights:
  - a StubTextEncoder is injected in place of the real BERT (so `transformers`
    is never imported), and
  - the vision backbone is built with pretrained=False (random init, no download).

Run from the repo root:
    python tests/test_text_fusion.py
    pytest tests/test_text_fusion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.models.multimodal import MultimodalClassifier
from src.training.trainer import _run_batch

B, L, META_DIM, TEXT_DIM = 2, 16, 12, 32
BACKBONE = "resnet50"  # lighter than convnext for a CPU forward pass


class StubTextEncoder(nn.Module):
    """Masked mean-pool over token embeddings - same interface as TextEncoder."""

    def __init__(self, feature_dim=TEXT_DIM, vocab=1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed = nn.Embedding(vocab, feature_dim)

    def forward(self, input_ids, attention_mask):
        emb = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1)


def _image():
    return torch.randn(B, 3, 224, 224)


def _text():
    input_ids = torch.randint(0, 1000, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    return input_ids, attention_mask


def _vision_dim(model):
    return model.vision.feature_dim


def _make(use_metadata, use_text):
    return MultimodalClassifier(
        backbone=BACKBONE,
        pretrained=False,
        use_metadata=use_metadata,
        metadata_dim=META_DIM,
        use_text=use_text,
        text_encoder=StubTextEncoder() if use_text else None,
    )


def test_vision_text_fusion_shapes():
    model = _make(use_metadata=False, use_text=True)
    assert model.classifier.in_features == _vision_dim(model) + TEXT_DIM
    out = model(_image(), text=_text())
    assert out.shape == (B,)


def test_vision_metadata_text_fusion_shapes():
    model = _make(use_metadata=True, use_text=True)
    expected = _vision_dim(model) + model.metadata_encoder.feature_dim + TEXT_DIM
    assert model.classifier.in_features == expected
    out = model(_image(), metadata=torch.randn(B, META_DIM), text=_text())
    assert out.shape == (B,)


def test_backward_compat_metadata_only():
    # Defaults (use_metadata=True, use_text=False) must match the old behaviour:
    # forward(image, metadata) and fused_dim = vision + metadata_output.
    model = MultimodalClassifier(backbone=BACKBONE, pretrained=False, metadata_dim=META_DIM)
    assert not model.use_text
    assert model.classifier.in_features == _vision_dim(model) + model.metadata_encoder.feature_dim
    out = model(_image(), metadata=torch.randn(B, META_DIM))
    assert out.shape == (B,)


def test_extract_features_dim_matches_classifier():
    model = _make(use_metadata=True, use_text=True)
    feats = model.extract_features(_image(), metadata=torch.randn(B, META_DIM), text=_text())
    assert feats.shape == (B, model.classifier.in_features)


def test_text_branch_actually_used():
    # Changing the text input must change the logits (text isn't silently dropped).
    model = _make(use_metadata=False, use_text=True)
    model.eval()
    img = _image()
    ids_a = torch.zeros(B, L, dtype=torch.long)
    ids_b = torch.full((B, L), 5, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.long)
    with torch.no_grad():
        out_a = model(img, text=(ids_a, mask))
        out_b = model(img, text=(ids_b, mask))
    assert not torch.allclose(out_a, out_b), "logits ignore the text input"


def test_run_batch_modes_match_dataset_layout():
    """The trainer's _run_batch must unpack each mode's batch tuple correctly.

    The tuple order here mirrors exactly what MultimodalTextDataset /
    MultimodalSkinDataset emit, so this catches dataset<->trainer drift.
    """
    labels = torch.randint(0, 2, (B,)).float()
    img = _image()
    meta = torch.randn(B, META_DIM)
    ids, mask = _text()

    # vision_text: (image, input_ids, attention_mask, label)
    m = _make(use_metadata=False, use_text=True)
    out, lab = _run_batch(m, (img, ids, mask, labels), "vision_text", "cpu")
    assert out.shape == (B,) and torch.equal(lab, labels)

    # vision_metadata_text: (image, metadata, input_ids, attention_mask, label)
    m = _make(use_metadata=True, use_text=True)
    out, lab = _run_batch(m, (img, meta, ids, mask, labels), "vision_metadata_text", "cpu")
    assert out.shape == (B,) and torch.equal(lab, labels)

    # vision_metadata: (image, metadata, label)
    m = _make(use_metadata=True, use_text=False)
    out, lab = _run_batch(m, (img, meta, labels), "vision_metadata", "cpu")
    assert out.shape == (B,) and torch.equal(lab, labels)


def test_run_batch_rejects_unknown_mode():
    m = _make(use_metadata=True, use_text=False)
    try:
        _run_batch(m, (_image(), torch.randn(B, META_DIM), torch.zeros(B)), "bogus", "cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
