"""Per-lesion prediction export for the reviewer analyses.

The paper's runs kept only summary metrics. Saving each trained model's logits on
the validation and test notes lets operating-point, calibration, decision-curve and
SRS-distribution analyses be recomputed without retraining. Logits (not
probabilities) are stored so that logit-scale variants of SRS stay finite for a
saturated model. torch is imported lazily.
"""
import json

import numpy as np


def collect_logits(model, loader, device):
    """Logits and labels for a loader yielding (image, input_ids, attention_mask, label)."""
    import torch

    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for img, ids, mask, y in loader:
            out = model(img.to(device), text=(ids.to(device), mask.to(device)))
            logits.append(out.float().cpu().numpy().reshape(-1))
            labels.append(y.numpy().reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def collect_paired_logits(model, paired_loader, device):
    """Logits on the concordant and the flipped note of every lesion (same image)."""
    import torch

    model.eval()
    conc, flip, labels = [], [], []
    with torch.no_grad():
        for img, ic, mc, iff, mf, y in paired_loader:
            img = img.to(device)
            conc.append(model(img, text=(ic.to(device), mc.to(device))).float().cpu().numpy().reshape(-1))
            flip.append(model(img, text=(iff.to(device), mf.to(device))).float().cpu().numpy().reshape(-1))
            labels.append(y.numpy().reshape(-1))
    return np.concatenate(conc), np.concatenate(flip), np.concatenate(labels)


def save_predictions(path, meta, **arrays):
    """Compressed .npz with the arrays plus a JSON metadata string (no pickling).
    Object arrays (pandas string columns) are stored as fixed-width unicode."""
    arrays = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else np.asarray(v))
              for k, v in arrays.items()}
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=str)), **arrays)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d
