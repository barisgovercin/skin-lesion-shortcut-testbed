"""Shortcut-reliance metrics: SRS (flip test) and FFilt-gap.

SRS applies the counterfactual token fairness gap of Garg et al. (2019,
"Counterfactual Fairness in Text Classification through Robustness") to
diagnosis-bearing clinical text. torch is imported lazily so the metrics run on a
plain CPU install (see scripts/srs_demo.py).
"""
import numpy as np


def srs_from_probs(p_orig, p_flip, threshold=0.5):
    """Shortcut Reliance Score from two probability arrays (original vs flipped
    leading language). Returns mean |delta prob| and the predicted-class flip rate."""
    p_orig = np.asarray(p_orig, float)
    p_flip = np.asarray(p_flip, float)
    flips = (p_orig >= threshold) != (p_flip >= threshold)
    return {"srs": float(np.mean(np.abs(p_orig - p_flip))),
            "flip_rate": float(np.mean(flips))}


def ffilt_gap_from_auroc(auroc_leaky, auroc_ffilt):
    """How much AUROC drops when the leading-language shortcut is removed."""
    return float(auroc_leaky - auroc_ffilt)


def srs_from_loader(model, paired_loader, device):
    """Run a PairedTextDataset loader; SRS + flip_rate over the whole set.

    Each batch is (image, ids_c, mask_c, ids_f, mask_f, label); the model is
    scored on the concordant and flipped text with the same image.
    """
    import torch

    model.eval()
    p_orig, p_flip = [], []
    with torch.no_grad():
        for img, ic, mc, iff, mf, _ in paired_loader:
            img = img.to(device)
            lo = model(img, text=(ic.to(device), mc.to(device)))
            lf = model(img, text=(iff.to(device), mf.to(device)))
            p_orig.extend(torch.sigmoid(lo).cpu().numpy().tolist())
            p_flip.extend(torch.sigmoid(lf).cpu().numpy().tolist())
    return srs_from_probs(p_orig, p_flip)
