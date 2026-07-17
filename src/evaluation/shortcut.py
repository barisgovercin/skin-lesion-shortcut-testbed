"""Shortcut-reliance metrics: SRS (flip test) and FFilt-gap."""
import numpy as np
import torch


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


@torch.no_grad()
def srs_from_loader(model, paired_loader, device):
    """Run a PairedTextDataset loader; SRS + flip_rate over the whole set.

    Each batch is (image, ids_c, mask_c, ids_f, mask_f, label); the model is
    scored on the concordant and flipped text with the same image.
    """
    model.eval()
    p_orig, p_flip = [], []
    for img, ic, mc, iff, mf, _ in paired_loader:
        img = img.to(device)
        lo = model(img, text=(ic.to(device), mc.to(device)))
        lf = model(img, text=(iff.to(device), mf.to(device)))
        p_orig.extend(torch.sigmoid(lo).cpu().numpy().tolist())
        p_flip.extend(torch.sigmoid(lf).cpu().numpy().tolist())
    return srs_from_probs(p_orig, p_flip)
