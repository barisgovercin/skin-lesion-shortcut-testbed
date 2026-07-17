"""Custom training steps for consistency and adversarial debiasing.

Batches come from PairedTextDataset: (image, ids_c, mask_c, ids_f, mask_f, label).
"""
import torch
import torch.nn as nn

from src.training.debias import consistency_loss


def _bce(pos_weight, device):
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))


def consistency_train_step(model, batch, optimizer, device, pos_weight, lambd):
    """One step: BCE on both text versions + a flip-invariance penalty."""
    model.train()
    img, ic, mc, iff, mf, y = batch
    img, y = img.to(device), y.to(device)
    crit = _bce(pos_weight, device)
    lo = model(img, text=(ic.to(device), mc.to(device)))
    lf = model(img, text=(iff.to(device), mf.to(device)))
    bce = 0.5 * (crit(lo, y) + crit(lf, y))
    cons = consistency_loss(lo, lf)
    loss = bce + lambd * cons
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": float(loss.item()), "bce": float(bce.item()),
            "consistency": float(cons.item())}


def adversarial_train_step(model, adv_head, batch, optimizer, device, pos_weight, lambd):
    """One step: main BCE + adversarial label-from-text head via gradient reversal."""
    model.train(); adv_head.train()
    img, ic, mc, iff, mf, y = batch
    img, y = img.to(device), y.to(device)
    ids, mask = ic.to(device), mc.to(device)
    crit = _bce(pos_weight, device)
    main_logit = model(img, text=(ids, mask))
    text_feat = model.text_encoder(ids, mask)
    adv_logit = adv_head(text_feat, lambd=lambd)
    loss = crit(main_logit, y) + crit(adv_logit, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"loss": float(loss.item())}
