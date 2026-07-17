"""Training and evaluation utilities for vision-only and multimodal models."""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np
from pathlib import Path
import time
import json


def _run_batch(model, batch, mode, device):
    """Unpack a batch for the given mode, move tensors to device, return (outputs, labels).

    Batch layouts (set by the dataset class):
      vision               -> (image, label)
      vision_metadata      -> (image, metadata, label)
      vision_text          -> (image, input_ids, attention_mask, label)
      vision_metadata_text -> (image, metadata, input_ids, attention_mask, label)
    """
    if mode == "vision":
        images, labels = batch
        outputs = model(images.to(device))
    elif mode == "vision_metadata":
        images, metadata, labels = batch
        outputs = model(images.to(device), metadata=metadata.to(device))
    elif mode == "vision_text":
        images, input_ids, attn, labels = batch
        outputs = model(images.to(device), text=(input_ids.to(device), attn.to(device)))
    elif mode == "vision_metadata_text":
        images, metadata, input_ids, attn, labels = batch
        outputs = model(
            images.to(device),
            metadata=metadata.to(device),
            text=(input_ids.to(device), attn.to(device)),
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return outputs, labels.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, mode="vision"):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        outputs, labels = _run_batch(model, batch, mode, device)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auroc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auroc


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode="vision"):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loader:
        outputs, labels = _run_batch(model, batch, mode, device)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    # Threshold for 95% sensitivity
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        threshold_95 = thresholds[idx[0]]
        specificity_95 = 1 - fpr[idx[0]]
        sensitivity_95 = tpr[idx[0]]
    else:
        threshold_95, specificity_95, sensitivity_95 = 0.5, 0.0, 0.0

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "ap": ap,
        "threshold_95_sens": float(threshold_95),
        "specificity_at_95_sens": float(specificity_95),
        "sensitivity": float(sensitivity_95),
    }


def train(model, train_loader, val_loader, config, device, mode="vision"):
    pos_weight = config.get("pos_weight", None)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 5)
    )

    save_dir = Path(config.get("save_dir", "models/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0
    history = []

    for epoch in range(config.get("epochs", 5)):
        start = time.time()

        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mode
        )
        val_metrics = evaluate(model, val_loader, criterion, device, mode)
        scheduler.step()

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{config['epochs']} ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} AUROC: {train_auroc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} AUROC: {val_metrics['auroc']:.4f} "
              f"AP: {val_metrics['ap']:.4f} Spec@95Sens: {val_metrics['specificity_at_95_sens']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_auroc": train_auroc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, save_dir / "best_model.pt")
            print(f"  -> New best model saved (AUROC: {best_auroc:.4f})")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history, best_auroc
