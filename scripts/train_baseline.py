"""Train baseline vision-only model on HAM10000."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from src.data.dataset import (
    SkinLesionDataset,
    get_transforms,
    prepare_ham10000,
    compute_class_weights,
)
from src.models.vision import VisionClassifier
from src.training.trainer import train, evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    with open("configs/baseline_ham10000.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n=== Preparing Dataset ===")
    train_df, val_df, test_df, img_dirs, _ = prepare_ham10000(
        data_dir=cfg["data"]["data_dir"], seed=seed
    )

    img_size = cfg["data"]["img_size"]
    train_dataset = SkinLesionDataset(train_df, img_dirs, get_transforms("train", img_size))
    val_dataset = SkinLesionDataset(val_df, img_dirs, get_transforms("val", img_size))
    test_dataset = SkinLesionDataset(test_df, img_dirs, get_transforms("test", img_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    class_weights = compute_class_weights(train_df)
    pos_weight = class_weights[1] / class_weights[0]
    print(f"\nClass weights - Neg: {class_weights[0]:.3f}, Pos: {class_weights[1]:.3f}")
    print(f"Pos weight for BCEWithLogitsLoss: {pos_weight:.3f}")

    print(f"\n=== Building Model: {cfg['model']['backbone']} ===")
    model = VisionClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\n=== Training ({cfg['training']['epochs']} epochs) ===")
    train_config = {
        **cfg["training"],
        "pos_weight": float(pos_weight),
    }

    history, best_auroc = train(model, train_loader, val_loader, train_config, device)

    print(f"\n=== Evaluating on Test Set ===")
    checkpoint = torch.load(
        Path(cfg["training"]["save_dir"]) / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  AUROC:                 {test_metrics['auroc']:.4f}")
    print(f"  Average Precision:     {test_metrics['ap']:.4f}")
    print(f"  Specificity @95% Sens: {test_metrics['specificity_at_95_sens']:.4f}")
    print(f"  Sensitivity:           {test_metrics['sensitivity']:.4f}")
    print(f"  Threshold:             {test_metrics['threshold_95_sens']:.4f}")

    import json
    results = {
        "experiment": cfg["experiment"]["name"],
        "backbone": cfg["model"]["backbone"],
        "seed": seed,
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
        "training_history": history,
    }
    results_path = Path("results") / f"{cfg['experiment']['name']}_seed{seed}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
