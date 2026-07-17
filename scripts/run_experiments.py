"""Run all experiments: vision-only and multimodal, across 5 seeds.

Follows Watson et al. (2026) protocol:
- 5 seeds for statistical robustness
- Report mean +/- 95% CI
- Vision-only baseline + Vision+Metadata multimodal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import json
import argparse
from torch.utils.data import DataLoader

from src.data.dataset import (
    SkinLesionDataset,
    MultimodalSkinDataset,
    MultimodalTextDataset,
    get_transforms,
    prepare_ham10000,
    prepare_isic2020,
    prepare_isic_dicm_17k,
    compute_class_weights,
)
from src.models.vision import VisionClassifier
from src.models.multimodal import MultimodalClassifier
from src.models.text import DEFAULT_TEXT_MODEL, build_tokenizer
from src.preprocessing.synthetic_text import add_synthetic_text
from src.training.trainer import train, evaluate


SEEDS = [42, 123, 456, 789, 1024]

PREPARE_FUNCS = {
    "ham10000": prepare_ham10000,
    "isic2020": prepare_isic2020,
    "isic_dicm_17k": prepare_isic_dicm_17k,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_ci(values, confidence=0.95):
    """Compute mean and 95% confidence interval."""
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0
    from scipy import stats
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, ci


def run_single_experiment(mode, seed, cfg, device):
    """Run a single experiment (one seed, one mode)."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Mode: {mode} | Seed: {seed}")
    print(f"{'='*60}")

    use_metadata = "metadata" in mode
    use_text = "text" in mode
    dataset_name = cfg["data"].get("dataset", "ham10000")
    prepare_fn = PREPARE_FUNCS.get(dataset_name)
    if prepare_fn is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(PREPARE_FUNCS)}")
    engineered = cfg["data"].get("engineered_metadata", False)
    prep_kwargs = dict(
        data_dir=cfg["data"]["data_dir"],
        seed=seed,
        include_metadata=use_metadata,
        engineered=engineered,
    )
    if dataset_name == "isic2020":
        prep_kwargs["patient_aggregates"] = cfg["data"].get("patient_aggregates", False)
    train_df, val_df, test_df, img_dirs, metadata_cols = prepare_fn(**prep_kwargs)

    img_size = cfg["data"]["img_size"]

    if use_text:
        # Synthesise the clinical note at the configured leading-language level,
        # then tokenise it inside the dataset.
        level = cfg["data"].get("text_filter_level", "Orig")
        text_model = cfg["model"].get("text_model", DEFAULT_TEXT_MODEL)
        tokenizer = build_tokenizer(text_model)
        max_length = cfg["data"].get("text_max_length", 128)
        print(f"Text modality: filter level={level}, model={text_model}")

        text_metadata_cols = metadata_cols if use_metadata else None
        train_df = add_synthetic_text(train_df, level=level, seed=seed)
        val_df = add_synthetic_text(val_df, level=level, seed=seed)
        test_df = add_synthetic_text(test_df, level=level, seed=seed)

        def make_text_ds(df, split):
            return MultimodalTextDataset(
                df, img_dirs, tokenizer, get_transforms(split, img_size),
                metadata_cols=text_metadata_cols, max_length=max_length,
            )

        train_dataset = make_text_ds(train_df, "train")
        val_dataset = make_text_ds(val_df, "val")
        test_dataset = make_text_ds(test_df, "test")
    elif use_metadata:
        train_dataset = MultimodalSkinDataset(
            train_df, img_dirs, get_transforms("train", img_size), metadata_cols
        )
        val_dataset = MultimodalSkinDataset(
            val_df, img_dirs, get_transforms("val", img_size), metadata_cols
        )
        test_dataset = MultimodalSkinDataset(
            test_df, img_dirs, get_transforms("test", img_size), metadata_cols
        )
    else:
        train_dataset = SkinLesionDataset(train_df, img_dirs, get_transforms("train", img_size))
        val_dataset = SkinLesionDataset(val_df, img_dirs, get_transforms("val", img_size))
        test_dataset = SkinLesionDataset(test_df, img_dirs, get_transforms("test", img_size))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    class_weights = compute_class_weights(train_df)
    pos_weight = float(class_weights[1] / class_weights[0])

    backbone = cfg["model"]["backbone"]
    if use_metadata or use_text:
        model = MultimodalClassifier(
            backbone=backbone,
            pretrained=cfg["model"]["pretrained"],
            use_metadata=use_metadata,
            metadata_dim=len(metadata_cols) if use_metadata else 0,
            use_text=use_text,
            text_model_name=cfg["model"].get("text_model", DEFAULT_TEXT_MODEL),
            text_freeze=cfg["model"].get("text_freeze", False),
        ).to(device)
    else:
        model = VisionClassifier(
            backbone=backbone,
            pretrained=cfg["model"]["pretrained"],
        ).to(device)

    save_dir = f"models/checkpoints/{dataset_name}_{mode}_{backbone}_seed{seed}"
    train_config = {
        **cfg["training"],
        "pos_weight": pos_weight,
        "save_dir": save_dir,
    }

    history, best_auroc = train(
        model, train_loader, val_loader, train_config, device, mode
    )

    # Evaluate on test set with best model
    checkpoint = torch.load(
        Path(save_dir) / "best_model.pt", map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    test_metrics = evaluate(model, test_loader, criterion, device, mode)

    print(f"\nTest Results (seed {seed}):")
    print(f"  AUROC:                 {test_metrics['auroc']:.4f}")
    print(f"  Average Precision:     {test_metrics['ap']:.4f}")
    print(f"  Specificity @95% Sens: {test_metrics['specificity_at_95_sens']:.4f}")
    print(f"  Sensitivity:           {test_metrics['sensitivity']:.4f}")

    return {
        "mode": mode,
        "seed": seed,
        "backbone": backbone,
        "best_val_auroc": best_auroc,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["vision", "vision_metadata", "vision_text", "vision_metadata_text", "all"],
        default="all",
        help="'all' runs vision + vision_metadata; text modes are opt-in (need a text-enabled dataset).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--config", default="configs/baseline_ham10000.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    modes = ["vision", "vision_metadata"] if args.mode == "all" else [args.mode]
    all_results = []

    for mode in modes:
        mode_results = []
        for seed in args.seeds:
            result = run_single_experiment(mode, seed, cfg, device)
            mode_results.append(result)
            all_results.append(result)

        # Aggregate results for this mode
        aurocs = [r["test_metrics"]["auroc"] for r in mode_results]
        aps = [r["test_metrics"]["ap"] for r in mode_results]
        specs = [r["test_metrics"]["specificity_at_95_sens"] for r in mode_results]

        auroc_mean, auroc_ci = compute_ci(aurocs)
        ap_mean, ap_ci = compute_ci(aps)
        spec_mean, spec_ci = compute_ci(specs)

        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS: {mode} ({len(args.seeds)} seeds)")
        print(f"{'='*60}")
        print(f"  AUROC:                 {auroc_mean:.4f} +/- {auroc_ci:.4f}")
        print(f"  Average Precision:     {ap_mean:.4f} +/- {ap_ci:.4f}")
        print(f"  Specificity @95% Sens: {spec_mean:.4f} +/- {spec_ci:.4f}")

    # Save all results. Text runs append the filter level so the four
    # leading-language ablations (Orig/CFilt/DFilt/FFilt) don't overwrite each other.
    dataset_name = cfg["data"].get("dataset", "ham10000")
    stem = f"experiment_results_{dataset_name}_{cfg['model']['backbone']}"
    if cfg["data"].get("engineered_metadata", False):
        stem += "_eng"
    if any("text" in m for m in modes):
        stem += f"_{cfg['data'].get('text_filter_level', 'Orig')}"
    results_path = Path("results") / f"{stem}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
