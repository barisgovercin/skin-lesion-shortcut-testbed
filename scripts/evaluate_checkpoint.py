"""Evaluate a released shortcut-testbed checkpoint on the real test split.

Rebuilds the test split for the checkpoint's seed, generates the test notes and the
facts-only (FFilt) probe, and computes visible AUROC, honest AUROC, FFilt-gap, SRS
and FlipRate with the same code as scripts/run_shortcut.py. Prints them next to the
values stored at training time and the paper's value for the same method and seed.

    python scripts/evaluate_checkpoint.py \\
        --checkpoint checkpoints/shortcut_ham10000_counterfactual_aug_p1.0_seed42.pt \\
        --data-dir data/raw/ham10000

Needs the dataset images and downloads the BioClinicalBERT tokenizer/config; a GPU
is optional (CPU takes a few minutes per model).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import compute_class_weights  # noqa: E402
from src.models.text import DEFAULT_TEXT_MODEL, build_tokenizer  # noqa: E402

METRICS = ("auroc", "auroc_ffilt", "ffilt_gap", "srs", "flip_rate", "ap", "spec95")


def _run_shortcut():
    spec = importlib.util.spec_from_file_location("run_shortcut", ROOT / "scripts" / "run_shortcut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def paper_reference(results_json, method, seed):
    """The paper's per-seed result row for (method, seed), or None."""
    path = Path(results_json)
    if not path.exists():
        return None
    rows = [r for r in json.loads(path.read_text()) if r["method"] == method and r["seed"] == seed]
    return rows[0] if rows else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", help="dataset folder (default: the one stored in the checkpoint)")
    ap.add_argument("--reference", help="paper results JSON (default: results/shortcut_<dataset>_bakeoff.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only N test images, half per class (smoke test)")
    ap.add_argument("--out", help="write the metrics to this JSON file")
    args = ap.parse_args(argv)

    rs = _run_shortcut()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    cfg = meta["config"]
    cfg["model"]["pretrained"] = False          # weights come from the checkpoint
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset, method, seed, p = meta["dataset"], meta["method"], meta["seed"], meta["leakage_p"]

    tr, _, te, img_dirs, _ = rs.PREP[dataset](data_dir=cfg["data"]["data_dir"], seed=seed)
    if args.limit:   # half per class, so AUROC stays defined in a smoke test
        te = te.groupby("label", group_keys=False).head(max(1, args.limit // 2))
    cw = compute_class_weights(rs.training_notes(cfg, tr, seed, p, method))
    tok = build_tokenizer(cfg["model"].get("text_model", DEFAULT_TEXT_MODEL))
    te_t, te_ffilt = rs.test_notes(cfg, te, seed, p)
    model = rs._build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    metrics = rs.evaluate_model(model, cfg, te_t, te_ffilt, img_dirs, tok, device,
                                float(cw[1] / cw[0]))

    ref = paper_reference(args.reference or ROOT / "results" / f"shortcut_{dataset}_bakeoff.json",
                          method, seed)
    stored = meta.get("metrics", {})
    print(f"\n{dataset} | {method} | p = {p} | seed {seed}"
          + (f" | {len(te)} test images (smoke test)" if args.limit else ""))
    print(f"  {'metric':<12}{'this run':>10}{'at training':>13}{'paper':>9}")
    for m in METRICS:
        cells = [metrics.get(m), stored.get(m), (ref or {}).get(m)]
        print(f"  {m:<12}" + "".join(f"{'' if v is None else f'{v:.4f}':>{w}}"
                                      for v, w in zip(cells, (10, 13, 9))))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": {k: v for k, v in meta.items() if k != "config"},
                                              "limit": args.limit, "metrics": metrics,
                                              "at_training": stored, "paper": ref}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
