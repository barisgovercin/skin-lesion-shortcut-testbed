"""Summarise near-duplicate leakage from the shipped duplicate-check outputs.

Reads results/split_checks/<dataset>_cross_split_candidates.csv (written by
scripts/check_splits.py) and the split manifests, and reports, per cosine
threshold, how many test images have a near-identical twin in the training set
(mean and range over the seeds). No images are needed.

    python scripts/summarize_duplicates.py

Writes results/split_checks/duplicates_summary.json and .md.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (PREPARE, SEEDS, load_manifest, manifest_path,  # noqa: E402
                             test_twin_counts)

THRESHOLDS = (0.99, 0.98, 0.97, 0.95)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+", default=list(PREPARE))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--checks", default=str(ROOT / "results" / "split_checks"))
    args = ap.parse_args(argv)

    checks = Path(args.checks)
    summary = {}
    lines = ["# Near-duplicate leakage: test images with a near-identical twin in train", "",
             "Cosine similarity of ImageNet ConvNeXt-Base embeddings; mean over seeds "
             "(min to max in brackets).", ""]
    for ds in args.datasets:
        pairs = pd.read_csv(checks / f"{ds}_cross_split_candidates.csv",
                            usecols=["a", "b", "similarity"], dtype={"a": str, "b": str})
        per_seed = [test_twin_counts(pairs, load_manifest(manifest_path(args.splits, ds, s)),
                                     THRESHOLDS) for s in args.seeds]
        summary[ds] = {}
        lines += [f"## {ds}", "", "| cosine >= | train-test pairs | test images with a train twin "
                  "| share of the test set |", "|---|---|---|---|"]
        for t in THRESHOLDS:
            vals = {k: [r[t][k] for r in per_seed] for k in per_seed[0][t]}
            mean = {k: sum(v) / len(v) for k, v in vals.items()}
            summary[ds][str(t)] = {f"{k}_mean": m for k, m in mean.items()}
            summary[ds][str(t)].update({f"{k}_range": [min(v), max(v)] for k, v in vals.items()})
            frac = vals["test_fraction"]
            lines.append(f"| {t} | {mean['train_test_pairs']:.1f} | {mean['test_images']:.1f} "
                         f"({min(vals['test_images'])} to {max(vals['test_images'])}) | "
                         f"{100 * mean['test_fraction']:.2f}% "
                         f"({100 * min(frac):.2f} to {100 * max(frac):.2f}%) |")
        lines.append("")
    (checks / "duplicates_summary.json").write_text(json.dumps(summary, indent=2))
    (checks / "duplicates_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
