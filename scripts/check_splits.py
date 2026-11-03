"""Leakage and duplicate-image checks on the saved split manifests.

Without --data-dir (nothing to download): ID-level checks on the shipped
manifests, i.e. no image and no lesion/patient appears in two splits, plus split
sizes and malignant ratios.

With --data-dir (images needed): also exact duplicates (identical files, md5) and
near duplicates (cosine similarity of ImageNet ConvNeXt-Base embeddings), counted
per seed for pairs whose two images fall in different splits, with a review sheet
of the most similar cross-split pairs for visual inspection.

    python scripts/check_splits.py --dataset all
    python scripts/check_splits.py --dataset ham10000 --data-dir data/raw/ham10000

Writes <out>/<dataset>.json and .md, and with images also
<dataset>_cross_split_candidates.csv and <dataset>_review.png.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (PREPARE, SEEDS, build_manifest, check_manifest,  # noqa: E402
                             cross_split_pairs, duplicate_report, embed_images,
                             exact_duplicate_pairs, image_paths, load_manifest,
                             manifest_path, near_duplicate_pairs, save_review_sheet)


def _image_checks(ds, manifests, args, out, embed_model=None):
    """Exact + near-duplicate checks; returns (summary, per-seed report)."""
    first = manifests[args.seeds[0]]
    info = first.set_index("image_id")
    _, img_dirs = build_manifest(ds, args.data_dir, args.seeds[0])
    paths = image_paths(first, img_dirs)
    print(f"[{ds}] hashing and embedding {len(paths)} images")
    exact = exact_duplicate_pairs(paths)
    ids, emb = embed_images(paths, model=embed_model, batch_size=args.batch_size,
                            device=args.device, num_workers=args.workers)
    threshold = args.threshold
    while True:
        try:
            near = near_duplicate_pairs(ids, emb, threshold=threshold)
            break
        except ValueError:
            threshold = round(threshold + 0.02, 2)
            print(f"[{ds}] too many near pairs; raising the threshold to {threshold}")
    thresholds = tuple(sorted({threshold} | {t for t in (0.95, 0.98, 0.99) if t > threshold}))
    per_seed = duplicate_report(manifests, exact, near, thresholds)

    # Near pairs that fall in different splits in at least one seed, most similar first.
    crossing = {}
    for seed, m in manifests.items():
        for a, b in cross_split_pairs(near, m)[["a", "b"]].itertuples(index=False):
            crossing.setdefault((a, b), []).append(str(seed))
    cand = near[[(a, b) in crossing for a, b in near[["a", "b"]].itertuples(index=False)]].copy()
    cand["crossing_seeds"] = [" ".join(crossing[(a, b)]) for a, b in cand[["a", "b"]].itertuples(index=False)]
    for side in ("a", "b"):
        cand[f"split_{side}"] = cand[side].map(info["split"])     # splits of the first seed
        cand[f"group_{side}"] = cand[side].map(info["group_id"])
    cand.to_csv(out / f"{ds}_cross_split_candidates.csv", index=False)
    save_review_sheet(cand, paths, out / f"{ds}_review.png", max_pairs=args.review_pairs)

    same_group = near["a"].map(info["group_id"]) == near["b"].map(info["group_id"])
    summary = {"n_images": len(paths), "exact_duplicate_pairs": int(len(exact)),
               "near_threshold": threshold, "near_duplicate_pairs": int(len(near)),
               "near_pairs_same_group": int(same_group.sum()),
               "cross_split_candidates_any_seed": int(len(cand))}
    return summary, per_seed


def _markdown(ds, report):
    lines = [f"# Split checks: {ds}", "",
             "| seed | train / val / test | malignant (train / val / test) | "
             "images in 2+ splits | lesions/patients in 2+ splits |",
             "|---|---|---|---|---|"]
    for seed, r in report["seeds"].items():
        lk = r["leakage"]
        sz, mal = lk["sizes"], lk["malignant"]
        lines.append(f"| {seed} | {sz['train']} / {sz['val']} / {sz['test']} | "
                     f"{mal['train']:.3f} / {mal['val']:.3f} / {mal['test']:.3f} | "
                     f"{lk['images_in_multiple_splits']} | {lk['groups_in_multiple_splits']} |")
    img = report.get("images")
    if img:
        lines += ["", f"Images checked: {img['n_images']}. Exact duplicate pairs: "
                  f"{img['exact_duplicate_pairs']}. Near-duplicate pairs (cosine >= "
                  f"{img['near_threshold']}): {img['near_duplicate_pairs']}, of which "
                  f"{img['near_pairs_same_group']} share a lesion/patient id.", "",
                  "| seed | exact pairs across splits | near pairs across splits, by cosine |",
                  "|---|---|---|"]
        for seed, r in report["seeds"].items():
            near = ", ".join(f">= {t}: {n}" for t, n in r["duplicates"]["near_cross_split"].items())
            lines.append(f"| {seed} | {r['duplicates']['exact_cross_split']} | {near} |")
    return "\n".join(lines) + "\n"


def main(argv=None, embed_model=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=[*PREPARE, "all"], required=True)
    ap.add_argument("--manifests", default=str(ROOT / "splits"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--data-dir", help="dataset folder; enables the image duplicate checks")
    ap.add_argument("--out", default=str(ROOT / "results" / "split_checks"))
    ap.add_argument("--threshold", type=float, default=0.90, help="near-duplicate cosine")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--review-pairs", type=int, default=40)
    args = ap.parse_args(argv)

    datasets = list(PREPARE) if args.dataset == "all" else [args.dataset]
    if args.data_dir and len(datasets) > 1:
        ap.error("--data-dir needs a single --dataset")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        manifests = {s: load_manifest(manifest_path(args.manifests, ds, s)) for s in args.seeds}
        report = {"dataset": ds,
                  "seeds": {str(s): {"leakage": check_manifest(m)} for s, m in manifests.items()}}
        if args.data_dir:
            summary, per_seed = _image_checks(ds, manifests, args, out, embed_model)
            report["images"] = summary
            for s, r in per_seed.items():
                r["near_cross_split"] = {str(t): n for t, n in r["near_cross_split"].items()}
                report["seeds"][str(s)]["duplicates"] = r
        (out / f"{ds}.json").write_text(json.dumps(report, indent=2))
        md = _markdown(ds, report)
        (out / f"{ds}.md").write_text(md)
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
