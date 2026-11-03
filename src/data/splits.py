"""Saved split manifests and leakage / duplicate checks for the three datasets.

A manifest is one row per image: image_id, group_id (lesion or patient), label,
split. Manifests are built with the same prepare_* functions that the training
runs use, so they are the exact partitions behind the paper's results.

Duplicate detection follows two published practices on these archives: exact
file duplicates (Cassidy et al., Med. Image Anal. 2022, ISIC) and near
duplicates by embedding cosine similarity with manual review of the candidate
pairs (Abhishek et al., Sci. Data 2025, HAM10000).
"""
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import (find_image_path, prepare_ham10000, prepare_isic2020,
                              prepare_isic_dicm_17k)

SPLITS = ("train", "val", "test")
SEEDS = [42, 123, 456, 789, 1024]   # the paper's five seeds
PREPARE = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020,
           "isic_dicm_17k": prepare_isic_dicm_17k}
GROUP_COL = {"ham10000": "lesion_id", "isic2020": "patient_id",
             "isic_dicm_17k": "lesion_id"}

# (train, val, test) image counts printed by the paper's training runs, per seed.
PAPER_SPLIT_SIZES = {
    "ham10000": {42: (5988, 1985, 2042), 123: (6023, 1979, 2013),
                 456: (5995, 2003, 2017), 789: (5954, 2002, 2059),
                 1024: (6044, 1971, 2000)},
    "isic2020": {42: (22732, 7699, 7217), 123: (22454, 7353, 7841),
                 456: (22404, 7441, 7803), 789: (23066, 7099, 7483),
                 1024: (22663, 7067, 7918)},
    "isic_dicm_17k": {42: (10202, 3350, 3508), 123: (10201, 3435, 3424),
                      456: (10322, 3326, 3412), 789: (10156, 3342, 3562),
                      1024: (10300, 3345, 3415)},
}


# ---- Manifests ---------------------------------------------------------------------

def build_manifest(dataset, data_dir, seed):
    """Return (manifest, img_dirs) for one dataset and seed."""
    train, val, test, img_dirs, _ = PREPARE[dataset](data_dir=data_dir, seed=seed)
    group = GROUP_COL[dataset]
    frames = [df.assign(split=name) for name, df in zip(SPLITS, (train, val, test))]
    manifest = (pd.concat(frames, ignore_index=True)
                [["image_id", group, "label", "split"]]
                .rename(columns={group: "group_id"}))
    manifest["label"] = manifest["label"].astype(int)
    return manifest, img_dirs


def image_paths(manifest, img_dirs):
    """image_id -> file path, searching img_dirs as the training loaders do."""
    return {i: find_image_path(img_dirs, i) for i in manifest["image_id"]}


def manifest_path(manifest_dir, dataset, seed):
    return Path(manifest_dir) / f"{dataset}_seed{seed}.csv"


def _canonical(manifest):
    """Stable row order and dtypes, so manifests compare and diff cleanly."""
    m = manifest.astype({"image_id": str, "group_id": str, "label": int, "split": str})
    order = m["split"].map({s: i for i, s in enumerate(SPLITS)})
    return m.assign(_o=order).sort_values(["_o", "image_id"]).drop(columns="_o") \
            .reset_index(drop=True)


def load_manifest(path):
    return _canonical(pd.read_csv(path, dtype={"image_id": str, "group_id": str}))


def write_manifests(dataset, data_dir, out_dir, seeds=SEEDS):
    """Write one manifest CSV per seed; returns the written paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        manifest, _ = build_manifest(dataset, data_dir, seed)
        path = manifest_path(out_dir, dataset, seed)
        _canonical(manifest).to_csv(path, index=False)
        written.append(path)
    return written


def verify_manifests(dataset, data_dir, manifest_dir, seeds=SEEDS):
    """Rebuild the splits from local data and compare with the shipped manifests."""
    report = {}
    for seed in seeds:
        rebuilt = _canonical(build_manifest(dataset, data_dir, seed)[0])
        shipped = load_manifest(manifest_path(manifest_dir, dataset, seed))
        joined = rebuilt.merge(shipped, on="image_id", how="outer", indicator=True,
                               suffixes=("", "_shipped"))
        differs = (joined["_merge"] != "both") | \
                  (joined["split"] != joined["split_shipped"]) | \
                  (joined["group_id"] != joined["group_id_shipped"]) | \
                  (joined["label"] != joined["label_shipped"])
        expected = PAPER_SPLIT_SIZES.get(dataset, {}).get(seed)
        report[seed] = {
            "identical": bool(not differs.any()),
            "n_differences": int(differs.sum()),
            "paper_sizes": compare_sizes(rebuilt, expected) if expected else None,
            "leakage": check_manifest(rebuilt),
        }
    return report


# ---- Leakage checks ----------------------------------------------------------------

def check_manifest(manifest):
    """ID-level leakage checks plus split sizes and malignant ratios."""
    per_image = manifest.groupby("image_id")["split"].nunique()
    per_group = manifest.groupby("group_id")["split"].nunique()
    return {
        "images_in_multiple_splits": int((per_image > 1).sum()),
        "duplicate_rows": int(manifest["image_id"].duplicated().sum()),
        "groups_in_multiple_splits": int((per_group > 1).sum()),
        "sizes": {s: int((manifest["split"] == s).sum()) for s in SPLITS},
        "malignant": {s: float(manifest.loc[manifest["split"] == s, "label"].mean())
                      for s in SPLITS},
    }


def compare_sizes(manifest, expected):
    """Compare (train, val, test) counts with the sizes logged by the paper's runs."""
    actual = tuple(int((manifest["split"] == s).sum()) for s in SPLITS)
    return {"expected": tuple(expected), "actual": actual, "match": actual == tuple(expected)}


# ---- Duplicate detection -------------------------------------------------------------

def _md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_duplicate_pairs(paths):
    """Pairs of image ids whose files are byte-identical."""
    by_hash = {}
    for image_id, path in paths.items():
        by_hash.setdefault(_md5(path), []).append(image_id)
    pairs = [tuple(sorted(p)) for ids in by_hash.values() if len(ids) > 1
             for p in combinations(ids, 2)]
    return pd.DataFrame(sorted(pairs), columns=["a", "b"])


def near_duplicate_pairs(ids, embeddings, threshold=0.90, chunk=2048, max_pairs=2_000_000):
    """Pairs (a, b) with cosine similarity >= threshold, each pair once, no self pairs.

    Raises ValueError rather than exhausting memory if more than `max_pairs`
    pairs pass the threshold.
    """
    emb = np.asarray(embeddings, np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.asarray(ids)
    rows, n_pairs = [], 0
    for start in range(0, len(emb), chunk):
        block = emb[start:start + chunk] @ emb[start:].T
        upper = np.arange(block.shape[1])[None, :] > np.arange(block.shape[0])[:, None]
        r, c = np.nonzero((block >= threshold) & upper)
        n_pairs += len(r)
        if n_pairs > max_pairs:
            raise ValueError(f"more than {max_pairs} pairs have cosine >= {threshold}; "
                             "raise the threshold")
        rows.append(pd.DataFrame({"a": ids[start + r], "b": ids[start + c],
                                  "similarity": block[r, c].astype(float)}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["a", "b", "similarity"])
    return out.sort_values("similarity", ascending=False, kind="stable").reset_index(drop=True)


def cross_split_pairs(pairs, manifest):
    """Keep only pairs whose two images sit in different splits (potential leakage)."""
    info = manifest.set_index("image_id")
    out = pairs.copy()
    for side in ("a", "b"):
        out[f"split_{side}"] = out[side].map(info["split"])
        out[f"group_{side}"] = out[side].map(info["group_id"])
    return out[out["split_a"] != out["split_b"]].reset_index(drop=True)


def test_twin_counts(pairs, manifest, thresholds=(0.99, 0.98, 0.97, 0.95)):
    """Per threshold: train-test pairs, and test images that have a twin in train."""
    split = manifest.set_index("image_id")["split"]
    n_test = int((manifest["split"] == "test").sum())
    out = {}
    for t in thresholds:
        p = pairs[pairs["similarity"] >= t]
        sa, sb = p["a"].map(split), p["b"].map(split)
        tt = p[((sa == "train") & (sb == "test")) | ((sa == "test") & (sb == "train"))]
        test_images = set(tt["a"][tt["a"].map(split) == "test"]) | \
            set(tt["b"][tt["b"].map(split) == "test"])
        out[t] = {"train_test_pairs": int(len(tt)), "test_images": len(test_images),
                  "test_fraction": len(test_images) / n_test}
    return out


class _ImageList:
    """Map-style dataset of image files (module level so worker processes can pickle it)."""

    def __init__(self, files, transform):
        self.files, self.transform = files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.open(self.files[i]).convert("RGB"))


def duplicate_report(manifests, exact_pairs, near_pairs, thresholds=(0.90, 0.95, 0.98)):
    """Per seed: how many exact / near-duplicate pairs fall in different splits."""
    report = {}
    for seed, manifest in manifests.items():
        near = cross_split_pairs(near_pairs, manifest)
        report[seed] = {
            "exact_cross_split": int(len(cross_split_pairs(exact_pairs, manifest))),
            "near_cross_split": {t: int((near["similarity"] >= t).sum()) for t in thresholds},
        }
    return report


def embed_images(paths, model=None, batch_size=64, device=None, num_workers=0):
    """L2-normalised image embeddings, in the order of `paths`.

    Default model: ImageNet ConvNeXt-Base with the classifier removed (1024-d),
    the same backbone family as the paper's image encoder.
    """
    import torch
    import torchvision
    from torchvision import transforms

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = torchvision.models.convnext_base(weights="IMAGENET1K_V1")
        model.classifier[2] = torch.nn.Identity()
    model = model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ids = list(paths)
    loader = torch.utils.data.DataLoader(_ImageList([paths[i] for i in ids], tf),
                                         batch_size=batch_size, num_workers=num_workers)
    feats = []
    with torch.no_grad():
        for x in loader:
            f = model(x.to(device)).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return ids, np.concatenate(feats)


def save_review_sheet(pairs, paths, out_png, max_pairs=40):
    """Side-by-side thumbnails of candidate pairs, for manual review."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    pairs = pairs.head(max_pairs)
    n = max(len(pairs), 1)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n), squeeze=False)
    for ax_row, (_, p) in zip(axes, pairs.iterrows()):
        for ax, side in zip(ax_row, ("a", "b")):
            ax.imshow(Image.open(paths[p[side]]).convert("RGB"))
            ax.set_title(f"{p[side]} ({p.get(f'split_{side}', '')})", fontsize=7)
            ax.axis("off")
        ax_row[0].text(1.0, -0.08, f"cos = {p['similarity']:.3f}", fontsize=7,
                       ha="center", transform=ax_row[0].transAxes)
    if pairs.empty:
        axes[0, 0].text(0.5, 0.5, "no candidate pairs", ha="center")
        for ax in axes[0]:
            ax.axis("off")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
