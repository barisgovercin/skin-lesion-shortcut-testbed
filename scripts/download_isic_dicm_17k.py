"""
Download the ISIC-DICM-17K dataset (Ahammed et al., CVPRW 2025).

The GitHub repo (mmu-dermatology-research/isic-dicm-17k) ships only the
train/valid metadata CSVs (image IDs + clinical metadata). The actual
dermoscopic images live in the public ISIC Archive S3 bucket and are fetched
here by their deterministic URL:

    https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg

Images are the curated/resized ISIC-DICM versions (~30-80 KB each, ~1 GB total),
so no resize step is needed — the training pipeline resizes to 224 anyway.

Usage:
    python scripts/download_isic_dicm_17k.py                 # download all 17,060
    python scripts/download_isic_dicm_17k.py --workers 24
    python scripts/download_isic_dicm_17k.py --limit 50      # smoke test
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "isic_dicm_17k"
IMG_DIR = DATA_DIR / "images"
S3_URL = "https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"


def load_ids() -> list[str]:
    """Union of isic_ids across the train and valid metadata CSVs."""
    ids: set[str] = set()
    for name in ("train-set-metadata.csv", "valid-set-metadata.csv"):
        df = pd.read_csv(DATA_DIR / name)
        ids.update(df["isic_id"].dropna().astype(str).tolist())
    return sorted(ids)


def download_one(isic_id: str, session: requests.Session, retries: int = 3) -> tuple[str, str]:
    """Download a single image; returns (isic_id, status)."""
    dest = IMG_DIR / f"{isic_id}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return isic_id, "skip"
    url = S3_URL.format(isic_id=isic_id)
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return isic_id, "ok"
            return isic_id, f"http_{r.status_code}"
        except requests.RequestException:
            if attempt == retries - 1:
                return isic_id, "error"
            time.sleep(1.5 * (attempt + 1))
    return isic_id, "error"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download ISIC-DICM-17K images from the ISIC Archive.")
    ap.add_argument("--workers", type=int, default=16, help="Concurrent download threads")
    ap.add_argument("--limit", type=int, default=0, help="Download only the first N ids (smoke test)")
    args = ap.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ids = load_ids()
    if args.limit:
        ids = ids[: args.limit]
    print(f"[ISIC-DICM-17K] {len(ids)} unique image ids -> {IMG_DIR}", flush=True)

    counts = {"ok": 0, "skip": 0, "error": 0}
    http_errs: list[str] = []
    t0 = time.time()

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, i, session) for i in ids]
        for n, fut in enumerate(as_completed(futures), 1):
            isic_id, status = fut.result()
            if status in counts:
                counts[status] += 1
            else:  # http_XXX
                counts["error"] += 1
                http_errs.append(f"{isic_id}:{status}")
            if n % 500 == 0 or n == len(ids):
                rate = n / max(time.time() - t0, 1e-6)
                print(
                    f"  {n}/{len(ids)}  ok={counts['ok']} skip={counts['skip']} "
                    f"err={counts['error']}  ({rate:.0f}/s)",
                    flush=True,
                )

    dt = time.time() - t0
    print(f"[DONE] ok={counts['ok']} skip={counts['skip']} err={counts['error']} in {dt/60:.1f} min", flush=True)
    if http_errs:
        print(f"[WARN] {len(http_errs)} failed ids (first 20): {http_errs[:20]}", flush=True)
        (DATA_DIR / "failed_ids.txt").write_text("\n".join(http_errs))
    n_files = len(list(IMG_DIR.glob("*.jpg")))
    print(f"[VERIFY] {n_files} jpg files on disk in {IMG_DIR}", flush=True)


if __name__ == "__main__":
    main()
