"""
Download HAM10000 (Tschandl et al., Sci. Data 2018) from the official Harvard
Dataverse record, doi:10.7910/DVN/DBW86T (version 4), into data/raw/ham10000:

    HAM10000_metadata.csv
    HAM10000_images_part_1/*.jpg
    HAM10000_images_part_2/*.jpg

The images (about 2.8 GB) are licensed CC BY-NC 4.0. The metadata CSV alone is
enough to rebuild and verify the split manifests (scripts/make_splits.py --verify).
The Kaggle copy kmader/skin-cancer-mnist-ham10000 gives identical splits.

Usage:
    python scripts/download_ham10000.py                  # metadata + images
    python scripts/download_ham10000.py --metadata-only  # the CSV only
"""

import argparse
import io
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://dataverse.harvard.edu/api/access/datafile/{}"
METADATA_FILE_ID = 4338392                 # HAM10000_metadata (served as .tab; ?format=original -> CSV)
IMAGE_ZIPS = {"HAM10000_images_part_1": 3172585, "HAM10000_images_part_2": 3172584}


def fetch(file_id: int, original: bool = False) -> bytes:
    url = API.format(file_id) + ("?format=original" if original else "")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content


def main() -> None:
    ap = argparse.ArgumentParser(description="Download HAM10000 from Harvard Dataverse.")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "ham10000"))
    ap.add_argument("--metadata-only", action="store_true", help="Fetch the metadata CSV and stop")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = data_dir / "HAM10000_metadata.csv"
    if not meta.exists():
        meta.write_bytes(fetch(METADATA_FILE_ID, original=True))
    print(f"[HAM10000] metadata: {meta} ({sum(1 for _ in open(meta)) - 1} images)", flush=True)
    if args.metadata_only:
        return

    for folder, file_id in IMAGE_ZIPS.items():
        target = data_dir / folder
        if target.exists() and any(target.glob("*.jpg")):
            print(f"[HAM10000] {folder}: already present, skipping", flush=True)
            continue
        target.mkdir(exist_ok=True)
        print(f"[HAM10000] downloading {folder} (about 1.4 GB)", flush=True)
        with zipfile.ZipFile(io.BytesIO(fetch(file_id))) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".jpg"):        # flatten any folder inside the zip
                    (target / Path(name).name).write_bytes(zf.read(name))
        print(f"[HAM10000] {folder}: {len(list(target.glob('*.jpg')))} images", flush=True)


if __name__ == "__main__":
    main()
