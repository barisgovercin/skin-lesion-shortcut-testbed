"""Pre-generate and cache LLM (or mock) clinical notes for the shortcut testbed.

Separates the (API-costed) note generation from GPU training: run this once per
(dataset, level, seed) to populate data/llm_notes_cache/, then run the shortcut
sweep with `data.text_source: llm` (configs/shortcut_ham10000_llm.yaml). Because
add_llm_text_columns caches on disk, the later training run reuses these notes
for free and is fully reproducible.

Examples:
  # offline dry run (no API key, deterministic mock):
  python scripts/generate_llm_notes.py --dataset ham10000 --level DFilt --seeds 42 --provider mock

  # real notes (needs: pip install anthropic ; setx ANTHROPIC_API_KEY ...):
  python scripts/generate_llm_notes.py --dataset ham10000 --level DFilt \
      --seeds 42 123 456 789 1024 --provider anthropic
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import prepare_ham10000, prepare_isic2020, prepare_isic_dicm_17k
from src.preprocessing.llm_text import add_llm_text_columns, DEFAULT_MODEL

PREP = {"ham10000": prepare_ham10000, "isic2020": prepare_isic2020,
        "isic_dicm_17k": prepare_isic_dicm_17k}
DATA_DIR = {"ham10000": "data/raw/ham10000", "isic2020": "data/raw/isic2020",
            "isic_dicm_17k": "data/raw/isic_dicm_17k"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PREP), default="ham10000")
    ap.add_argument("--level", default="DFilt", choices=["Orig", "CFilt", "DFilt", "FFilt"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--leakage-p", type=float, default=1.0)
    ap.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    prep = PREP[args.dataset]
    grand_total = 0
    for seed in args.seeds:
        tr, va, te, _, _ = prep(data_dir=DATA_DIR[args.dataset], seed=seed)
        seed_total = 0
        for split_name, df in [("train", tr), ("val", va), ("test", te)]:
            out = add_llm_text_columns(
                df, level=args.level, seed=seed, leakage_p=args.leakage_p,
                pair=(split_name != "val"), provider=args.provider, model=args.model)
            seed_total += len(out)
            if split_name == "train":
                print(f"[seed {seed}] sample {args.level} notes ({args.provider}):")
                for note in out["text"].head(3).tolist():
                    print(f"    - {note}")
        grand_total += seed_total
        print(f"[seed {seed}] generated/cached notes across train+val+test: {seed_total} rows")
    print(f"Done. {grand_total} note rows across {len(args.seeds)} seed(s) "
          f"-> cache in data/llm_notes_cache/ (provider={args.provider}).")


if __name__ == "__main__":
    main()
