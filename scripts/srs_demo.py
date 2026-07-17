"""Live SRS demo: shows exactly what the Shortcut Reliance Score does.

No GPU, no model download. Runs in seconds on real HAM10000 metadata.

It does three things a supervisor can watch live:
  1. Build a REAL (concordant, flipped) synthetic-note pair for a real lesion and
     show which sentence flips while the patient facts stay identical.
  2. Compute SRS for two hypothetical models on that pair - a "shortcut" model
     that copies the note (SRS -> 1) and an "honest" model that uses the image
     (SRS -> 0) - so the metric's meaning is concrete.
  3. Print the REAL measured SRS numbers from our 5-seed experiments
     (dose-response + mitigation bake-off) straight from the result JSONs.

Usage:
    python scripts/srs_demo.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Make the demo runnable from anywhere (repo root, scripts/, double-click) by
# putting the project root on sys.path before importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.synthetic_text import generate_pair, DFILT, facts_only
from src.evaluation.shortcut import srs_from_probs

ROOT = Path(__file__).resolve().parents[1]
HAM_META = ROOT / "data" / "raw" / "ham10000" / "HAM10000_metadata.csv"
DOSE_JSON = ROOT / "results" / "shortcut_ham10000_dose.json"
BAKEOFF_JSON = ROOT / "results" / "shortcut_ham10000_bakeoff.json"

RULE = "=" * 70


def _word_diff(a: str, b: str):
    """Return the trailing clause that differs between two notes (the leak)."""
    aw, bw = a.split(), b.split()
    i = 0
    while i < min(len(aw), len(bw)) and aw[i] == bw[i]:
        i += 1
    return " ".join(aw[i:]), " ".join(bw[i:])


def demo_pair(row, title):
    """Show one real note pair and the SRS it produces for two model types."""
    concordant, flipped = generate_pair(row, level=DFILT, seed=42)
    shared, leaked_c = _word_diff(flipped, concordant)  # common prefix vs leak

    print(RULE)
    print(f"{title}  (true label: {'MALIGNANT' if _is_mal(row) else 'benign'})")
    print(RULE)
    print("Patient FACTS (identical in both notes, survive FFilt):")
    print(f"   {facts_only(row, seed=42)}")
    print()
    print("CONCORDANT note (leak matches the TRUE label):")
    print(f"   ...{_leak_clause(concordant)}")
    print("FLIPPED note (same facts, leak says the OPPOSITE):")
    print(f"   ...{_leak_clause(flipped)}")
    print()

    # Two hypothetical models scored on (concordant, flipped) for this lesion.
    # Shortcut model: prediction follows the words -> high on concordant, low on flipped.
    p_orig_shortcut, p_flip_shortcut = [0.97], [0.03]
    # Honest model: looks at the image, ignores the words -> same both times.
    p_orig_honest, p_flip_honest = [0.88], [0.86]

    s_short = srs_from_probs(p_orig_shortcut, p_flip_shortcut)
    s_honest = srs_from_probs(p_orig_honest, p_flip_honest)
    print("If a SHORTCUT model sees this pair:")
    print(f"   p(concordant)=0.97  p(flipped)=0.03  ->  SRS = {s_short['srs']:.2f}  (BAD: reads the note)")
    print("If an HONEST model sees this pair:")
    print(f"   p(concordant)=0.88  p(flipped)=0.86  ->  SRS = {s_honest['srs']:.2f}  (GOOD: uses the image)")
    print()


def _is_mal(row):
    return str(row["dx"]).lower() in {"mel", "bcc", "akiec"}


def _leak_clause(note):
    """Heuristic: show the last two sentences (the verdict + leading clause)."""
    parts = [s.strip() for s in note.split(".") if s.strip()]
    return ". ".join(parts[-2:]) + "."


def real_numbers():
    """Print the REAL measured SRS from the 5-seed experiment JSONs."""
    dose = pd.DataFrame(json.loads(DOSE_JSON.read_text()))
    bake = pd.DataFrame(json.loads(BAKEOFF_JSON.read_text()))

    print(RULE)
    print("REAL measured SRS - dose-response (naive model, mean over 5 seeds)")
    print(RULE)
    g = dose.groupby("leakage_p").agg(
        honest_auroc=("auroc_ffilt", "mean"),
        visible_auroc=("auroc", "mean"),
        srs=("srs", "mean"),
    )
    print(f"{'leakage p':>10} {'visible AUROC':>14} {'honest AUROC':>14} {'SRS':>8}")
    for p, r in g.iterrows():
        print(f"{p:>10.1f} {r.visible_auroc:>14.3f} {r.honest_auroc:>14.3f} {r.srs:>8.3f}")
    print("  -> as p rises, visible AUROC looks great but honest AUROC collapses")
    print("     and SRS climbs to ~1.0: the model abandoned the image.")
    print()

    print(RULE)
    print("REAL measured SRS - mitigation bake-off (p=1.0, mean over 5 seeds)")
    print(RULE)
    gb = bake.groupby("method").agg(
        honest_auroc=("auroc_ffilt", "mean"),
        srs=("srs", "mean"),
    ).reindex(["naive", "ffilt_only", "counterfactual_aug", "consistency", "adversarial"])
    print(f"{'method':>20} {'honest AUROC':>14} {'SRS':>8}")
    for m, r in gb.iterrows():
        print(f"{m:>20} {r.honest_auroc:>14.3f} {r.srs:>8.3f}")
    print("  -> counterfactual_aug gives the best honest AUROC AND near-zero SRS,")
    print("     beating Watson-style filtering (ffilt_only).")


def main():
    meta = pd.read_csv(HAM_META)
    mal = meta[meta["dx"] == "mel"].iloc[0]
    ben = meta[meta["dx"] == "nv"].iloc[0]

    print("\nSHORTCUT RELIANCE SCORE (SRS) - LIVE DEMO\n")
    demo_pair(mal, "Example 1: a melanoma")
    demo_pair(ben, "Example 2: a benign naevus")
    real_numbers()
    print()


if __name__ == "__main__":
    main()
