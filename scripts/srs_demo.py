"""GPU-free demo of the Shortcut Reliance Score (SRS) and the counterfactual flip test.

Runs in about 10 seconds on a standard CPU and needs only numpy, pandas,
scikit-learn and scipy: no GPU, no dataset download, no model weights.

SRS is the mean absolute change in the predicted probability of malignancy when
only the diagnosis-bearing text of the clinical note is flipped, with the same
image and the same patient facts. It applies the counterfactual token fairness gap
of Garg et al. (2019) to diagnosis-bearing clinical text.

The demo uses TOY data (synthetic patients and a synthetic image signal) to show
the mechanism with the paper's own note engine and metric code. It does not
reproduce the paper's numbers; those come from the real 5-seed experiments and are
printed in section 4 from results/shortcut_*_bakeoff.json.

    [1] The flip test: one patient, original vs flipped note, automatic checks
    [2] Known-answer controls: image-only model (SRS = 0), text-only model (SRS ~ 1)
    [3] Mini testbed: dose-response over leakage p, mitigation at p = 1.0
    [4] Paper results: ConvNeXt-Base + BioClinicalBERT, 3 datasets, 5 seeds

Usage:
    python scripts/srs_demo.py
"""

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Make the demo runnable from anywhere by putting the project root on sys.path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.shortcut import ffilt_gap_from_auroc, srs_from_probs  # noqa: E402
from src.preprocessing.synthetic_text import (DFILT, FFILT, MALIGNANT_DX,  # noqa: E402
                                              add_text_columns, build_counterfactual_df,
                                              facts_only, generate_pair)

SEEDS = [42, 123, 456, 789, 1024]          # the paper's five seeds
LEAKAGE_PS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TEXT_LEVEL = DFILT                          # the paper's testbed level (leading language kept)
METHODS = ["naive", "ffilt_only", "counterfactual_aug"]

# HAM10000 class and site counts (Tschandl et al. 2018), used as sampling weights.
DX_COUNTS = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327,
             "vasc": 142, "df": 115}
SITE_COUNTS = {"back": 2192, "lower extremity": 2077, "trunk": 1404,
               "upper extremity": 1118, "abdomen": 1022, "face": 745, "chest": 407,
               "foot": 319, "unknown": 234, "neck": 168, "scalp": 128, "hand": 90,
               "ear": 56, "genital": 48, "acral": 7}

# Toy image: 16 features, 4 of which shift with malignancy (image-only AUROC ~0.84).
IMG_DIM, IMG_SIGNAL_DIMS, IMG_SHIFT = 16, 4, 0.75

PAPER_DATASETS = {"ham10000": "HAM10000", "isic2020": "ISIC 2020",
                  "isic_dicm_17k": "ISIC-DICM-17K"}
PAPER_METHODS = ["naive", "ffilt_only", "counterfactual_aug", "consistency", "adversarial"]

RULE = "=" * 78
WIDTH = 78


# ---- Toy data ------------------------------------------------------------------

def make_patients(n, seed):
    """Synthetic HAM10000-like patients plus a toy image feature matrix.

    Returns (df, img): df has one row per lesion with a `row` column indexing img.
    """
    rng = np.random.default_rng(seed)

    def draw(counts):
        p = np.array(list(counts.values()), float)
        return rng.choice(list(counts), size=n, p=p / p.sum())

    dx = draw(DX_COUNTS)
    label = np.isin(dx, list(MALIGNANT_DX)).astype(int)
    img = rng.normal(size=(n, IMG_DIM))
    img[:, :IMG_SIGNAL_DIMS] += label[:, None] * IMG_SHIFT
    df = pd.DataFrame({
        "row": np.arange(n),
        "image_id": [f"TOY_{seed}_{i:05d}" for i in range(n)],
        "dx": dx,
        "age": np.clip(np.round(rng.normal(52, 17, n) / 5) * 5, 5, 85).astype(int),
        "sex": rng.choice(["male", "female"], size=n, p=[0.54, 0.46]),
        "localization": draw(SITE_COUNTS),
        "label": label,
    })
    return df, img


def split(df, seed):
    """The paper's 60/20/20 stratified split; the toy model needs no validation set."""
    train, rest = train_test_split(np.arange(len(df)), test_size=0.4,
                                   stratify=df["label"], random_state=seed)
    _, test = train_test_split(rest, test_size=0.5, stratify=df["label"].iloc[rest],
                               random_state=seed)
    return train, test


class FusionModel:
    """Toy late fusion: [image features | TF-IDF of the note] -> logistic regression.

    `inputs` selects "fusion", "image" (never reads the note) or "text" (never sees
    the image). Class weighting is balanced, as in the paper.
    """

    def __init__(self, inputs="fusion"):
        self.inputs = inputs

    def _features(self, img, text):
        parts = []
        if self.inputs in ("fusion", "image"):
            parts.append(sparse.csr_matrix(img))
        if self.inputs in ("fusion", "text"):
            parts.append(self.vectorizer.transform(text))
        return sparse.hstack(parts).tocsr()

    def fit(self, img, text, y):
        if self.inputs in ("fusion", "text"):
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(text)
        self.clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        self.clf.fit(self._features(img, text), y)
        return self

    def predict(self, img, text):
        return self.clf.predict_proba(self._features(img, text))[:, 1]


def run(df, img, seed, leakage_p, method, inputs="fusion"):
    """Train one toy model and score it, mirroring scripts/run_shortcut.py::run_seed."""
    train_idx, test_idx = split(df, seed)
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    if method == "ffilt_only":
        train = add_text_columns(train, level=FFILT, seed=seed, leakage_p=leakage_p)
    else:
        train = add_text_columns(train, level=TEXT_LEVEL, seed=seed,
                                 leakage_p=leakage_p, pair=True)
        if method == "counterfactual_aug":
            train = build_counterfactual_df(train)
    test_ffilt = add_text_columns(test, level=FFILT, seed=seed)
    test = add_text_columns(test, level=TEXT_LEVEL, seed=seed, leakage_p=leakage_p, pair=True)

    # Images are looked up by patient (`row`), as the real loader does by image_id.
    model = FusionModel(inputs).fit(img[train["row"]], train["text"], train["label"])
    x, y = img[test["row"]], test["label"]
    visible = roc_auc_score(y, model.predict(x, test["text"]))
    honest = roc_auc_score(y, model.predict(x, test_ffilt["text"]))
    s = srs_from_probs(model.predict(x, test["text_concordant"]),
                       model.predict(x, test["text_flipped"]))
    return {"visible_auroc": float(visible), "honest_auroc": float(honest),
            "ffilt_gap": ffilt_gap_from_auroc(visible, honest), **s}


def _mean_over_seeds(data, seeds, jobs):
    """jobs: list of (label, leakage_p, method, inputs) -> DataFrame of seed means."""
    rows = []
    for label, p, method, inputs in jobs:
        per_seed = [run(*data[s], s, p, method, inputs=inputs) for s in seeds]
        rows.append({"label": label, "p": p,
                     **pd.DataFrame(per_seed).mean().to_dict()})
    return pd.DataFrame(rows).set_index("label")


def controls(data, seeds):
    return _mean_over_seeds(data, seeds, [("image-only", 1.0, "naive", "image"),
                                          ("text-only", 1.0, "naive", "text")])


def dose_response(data, seeds, ps=LEAKAGE_PS):
    return _mean_over_seeds(data, seeds, [(p, p, "naive", "fusion") for p in ps])


def mitigation(data, seeds, methods=METHODS):
    return _mean_over_seeds(data, seeds, [(m, 1.0, m, "fusion") for m in methods])


# ---- The flip test -----------------------------------------------------------------

def check_pair(concordant, flipped, facts):
    """Both notes must start with the identical patient facts and differ after them."""
    return {"facts_identical": concordant.startswith(facts) and flipped.startswith(facts),
            "diagnosis_differs": concordant[len(facts):] != flipped[len(facts):]}


def flip_example(seed=42):
    """A real (concordant, flipped) note pair for the first toy melanoma."""
    df, _ = make_patients(200, seed)
    row = df[df["dx"] == "mel"].iloc[0]
    concordant, flipped = generate_pair(row, level=TEXT_LEVEL, seed=seed)
    facts = facts_only(row, seed=seed)
    return {"row": row, "concordant": concordant, "flipped": flipped, "facts": facts,
            "checks": check_pair(concordant, flipped, facts)}


# ---- Paper results -----------------------------------------------------------------

def paper_results(results_dir):
    """Seed means per method, plus a paired t-test of counterfactual_aug vs ffilt_only
    (honest AUROC, paired by seed), from results/shortcut_<dataset>_bakeoff.json."""
    out = {}
    for ds in PAPER_DATASETS:
        path = Path(results_dir) / f"shortcut_{ds}_bakeoff.json"
        if not path.exists():
            continue
        df = pd.DataFrame(json.loads(path.read_text()))
        means = df.groupby("method")[["auroc_ffilt", "srs"]].mean()
        methods = {m: {"honest_auroc": float(means.loc[m, "auroc_ffilt"]),
                       "srs": float(means.loc[m, "srs"])}
                   for m in PAPER_METHODS if m in means.index}
        test = None
        by_seed = df.pivot_table(index="seed", columns="method", values="auroc_ffilt")
        if {"counterfactual_aug", "ffilt_only"} <= set(by_seed.columns):
            pairs = by_seed[["counterfactual_aug", "ffilt_only"]].dropna()
            t = stats.ttest_rel(pairs["counterfactual_aug"], pairs["ffilt_only"])
            test = {"delta": float((pairs["counterfactual_aug"] - pairs["ffilt_only"]).mean()),
                    "t": float(t.statistic), "p": float(t.pvalue), "n": len(pairs)}
        out[ds] = {"methods": methods, "cf_vs_ffilt": test}
    return out


# ---- Printing ----------------------------------------------------------------------

def _header(title):
    print(f"\n{title}\n{'-' * WIDTH}")


def _wrap(text, indent="  "):
    print(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))


def _print_flip(ex):
    _header("[1] The flip test: one patient, two notes")
    r = ex["row"]
    print(f"  Patient: {r['age']}-year-old {r['sex']}, {r['localization']}, "
          f"true diagnosis: melanoma\n")
    for name, note in (("ORIGINAL", ex["concordant"]), ("FLIPPED ", ex["flipped"])):
        print(textwrap.fill(note, WIDTH, initial_indent=f"  {name}: ",
                            subsequent_indent=" " * 12))
    print()
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    print(f"  check: patient facts identical in both notes ...... "
          f"{ok(ex['checks']['facts_identical'])}")
    print(f"  check: only the diagnosis-bearing sentence differs  "
          f"{ok(ex['checks']['diagnosis_differs'])}")
    print()
    _wrap("SRS = mean |P(malignant | original) - P(malignant | flipped)| over the "
          "test set. FlipRate = share of lesions whose predicted class flips.")


def _print_controls(c):
    _header("[2] Known-answer controls (p = 1.0, mean over seeds)")
    print(f"  {'model':<14}{'SRS':>8}{'FlipRate':>10}   expected")
    expected = {"image-only": "0  (never reads the note)",
                "text-only": "~1 (copies the note)"}
    for name, r in c.iterrows():
        print(f"  {name:<14}{r.srs:>8.3f}{r.flip_rate:>10.3f}   {expected[name]}")


def _print_testbed(dose, mit):
    _header("[3] Mini testbed: image + note fusion model (mean over seeds)")
    print("  a) Dose-response: p = how often the note's verdict matches the truth")
    print(f"  {'p':>5}{'visible AUROC':>15}{'honest AUROC':>14}{'FFilt-gap':>11}"
          f"{'SRS':>8}{'FlipRate':>10}")
    for _, r in dose.iterrows():
        print(f"  {r.p:>5.1f}{r.visible_auroc:>15.3f}{r.honest_auroc:>14.3f}"
              f"{r.ffilt_gap:>11.3f}{r.srs:>8.3f}{r.flip_rate:>10.3f}")
    _wrap("Visible AUROC uses the leaky note; honest AUROC uses the facts-only "
          "(FFilt) note. As p rises the model leans on the note and SRS climbs. "
          "Toy limitation: the toy image signal is fixed, so honest AUROC cannot "
          "collapse here. In the real networks it does (naive rows in section 4), "
          "consistent with the image being learned less well when the note "
          "already gives the answer.")
    print("\n  b) Mitigation at p = 1.0")
    print(f"  {'method':<20}{'visible AUROC':>14}{'honest AUROC':>14}{'SRS':>9}")
    for name, r in mit.iterrows():
        print(f"  {name:<20}{r.visible_auroc:>14.3f}{r.honest_auroc:>14.3f}{r.srs:>9.4f}")
    _wrap("ffilt_only never sees diagnosis words in training, so it cannot react "
          "to them. counterfactual_aug sees them, but every note appears with both "
          "verdicts, so it learns they carry no information. Both remove the "
          "reliance here; section 4 tests on real data which mitigation works "
          "better.")


def _print_paper(res, results_dir):
    _header("[4] Paper results: real data, ConvNeXt-Base + BioClinicalBERT, "
            "p = 1.0, 5 seeds")
    if not res:
        _wrap(f"No results/shortcut_*_bakeoff.json found in {results_dir}. Run "
              "scripts/run_shortcut.py --mode bakeoff to produce them.")
        return
    print(f"  {'dataset':<16}{'method':<20}{'honest AUROC':>14}{'SRS':>9}")
    for ds, r in res.items():
        for m, v in r["methods"].items():
            print(f"  {PAPER_DATASETS[ds]:<16}{m:<20}{v['honest_auroc']:>14.3f}"
                  f"{v['srs']:>9.4f}")
        print()
    print("  counterfactual_aug vs ffilt_only, honest AUROC, paired over seeds:")
    for ds, r in res.items():
        t = r["cf_vs_ffilt"]
        if t:
            print(f"  {PAPER_DATASETS[ds]:<16}delta = {t['delta']:+.4f}   "
                  f"t = {t['t']:.2f}   p = {t['p']:.4f}   (n = {t['n']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000, help="toy patients per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    start = time.time()
    print(f"{RULE}\nSHORTCUT RELIANCE SCORE (SRS): GPU-free demo\n{RULE}")
    _wrap("Toy data (synthetic patients, synthetic image signal) with the paper's "
          "note engine and metric code. This shows the MECHANISM; it does not "
          "reproduce the paper's numbers (see section 4). SRS applies the "
          "counterfactual token fairness gap of Garg et al. (2019) to "
          "diagnosis-bearing clinical text.", indent="")

    data = {s: make_patients(args.n, s) for s in args.seeds}
    _print_flip(flip_example(seed=args.seeds[0]))
    _print_controls(controls(data, args.seeds))
    _print_testbed(dose_response(data, args.seeds), mitigation(data, args.seeds))
    _print_paper(paper_results(args.results_dir), args.results_dir)
    print(f"\nDone in {time.time() - start:.1f} s on CPU "
          f"({args.n} toy patients x {len(args.seeds)} seeds).")


if __name__ == "__main__":
    main()
