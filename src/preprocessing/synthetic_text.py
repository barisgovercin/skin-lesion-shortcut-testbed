"""Synthetic GP-style referral notes with controllable 'leading language'.

The original NHS dataset with real clinical freetext is GDPR-restricted, so we
synthesise referral notes from HAM10000 / ISIC 2020 metadata + the known
diagnosis, and inject diagnosis-leaking 'leading language' at four levels matching
Watson et al. (2026), Table 3:

    Orig  : full note (condition name + benign/malignant verdict + referral/
            treatment phrasing + patient-obtainable facts)
    CFilt : ConditionFiltering  -> remove skin-condition NAMES
    DFilt : DiagnosisFiltering  -> CFilt + remove the words 'benign'/'malignant'
    FFilt : FullyFiltered       -> remove ALL leading language (incl. referral and
            proposed treatment); keep only facts a patient could provide

Beyond the filter levels, leakage is *continuously* controllable via `leakage_p`
(the concordance probability): the diagnosis-revealing segments (condition /
verdict / leading) match the TRUE label with probability `leakage_p` and the
OPPOSITE label otherwise. Patient `facts` are always truthful. At `leakage_p=1.0`
the output is byte-identical to the original deterministic generator.

This gives a public, reproducible testbed for measuring and mitigating
leading-language shortcut learning.
"""

import hashlib
import random

ORIG, CFILT, DFILT, FFILT = "Orig", "CFilt", "DFilt", "FFilt"
LEVELS = [ORIG, CFILT, DFILT, FFILT]

# Which filter levels KEEP each segment category.
KEEP = {
    "facts": {ORIG, CFILT, DFILT, FFILT},
    "condition": {ORIG},
    "verdict": {ORIG, CFILT},
    "leading": {ORIG, CFILT, DFILT},
}

MALIGNANT_DX = {"mel", "bcc", "akiec"}

DX_NAME = {
    "mel": "melanoma",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis",
    "nv": "melanocytic naevus",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular lesion",
}


def _norm(row):
    """Map ISIC field names onto the HAM-style keys this module expects.

    Accepts a dict or a pandas Series. Idempotent for HAM rows.
    """
    g = dict(row)
    if g.get("image_id") in (None, ""):
        g["image_id"] = g.get("image_name", g.get("image_id", ""))
    if g.get("age") is None:
        g["age"] = g.get("age_approx")
    if g.get("localization") is None:
        # ISIC 2020 uses 'anatom_site_general_challenge'; ISIC-DICM-17K uses
        # 'anatom_site_general'. Accept either so all datasets get a site clause.
        g["localization"] = (g.get("anatom_site_general_challenge")
                             or g.get("anatom_site_general"))
    if g.get("label") is None and g.get("target") is not None:
        g["label"] = int(g["target"])
    return g


def _rng_for(row, seed):
    """Deterministic phrasing RNG per (lesion, seed) - independent of leakage_p."""
    key = f"{seed}|{row.get('image_id', '')}|{row.get('dx', '')}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return random.Random(int(digest, 16))


def _concordance_u(row, seed):
    """Deterministic uniform in [0, 1) per (lesion, seed); separate from phrasing.

    A row is 'concordant' (diagnosis segments match the true label) iff
    `u < leakage_p`. Because `u` does not depend on `leakage_p`, concordance sets
    are nested across `p` and the phrasing RNG is never perturbed.
    """
    key = f"{seed}|{row.get('image_id', '')}|conc"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) / float(1 << 128)


def _true_malignant(row):
    dx = str(row.get("dx", "")).lower()
    if row.get("label") is not None:
        return bool(row["label"])
    return dx in MALIGNANT_DX


def _sex_word(sex):
    return {"male": "man", "female": "woman"}.get(str(sex).lower(), "patient")


def _age_phrase(age):
    try:
        a = int(float(age))
        return f"A {a}-year-old"
    except (TypeError, ValueError):
        return "An adult"


def _loc_clause(loc):
    if loc is None:
        return "at an unspecified site"
    loc = str(loc).strip().lower()
    if not loc or loc in {"nan", "none", "unknown"}:
        return "at an unspecified site"
    return f"on the {loc}"


def _fact_segments(row, rng):
    """Patient-obtainable facts (truthful, survive even FFilt). 1 RNG draw."""
    return [
        ("facts",
         f"{_age_phrase(row.get('age'))} {_sex_word(row.get('sex'))} presents with a "
         f"skin lesion {_loc_clause(row.get('localization'))}."),
        ("facts", rng.choice([
            "The lesion has been present for several months.",
            "The patient reports the lesion has been present for some time.",
            "The patient first noticed the lesion a while ago.",
        ])),
    ]


def _dx_segments(row, rng, malignant):
    """Diagnosis-revealing segments for the given malignant flag. 3 RNG draws.

    `condition` always uses the true disease name (removed at CFilt anyway);
    `verdict` and `leading` follow the supplied `malignant` flag.
    """
    dx = str(row.get("dx", "")).lower()
    condition = DX_NAME.get(dx, "skin lesion")
    segs = [("condition", rng.choice([
        f"The appearance is consistent with {condition}.",
        f"Clinical impression: {condition}.",
        f"Most likely a {condition}.",
    ]))]
    if malignant:
        segs.append(("verdict", rng.choice([
            "The lesion is considered malignant.",
            "Findings are in keeping with a malignant lesion.",
        ])))
        segs.append(("leading", rng.choice([
            "Urgent two-week-wait referral to dermatology.",
            "Excision is recommended.",
            "Clinically concerning; refer to secondary care.",
        ])))
    else:
        segs.append(("verdict", rng.choice([
            "The lesion appears benign.",
            "Findings are in keeping with a benign lesion.",
        ])))
        segs.append(("leading", rng.choice([
            "Routine monitoring advised.",
            "No urgent action required.",
            "Reassurance given with safety-net advice.",
        ])))
    return segs


def _assemble(level, fact_segs, dx_segs):
    segs = fact_segs + dx_segs
    return " ".join(text for cat, text in segs if level in KEEP[cat])


def _note(row, level, seed, malignant):
    """Assemble one note at `level` with diagnosis segments for `malignant`.

    RNG draw order (duration, condition, verdict, leading) matches the original
    generator, so `malignant == _true_malignant(row)` reproduces legacy output.
    """
    rng = _rng_for(row, seed)
    return _assemble(level, _fact_segments(row, rng), _dx_segments(row, rng, malignant))


def facts_only(row, seed=0):
    """The FFilt (facts-only) note - the 'honest' / clean probe text."""
    row = _norm(row)
    rng = _rng_for(row, seed)
    return _assemble(FFILT, _fact_segments(row, rng), [])


def generate_pair(row, level=ORIG, seed=0):
    """(concordant, flipped): identical facts, diagnosis segments for the true vs
    the opposite label. Used for SRS (flip test) and counterfactual augmentation."""
    row = _norm(row)
    true_mal = _true_malignant(row)
    return (_note(row, level, seed, true_mal),
            _note(row, level, seed, not true_mal))


def generate_note(row, level=ORIG, seed=0, leakage_p=1.0):
    """Single note at `level`; diagnosis segments concordant with prob `leakage_p`."""
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'. Options: {LEVELS}")
    row = _norm(row)
    true_mal = _true_malignant(row)
    concordant = _concordance_u(row, seed) < leakage_p
    eff_mal = true_mal if concordant else (not true_mal)
    return _note(row, level, seed, eff_mal)


def generate_all_levels(row, seed=0):
    """Generate the SAME (truthful) note filtered at all four levels."""
    row = _norm(row)
    true_mal = _true_malignant(row)
    return {level: _note(row, level, seed, true_mal) for level in LEVELS}


def add_text_columns(df, level=ORIG, seed=0, leakage_p=1.0, pair=False, col="text"):
    """Return a copy of df with synthetic-note column(s).

    - `col`: note at `level` with diagnosis concordance `leakage_p`.
    - `pair=True` also adds `text_concordant` / `text_flipped` (true vs opposite
      diagnosis, identical facts) for SRS and counterfactual augmentation.
    """
    result = df.copy()
    result[col] = [generate_note(r, level=level, seed=seed, leakage_p=leakage_p)
                   for _, r in result.iterrows()]
    if pair:
        pairs = [generate_pair(r, level=level, seed=seed) for _, r in result.iterrows()]
        result["text_concordant"] = [c for c, _ in pairs]
        result["text_flipped"] = [f for _, f in pairs]
    return result


def add_synthetic_text(df, level=ORIG, seed=0, col="text"):
    """Legacy wrapper: full-leakage (p=1.0) note at the given filter level."""
    return add_text_columns(df, level=level, seed=seed, leakage_p=1.0, col=col)
