"""LLM-generated realistic clinical referral notes with controllable leakage.

A drop-in, higher-realism alternative to the template engine in
``synthetic_text.py``. The patient FACTS stay deterministic (reused from
``synthetic_text``, so they are byte-identical across a counterfactual pair and
survive FFilt); the diagnosis-revealing portion is drawn from an LLM-generated
POOL of varied natural phrasings for each (level, impression).

Why a pool: generating a bespoke sentence for every one of ~10k rows would be
tens of thousands of sequential API calls (hours, dollars). Instead we generate
a modest pool of diverse clauses per (level, impression) ONCE, then assign one to
each row deterministically by hash(seed, lesion). That is a couple of hundred
calls total, cached on disk, and still makes the leak read like real, varied
clinical language rather than a fixed template - the whole point, since keyword
filtering (FFilt) should no longer trivially remove a naturally phrased leak.

Every shortcut-testbed invariant is preserved: the ``leakage_p`` concordance
draw, the four filter levels, and the identical-facts SRS / counterfactual pairs
(concordant vs flipped use the same hash slot in the true vs opposite pool).

Backends:
  - "mock"      : offline, deterministic stub (no network); used by tests.
  - "anthropic" : a hosted LLM API via the Anthropic SDK (``pip install anthropic`` + ANTHROPIC_API_KEY).
                  The backend is created lazily, so once the pools are cached a
                  training run consumes them WITHOUT needing the API key.

Interface mirrors ``synthetic_text.add_text_columns`` so ``scripts/run_shortcut.py``
consumes these notes with a one-line config swap (``data.text_source: llm``).
"""
import hashlib
import json
import os
import random
from pathlib import Path

from src.preprocessing.synthetic_text import (
    _norm, _rng_for, _concordance_u, _true_malignant, _fact_segments,
    DX_NAME, KEEP, ORIG, CFILT, DFILT, FFILT, LEVELS,
)

CACHE_DIR = Path("data/llm_notes_cache")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "")
DEFAULT_POOL_SIZE = 120


def _facts_text(row, seed):
    """Deterministic patient-facts sentences (identical across a pair, survive FFilt)."""
    rng = _rng_for(row, seed)
    return " ".join(text for _, text in _fact_segments(row, rng))


def _diagnosis_spec(level, malignant):
    """What the diagnosis clause may reveal at this level (mirrors synthetic_text.KEEP)."""
    return {
        "name_ok": level in KEEP["condition"],   # may name the likely condition
        "verdict_ok": level in KEEP["verdict"],  # may use benign/malignant
        "leading_ok": level in KEEP["leading"],  # may give management/referral
        "malignant": malignant,
    }


def _revealing(spec):
    return spec["name_ok"] or spec["verdict_ok"] or spec["leading_ok"]


_VERDICT_WORDS = ["malignant", "malignancy", "benign", "cancer", "cancerous",
                  "carcinoma", "tumour", "tumor", "metastatic", "metastasis"]
_VERDICT_STEMS = ["malignan", "benign", "cancer", "carcinoma", "tumour", "tumor", "metasta"]
_CONDITION_WORDS = ["melanoma", "naevus", "nevus", "keratosis", "dermatofibroma",
                    "basal cell", "actinic", "vascular lesion"]


def _forbidden_words(spec):
    """Human-readable words banned at this level (for the prompt)."""
    words = []
    if not spec["verdict_ok"]:
        words += _VERDICT_WORDS
    if not spec["name_ok"]:
        words += _CONDITION_WORDS
    return words


def _violates(clause, spec):
    """True if a generated clause leaks a verdict/condition it must not, at this level."""
    c = clause.lower()
    stems = []
    if not spec["verdict_ok"]:
        stems += _VERDICT_STEMS
    if not spec["name_ok"]:
        stems += _CONDITION_WORDS
    return any(s in c for s in stems)


def _prompt(row, spec):
    condition = DX_NAME.get(str(row.get("dx", "")).lower(), "skin lesion")
    impression = "malignant / clinically concerning" if spec["malignant"] else "benign / reassuring"
    allow = [
        (f'you MAY name the likely condition ("{condition}")' if spec["name_ok"]
         else "do NOT name any specific skin condition"),
        ('you MAY use the words "benign" or "malignant"' if spec["verdict_ok"]
         else 'do NOT state any diagnostic verdict'),
        ("you MAY give a management/referral recommendation implying urgency or reassurance"
         if spec["leading_ok"]
         else "do NOT give any management, referral, or follow-up recommendation"),
    ]
    forbidden = _forbidden_words(spec)
    extra = (f" Do NOT use any of these words or their variants: {', '.join(forbidden)}."
             if forbidden else "")
    return ("Write ONE short, natural sentence (occasionally two) as the clinical-impression "
            "part of a UK GP dermatology referral letter. "
            f"Overall impression to convey: {impression}. "
            f"Constraints: {'; '.join(allow)}.{extra} Convey the impression only through the "
            "level of clinical urgency and the management tone. Vary the phrasing and hedging so "
            "it reads like a real clinician; do not restate the patient's age, sex, or lesion site. "
            "Output only the sentence(s), no preamble.")


def _mock_clause(row, spec, rng):
    """Deterministic offline stand-in with the SAME leakage semantics as the prompt."""
    if not _revealing(spec):
        return ""
    if spec["name_ok"]:  # Orig: may name the condition
        cond = DX_NAME.get(str(row.get("dx", "")).lower(), "skin lesion")
        return (rng.choice([f"Appearances are concerning for {cond}; please assess urgently.",
                            f"This looks like it could be {cond} and I would refer promptly.",
                            f"Strong suspicion of {cond}; would value an urgent opinion."])
                if spec["malignant"] else
                rng.choice([f"Appearances suggest a {cond}, most likely benign.",
                            f"Features are in keeping with a {cond}; reassuring overall.",
                            f"Looks like a straightforward {cond}."]))
    if spec["verdict_ok"]:  # CFilt: may say benign/malignant, no condition name
        return (rng.choice(["The features look malignant and warrant urgent review.",
                            "On balance the lesion appears malignant to me.",
                            "I am concerned this is malignant."])
                if spec["malignant"] else
                rng.choice(["The features look benign and reassuring.",
                            "This appears benign on examination.",
                            "Most likely a benign lesion."]))
    # DFilt: leading/management only, no condition name, no benign/malignant word
    return (rng.choice(["I would have a low threshold for specialist review here.",
                        "Suggest prompt dermatology input given the appearance.",
                        "I would value an urgent specialist opinion on this one.",
                        "Given the appearance I would refer without delay."])
            if spec["malignant"] else
            rng.choice(["Happy to reassure and review only if it changes.",
                        "No immediate concern; safety-net advice given.",
                        "Reasonable to keep this under routine observation.",
                        "I would simply keep an eye on it for now."]))


def _pool_key(level, malignant, row):
    cond = (DX_NAME.get(str(row.get("dx", "")).lower(), "skin lesion")
            if level in KEEP["condition"] else "")
    return f"{level}|{int(malignant)}|{cond}"


class _AnthropicBackend:
    def __init__(self, model=DEFAULT_MODEL):
        import anthropic  # lazy; only when a pool actually needs generating
        self.client = anthropic.Anthropic()
        self.model = model

    def clause(self, prompt):
        msg = self.client.messages.create(
            model=self.model, max_tokens=120, temperature=1.0,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()


def _generate_pool(level, malignant, row, provider, get_backend, pool_size):
    spec = _diagnosis_spec(level, malignant)
    if level == FFILT or not _revealing(spec):
        return [""]
    if provider == "mock":
        key = _pool_key(level, malignant, row)
        pool = []
        for i in range(pool_size * 6):
            c = _mock_clause(row, spec, random.Random(f"{key}|{i}"))
            if c and c not in pool:
                pool.append(c)
            if len(pool) >= pool_size:
                break
        return pool or [""]
    backend = get_backend()
    prompt = _prompt(row, spec)
    pool, attempts, max_attempts = [], 0, pool_size * 3
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
            c = backend.clause(prompt)
        except Exception as exc:
            if not pool:
                raise
            print(f"  pool generation stopped early ({type(exc).__name__}); kept {len(pool)}")
            break
        if c and c not in pool and not _violates(c, spec):
            pool.append(c)
    return pool or [""]


def _assign(pool, row, seed):
    if not pool:
        return ""
    h = int(hashlib.md5(f"{seed}|{row.get('image_id', '')}".encode()).hexdigest(), 16)
    return pool[h % len(pool)]


def _clause(row, level, malignant, seed, provider, model, get_backend, pools, pool_size):
    if level == FFILT:
        return ""
    key = f"{provider}|{model}|{_pool_key(level, malignant, row)}"
    if key not in pools:
        pools[key] = _generate_pool(level, malignant, row, provider, get_backend, pool_size)
    return _assign(pools[key], row, seed)


def _note(row, level, seed, malignant, provider, model, get_backend, pools, pool_size):
    facts = _facts_text(row, seed)
    clause = _clause(row, level, malignant, seed, provider, model, get_backend, pools, pool_size)
    return f"{facts} {clause}".strip() if clause else facts


def add_llm_text_columns(df, level=ORIG, seed=0, leakage_p=1.0, pair=False,
                         provider="mock", model=DEFAULT_MODEL, col="text",
                         cache_path=None, pool_size=DEFAULT_POOL_SIZE):
    """LLM (or mock) analogue of ``synthetic_text.add_text_columns``.

    Adds ``col`` (diagnosis segments concordant with the true label with
    probability ``leakage_p``) and, if ``pair=True``, ``text_concordant`` /
    ``text_flipped`` (true vs opposite impression, identical facts) for SRS and
    counterfactual augmentation. Clause pools are cached on disk, so the first
    call may hit the API and every later call (other seeds/splits/methods) is free.
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'. Options: {LEVELS}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    cache_path = Path(cache_path or CACHE_DIR / f"pools_{provider}_{safe_model}.json")
    pools = json.load(open(cache_path)) if cache_path.exists() else {}

    holder = {}

    def get_backend():
        if "b" not in holder:
            holder["b"] = _AnthropicBackend(model)
        return holder["b"]

    result = df.copy()
    notes, concordant_notes, flipped_notes = [], [], []
    for _, raw_row in result.iterrows():
        row = _norm(raw_row)
        true_mal = _true_malignant(row)
        concordant = _concordance_u(row, seed) < leakage_p
        eff_mal = true_mal if concordant else (not true_mal)
        notes.append(_note(row, level, seed, eff_mal, provider, model, get_backend, pools, pool_size))
        if pair:
            concordant_notes.append(
                _note(row, level, seed, true_mal, provider, model, get_backend, pools, pool_size))
            flipped_notes.append(
                _note(row, level, seed, not true_mal, provider, model, get_backend, pools, pool_size))

    result[col] = notes
    if pair:
        result["text_concordant"] = concordant_notes
        result["text_flipped"] = flipped_notes
    json.dump(pools, open(cache_path, "w"), indent=2)
    return result
