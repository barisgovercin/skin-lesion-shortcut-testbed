"""Tests for scripts/reproduce_tables.py: the rules that compare manuscript numbers
with values recomputed from results/, and one run on the shipped result files."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "reproduce_tables", ROOT / "scripts" / "reproduce_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rt = _load()


# ---- Comparison rules -------------------------------------------------------------

@pytest.mark.parametrize("printed, value, ok", [
    ("0.928", 0.92755, True),       # rounds to the printed digits
    ("0.928", 0.9286, False),
    ("+0.010", 0.01003, True),      # explicit sign
    ("-0.149", -0.1489, True),
    ("1.000", 0.99999, True),
    ("2.4", 2.40, True),
])
def test_round_rule(printed, value, ok):
    assert rt.check("round", printed, value) is ok


def test_mean_ci_rule():
    assert rt.check("pm", "0.906±0.008", (0.9057, 0.0081)) is True
    assert rt.check("pm", "0.982±0.005", (0.98262, 0.0049)) is False


def test_threshold_rules():
    assert rt.check("lt", "<0.001", 0.0008) is True
    assert rt.check("lt", "<0.001", 0.0067) is False
    assert rt.check("gt", ">2", 2.30) is True
    assert rt.check("not_significant", "p>=0.05", 0.0425) is False
    assert rt.check("not_significant", "p>=0.05", 0.1758) is True


def test_range_rule():
    assert rt.check("range", "0.90-0.92", [0.917, 0.903]) is True
    assert rt.check("range", "0.90-0.92", [0.917, 0.656]) is False


def test_best_rule_compares_at_printed_precision():
    # lowest SRS wins; ties at 3 decimals count as winning
    assert rt.check("lowest", "3", (0.0010, [0.0008, 0.0021])) is True    # 0.001 vs 0.001
    assert rt.check("lowest", "3", (0.0036, [0.0024, 0.0067])) is False   # 0.004 vs 0.002
    assert rt.check("highest", "3", (0.9276, [0.9242, 0.9176])) is True


# ---- Against the shipped results ---------------------------------------------------

@pytest.fixture(scope="module")
def facts():
    return rt.compute(ROOT / "results")


def test_compute_recovers_known_headline_values(facts):
    assert f"{facts['vision.ham10000'][0]:.3f}" == "0.906"
    assert f"{facts['ham10000.bake.counterfactual_aug.honest']:.3f}" == "0.928"
    assert f"{facts['ham10000.paired.honest.ffilt_only.p']:.3f}" == "0.006"


def test_a_planted_wrong_claim_is_reported(facts):
    claims = [{"where": "true", "rule": "round", "printed": "0.928",
               "key": "ham10000.bake.counterfactual_aug.honest"},
              {"where": "planted", "rule": "round", "printed": "0.930",
               "key": "ham10000.bake.counterfactual_aug.honest"}]
    results = rt.run_checks(facts, claims)
    assert [r["ok"] for r in results] == [True, False]


def test_every_manuscript_claim_has_a_computed_value(facts):
    for claim in rt.CLAIMS:
        keys = claim["key"] if isinstance(claim["key"], list) else [claim["key"]]
        for key in keys:
            assert key in facts, f"{claim['where']}: no value for {key}"
