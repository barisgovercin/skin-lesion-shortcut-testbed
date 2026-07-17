"""Offline tests for the results-aggregation / comparison module.

Builds synthetic experiment_results_*.json files in a temp dir and validates
filename parsing, loading, mean+/-95% CI aggregation, and ordering. No GPU,
dataset, or model needed. Run from the repo root:

    python tests/test_evaluation.py
    pytest tests/test_evaluation.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.compare import (
    parse_result_filename, load_results, aggregate, format_markdown, _none_if_nan,
)


def _rec(mode, seed, backbone, auroc, ap, spec):
    return {
        "mode": mode, "seed": seed, "backbone": backbone,
        "best_val_auroc": auroc,
        "test_metrics": {
            "auroc": auroc, "ap": ap, "specificity_at_95_sens": spec,
            "sensitivity": 0.95, "threshold_95_sens": 0.3, "loss": 0.1,
        },
    }


def _write(dirpath, fname, records):
    (Path(dirpath) / fname).write_text(json.dumps(records))


def test_parse_filename():
    assert parse_result_filename("experiment_results_ham10000_convnext_base.json") == (
        "ham10000", "convnext_base", None)
    assert parse_result_filename("experiment_results_isic2020_convnext_large.json") == (
        "isic2020", "convnext_large", None)
    assert parse_result_filename("experiment_results_ham10000_convnext_base_FFilt.json") == (
        "ham10000", "convnext_base", "FFilt")
    assert parse_result_filename("experiment_results_ham10000_resnet50_DFilt.json") == (
        "ham10000", "resnet50", "DFilt")
    # Non-matching files are ignored.
    assert parse_result_filename("baseline_vision_ham10000_seed42.json") is None
    assert parse_result_filename("notes.txt") is None
    # Unknown dataset prefix -> not parsed.
    assert parse_result_filename("experiment_results_foo_convnext_base.json") is None


def test_load_and_aggregate_grouping():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "experiment_results_ham10000_convnext_base.json", [
            _rec("vision", s, "convnext_base", 0.90, 0.65, 0.60) for s in (42, 123, 456)
        ] + [
            _rec("vision_metadata", s, "convnext_base", 0.93, 0.70, 0.66) for s in (42, 123, 456)
        ])
        _write(tmp, "experiment_results_ham10000_convnext_base_Orig.json", [
            _rec("vision_text", s, "convnext_base", 0.96, 0.80, 0.75) for s in (42, 123, 456)
        ])
        _write(tmp, "experiment_results_ham10000_convnext_base_FFilt.json", [
            _rec("vision_text", s, "convnext_base", 0.94, 0.74, 0.70) for s in (42, 123, 456)
        ])

        df = load_results(tmp)
        assert len(df) == 12  # 4 mode/level combos x 3 seeds
        agg = aggregate(df)

    assert len(agg) == 4
    assert agg["n_seeds"].tolist() == [3, 3, 3, 3]
    # Sort order: vision -> vision_metadata -> vision_text(Orig) -> vision_text(FFilt)
    assert agg["mode"].tolist() == ["vision", "vision_metadata", "vision_text", "vision_text"]
    # Non-text modes have no level (NaN); text modes carry the filter level.
    assert [_none_if_nan(x) for x in agg["level"].tolist()] == [None, None, "Orig", "FFilt"]


def test_ci_zero_for_constant_values():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "experiment_results_ham10000_convnext_base.json", [
            _rec("vision", s, "convnext_base", 0.9, 0.5, 0.4) for s in (42, 123, 456, 789, 1024)
        ])
        agg = aggregate(load_results(tmp))
    row = agg.iloc[0]
    assert abs(row["auroc_mean"] - 0.9) < 1e-9
    assert row["auroc_ci"] == 0.0  # identical values -> zero-width CI
    assert row["n_seeds"] == 5


def test_single_seed_ci_zero():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "experiment_results_isic2020_convnext_base.json", [
            _rec("vision", 42, "convnext_base", 0.88, 0.55, 0.5)
        ])
        agg = aggregate(load_results(tmp))
    assert agg.iloc[0]["auroc_ci"] == 0.0
    assert agg.iloc[0]["dataset"] == "isic2020"


def test_markdown_renders_rows_and_empty():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "experiment_results_ham10000_convnext_base.json", [
            _rec("vision", s, "convnext_base", 0.9, 0.6, 0.55) for s in (42, 123)
        ])
        md = format_markdown(aggregate(load_results(tmp)))
    assert "AUROC" in md and "vision" in md and "0.9000" in md
    # Empty results -> graceful message, no crash.
    with tempfile.TemporaryDirectory() as empty:
        md_empty = format_markdown(aggregate(load_results(empty)))
    assert "No results found" in md_empty


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
