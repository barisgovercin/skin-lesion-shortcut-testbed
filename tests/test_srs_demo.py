"""Offline tests for the GPU-free SRS demo (scripts/srs_demo.py). CPU only, no data."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


def _load_demo():
    spec = importlib.util.spec_from_file_location("srs_demo", ROOT / "scripts" / "srs_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo()
SEEDS = [42, 123]
N = 1500


@pytest.fixture(scope="module")
def data():
    return {s: demo.make_patients(N, s) for s in SEEDS}


def _mean(data, leakage_p, method, inputs="fusion"):
    rows = [demo.run(*data[s], s, leakage_p, method, inputs=inputs) for s in SEEDS]
    return {k: float(np.mean([r[k] for r in rows]))
            for k in ("visible_auroc", "honest_auroc", "srs", "flip_rate")}


# ---- Known-answer controls ---------------------------------------------------

def test_image_only_model_has_zero_srs(data):
    m = _mean(data, 1.0, "naive", inputs="image")
    assert m["srs"] == 0.0 and m["flip_rate"] == 0.0


def test_text_only_model_has_high_srs(data):
    m = _mean(data, 1.0, "naive", inputs="text")
    assert m["srs"] > 0.5 and m["flip_rate"] > 0.9


# ---- Mini testbed --------------------------------------------------------------

def test_srs_rises_with_leakage(data):
    low, high = _mean(data, 0.5, "naive"), _mean(data, 1.0, "naive")
    assert high["srs"] > low["srs"] + 0.4
    assert high["visible_auroc"] > 0.99


def test_counterfactual_aug_removes_reliance_but_keeps_image_signal(data):
    naive = _mean(data, 1.0, "naive")
    cf = _mean(data, 1.0, "counterfactual_aug")
    image_only = _mean(data, 1.0, "naive", inputs="image")
    assert naive["srs"] > 0.5
    assert cf["srs"] < 0.05
    # Duplicated training rows must keep each image with its own patient.
    assert abs(cf["honest_auroc"] - image_only["honest_auroc"]) < 0.03


def test_ffilt_only_removes_reliance(data):
    assert _mean(data, 1.0, "ffilt_only")["srs"] < 0.05


def test_run_is_deterministic(data):
    a = demo.run(*data[42], 42, 0.8, "naive")
    b = demo.run(*data[42], 42, 0.8, "naive")
    assert a == b


# ---- Flip-pair checks -------------------------------------------------------------

def test_check_pair_detects_changed_facts():
    out = demo.check_pair("A 50-year-old man. Refer urgently.",
                          "A 51-year-old man. Routine follow-up.",
                          facts="A 50-year-old man.")
    assert out["facts_identical"] is False


def test_check_pair_detects_unchanged_diagnosis():
    out = demo.check_pair("A 50-year-old man. Refer urgently.",
                          "A 50-year-old man. Refer urgently.",
                          facts="A 50-year-old man.")
    assert out["facts_identical"] is True and out["diagnosis_differs"] is False


def test_flip_example_passes_both_checks():
    ex = demo.flip_example(seed=42)
    assert ex["concordant"] != ex["flipped"]
    assert ex["checks"] == {"facts_identical": True, "diagnosis_differs": True}


# ---- Paper results -----------------------------------------------------------------

def test_paper_results_pairs_methods_by_seed(tmp_path):
    cf = {42: 0.92, 123: 0.91, 456: 0.93, 789: 0.94, 1024: 0.93}
    ff = {42: 0.91, 123: 0.90, 456: 0.92, 789: 0.92, 1024: 0.92}
    rows = [{"seed": s, "method": "ffilt_only", "auroc_ffilt": v, "srs": 0.006} for s, v in ff.items()]
    # Reverse the seed order so pairing by position would give the wrong answer.
    rows += [{"seed": s, "method": "counterfactual_aug", "auroc_ffilt": v, "srs": 0.004}
             for s, v in reversed(list(cf.items()))]
    (tmp_path / "shortcut_ham10000_bakeoff.json").write_text(json.dumps(rows))

    out = demo.paper_results(tmp_path)["ham10000"]
    assert out["methods"]["counterfactual_aug"]["honest_auroc"] == pytest.approx(0.926)
    assert out["methods"]["ffilt_only"]["srs"] == pytest.approx(0.006)
    # d = [.01, .01, .01, .02, .01]: mean .012, sd sqrt(2e-5) -> t = 6.0 on 4 df.
    assert out["cf_vs_ffilt"]["delta"] == pytest.approx(0.012)
    assert out["cf_vs_ffilt"]["t"] == pytest.approx(6.0)
    assert out["cf_vs_ffilt"]["n"] == 5


def test_paper_results_missing_dir_is_empty(tmp_path):
    assert demo.paper_results(tmp_path / "nope") == {}


# ---- End to end --------------------------------------------------------------------

BLOCK_TORCH = """
import importlib.abc, sys
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "torch":
            raise ModuleNotFoundError(f"No module named {name!r}")
sys.meta_path.insert(0, _NoTorch())
"""


def test_demo_runs_without_torch(tmp_path):
    """The demo must run on a plain CPU install (numpy/pandas/scikit-learn/scipy)."""
    code = (BLOCK_TORCH + "import runpy\n"
            f"sys.argv = ['srs_demo.py', '--n', '400', '--seeds', '42', '--results-dir', {str(tmp_path)!r}]\n"
            f"runpy.run_path({str(ROOT / 'scripts' / 'srs_demo.py')!r}, run_name='__main__')\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
