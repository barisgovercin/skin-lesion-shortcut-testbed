"""Unit tests for split manifests and leakage/duplicate checks on small synthetic inputs.

The real-data run is scripts/make_splits.py + scripts/check_splits.py, and
tests/test_real_data_splits.py checks the shipped manifests against real data
whenever the datasets are present.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import prepare_ham10000
from src.data import splits


def _fake_ham(tmp_path, n_lesions=60, seed=0):
    """HAM10000-style metadata: 1-3 images per lesion, ~25% malignant lesions."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for lid in range(n_lesions):
        dx = "mel" if lid % 4 == 0 else "nv"
        for _ in range(int(rng.integers(1, 4))):
            rows.append({"lesion_id": f"HAM_{lid:07d}", "image_id": f"ISIC_{k:07d}",
                         "dx": dx, "dx_type": "histo", "age": 50.0, "sex": "male",
                         "localization": "back"})
            k += 1
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)
    return tmp_path


def _manifest(rows):
    return pd.DataFrame(rows, columns=["image_id", "group_id", "label", "split"])


# ---- Manifests ---------------------------------------------------------------------

def test_ham_manifest_has_every_image_once_and_maps_labels(tmp_path):
    data_dir = _fake_ham(tmp_path)
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    m, _ = splits.build_manifest("ham10000", data_dir, seed=42)
    assert list(m.columns) == ["image_id", "group_id", "label", "split"]
    assert sorted(m["image_id"]) == sorted(meta["image_id"])
    assert set(m["split"]) == {"train", "val", "test"}
    mel_images = set(meta.loc[meta["dx"] == "mel", "image_id"])
    assert set(m.loc[m["label"] == 1, "image_id"]) == mel_images


def test_ham_manifest_groups_by_lesion(tmp_path):
    data_dir = _fake_ham(tmp_path)
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    m, _ = splits.build_manifest("ham10000", data_dir, seed=42)
    lesion_of = dict(zip(meta["image_id"], meta["lesion_id"]))
    assert (m["group_id"] == m["image_id"].map(lesion_of)).all()


def test_manifest_test_split_is_the_training_code_test_split(tmp_path):
    data_dir = _fake_ham(tmp_path)
    m, _ = splits.build_manifest("ham10000", data_dir, seed=123)
    _, _, test_df, _, _ = prepare_ham10000(data_dir=data_dir, seed=123)
    assert set(m.loc[m["split"] == "test", "image_id"]) == set(test_df["image_id"])


# ---- Leakage checks ----------------------------------------------------------------

def test_check_manifest_flags_group_in_two_splits():
    m = _manifest([("a", "G1", 0, "train"), ("b", "G1", 0, "test"),
                   ("c", "G2", 1, "val")])
    out = splits.check_manifest(m)
    assert out["groups_in_multiple_splits"] == 1
    assert out["images_in_multiple_splits"] == 0


def test_check_manifest_flags_image_listed_twice():
    m = _manifest([("a", "G1", 0, "train"), ("a", "G2", 0, "test"),
                   ("c", "G3", 1, "val")])
    out = splits.check_manifest(m)
    assert out["images_in_multiple_splits"] == 1
    assert out["duplicate_rows"] == 1


def test_check_manifest_reports_sizes_and_malignant_ratio():
    m = _manifest([("a", "G1", 1, "train"), ("b", "G2", 0, "train"),
                   ("c", "G3", 0, "val"), ("d", "G4", 1, "test")])
    out = splits.check_manifest(m)
    assert out["sizes"] == {"train": 2, "val": 1, "test": 1}
    assert out["malignant"] == {"train": 0.5, "val": 0.0, "test": 1.0}
    assert out["groups_in_multiple_splits"] == 0


def test_compare_sizes_with_paper_logs():
    m = _manifest([("a", "G1", 1, "train"), ("b", "G2", 0, "train"),
                   ("c", "G3", 0, "val"), ("d", "G4", 1, "test")])
    assert splits.compare_sizes(m, expected=(2, 1, 1))["match"] is True
    out = splits.compare_sizes(m, expected=(3, 1, 0))
    assert out["match"] is False and out["actual"] == (2, 1, 1)


def test_paper_sizes_cover_all_datasets_and_seeds():
    for ds in ("ham10000", "isic2020", "isic_dicm_17k"):
        assert sorted(splits.PAPER_SPLIT_SIZES[ds]) == [42, 123, 456, 789, 1024]


# ---- Duplicate detection -------------------------------------------------------------

def test_exact_duplicates_finds_identical_files_only(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"same-bytes")
    (tmp_path / "b.jpg").write_bytes(b"same-bytes")
    (tmp_path / "c.jpg").write_bytes(b"other-bytes")
    paths = {i: tmp_path / f"{i}.jpg" for i in ("a", "b", "c")}
    pairs = splits.exact_duplicate_pairs(paths)
    assert [tuple(r) for r in pairs[["a", "b"]].to_numpy()] == [("a", "b")]


def test_near_duplicates_respects_threshold_without_self_or_mirror_pairs():
    ids = ["x", "y", "z"]
    emb = np.array([[1.0, 0.0], [0.99, 0.1411], [0.0, 1.0]])   # cos(x, y) ~ 0.99
    for chunk in (1, 2, 10):
        pairs = splits.near_duplicate_pairs(ids, emb, threshold=0.95, chunk=chunk)
        assert [tuple(r) for r in pairs[["a", "b"]].to_numpy()] == [("x", "y")]
        assert pairs["similarity"].iloc[0] == pytest.approx(0.99, abs=1e-3)


def test_near_duplicates_refuses_to_build_an_oversized_pair_list():
    emb = np.ones((5, 2))                       # every pair is identical: 10 pairs
    with pytest.raises(ValueError, match="threshold"):
        splits.near_duplicate_pairs(list("abcde"), emb, threshold=0.9, max_pairs=3)


def test_cross_split_pairs_keeps_only_pairs_across_splits():
    m = _manifest([("a", "G1", 0, "train"), ("b", "G2", 0, "test"),
                   ("c", "G3", 1, "train")])
    pairs = pd.DataFrame({"a": ["a", "a"], "b": ["b", "c"], "similarity": [0.97, 0.96]})
    out = splits.cross_split_pairs(pairs, m)
    assert out[["a", "b", "split_a", "split_b"]].values.tolist() == [["a", "b", "train", "test"]]


def test_embed_images_returns_unit_vectors_in_input_order(tmp_path):
    import torch.nn as nn

    colours = {"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255)}
    paths = {}
    for name, rgb in colours.items():
        Image.new("RGB", (40, 30), rgb).save(tmp_path / f"{name}.png")
        paths[name] = tmp_path / f"{name}.png"
    mean_colour = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    ids, emb = splits.embed_images(paths, model=mean_colour, batch_size=2)
    assert ids == ["r", "g", "b"]
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert int(np.argmax(emb[0])) == 0 and int(np.argmax(emb[2])) == 2


def test_image_paths_finds_files_across_dirs(tmp_path):
    (tmp_path / "part_1").mkdir()
    (tmp_path / "part_2").mkdir()
    (tmp_path / "part_1" / "a.jpg").write_bytes(b"x")
    (tmp_path / "part_2" / "b.png").write_bytes(b"y")
    m = _manifest([("a", "G1", 0, "train"), ("b", "G2", 1, "test")])
    paths = splits.image_paths(m, [tmp_path / "part_1", tmp_path / "part_2"])
    assert paths == {"a": tmp_path / "part_1" / "a.jpg", "b": tmp_path / "part_2" / "b.png"}


def test_review_sheet_is_written(tmp_path):
    for name in ("a", "b"):
        Image.new("RGB", (20, 20), (120, 80, 60)).save(tmp_path / f"{name}.png")
    paths = {n: tmp_path / f"{n}.png" for n in ("a", "b")}
    pairs = pd.DataFrame({"a": ["a"], "b": ["b"], "similarity": [0.98],
                          "split_a": ["train"], "split_b": ["test"]})
    out = tmp_path / "review.png"
    splits.save_review_sheet(pairs, paths, out)
    assert out.exists() and out.stat().st_size > 0


# ---- Writing and verifying shipped manifests ------------------------------------------

def test_written_manifests_verify_as_identical(tmp_path):
    data_dir = _fake_ham(tmp_path / "data")
    written = splits.write_manifests("ham10000", data_dir, tmp_path / "splits", seeds=[42, 123])
    assert [p.name for p in written] == ["ham10000_seed42.csv", "ham10000_seed123.csv"]
    report = splits.verify_manifests("ham10000", data_dir, tmp_path / "splits", seeds=[42, 123])
    for seed in (42, 123):
        assert report[seed]["identical"] is True
        assert report[seed]["leakage"]["groups_in_multiple_splits"] == 0
        # The fake data is not the real HAM10000, so the paper's logged sizes differ.
        assert report[seed]["paper_sizes"]["match"] is False


def test_verify_detects_a_moved_image(tmp_path):
    data_dir = _fake_ham(tmp_path / "data")
    splits.write_manifests("ham10000", data_dir, tmp_path / "splits", seeds=[42])
    path = tmp_path / "splits" / "ham10000_seed42.csv"
    m = pd.read_csv(path)
    m.loc[m.index[0], "split"] = "val" if m.loc[m.index[0], "split"] != "val" else "test"
    m.to_csv(path, index=False)
    report = splits.verify_manifests("ham10000", data_dir, tmp_path / "splits", seeds=[42])
    assert report[42]["identical"] is False and report[42]["n_differences"] == 1


def test_duplicate_report_counts_cross_split_pairs_per_seed():
    seed1 = _manifest([("a", "G1", 0, "train"), ("b", "G2", 0, "test"),
                       ("c", "G3", 1, "val"), ("d", "G4", 1, "val")])
    seed2 = _manifest([("a", "G1", 0, "train"), ("b", "G2", 0, "train"),
                       ("c", "G3", 1, "val"), ("d", "G4", 1, "test")])
    exact = pd.DataFrame({"a": ["a"], "b": ["b"]})
    near = pd.DataFrame({"a": ["a", "c"], "b": ["b", "d"], "similarity": [0.97, 0.91]})
    out = splits.duplicate_report({1: seed1, 2: seed2}, exact, near, thresholds=(0.90, 0.95))
    assert out[1] == {"exact_cross_split": 1, "near_cross_split": {0.90: 1, 0.95: 1}}
    assert out[2] == {"exact_cross_split": 0, "near_cross_split": {0.90: 1, 0.95: 0}}


# ---- Command-line scripts ---------------------------------------------------------

def _run(script, *args):
    import subprocess
    root = Path(__file__).parent.parent
    return subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, cwd=root, timeout=300)


def test_make_splits_cli_writes_then_verifies(tmp_path):
    data_dir = _fake_ham(tmp_path / "data")
    out = tmp_path / "splits"
    wrote = _run("make_splits.py", "--dataset", "ham10000", "--data-dir", data_dir,
                 "--out", out, "--seeds", 42)
    assert wrote.returncode == 0, wrote.stderr[-1500:]
    assert (out / "ham10000_seed42.csv").exists()
    ok = _run("make_splits.py", "--dataset", "ham10000", "--data-dir", data_dir,
              "--out", out, "--seeds", 42, "--verify", "--no-paper-check")
    assert ok.returncode == 0, ok.stdout[-1500:]


def test_make_splits_cli_verify_fails_on_tampered_manifest(tmp_path):
    data_dir = _fake_ham(tmp_path / "data")
    out = tmp_path / "splits"
    _run("make_splits.py", "--dataset", "ham10000", "--data-dir", data_dir,
         "--out", out, "--seeds", 42)
    path = out / "ham10000_seed42.csv"
    m = pd.read_csv(path)
    m = m.iloc[1:]                      # drop one image
    m.to_csv(path, index=False)
    bad = _run("make_splits.py", "--dataset", "ham10000", "--data-dir", data_dir,
               "--out", out, "--seeds", 42, "--verify", "--no-paper-check")
    assert bad.returncode != 0


def test_check_splits_cli_runs_on_manifests_alone(tmp_path):
    data_dir = _fake_ham(tmp_path / "data")
    out = tmp_path / "splits"
    _run("make_splits.py", "--dataset", "ham10000", "--data-dir", data_dir,
         "--out", out, "--seeds", 42)
    rep = tmp_path / "reports"
    res = _run("check_splits.py", "--dataset", "ham10000", "--manifests", out,
               "--seeds", 42, "--out", rep)
    assert res.returncode == 0, res.stderr[-1500:]
    import json
    report = json.loads((rep / "ham10000.json").read_text())
    assert report["seeds"]["42"]["leakage"]["groups_in_multiple_splits"] == 0


def test_image_checks_find_a_planted_duplicate(tmp_path):
    """Two different lesions share a byte-identical image file: both checks must see it."""
    import importlib.util
    import json
    import torch.nn as nn

    data_dir = _fake_ham(tmp_path / "data", n_lesions=30)
    meta = pd.read_csv(data_dir / "HAM10000_metadata.csv")
    img_dir = data_dir / "HAM10000_images_part_1"
    img_dir.mkdir()
    rng = np.random.default_rng(1)
    for image_id in meta["image_id"]:
        Image.fromarray(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)).save(
            img_dir / f"{image_id}.png")
    first, other = meta.groupby("lesion_id")["image_id"].first().iloc[[0, 1]]
    (img_dir / f"{other}.png").write_bytes((img_dir / f"{first}.png").read_bytes())

    out = tmp_path / "splits"
    splits.write_manifests("ham10000", data_dir, out, seeds=[42, 123])
    spec = importlib.util.spec_from_file_location(
        "check_splits", Path(__file__).parent.parent / "scripts" / "check_splits.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    mean_colour = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten())
    rep = tmp_path / "reports"
    cli.main(["--dataset", "ham10000", "--manifests", str(out), "--seeds", "42", "123",
              "--data-dir", str(data_dir), "--out", str(rep), "--device", "cpu"],
             embed_model=mean_colour)

    report = json.loads((rep / "ham10000.json").read_text())
    assert report["images"]["exact_duplicate_pairs"] == 1
    assert report["images"]["near_duplicate_pairs"] >= 1
    for seed in ("42", "123"):
        m = splits.load_manifest(splits.manifest_path(out, "ham10000", int(seed)))
        split_of = dict(zip(m["image_id"], m["split"]))
        crosses = int(split_of[first] != split_of[other])
        assert report["seeds"][seed]["duplicates"]["exact_cross_split"] == crosses
    assert (rep / "ham10000_review.png").exists()


def test_test_twin_counts_counts_each_test_image_once():
    m = _manifest([("a", "G1", 0, "train"), ("b", "G2", 0, "test"), ("c", "G3", 1, "test"),
                   ("d", "G4", 1, "train"), ("e", "G5", 0, "val")])
    pairs = pd.DataFrame({"a": ["a", "d", "a", "b", "d"],
                          "b": ["b", "c", "e", "c", "b"],
                          "similarity": [0.99, 0.96, 0.99, 0.99, 0.985]})
    out = splits.test_twin_counts(pairs, m, thresholds=(0.98, 0.95))
    # >= 0.98: train-test pairs a-b and d-b; only test image b has a train twin
    assert out[0.98] == {"train_test_pairs": 2, "test_images": 1, "test_fraction": 0.5}
    # >= 0.95 adds d-c; train-val (a-e) and test-test (b-c) pairs never count
    assert out[0.95] == {"train_test_pairs": 3, "test_images": 2, "test_fraction": 1.0}


def test_summarize_duplicates_cli_writes_table_from_shipped_files(tmp_path):
    import json
    sp, ck = tmp_path / "splits", tmp_path / "checks"
    sp.mkdir(); ck.mkdir()
    _manifest([("a", "G1", 0, "train"), ("b", "G2", 0, "test"), ("c", "G3", 1, "test"),
               ("d", "G4", 1, "train")]).to_csv(sp / "ham10000_seed42.csv", index=False)
    pd.DataFrame({"a": ["a", "d"], "b": ["b", "c"], "similarity": [0.99, 0.96]}).to_csv(
        ck / "ham10000_cross_split_candidates.csv", index=False)
    res = _run("summarize_duplicates.py", "--datasets", "ham10000", "--seeds", 42,
               "--splits", sp, "--checks", ck)
    assert res.returncode == 0, res.stderr[-1500:]
    summary = json.loads((ck / "duplicates_summary.json").read_text())
    assert summary["ham10000"]["0.98"]["test_images_mean"] == 1
    assert summary["ham10000"]["0.95"]["test_fraction_mean"] == 1.0
    assert (ck / "duplicates_summary.md").exists()
