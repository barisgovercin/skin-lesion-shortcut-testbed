"""Write, or verify, the saved 60/20/20 split manifests for the paper's five seeds.

The splits are built with the same prepare_* functions the training runs use
(lesion-level for HAM10000 and ISIC-DICM-17K, patient-level for ISIC 2020), and
need only the metadata CSVs, not the images.

    # write splits/<dataset>_seed<seed>.csv
    python scripts/make_splits.py --dataset ham10000 --data-dir data/raw/ham10000

    # rebuild from your own download and compare with the shipped manifests and
    # the split sizes logged by the paper's runs; exit code 1 on any mismatch
    python scripts/make_splits.py --dataset all --verify
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import PREPARE, SEEDS, verify_manifests, write_manifests  # noqa: E402


def _print_verify(dataset, report, paper_check=True):
    ok = True
    print(f"\n{dataset}")
    print(f"  {'seed':>6}  {'identical to shipped':>21}  {'sizes (train/val/test)':>24}"
          f"  {'paper sizes':>11}  {'lesion/patient leak':>19}")
    for seed, r in report.items():
        sizes = r["leakage"]["sizes"]
        paper = r["paper_sizes"]
        leak = r["leakage"]["groups_in_multiple_splits"] + r["leakage"]["images_in_multiple_splits"]
        paper_txt = "n/a" if paper is None else ("match" if paper["match"] else "DIFFER")
        print(f"  {seed:>6}  {('yes' if r['identical'] else 'NO (%d rows)' % r['n_differences']):>21}"
              f"  {'%d/%d/%d' % (sizes['train'], sizes['val'], sizes['test']):>24}"
              f"  {paper_txt:>11}  {('none' if leak == 0 else 'FOUND %d' % leak):>19}")
        ok &= r["identical"] and leak == 0 and (not paper_check or paper is None or paper["match"])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=[*PREPARE, "all"], required=True)
    ap.add_argument("--data-dir", help="dataset folder (default: data/raw/<dataset>)")
    ap.add_argument("--out", default=str(ROOT / "splits"), help="manifest folder")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--verify", action="store_true",
                    help="compare with the manifests in --out instead of writing them")
    ap.add_argument("--no-paper-check", action="store_true",
                    help="with --verify, do not require the paper's logged split sizes "
                         "(e.g. for a modified dataset)")
    args = ap.parse_args(argv)

    datasets = list(PREPARE) if args.dataset == "all" else [args.dataset]
    if args.data_dir and len(datasets) > 1:
        ap.error("--data-dir needs a single --dataset")
    ok = True
    for ds in datasets:
        data_dir = args.data_dir or str(ROOT / "data" / "raw" / ds)
        if args.verify:
            ok &= _print_verify(ds, verify_manifests(ds, data_dir, args.out, args.seeds),
                                paper_check=not args.no_paper_check)
        else:
            for path in write_manifests(ds, data_dir, args.out, args.seeds):
                print(f"wrote {path}")
    if args.verify:
        print("\nVERIFIED: splits match" if ok else "\nVERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
