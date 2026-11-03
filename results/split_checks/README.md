# Split and duplicate checks

Real-data checks of the saved split manifests in `splits/` (5 seeds x 3 datasets),
run on Kaggle with the official downloads (HAM10000 from Harvard Dataverse,
ISIC 2020 variant `nroman/melanoma-external-malignant-256`, ISIC-DICM-17K from
the ISIC Archive). Logs, test output and the software environment of that run
are in `kaggle_run/`.

## 1. Lesion / patient overlap and split sizes

For every dataset and seed, no image and no lesion (HAM10000, ISIC-DICM-17K) or
patient (ISIC 2020) appears in more than one of train / validation / test, and
the split sizes equal those logged by the paper's training runs
(`kaggle_run/verify_splits.log`, `<dataset>.md`). The HAM10000 and ISIC-DICM-17K
manifests rebuilt on Kaggle are byte-identical to the shipped ones.

## 2. Duplicate images across splits

- **Byte-identical files** (md5): none across splits in any dataset or seed.
- **Near-identical images**: cosine similarity of ImageNet ConvNeXt-Base
  embeddings, following the embedding-plus-visual-review approach of Abhishek
  et al. (Sci. Data 2025) for HAM10000. Candidate pairs that fall in different
  splits are in `<dataset>_cross_split_candidates.csv` (cosine >= 0.95) and the
  40 most similar ones are shown in `<dataset>_review.jpg`.

### Visual review

The most similar cross-split pairs were inspected (all 40 per dataset at reduced
size, the top 6 at full size in `<dataset>_review_top6.jpg`):

| Dataset | Top pairs (cosine) | Finding |
|---|---|---|
| ISIC-DICM-17K | 1.000 | The same photograph stored under two ISIC identifiers with different lesion ids (re-uploads to the archive). |
| ISIC 2020 | 0.983 to 0.996 | The same photograph: a downsampled ISIC 2019 copy and its ISIC 2020 re-upload. A different lesion already appears at 0.982. |
| HAM10000 | 0.969 to 0.979 | Mostly separate photographs of the same lesion that carry different `lesion_id`s; one pair at 0.969 shows two different lesions. |

### How much of the test set is affected

`duplicates_summary.md` (from `python scripts/summarize_duplicates.py`) gives,
per cosine threshold, the test images that have a near-identical twin in the
training set, averaged over the 5 seeds. Reading it with the visual review:

| Dataset | Lower bound: near-identical (cosine >= 0.98) | Upper bound | Upper-bound threshold |
|---|---|---|---|
| HAM10000 | 0.00% | 1.61% | 0.95 (same-lesion photographs reach down to 0.969) |
| ISIC 2020 | 0.02% | 0.48% | 0.97 |
| ISIC-DICM-17K | 0.86% | 1.22% | 0.97 |

Fewer than 2% of test images are affected in every dataset. All methods are
compared on the same partitions, so the method comparisons (SRS, mitigation
ranking) are not affected.

## Reproduce

```bash
python scripts/check_splits.py --dataset all                 # ID-level checks, no download
python scripts/check_splits.py --dataset ham10000 --data-dir data/raw/ham10000   # with images
python scripts/summarize_duplicates.py                       # table above, no images needed
```
