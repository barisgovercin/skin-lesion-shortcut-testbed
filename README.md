# Detecting and mitigating language shortcuts in multimodal skin cancer AI

Code, data splits, results and released models for the article

> Govercin M.B., Wishart G.C., Raza H. *Detecting and mitigating language shortcuts in
> multimodal AI for skin cancer diagnosis.* Submitted to Scientific Reports.

Multimodal skin cancer models that read the dermoscopic image together with a clinical
note can reach a near-perfect AUROC by reading the diagnosis in the note instead of
examining the lesion. This repository provides a public, reproducible testbed to
measure and mitigate that failure: a synthetic clinical-text engine with a
controllable leakage dial, the **Shortcut Reliance Score (SRS)**, and five mitigation
methods, evaluated on three public datasets.

**Contents.**
[Overview](#1-overview) ·
[System requirements](#2-system-requirements) ·
[Installation](#3-installation) ·
[GPU-free demo](#4-gpu-free-demo) ·
[Verifying the results](#5-verifying-the-papers-results) ·
[Datasets](#6-datasets) ·
[Synthetic clinical text](#7-synthetic-clinical-text-engine) ·
[Training and evaluation](#8-training-and-evaluation) ·
[Released models](#9-released-models) ·
[Results](#10-results) ·
[Repository structure](#11-repository-structure) ·
[Citation and licences](#12-citation-and-licences)

## 1. Overview

**Research objectives.** (1) Build a privacy-preserving, fully reproducible
substitute for private clinical free text on public skin-lesion data. (2) Measure, per
patient, how much a multimodal model relies on diagnosis-bearing language. (3) Show how
reliance grows as the text leaks the label more reliably. (4) Compare mitigations.
(5) Test whether the findings hold across three datasets.

**Method in brief.**

- **Model:** late fusion as in Watson et al. (2026). ConvNeXt-Base encodes the image
  (224 x 224, 1024-d), BioClinicalBERT encodes the note (768-d), and one linear layer on
  the concatenated vector outputs P(malignant).
- **Synthetic notes:** built from each lesion's label and metadata (age, sex, site).
  Four filter levels as in Watson et al.: Orig, CFilt (condition names removed), DFilt
  (also the words benign/malignant removed) and FFilt (facts only). A leakage dial
  p in [0.5, 1.0] sets how often the diagnosis-bearing sentences match the true label.
  Every note has a flipped twin with identical facts and the opposite verdict.
- **Shortcut Reliance Score:** for each test lesion, the model scores the same image
  with the original and with the flipped note;
  `SRS = mean |p_orig - p_flip|`, and `FlipRate` is the share of lesions whose
  predicted class changes. SRS applies the counterfactual token fairness gap of
  Garg et al. (AIES 2019) to diagnosis-bearing clinical text. **FFilt-gap** is the drop
  in AUROC when the model sees facts-only notes; the AUROC on those notes is the
  *honest* AUROC.
- **Mitigations:** naive, filtering (`ffilt_only`), counterfactual augmentation,
  consistency regularisation and adversarial debiasing with gradient reversal.

## 2. System requirements

| | Tested |
|---|---|
| Operating systems | Windows 11 (all paper experiments), Linux (Kaggle: released checkpoints, split and duplicate checks), macOS on Apple silicon (tests, demo, table reproduction) |
| Python | 3.11, 3.12, 3.13 |
| Packages | `requirements.txt`, pinned to the environment in which the released checkpoints were trained (PyTorch 2.11.0, transformers 5.20.2). The paper's runs used PyTorch 2.11.0 + CUDA 12.8, transformers 5.10.2, Python 3.11.9. |
| Hardware | Demo, tests, table and figure reproduction: any CPU. Training: an NVIDIA GPU with 16 GB (RTX 5080 for the paper, Tesla T4 for the released checkpoints). |

## 3. Installation

```bash
git clone https://github.com/barisgovercin/skin-lesion-shortcut-testbed.git
cd skin-lesion-shortcut-testbed
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a GPU, install the CUDA build of `torch` and `torchvision` for your platform first
(<https://pytorch.org/get-started/locally/>). **Typical install time:** about one minute
(43 s measured for the full requirements in a fresh environment on a laptop).

## 4. GPU-free demo

The demo shows the SRS flip test and the counterfactual note pairs without any dataset,
GPU or model download.

```bash
pip install numpy pandas scikit-learn scipy     # enough for the demo; PyTorch not needed
python scripts/srs_demo.py
```

- **Run time:** about 10 s on a laptop CPU (9.8 s measured on an Apple M4 Pro).
- **Data:** toy data only (synthetic patients and a synthetic image signal), run through
  the paper's own note engine and metric code. It shows the mechanism; it does not
  reproduce the paper's numbers. Part 4 of the output prints the paper's real 5-seed
  results from `results/`.

The output has four parts: (1) the flip test for one patient, with automatic checks that
the patient facts are identical and only the diagnosis-bearing sentence differs;
(2) known-answer controls (an image-only model must score SRS = 0, a text-only model
close to 1); (3) a mini testbed with a dose-response over `p` and the mitigations at
`p = 1.0`; (4) the paper's results with a paired test of counterfactual augmentation
against filtering. Expected output (abridged; the last digit can vary across library
versions):

```
[2] Known-answer controls (p = 1.0, mean over seeds)
  model              SRS  FlipRate   expected
  image-only       0.000     0.000   0  (never reads the note)
  text-only        0.933     1.000   ~1 (copies the note)

  b) Mitigation at p = 1.0
  method               visible AUROC  honest AUROC      SRS
  naive                        1.000         0.852   0.8720
  ffilt_only                   0.852         0.852   0.0001
  counterfactual_aug           0.849         0.848   0.0022

Done in 9.8 s on CPU (3000 toy patients x 5 seeds).
```

To score your own model, pass its predicted probabilities on the original and flipped
notes to the metric:

```python
from src.evaluation.shortcut import srs_from_probs
srs_from_probs(p_original, p_flipped)   # {"srs": ..., "flip_rate": ...}
```

## 5. Verifying the paper's results

Five levels, from no download at all to full retraining:

| Level | What it shows | Command | Needs | Time |
|---|---|---|---|---|
| 1. Code works | The metric, note engine and pipeline behave as specified | `python scripts/srs_demo.py` and `pytest tests/` | nothing | 10 s and 30 s |
| 2. Splits | Train/val/test partitions are the paper's, with no lesion or patient overlap | `python scripts/check_splits.py --dataset all`, then `python scripts/make_splits.py --dataset all --verify` | nothing, then the metadata CSVs (a few MB) | seconds |
| 3. Tables and figures | Every number in the manuscript follows from the logged per-seed results | `python scripts/reproduce_tables.py` and `python scripts/make_paper_figures.py` | nothing | seconds |
| 4. Released models | The released models give the reported metrics on the real test images | `python scripts/evaluate_checkpoint.py --checkpoint <file> --data-dir data/raw/ham10000` | HAM10000 images + checkpoints | minutes (GPU) |
| 5. Full retraining | The experiments reproduce from scratch | see [Training and evaluation](#8-training-and-evaluation) | images + GPU | hours |

- `reproduce_tables.py` recomputes the four manuscript tables and checks 140 reported
  numbers and statements against `results/` (`results/reports/manuscript_check.md`); it
  exits with code 1 on any mismatch.
- `make_splits.py --verify` rebuilds the splits from your own download and compares them
  with the shipped manifests and with the split sizes logged by the paper's runs.
- `tests/test_real_data_splits.py` runs the same check inside `pytest` whenever a
  dataset is present under `data/raw/`.

## 6. Datasets

Images are not redistributed. The repository ships only image identifiers
(`splits/`) and results.

| Dataset | Version used | Source | Licence | Cite | Split unit |
|---|---|---|---|---|---|
| HAM10000 (10,015 images, 19.5% malignant) | Harvard Dataverse version 4 | doi:[10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T) | CC BY-NC 4.0 | Tschandl et al., Sci. Data 5, 180161 (2018) | lesion |
| ISIC 2020 (37,648 images, 13.6% malignant) | Kaggle [`nroman/melanoma-external-malignant-256`](https://www.kaggle.com/datasets/nroman/melanoma-external-malignant-256): the ISIC 2020 training set plus external malignant images from earlier ISIC challenges, resized to 256 x 256 | built from the SIIM-ISIC 2020 challenge, doi:[10.34970/2020-ds01](https://doi.org/10.34970/2020-ds01) | CC BY-NC 4.0 (ISIC) | Rotemberg et al., Sci. Data 8, 34 (2021) | patient |
| ISIC-DICM-17K (17,060 images, 50% melanoma) | metadata from [`mmu-dermatology-research/isic-dicm-17k`](https://github.com/mmu-dermatology-research/isic-dicm-17k), images from the ISIC Archive | doi:[10.34970/233480](https://doi.org/10.34970/233480) | per image: CC-0, CC BY 4.0 or CC BY-NC 4.0 | Ahammed et al., CVPRW 2025 | lesion |

Note that the official ISIC 2020 challenge download (33,126 training images) is not the
variant used in the paper; use the Kaggle dataset above to reproduce the ISIC 2020
results.

**Download.** The metadata alone is enough for split verification (level 2).

```bash
python scripts/download_ham10000.py                    # add --metadata-only for the CSV only
python scripts/download_isic_dicm_17k.py               # add --metadata-only for the CSVs only
kaggle datasets download nroman/melanoma-external-malignant-256 -p data/raw/isic2020 --unzip
#   CSV only: add  -f train_concat.csv   (needs a Kaggle API token)
```

Expected layout (the loaders also accept a few common alternatives):

```
data/raw/ham10000/      HAM10000_metadata.csv, HAM10000_images_part_1/, HAM10000_images_part_2/
data/raw/isic2020/      train_concat.csv, train/train/*.jpg
data/raw/isic_dicm_17k/ train-set-metadata.csv, valid-set-metadata.csv, images/
```

**Preprocessing and label mapping** (`src/data/dataset.py`, `prepare_*`):
HAM10000 `mel`, `bcc` and `akiec` are malignant and the other four classes benign;
ISIC 2020 uses `target`; ISIC-DICM-17K uses `class` (the released train and validation
CSVs are pooled and re-split). Images without a patient or lesion identifier become
their own group. Images are resized to 224 x 224 and normalised with ImageNet
statistics; training adds random resized crops, flips, rotation and colour jitter.
Age, sex and site are used only by the note engine, not as a separate model input.

**Splits** (`splits/<dataset>_seed<seed>.csv`, `scripts/make_splits.py`). Stratified
60/20/20 group splits, by lesion for HAM10000 and ISIC-DICM-17K and by patient for
ISIC 2020, for the seeds 42, 123, 456, 789 and 1024. All 15 manifests reproduce the
split sizes logged by the paper's runs and keep every lesion or patient in one split.
An image-level audit (`results/split_checks/README.md`) found no byte-identical images
across splits; near-identical images (re-uploads under a second identifier, and in
HAM10000 separate photographs of the same lesion under different lesion ids) affect
fewer than 2% of test images in every dataset.

## 7. Synthetic clinical-text engine

`src/preprocessing/synthetic_text.py`. Each note is assembled from labelled segments:
Facts (age, sex, site; always truthful), Condition (disease name), Verdict (benign or
malignant) and Leading (referral or treatment wording). The filter level decides which
segments survive; the leakage dial `p` decides how often Condition, Verdict and Leading
match the true label. `generate_pair` returns the concordant note and its flipped twin
(same facts, opposite verdict); `build_counterfactual_df` turns pairs into the
counterfactual-augmentation training set. Generation is deterministic: the wording and
the concordance of each lesion are drawn from a hash of the lesion id and the seed.

**LLM-generated variant (HAM10000).** `src/preprocessing/llm_text.py` keeps the facts
but draws the diagnosis clause from a pool of varied natural phrasings written by a
large language model (Claude Haiku) per filter level and impression. The prompt is in
`_prompt`; condition and verdict words are banned in the prompt and removed by a
post-filter (`_violates`). The generated pool (120 clauses per impression) is shipped in
`data/llm_notes_cache/`, so the LLM experiments run offline:

```bash
python scripts/run_shortcut.py --config configs/shortcut_ham10000_llm.yaml --mode bakeoff
```

To regenerate the pool, bring your own API key; it is read only from the environment and
never written to the repository:

```bash
export ANTHROPIC_API_KEY=your-own-key
export LLM_MODEL=your-provider-model-id
python scripts/generate_llm_notes.py --dataset ham10000 --seeds 42 --provider anthropic
```

## 8. Training and evaluation

**Hyperparameters** (`configs/`): AdamW, learning rate 1e-5, weight decay 0.01, cosine
schedule, 5 epochs, batch size 32 (image only) or 16 (image + text), notes truncated at
128 tokens, class-weighted binary cross-entropy with `w_i = N / (2 n_i)`, the checkpoint
with the best validation AUROC (consistency and adversarial training keep the final
epoch), consistency and adversarial weight 1.0, seeds 42, 123, 456, 789 and 1024.
Reported metrics: AUROC, average precision and specificity at 95% sensitivity (at the
threshold where the evaluated split first reaches 95% sensitivity), plus SRS, FlipRate
and FFilt-gap for the shortcut experiments. Results are mean and 95% t-interval over the
five seeds; methods are compared with paired t-tests (Wilcoxon as a check), Cohen's
d_z and Spearman trend tests.

**Commands, by manuscript item.** Each run writes per-seed JSON to `results/`; running
an experiment overwrites the shipped file of the same name (restore it with
`git checkout results/`).

| Manuscript item | Command |
|---|---|
| Table 1, image only and image + metadata | `python scripts/run_experiments.py --mode all --config configs/baseline_ham10000.yaml` (and `isic2020.yaml`, `isic_dicm_17k.yaml`) |
| Table 1, image + synthetic text | `python scripts/run_experiments.py --mode vision_text --config configs/text_ham10000.yaml` (filter levels: `text_ham10000_{CFilt,DFilt,FFilt}.yaml`) |
| Dose-response (Fig. 2, Table 3) | `python scripts/run_shortcut.py --config configs/shortcut_ham10000.yaml --mode dose` (and `shortcut_isic2020.yaml`, `shortcut_isic_dicm_17k.yaml`) |
| Mitigation bake-off (Table 2, Fig. 3) | same configs with `--mode bakeoff` |
| LLM-generated notes (Fig. 4) | `configs/shortcut_ham10000_llm.yaml` with `--mode dose` and `--mode bakeoff` |
| Net benefit (Table 4) | `python scripts/clinical_eval.py`, then `python scripts/clinical_stats.py` (uses the image-only checkpoints from Table 1) |
| Paired tests | `python scripts/significance_tests.py` |
| Tables and figures from results | `python scripts/reproduce_tables.py`, `python scripts/make_paper_figures.py` |

`--seeds 42` runs a single seed; `--keep-checkpoints DIR` keeps the trained weights.
A training-budget control, `ffilt_only_2x` (filtered notes, every training row twice per
epoch, the same number of steps as counterfactual augmentation), can be added to
`shortcut.methods` in a config; its HAM10000 results are in `results/training_budget_control/`.

**Compute.** One seed of the HAM10000 bake-off takes about 45 min for naive or
`ffilt_only` and about 85 min for counterfactual augmentation on a Tesla T4; an epoch of
the ISIC-DICM-17K text model (10,200 training images) takes about 3 min on an RTX 5080.
A full dose-response and bake-off sweep is 55 training runs per dataset.

**Expected variation across GPUs.** Retraining on a Tesla T4 reproduced the paper's
RTX 5080 runs closely: counterfactual augmentation, seed 42, honest AUROC 0.9241
against 0.9239; filtering, seed 42, 0.9093 against 0.9129. The honest AUROC of the
naive model is unstable by nature (0.636 to 0.758 across the paper's seeds) and moved
from 0.709 to 0.740, while its SRS (0.998) did not change.

## 9. Released models

Two retrained HAM10000 models (seed 42, `p = 1.0`): `naive` and `counterfactual_aug`,
about 0.8 GB each, archived on Zenodo (DOI added at release). See `MODEL_CARD.md` for
intended use, training data and limitations. **Research use only, not for clinical
use.**

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint shortcut_ham10000_counterfactual_aug_p1.0_seed42.pt --data-dir data/raw/ham10000
```

| Model | Visible AUROC | Honest AUROC | SRS | FlipRate |
|---|---|---|---|---|
| naive | 1.000 | 0.740 | 0.998 | 1.000 |
| counterfactual_aug | 0.924 | 0.924 | 0.003 | 0.001 |

Re-evaluating a saved model gives exactly the metrics computed at the end of its
training (`results/checkpoint_eval/`).

## 10. Results

Mitigation bake-off at `p = 1.0`, honest AUROC / SRS, mean of five seeds
(Table 2 of the manuscript; regenerate with `python scripts/reproduce_tables.py`):

| Method | HAM10000 | ISIC 2020 | ISIC-DICM-17K |
|---|---|---|---|
| naive | 0.711 / 0.998 | 0.897 / 0.999 | 0.679 / 0.999 |
| ffilt_only (filtering) | 0.918 / 0.007 | 0.985 / 0.002 | 0.880 / 0.006 |
| counterfactual_aug | 0.928 / 0.004 | 0.987 / 0.001 | 0.889 / 0.003 |
| consistency | 0.924 / 0.002 | 0.986 / 0.001 | 0.887 / 0.003 |
| adversarial | 0.898 / 0.102 | 0.971 / 0.124 | 0.784 / 0.679 |

Image-only AUROC is 0.906, 0.982 and 0.863; adding the synthetic note raises it to
1.000 on all three datasets, which is the shortcut the SRS detects.

## 11. Repository structure

```
src/data/            datasets, splits (lesion/patient level), split manifests and leakage checks
src/preprocessing/   synthetic note engine and LLM note variant
src/models/          ConvNeXt image encoder, BioClinicalBERT text encoder, late fusion
src/training/        trainer, consistency and adversarial (gradient reversal) training
src/evaluation/      SRS / FlipRate / FFilt-gap, attribution, calibration and net benefit
scripts/             experiment runners, downloads, splits, checks, tables, figures, demo
configs/             experiment configurations
splits/              split manifests (image ids only), 3 datasets x 5 seeds
results/             per-seed result JSONs, regenerated tables, figures, split checks,
                     released-model evaluation
data/llm_notes_cache/  cached LLM note pool
tests/               offline unit tests, plus real-data split tests when data is present
```

## 12. Citation and licences

Please cite the article and this software; `CITATION.cff` (GitHub: "Cite this
repository") holds the metadata, and the archived release is on Zenodo.

- **Code:** MIT licence (`LICENSE`).
- **Datasets:** under their own licences (Section 6); not redistributed here.
- **Released model weights:** trained on HAM10000 (CC BY-NC 4.0); non-commercial
  research use only.

Questions about the code: please open a GitHub issue. Correspondence about the article:
the corresponding author listed in the paper.
