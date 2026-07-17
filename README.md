# Multimodal Skin Lesion Classification and Shortcut-Learning Testbed

A research codebase for multimodal (image + clinical free text + metadata) skin
lesion classification, with a controllable testbed for studying and mitigating
**shortcut learning** from clinical free text.

Multimodal skin cancer models can reach very high accuracy by exploiting leading
language in the accompanying clinical notes rather than the lesion image itself.
This repository provides:

- A late-fusion multimodal pipeline (vision + text + metadata).
- A synthetic clinical-text engine with four filtering levels and a controllable
  leakage probability `p`.
- A **Shortcut Reliance Score (SRS)** flip-test metric that measures how much a
  model depends on the text shortcut.
- A **mitigation bake-off**: naive / filtered-text-only / counterfactual
  augmentation / consistency / adversarial training.

## Datasets

The code supports three public datasets (downloaded separately; not included):

- **HAM10000** (10,015 dermatoscopic images, 7 classes collapsed to benign/malignant).
- **ISIC 2020** (binary melanoma classification).
- **ISIC-DICM-17K** (17,060 images, balanced melanoma / non-melanoma).

Data lives under `data/raw/` and is git-ignored.

## Architecture

```
Image (224x224) -> ConvNeXt (pretrained) -> vision features ─┐
Clinical text    -> BioClinicalBERT       -> text features   ─┼─> concat -> linear -> P(malignant)
Metadata         -> MLP                    -> meta features   ─┘
```

## Project structure

```
src/
  data/          datasets, transforms, per-patient splits, metadata encoding
  models/        vision (ConvNeXt/EfficientNet/ResNet), text (BioClinicalBERT), fusion
  preprocessing/ synthetic clinical text (4 filter levels, controllable leakage), LLM-realistic notes
  training/      trainer, debiasing (consistency / adversarial), gradient reversal
  evaluation/    SRS flip test, Integrated-Gradients attribution, calibration + decision-curve analysis, plots
scripts/         experiment runners, shortcut sweeps, figures, clinical evaluation, demo
configs/         YAML experiment configs
tests/           offline unit tests (no GPU / data / network required)
```

## Setup

```bash
pip install -r requirements.txt
# Install a CUDA-matched PyTorch build separately for your platform.
```

## Usage

```bash
# Vision-only and vision+metadata baselines (5 seeds)
python scripts/run_experiments.py --mode all --config configs/baseline_ham10000.yaml

# Shortcut testbed: dose-response (vary leakage_p) and mitigation bake-off
python scripts/run_shortcut.py --config configs/shortcut_ham10000.yaml --mode dose
python scripts/run_shortcut.py --config configs/shortcut_ham10000.yaml --mode bakeoff

# Text modality (BioClinicalBERT), leading-language level via data.text_filter_level
python scripts/run_experiments.py --mode vision_text --config configs/text_ham10000.yaml

# Clinical evaluation: calibration + decision-curve analysis
python scripts/clinical_eval.py

# Publication figures from cached results (no retraining)
python scripts/make_figures.py --what all

# Interactive SRS demo (CPU, no GPU needed)
python -m scripts.srs_demo
```

The experiment runner picks the dataset from `data.dataset` in the config
(`ham10000`, `isic2020`, `isic_dicm_17k`) and writes per-dataset result JSONs so
runs never overwrite each other.

## Tests

Offline unit tests (no GPU, dataset, or model download needed):

```bash
pytest tests/
```

## License

MIT
