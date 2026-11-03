# Model card: released HAM10000 shortcut-testbed checkpoints

Structure follows Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).

## Model details

- **Models:** two image + clinical-note late-fusion classifiers, `naive` and
  `counterfactual_aug`, from the mitigation bake-off of the accompanying article
  ("Detecting and mitigating language shortcuts in multimodal AI for skin cancer
  diagnosis", Govercin, Wishart and Raza, submitted to Scientific Reports).
- **Architecture:** ConvNeXt-Base image encoder (ImageNet-pretrained, 1024-d) and
  BioClinicalBERT note encoder (768-d, [CLS] token), concatenated and passed to one
  linear layer with a sigmoid output, P(malignant). About 198M parameters.
- **Files:** `shortcut_ham10000_naive_p1.0_seed42.pt` and
  `shortcut_ham10000_counterfactual_aug_p1.0_seed42.pt` (about 0.8 GB each, weights
  only, PyTorch `state_dict` plus run metadata). Archived on Zenodo (link in README).
- **Training:** HAM10000, lesion-level 60/20/20 split for seed 42 (`splits/`),
  synthetic notes at the DFilt level with leakage p = 1.0, AdamW (learning rate 1e-5,
  weight decay 0.01, cosine schedule), 5 epochs, batch size 16, class-weighted
  binary cross-entropy, checkpoint with the best validation AUROC.
  `counterfactual_aug` sees every training note twice, once with the true and once
  with the flipped diagnosis-bearing text, both labelled with the true diagnosis.
- **Provenance:** retrained with the released code on Kaggle (Tesla T4, PyTorch 2.11,
  CUDA 13.0); not the exact weights behind the article's tables, which were trained
  on an RTX 5080 and not kept. See `results/checkpoint_eval/` for the comparison.
- **Licence:** code MIT. The weights are derived from HAM10000 (CC BY-NC 4.0) and are
  released for non-commercial research use only.

## Intended use

- Reproducing and inspecting the article's results without retraining.
- Research on shortcut learning: computing the Shortcut Reliance Score (SRS),
  FlipRate and FFilt-gap, and comparing a shortcut-reliant model (`naive`) with a
  mitigated one (`counterfactual_aug`).

## Out-of-scope use

- **Not for clinical use.** The models must not be used to diagnose, triage or
  inform the care of any patient.
- Not for real clinical notes: they were trained on synthetic notes generated from
  public labels and metadata.
- `naive` is deliberately shortcut-reliant (SRS about 1.0): its output follows the
  note, not the lesion. It is released only as a negative control.

## Evaluation

HAM10000 test split for seed 42 (2,042 images), notes at DFilt with p = 1.0.
Visible AUROC uses the leaky notes; honest AUROC uses facts-only (FFilt) notes;
SRS and FlipRate come from the concordant/flipped note pairs.

| Model | Visible AUROC | Honest AUROC | SRS | FlipRate |
|---|---|---|---|---|
| naive | 1.000 | 0.740 | 0.998 | 1.000 |
| counterfactual_aug | 0.924 | 0.924 | 0.003 | 0.001 |

Reproduce with `python scripts/evaluate_checkpoint.py --checkpoint <file> --data-dir
data/raw/ham10000`. Re-evaluating a saved checkpoint gives exactly the metrics computed
at the end of training. Against the article's seed-42 runs, `counterfactual_aug`
agrees within 0.001 on AUROC and SRS; the naive model's honest AUROC (0.740 against
0.709) lies inside its seed-to-seed range in the article (0.636 to 0.758).

## Factors, limitations and ethical considerations

- **Synthetic text.** The note encoder only ever saw template notes; behaviour on
  real clinician-written text is unknown.
- **Population.** HAM10000 comes from two centres (Vienna and Queensland) and does not
  record skin type; its populations are predominantly fair-skinned, so performance on
  darker skin types, other populations and other devices is untested.
- **Image artefacts.** The honest AUROC isolates the model from the text leak but not
  from image artefacts such as rulers or ink markings that correlate with malignancy
  in dermoscopy archives.
- **Single seed and hardware.** These are single-seed models; results vary across
  seeds and GPUs (see `results/checkpoint_eval/README.md`).
