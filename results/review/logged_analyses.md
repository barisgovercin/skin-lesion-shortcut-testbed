# Review analyses from the logged per-seed results

## Bake-off comparisons with Holm correction

Paired t-tests over the five shared seeds, counterfactual augmentation minus the other method. Holm adjustment within each dataset (8 tests) and across all 32.

| dataset | metric | comparison | mean diff | p | Holm (dataset) | Holm (all) |
|---|---|---|---|---|---|---|
| HAM10000 | honest AUROC | vs consistency | +0.0034 | 0.042 | 0.042 | 0.47 |
| HAM10000 | honest AUROC | vs ffilt_only | +0.0100 | 0.0058 | 0.029 | 0.11 |
| HAM10000 | honest AUROC | vs adversarial | +0.0297 | 0.017 | 0.035 | 0.24 |
| HAM10000 | honest AUROC | vs naive | +0.2162 | 7.8e-04 | 0.0055 | 0.02 |
| HAM10000 | SRS | vs consistency | +0.0012 | 0.0036 | 0.021 | 0.075 |
| HAM10000 | SRS | vs ffilt_only | -0.0031 | 0.0088 | 0.035 | 0.15 |
| HAM10000 | SRS | vs adversarial | -0.0981 | 0.0095 | 0.035 | 0.15 |
| HAM10000 | SRS | vs naive | -0.9943 | 1.5e-13 | 1.2e-12 | 4.4e-12 |
| ISIC 2020 | honest AUROC | vs consistency | +0.0010 | 0.18 | 0.35 | 0.7 |
| ISIC 2020 | honest AUROC | vs ffilt_only | +0.0024 | 8.1e-04 | 0.0057 | 0.02 |
| ISIC 2020 | honest AUROC | vs adversarial | +0.0161 | 0.051 | 0.17 | 0.47 |
| ISIC 2020 | honest AUROC | vs naive | +0.0907 | 0.0067 | 0.034 | 0.12 |
| ISIC 2020 | SRS | vs consistency | +0.0002 | 0.23 | 0.35 | 0.7 |
| ISIC 2020 | SRS | vs ffilt_only | -0.0012 | 0.0037 | 0.022 | 0.075 |
| ISIC 2020 | SRS | vs adversarial | -0.1234 | 0.044 | 0.17 | 0.47 |
| ISIC 2020 | SRS | vs naive | -0.9981 | 5.6e-15 | 4.5e-14 | 1.8e-13 |
| ISIC-DICM-17K | honest AUROC | vs consistency | +0.0016 | 0.43 | 0.86 | 0.86 |
| ISIC-DICM-17K | honest AUROC | vs ffilt_only | +0.0088 | 0.037 | 0.15 | 0.44 |
| ISIC-DICM-17K | honest AUROC | vs adversarial | +0.1043 | 0.044 | 0.15 | 0.47 |
| ISIC-DICM-17K | honest AUROC | vs naive | +0.2090 | 4.6e-04 | 0.0028 | 0.012 |
| ISIC-DICM-17K | SRS | vs consistency | +0.0001 | 0.92 | 0.92 | 0.92 |
| ISIC-DICM-17K | SRS | vs ffilt_only | -0.0031 | 0.012 | 0.06 | 0.18 |
| ISIC-DICM-17K | SRS | vs adversarial | -0.6762 | 1.3e-05 | 8.8e-05 | 3.5e-04 |
| ISIC-DICM-17K | SRS | vs naive | -0.9955 | 7.4e-14 | 5.9e-13 | 2.3e-12 |
| HAM10000, LLM notes | honest AUROC | vs consistency | +0.0032 | 0.048 | 0.14 | 0.47 |
| HAM10000, LLM notes | honest AUROC | vs ffilt_only | +0.0106 | 0.0027 | 0.013 | 0.059 |
| HAM10000, LLM notes | honest AUROC | vs adversarial | +0.0104 | 0.019 | 0.076 | 0.25 |
| HAM10000, LLM notes | honest AUROC | vs naive | +0.2723 | 1.6e-04 | 0.0011 | 0.0043 |
| HAM10000, LLM notes | SRS | vs consistency | +0.0017 | 0.071 | 0.14 | 0.47 |
| HAM10000, LLM notes | SRS | vs ffilt_only | +0.0016 | 0.07 | 0.14 | 0.47 |
| HAM10000, LLM notes | SRS | vs adversarial | -0.0764 | 0.0022 | 0.013 | 0.05 |
| HAM10000, LLM notes | SRS | vs naive | -0.9916 | 1.7e-12 | 1.4e-11 | 5.0e-11 |

Significant before but not after the within-dataset Holm correction: 6 of 25.
- ISIC 2020, SRS, counterfactual_aug vs adversarial: p = 0.044 -> 0.17
- ISIC-DICM-17K, honest AUROC, counterfactual_aug vs ffilt_only: p = 0.037 -> 0.15
- ISIC-DICM-17K, honest AUROC, counterfactual_aug vs adversarial: p = 0.044 -> 0.15
- ISIC-DICM-17K, SRS, counterfactual_aug vs ffilt_only: p = 0.012 -> 0.06
- HAM10000, LLM notes, honest AUROC, counterfactual_aug vs consistency: p = 0.048 -> 0.14
- HAM10000, LLM notes, honest AUROC, counterfactual_aug vs adversarial: p = 0.019 -> 0.076

## Early warning in the dose-response (naive model)

One-sided paired t-test of each leakage level against p = 0.5 (SRS rising, honest AUROC falling), Holm-adjusted across levels. 'Share' is the fraction of the full p = 0.5 to 1.0 change already reached.


**HAM10000.** First significant level: SRS p = 0.6, honest AUROC p = 0.6.

| p | SRS | share | Holm p | honest AUROC | share | Holm p |
|---|---|---|---|---|---|---|
| 0.6 | 0.086 | 0.07 | 1.0e-04 | 0.916 | 0.01 | 0.034 |
| 0.7 | 0.174 | 0.16 | 1.0e-05 | 0.914 | 0.02 | 0.0065 |
| 0.8 | 0.297 | 0.29 | 4.6e-06 | 0.911 | 0.03 | 0.0037 |
| 0.9 | 0.467 | 0.46 | 1.7e-07 | 0.903 | 0.07 | 0.0023 |
| 1.0 | 0.998 | 1.00 | 2.2e-10 | 0.711 | 1.00 | 0.0019 |

**ISIC 2020.** First significant level: SRS p = 0.6, honest AUROC p = 0.7.

| p | SRS | share | Holm p | honest AUROC | share | Holm p |
|---|---|---|---|---|---|---|
| 0.6 | 0.040 | 0.03 | 2.6e-05 | 0.985 | 0.00 | 0.21 |
| 0.7 | 0.083 | 0.08 | 2.6e-05 | 0.985 | 0.01 | 0.029 |
| 0.8 | 0.144 | 0.14 | 1.1e-05 | 0.984 | 0.01 | 0.013 |
| 0.9 | 0.247 | 0.24 | 2.6e-05 | 0.983 | 0.03 | 0.029 |
| 1.0 | 0.999 | 1.00 | 2.0e-11 | 0.897 | 1.00 | 0.015 |

**ISIC-DICM-17K.** First significant level: SRS p = 0.6, honest AUROC p = 0.6.

| p | SRS | share | Holm p | honest AUROC | share | Holm p |
|---|---|---|---|---|---|---|
| 0.6 | 0.120 | 0.10 | 8.8e-06 | 0.879 | 0.00 | 0.011 |
| 0.7 | 0.246 | 0.23 | 2.2e-06 | 0.877 | 0.01 | 0.011 |
| 0.8 | 0.400 | 0.39 | 2.4e-06 | 0.870 | 0.04 | 0.0016 |
| 0.9 | 0.612 | 0.61 | 2.2e-06 | 0.863 | 0.08 | 0.011 |
| 1.0 | 0.999 | 1.00 | 8.2e-10 | 0.679 | 1.00 | 0.0016 |

**HAM10000, LLM notes.** First significant level: SRS p = 0.6, honest AUROC p = 0.7.

| p | SRS | share | Holm p | honest AUROC | share | Holm p |
|---|---|---|---|---|---|---|
| 0.6 | 0.094 | 0.08 | 2.6e-04 | 0.916 | 0.00 | 0.1 |
| 0.7 | 0.181 | 0.17 | 9.4e-05 | 0.914 | 0.01 | 0.031 |
| 0.8 | 0.305 | 0.29 | 2.0e-05 | 0.908 | 0.03 | 0.011 |
| 0.9 | 0.472 | 0.46 | 1.9e-06 | 0.903 | 0.05 | 0.011 |
| 1.0 | 0.998 | 1.00 | 3.8e-09 | 0.656 | 1.00 | 3.9e-04 |
