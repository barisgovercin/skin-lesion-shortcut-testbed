# Manuscript tables, regenerated from results/

## Table 1. Replication (test AUROC, mean ± 95% CI over 5 seeds)

| Configuration | HAM10000 | ISIC 2020 | ISIC-DICM-17K |
|---|---|---|---|
| Vision only (ConvNeXt-Base) | 0.906 ± 0.008 | 0.982 ± 0.005 | 0.863 ± 0.005 |
| Vision + metadata | 0.907 ± 0.008 | 0.983 ± 0.005 | 0.862 ± 0.008 |
| Vision + synthetic text | 1.000 | 1.000 | 1.000 |

## Table 2. Mitigation bake-off at p = 1.0 (honest AUROC / SRS, mean of 5 seeds)

| Method | HAM10000 | ISIC 2020 | ISIC-DICM-17K |
|---|---|---|---|
| naive | 0.711 / 0.998 | 0.897 / 0.999 | 0.679 / 0.999 |
| ffilt_only | 0.918 / 0.007 | 0.985 / 0.002 | 0.880 / 0.006 |
| counterfactual_aug | 0.928 / 0.004 | 0.987 / 0.001 | 0.889 / 0.003 |
| consistency | 0.924 / 0.002 | 0.986 / 0.001 | 0.887 / 0.003 |
| adversarial | 0.898 / 0.102 | 0.971 / 0.124 | 0.784 / 0.679 |

## Table 3. Generalisation scorecard

| Finding | HAM10000 | ISIC 2020 | ISIC-DICM-17K |
|---|---|---|---|
| Vision baseline (AUROC) | 0.906 | 0.982 | 0.863 |
| Vision + text leaks | 1.000 | 1.000 | 1.000 |
| Honest AUROC at p = 1.0 | 0.711 | 0.897 | 0.679 |
| SRS at p = 1.0 | 0.998 | 0.999 | 0.999 |
| counterfactual_aug honest / SRS | 0.928 / 0.004 | 0.987 / 0.001 | 0.889 / 0.003 |

## Table 4. Net benefit of the calibrated image model

| Dataset | p_t | Model NB | Refer-all NB | Refer-none NB |
|---|---|---|---|---|
| HAM10000 | 0.10 | 0.153 | 0.106 | 0 |
| HAM10000 | 0.30 | 0.101 | -0.149 | 0 |
| ISIC 2020 | 0.10 | 0.118 | 0.036 | 0 |
| ISIC 2020 | 0.30 | 0.110 | -0.239 | 0 |
| ISIC-DICM-17K | 0.10 | 0.463 | 0.453 | 0 |
| ISIC-DICM-17K | 0.30 | 0.379 | 0.296 | 0 |

## LLM-generated notes (HAM10000), bake-off at p = 1.0

| Method | honest AUROC | SRS |
|---|---|---|
| naive | 0.656 | 0.998 |
| ffilt_only | 0.918 | 0.005 |
| counterfactual_aug | 0.928 | 0.006 |
| consistency | 0.925 | 0.004 |
| adversarial | 0.918 | 0.083 |
