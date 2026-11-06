# Training-budget control: ffilt_only_2x (HAM10000, p = 1.0, 5 seeds)

counterfactual_aug presents every training example twice per epoch (concordant and
flipped note), so in the same 5 epochs it takes twice as many optimisation steps as
ffilt_only. `ffilt_only_2x` gives filtering the same budget: filtered notes, every
training example twice per epoch (now built into `scripts/run_shortcut.py`).

Run on Kaggle (2 x Tesla T4) with the code of commit 71d18c6 plus this method;
`summary.json` and `gpu*.log` are the raw outputs. The other arms are the paper's
RTX 5080 runs in `results/shortcut_ham10000_bakeoff.json`.

| Seed | ffilt_only (paper) | ffilt_only_2x (control) | counterfactual_aug (paper) |
|---|---|---|---|
| 42 | 0.9129 | 0.9223 | 0.9239 |
| 123 | 0.9091 | 0.9133 | 0.9133 |
| 456 | 0.9255 | 0.9345 | 0.9331 |
| 789 | 0.9217 | 0.9304 | 0.9366 |
| 1024 | 0.9186 | 0.9297 | 0.9310 |

Honest AUROC (FFilt test notes), paired over the five seeds:

| Comparison | Mean difference | 95% CI | paired t p |
|---|---|---|---|
| counterfactual_aug minus ffilt_only | +0.0100 | [+0.0048, +0.0152] | 0.0058 |
| counterfactual_aug minus ffilt_only_2x | +0.0015 | [-0.0020, +0.0051] | 0.2968 |
| ffilt_only_2x minus ffilt_only | +0.0085 | [+0.0053, +0.0117] | 0.0018 |
| SRS: counterfactual_aug minus ffilt_only_2x | -0.0020 | [-0.0033, -0.0006] | 0.0147 |

Hardware check: plain ffilt_only, seed 42, gives 0.9093 on the Tesla T4 against 0.9129 on the RTX 5080 (-0.0036).

Reading: at a matched training budget, filtering reaches about the same honest AUROC as
counterfactual augmentation; counterfactual augmentation keeps a lower SRS.
