# Released checkpoints: re-evaluation and cross-GPU reproduction

Retrained on Kaggle (Tesla T4, `environment.txt`) with the released code, HAM10000,
leakage p = 1.0, seed 42 (`train_gpu*.log`). Each saved checkpoint was then re-evaluated
with `scripts/evaluate_checkpoint.py` (`eval_*.json`): the re-evaluation equals the
metrics computed at the end of training for every metric, for both models.

Note: the `paper` field inside `eval_*.json` is not the paper's value. In this run the
training step overwrote the reference file it pointed to (fixed in the notebook since).
The comparison with the paper (RTX 5080 runs, `results/shortcut_ham10000_bakeoff.json`) is below.

| Model | Metric | Released checkpoint (T4) | Paper, seed 42 (RTX 5080) | Paper, 5 seeds (min to max) |
|---|---|---|---|---|
| naive | visible AUROC | 1.0000 | 1.0000 | 1.0000 to 1.0000 |
| naive | honest AUROC | 0.7398 | 0.7092 | 0.6357 to 0.7579 |
| naive | SRS | 0.9976 | 0.9976 | 0.9976 to 0.9980 |
| naive | FlipRate | 1.0000 | 1.0000 | 1.0000 to 1.0000 |
| counterfactual_aug | visible AUROC | 0.9239 | 0.9238 | 0.9149 to 0.9357 |
| counterfactual_aug | honest AUROC | 0.9241 | 0.9239 | 0.9133 to 0.9366 |
| counterfactual_aug | SRS | 0.0031 | 0.0038 | 0.0022 to 0.0044 |
| counterfactual_aug | FlipRate | 0.0010 | 0.0039 | 0.0024 to 0.0060 |

counterfactual_aug reproduces within 0.001 on visible AUROC, honest AUROC and SRS; its
FlipRate is 0.0010 against 0.0039 (about 2 against 8 of the 2,042 test lesions). The naive model's honest
AUROC differs by 0.03 but stays inside its own seed-to-seed range in the paper
(0.64 to 0.76): with the label available from the note, how much the image branch learns
is unstable, while its SRS, visible AUROC and FlipRate reproduce exactly.

The checkpoints themselves (about 0.8 GB each, weights only) are archived on Zenodo.
