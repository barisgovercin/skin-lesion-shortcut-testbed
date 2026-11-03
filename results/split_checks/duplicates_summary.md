# Near-duplicate leakage: test images with a near-identical twin in train

Cosine similarity of ImageNet ConvNeXt-Base embeddings; mean over seeds (min to max in brackets).

## ham10000

| cosine >= | train-test pairs | test images with a train twin | share of the test set |
|---|---|---|---|
| 0.99 | 0.0 | 0.0 (0 to 0) | 0.00% (0.00 to 0.00%) |
| 0.98 | 0.0 | 0.0 (0 to 0) | 0.00% (0.00 to 0.00%) |
| 0.97 | 1.2 | 1.2 (0 to 3) | 0.06% (0.00 to 0.15%) |
| 0.95 | 49.6 | 32.6 (26 to 37) | 1.61% (1.30 to 1.83%) |

## isic2020

| cosine >= | train-test pairs | test images with a train twin | share of the test set |
|---|---|---|---|
| 0.99 | 0.4 | 0.4 (0 to 1) | 0.01% (0.00 to 0.01%) |
| 0.98 | 1.6 | 1.6 (1 to 3) | 0.02% (0.01 to 0.04%) |
| 0.97 | 48.4 | 36.6 (31 to 40) | 0.48% (0.41 to 0.53%) |
| 0.95 | 3919.0 | 682.2 (643 to 719) | 8.94% (8.24 to 9.96%) |

## isic_dicm_17k

| cosine >= | train-test pairs | test images with a train twin | share of the test set |
|---|---|---|---|
| 0.99 | 24.0 | 23.0 (20 to 28) | 0.66% (0.59 to 0.80%) |
| 0.98 | 33.2 | 29.8 (25 to 37) | 0.86% (0.73 to 1.05%) |
| 0.97 | 48.8 | 42.4 (35 to 49) | 1.22% (1.02 to 1.43%) |
| 0.95 | 249.2 | 119.0 (110 to 126) | 3.44% (3.09 to 3.69%) |

