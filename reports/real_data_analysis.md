# Real-Data Analysis

## Dataset

Manifest: `data/real_castles_gzh/manifest.csv`

| Split | Total | Lens | Non-lens | CASTLES HST | Galaxy Zoo: Hubble |
|---|---:|---:|---:|---:|---:|
| train | 208 | 104 | 104 | 104 | 104 |
| val | 30 | 15 | 15 | 15 | 15 |
| test | 120 | 24 | 96 | 24 | 96 |

The held-out test split is intentionally imbalanced at 1:4 lens prevalence.

## Test Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.1351 | 0.0246 |
| ViT-B/16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.2461 | 0.0634 |

## Calibration

| Model | Temperature | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 0.0428 | 0.1351 | 0.0004 | 0.0246 | 0.0000 |
| ViT-B/16 | 0.0010 | 0.2461 | 0.0000 | 0.0634 | 0.0000 |

## Interpretation

- Both models achieved perfect discrimination on this held-out split.
- ResNet18 had substantially better raw calibration than ViT-B/16 before temperature scaling.
- The top uncertain examples are still mostly positive CASTLES systems with raw probabilities around 0.54 to 0.66, which indicates some intra-lens variation remains despite the perfect thresholded metrics.
- The results are almost certainly inflated by source bias and dataset simplicity: positives are CASTLES lens systems and negatives are Galaxy Zoo: Hubble disk galaxies, which is useful for a proof of pipeline integration but not a realistic estimate of Rubin/LSST survey performance.

## HITL Queue Samples

### ResNet18 top uncertain examples

- `B0218_02.png` label=1 prob=0.537308 review_score=0.995980
- `B1422_02.png` label=1 prob=0.546055 review_score=0.993871
- `B0218_00.png` label=1 prob=0.597179 review_score=0.972577
- `Q1009_02.png` label=1 prob=0.620534 review_score=0.957664
- `B1422_00.png` label=1 prob=0.658575 review_score=0.926176

### ViT-B/16 top uncertain examples

- `HE1104_02.png` label=1 prob=0.565196 review_score=0.987701
- `gzh_test_0003.png` label=0 prob=0.411868 review_score=0.977471
- `B1422_03.png` label=1 prob=0.599641 review_score=0.971160
- `Q1009_03.png` label=1 prob=0.603453 review_score=0.968895
- `B1422_04.png` label=1 prob=0.618091 review_score=0.959379


## Source Links

- CASTLES lens database: https://www.cfa.harvard.edu/castles/
- Galaxy Zoo data portal: https://data.galaxyzoo.org/index.html
- Galaxy Zoo: Hubble sample images: https://data.galaxyzoo.org/data/gzh/samples/low_redshift_disks.html
