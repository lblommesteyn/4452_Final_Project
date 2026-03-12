# Strong Gravitational Lens Detection

This repository turns the project proposal into a working deep-learning scaffold for automated strong gravitational lens detection. It includes:

- A manifest-based image classification pipeline for astronomy cutouts
- ResNet and Vision Transformer (ViT) model options through `torchvision`
- Calibration metrics and temperature scaling
- A simple human-in-the-loop (HITL) review queue generator
- Synthetic data generation so the project can be tested without downloading sky-survey data first

## Project layout

- `plan.txt`: expanded project proposal
- `configs/`: baseline experiment configurations
- `scripts/`: command-line entry points
- `src/lensing/`: reusable training, data, model, metric, and HITL code
- `tests/`: smoke test using synthetic data

## Quick start

Generate a synthetic dataset:

```powershell
python scripts/make_synthetic_dataset.py --output-dir data/synthetic
```

Train a ResNet baseline:

```powershell
python scripts/train.py --config configs/resnet18_default.yaml
```

Evaluate a saved checkpoint:

```powershell
python scripts/evaluate.py --config configs/resnet18_default.yaml --checkpoint outputs/resnet18_lens/best_model.pt --split test
```

Create a HITL review queue from prediction scores:

```powershell
python scripts/build_hitl_queue.py --predictions outputs/resnet18_lens/test_predictions.csv --top-k 25
```

Run the synthetic smoke test:

```powershell
python -m unittest tests.test_smoke
```

## Real-data workflow

Build the real-image dataset used in the analysis:

```powershell
python scripts/download_real_dataset.py --output-dir data/real_castles_gzh
```

Train the real-data ResNet baseline:

```powershell
python scripts/train.py --config configs/resnet18_real.yaml
```

Train the real-data ViT baseline:

```powershell
python scripts/train.py --config configs/vit_b16_real.yaml
```

Write the comparison report:

```powershell
python scripts/write_real_analysis_report.py
```

## Manifest format

The training pipeline expects a CSV manifest with these columns:

- `image_path`: path to the image, relative to the manifest file or absolute
- `label`: `1` for lens and `0` for non-lens
- `split`: `train`, `val`, or `test`
- `source`: free-text dataset name such as `strong_lens_challenge`, `galaxy_zoo`, or `synthetic`

Example:

```csv
image_path,label,split,source
images/train/lens_0001.png,1,train,strong_lens_challenge
images/train/non_lens_0001.png,0,train,galaxy_zoo
```

## Real data plan

The code is designed to work with public lens-finding resources mentioned in the proposal:

- Galaxy Zoo Data: https://data.galaxyzoo.org/index.html
- Galaxy Zoo: Hubble sample pages: https://data.galaxyzoo.org/data/gzh/samples/low_redshift_disks.html
- Rubin/LSST Strong Lensing Science Collaboration: https://sites.google.com/view/lsst-stronglensing
- Strong Lens Challenge data release: https://slchallenge.cbpf.br/public/Strong_Lens_Challenge_Data_Release.pdf
- Bologna Lens Factory challenge portal: http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html
- CASTLES lens database: https://www.cfa.harvard.edu/castles/

Recommended workflow for real data:

1. Download or export lens and non-lens cutouts from the chosen public sources.
2. Convert them into a unified image format such as PNG or JPEG.
3. Build a manifest CSV that maps each image to a label, split, and source.
4. Point a config file at the manifest and train the selected backbone.
5. Use the saved predictions to build a HITL review queue for ambiguous candidates.

## Notes

- `resnet18` is the practical CPU baseline.
- `vit_b_16` is supported, but it is heavier and benefits from GPU training.
- Temperature scaling is fitted on the validation split and then applied to test probabilities when enabled.
