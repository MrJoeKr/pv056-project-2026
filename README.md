# Plant Disease Classification with Triplet Loss

PV056 ML & DM Semestral Project (MUNI, 2026)

## Overview

Plant disease classification using contrastive/triplet learning on the PlantVillage dataset (15,466 images, 12 classes).

- **Subtask a**: Supervised classification with triplet loss + 5-fold CV (target: F1macro > 0.76)
- **Subtask b**: Unknown disease detection — exclude `Tomato_Bacterial_spot`, detect it as unknown via Mahalanobis distance thresholding

## Approach

- **Backbone**: ResNet-50 (pretrained ImageNet) with 512-dim embedding head + L2 normalization
- **Loss**: TripletMarginLoss with BatchHardMiner (pytorch-metric-learning)
- **Sampling**: MPerClassSampler for effective mining
- **Classification**: Prototype network (nearest class centroid in embedding space)
- **Unknown detection**: Mahalanobis distance from test embeddings to class prototypes

## Setup

Requires Python 3.14+.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows
source .venv/bin/activate            # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

```bash
# 1. Exploratory Data Analysis
python scripts/01_eda.py

# 2. 5-Fold CV Training
python scripts/02_train.py

# 3. Hyperparameter Optimization
python scripts/03_hpo.py

# 4. Evaluation (confusion matrix, Grad-CAM, UMAP)
python scripts/04_evaluate.py

# 5. Unknown Disease Detection
python scripts/05_unknown.py
```

## Project Structure

```
src/
  config.py          — Central configuration (paths, hyperparams)
  dataset.py         — Dataset class, transforms, fold splitting
  model.py           — ResNet-50 backbone + embedding head
  losses.py          — Triplet loss + miner setup
  prototype.py       — Prototype classifier + Mahalanobis distance
  trainer.py         — Training loop with early stopping
  outlier_detection.py — Pixel statistics + z-score outlier detection
  unknown_detection.py — Subtask b pipeline
  visualization.py   — All plotting functions
  utils.py           — Seed, device, checkpoint helpers

scripts/
  01_eda.py          — R1a + R1b: data analysis
  02_train.py        — R2b: 5-fold CV training
  03_hpo.py          — R2a: Optuna HPO
  04_evaluate.py     — R3a + R3b: evaluation + explainability
  05_unknown.py      — Subtask b: unknown detection
```

## Dataset

PlantVillage subset — 12 classes:

| Class | Count |
|-------|-------|
| Pepper bell Bacterial spot | 997 |
| Pepper bell healthy | 1,478 |
| Potato Early blight | 1,000 |
| Potato Late blight | 1,000 |
| Potato healthy | 152 |
| Tomato Bacterial spot | 2,127 |
| Tomato Early blight | 1,000 |
| Tomato Late blight | 1,909 |
| Tomato Leaf Mold | 952 |
| Tomato Septoria leaf spot | 1,771 |
| Tomato Spider mites | 1,676 |
| Tomato Target Spot | 1,404 |
| **Total** | **15,466** |
