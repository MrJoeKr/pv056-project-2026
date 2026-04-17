# Plant Disease Classification with Triplet Loss

PV056 ML & DM Semestral Project (MUNI, 2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrJoeKr/pv056-project-2026/blob/main/colab.ipynb)
![Python](https://img.shields.io/badge/python-3.12%2B-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76b900?logo=nvidia&logoColor=white)
![Dataset: PlantVillage](https://img.shields.io/badge/dataset-PlantVillage-2e8b57)

## Overview

Plant disease classification using contrastive/triplet learning on the PlantVillage dataset (20,638 images, 15 classes).

- **Subtask a**: Supervised classification with triplet loss + 5-fold CV (target: F1macro > 0.76)
- **Subtask b**: Unknown disease detection — exclude `Tomato_Bacterial_spot`, detect it as unknown via Mahalanobis distance thresholding

## Results

| Task | Metric | Value |
|------|--------|-------|
| Closed-set (5-fold CV) | F1-macro | **0.9985 ± 0.0005** |
| Unknown detection | AUROC | **0.9933** |
| Unknown detection | PR-AUC | 0.9899 |
| Unknown detection | F1 @ τ=8.41 | 0.9505 |

Substantially above the 0.76 F1-macro baseline. The report covers the full breakdown, Grad-CAM analysis, and a background-shortcut ablation (leaf-masked vs. background-only).

## Approach

- **Backbone**: ResNet-50 (pretrained ImageNet) with 512-dim embedding head + L2 normalization
- **Loss**: TripletMarginLoss with BatchHardMiner (pytorch-metric-learning)
- **Sampling**: MPerClassSampler for effective mining
- **Classification**: Prototype network (nearest class centroid in embedding space)
- **Unknown detection**: Mahalanobis distance from test embeddings to class prototypes

## Setup

Requires Python 3.12+.

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

PlantVillage subset — 15 classes:

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
| Tomato YellowLeaf Curl Virus | 3,208 |
| Tomato mosaic virus | 373 |
| Tomato healthy | 1,591 |
| **Total** | **20,638** |

## Authors

Faculty of Informatics, Masaryk University — Brno, Czech Republic.

- Jozef Kraus
- Richard Schwarz
- David Charvát
- Matěj Kubík
