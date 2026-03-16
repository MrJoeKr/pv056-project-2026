# CLAUDE.md

## Project Overview

PV056 ML & DM semestral project — Plant disease classification using triplet/contrastive learning on the PlantVillage dataset. Deadline: April 19, 2026.

Two subtasks:
- **Subtask a**: Classify plant diseases with triplet loss, 5-fold stratified CV, beat 0.76 F1macro baseline
- **Subtask b**: Detect unknown diseases — exclude `Tomato_Bacterial_spot` from training, detect it via Mahalanobis distance (NOT softmax probabilities)

## Setup

```bash
# Python 3.12+
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

NVIDIA CUDA GPU available locally.

## Dataset

PlantVillage at `data/PlantVillage/` — 12 classes, 15,466 images. Already gitignored.
- Smallest class: `Potato___healthy` (152 images)
- Largest class: `Tomato_Bacterial_spot` (2,127 images) — this is the unknown class for subtask b
- Class folder names use underscores/double-underscores inconsistently — always use `config.classes` list

## Architecture

```
ResNet-50 (pretrained ImageNet, backbone)
  -> Linear(2048, embedding_dim) + BatchNorm1d
  -> L2 normalization
  -> TripletMarginLoss + BatchHardMiner
  -> Prototype classifier (nearest class centroid)
```

Key design: `MPerClassSampler` (4 samples/class/batch) is required for triplet mining to work.

## Running Scripts

```bash
python scripts/01_eda.py          # EDA: class distribution, outlier detection
python scripts/02_train.py        # 5-fold CV training with early stopping
python scripts/03_hpo.py          # Optuna HPO (30 trials, fold 0)
python scripts/04_evaluate.py     # Confusion matrix, Grad-CAM, UMAP, per-class F1
python scripts/05_unknown.py      # Subtask b: unknown disease detection
```

Scripts must be run from the project root. All output goes to `results/` (plots, models, tables, logs).

## Code Layout

- `src/config.py` — `Config` dataclass with all hyperparameters and paths. HPO overrides fields here.
- `src/dataset.py` — `PlantVillageDataset`, transforms, `get_stratified_folds()`, supports `exclude_classes` for subtask b
- `src/model.py` — `EmbeddingNet` wrapping ResNet-50 backbone + embedding head
- `src/losses.py` — Factory returning `(loss_fn, miner_fn)` based on `mining_strategy` config
- `src/trainer.py` — `Trainer` class with `train_one_epoch()`, `validate()` (prototype-based), `fit()` with early stopping
- `src/prototype.py` — `PrototypeClassifier` with `fit()`, `predict()`, `mahalanobis_distance()`, `compute_thresholds()`
- `src/outlier_detection.py` — Pixel-level statistics + z-score outlier detection
- `src/unknown_detection.py` — Full subtask b pipeline: exclude class, retrain, Mahalanobis evaluation
- `src/visualization.py` — All plotting functions (class distribution, training curves, confusion matrix, Grad-CAM, UMAP, ROC, etc.)
- `src/utils.py` — `set_seed()`, `get_device()`, `save_checkpoint()`, `load_checkpoint()`, `compute_all_embeddings()`

## Key Libraries

- `pytorch-metric-learning` — TripletMarginLoss, BatchHardMiner, MPerClassSampler
- `grad-cam` — Grad-CAM explainability (package name is `grad-cam`, NOT `pytorch-grad-cam`)
- `optuna` — Hyperparameter optimization with MedianPruner
- `umap-learn` — Embedding space visualization

## Important Notes

- Classification uses prototype network (nearest centroid), not a softmax classifier
- Mahalanobis distance uses shared (pooled) covariance with ridge regularization for invertibility
- The `data/` directory and `results/models/` are gitignored — don't commit large files
- Config defaults: embedding_dim=512, margin=0.2, lr=1e-4, batch_size=64, 30 epochs, patience=7
- Cross-validation: 5 stratified folds, report mean +/- std F1macro
