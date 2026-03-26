# CLAUDE.md

## Project Overview

PV056 ML & DM semestral project — Plant disease classification using triplet/contrastive learning on the PlantVillage dataset. Deadline: April 19, 2026.

Two subtasks:
- **Subtask a**: Classify plant diseases with triplet loss, 5-fold stratified CV, beat 0.76 F1macro baseline
- **Subtask b**: Detect unknown diseases — exclude `Tomato_Bacterial_spot` from training, detect it via Mahalanobis distance (NOT softmax probabilities)

## Solution Requirements

All of the following must be completed (R1a–R3b):

**R1 — Data analysis**
- R1a: Analyze and visualize class label distribution → `scripts/01_eda.py`
- R1b: Find and visualize outliers/anomalies → `scripts/01_eda.py` + `src/outlier_detection.py`

**R2 — Model training**
- R2a: Use HPO to find hyperparameters → `scripts/03_hpo.py` (Optuna, 30 trials)
- R2b: Visualize training progress, train to convergence → `scripts/02_train.py` + training curves in `src/visualization.py`

**R3 — Results analysis**
- R3a: Explore successful/failed cases via confusion matrix and explainability (Grad-CAM) → `scripts/04_evaluate.py`
- R3b: Present results with appropriate tables/plots; consider statistical tests to support claims

## Deliverables

1. **Google Colab notebook** — publicly shared link submitted to IS MU by April 19, 2026
2. **Technical report** — max 2 pages + appendices, Springer template (LaTeX or Word), submitted as PDF to IS MU vault
   - Sections: Introduction & Related Work (≥3 references), Data & Methods, Results & Discussion, Appendix (author contributions)
   - No abstract, no keywords
3. **10-minute presentation** — starting May 4, 2026; focus on approach and results

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

PlantVillage at `data/PlantVillage/` — 15 classes, 20,638 images. Already gitignored.
- Smallest class: `Potato___healthy` (152 images)
- Largest class: `Tomato__Tomato_YellowLeaf__Curl_Virus` (3,208 images)
- Unknown class for subtask b: `Tomato_Bacterial_spot` (2,127 images)
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

## Remote Execution

To run scripts on the MUNI nymfe GPU workstations, read `remote_setup.md` for full agent instructions (how to connect, check GPU availability, run scripts, sync code).

## Autoresearch Mode

For autonomous speed optimization experiments, read `autoresearch/program.md` for full instructions.
