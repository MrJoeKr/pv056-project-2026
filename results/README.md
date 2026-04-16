# Results Directory Structure

## Naming Convention

Results are organized by embedding dimension and run:

| Directory suffix | Description |
|------------------|-------------|
| `emb512_old`     | Very first training run (2026-03-16). Used **rule-of-thumb hyperparameters** before any HPO was conducted (embedding_dim=512, batch_hard mining, default LR). Kept for reference only. |
| `emb256`         | Training Run 1 (2026-03-25/26). Used hyperparameters from **HPO Run 1** (embedding_dim=256, batch_hard, lr=6.52e-4). 5-fold CV Mean F1: 0.9974. Eval on fold 0. |
| `emb512`         | Training Run 2 (2026-03-31). Used hyperparameters from **HPO Run 2** (embedding_dim=512, multi_similarity, lr=2.808e-4). 5-fold CV Mean F1: **0.9985**. Eval + unknown detection on **fold 1** (not fold 0) to avoid HPO bias. Current best run. |

## Key Results (emb512)

**Subtask a — Classification (fold 1 validation):**
- F1 Macro: **0.9980** (baseline 0.76)
- 5-fold CV Mean F1: 0.9985 ± 0.0005

**Subtask b — Unknown Detection (fold 1, trained from scratch on 14 known classes):**
- AUROC: **0.9933**, PR-AUC: **0.9899**
- F1 @ optimal threshold (8.41): **0.9505** (Precision: 0.9489, Recall: 0.9520)
- Known-class Mahalanobis distance: mean=4.67, std=1.57
- Unknown-class Mahalanobis distance: mean=21.74, std=9.40
- Mann-Whitney U: p-value ~ 0 (highly significant separation)

## Why fold 1, not fold 0?

HPO was run on fold 0's validation set (Optuna trials, 10 epochs/trial, fold 0 only). Evaluating
on fold 0 again would double-dip into the same data and introduce optimistic bias from the
hyperparameter search. **Fold 1 is a clean held-out split** — same hyperparameters, but the
validation samples were never seen during HPO.

For reference: emb512 fold 0 eval gave F1=0.9994 (essentially perfect). Fold 1 gives F1=0.9980.
The ~0.0014 drop quantifies the HPO bias.

## Directory Layout

```
results/
├── logs/
│   ├── emb256/               # Run 1 training log (training_log.json)
│   └── emb512/               # Run 2 logs
│       ├── training_log.json     # per-epoch train_loss / val_f1_macro for 5 folds
│       ├── train_emb512.log      # clean training script output
│       ├── evaluate_emb512.log   # eval script output (F1, classification report)
│       └── unknown_emb512.log    # unknown detection output (AUROC, distances)
├── models/
│   └── train/                # Model checkpoints (on remote only, ~310 MB each)
│                             # best_model_fold{0-4}.pt for Run 2 emb512
├── plots/
│   ├── eda/                  # EDA: class distribution, outliers, pixel stats
│   ├── emb256/               # Run 1 training curves + fold 0 eval plots
│   ├── emb512/               # Run 2 plots (HPO-tuned, fold 1 eval)
│   │   ├── training_curves_fold{0-4}.png
│   │   ├── cv_results.png
│   │   ├── confusion_matrix.png
│   │   ├── per_class_f1.png
│   │   ├── umap_embeddings.png
│   │   ├── gradcam_samples.png
│   │   ├── unknown_distance_histogram.png
│   │   ├── unknown_roc_curve.png
│   │   ├── unknown_confusion_matrix.png
│   │   ├── unknown_umap_known_only.png
│   │   └── unknown_umap.png
│   ├── emb512_old/           # First rule-of-thumb run (reference only)
│   └── unknown/              # Unknown detection plots from emb256 (superseded by emb512)
├── tables/
│   ├── emb256/               # Run 1 CV results + eval summary
│   ├── emb512/               # Run 2 tables
│   │   ├── cv_results.csv                # per-fold F1 + mean/std
│   │   ├── results_summary.csv           # per-class F1 for fold 1
│   │   └── unknown_detection_results.csv # AUROC, PR-AUC, thresholds, distances
│   ├── hpo/                  # HPO trial results (best_hpo_params.json, trial log)
│   └── unknown/              # Superseded unknown detection results
└── README.md                 # This file
```

## Configuration used for emb512

From HPO Run 2 (30 Optuna trials, 7h timeout, fold 0):
- `embedding_dim`: 512
- `mining_strategy`: multi_similarity
- `margin`: 0.1776
- `lr`: 2.808e-4
- `weight_decay`: 5.372e-5
- `lr_backbone_factor`: 0.3088
- `batch_size`: 32
- Training: 50 max epochs, patience=7, ReduceLROnPlateau scheduler
- Backbone: ResNet-50 (ImageNet pretrained) + Linear(2048, 512) + BatchNorm1d + L2-normalize
