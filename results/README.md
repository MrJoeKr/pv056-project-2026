# Results Directory Structure

## Naming Convention

Results are organized by embedding dimension and run:

| Directory suffix | Description |
|------------------|-------------|
| `emb512_old`     | Very first training run (2026-03-16). Used **rule-of-thumb hyperparameters** before any HPO was conducted (embedding_dim=512, batch_hard mining, default LR). Kept for reference only. |
| `emb256`         | Training Run 1 (2026-03-25/26). Used hyperparameters from **HPO Run 1** (embedding_dim=256, batch_hard, lr=6.52e-4). Mean F1: 0.9974. |
| `emb512`         | Training Run 2 (2026-03-31). Used hyperparameters from **HPO Run 2** (embedding_dim=512, multi_similarity, lr=2.808e-4). Mean F1: 0.9985. This is the current best run. |

## Directory Layout

```
results/
├── logs/
│   ├── emb256/           # Run 1 training log
│   └── emb512/           # Run 2 training log (HPO-tuned)
├── models/
│   └── train/            # Model checkpoints (on remote only, ~300MB each)
├── plots/
│   ├── eda/              # EDA visualizations (class distribution, outliers, etc.)
│   ├── emb256/           # Run 1 training curves + eval plots
│   ├── emb512/           # Run 2 training curves + eval plots (HPO-tuned)
│   ├── emb512_old/       # First rule-of-thumb run training curves
│   └── unknown/          # Unknown detection plots (subtask b)
├── tables/
│   ├── emb256/           # Run 1 CV results + eval summary
│   ├── emb512/           # Run 2 CV results + eval summary (HPO-tuned)
│   ├── hpo/              # HPO trial results
│   └── unknown/          # Unknown detection results
└── README.md             # This file
```
