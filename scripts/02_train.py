"""
02_train.py — 5-Fold Cross-Validation Training (R2b)

Generates:
  - results/models/best_model_fold{i}.pt
  - results/plots/training_curves_fold{i}.png
  - results/plots/cv_results.png
  - results/tables/cv_results.csv
  - results/logs/training_log.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import (
    PlantVillageDataset,
    get_train_transform,
    get_val_transform,
    get_stratified_folds,
    create_samplers,
)
from src.model import EmbeddingNet
from src.losses import get_loss_and_miner
from src.trainer import Trainer
from src.utils import set_seed
from src.visualization import plot_training_curves, plot_cv_results


def train_fold(config: Config, fold: int, train_idx, val_idx):
    """Train a single fold and return history."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}")
    print(f"{'='*60}")

    # Datasets
    train_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_train_transform(config.img_size),
        indices=train_idx,
    )
    val_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_val_transform(config.img_size),
        indices=val_idx,
    )
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Sampler for metric learning
    train_sampler = create_samplers(train_dataset.labels, config.samples_per_class)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    # Separate loader for validation (no augmentation, no sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    # Train loader without sampler for prototype fitting
    train_eval_loader = DataLoader(
        PlantVillageDataset(
            config.data_dir, config.classes,
            transform=get_val_transform(config.img_size),
            indices=train_idx,
        ),
        batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    model = EmbeddingNet(config.embedding_dim, config.pretrained)
    loss_fn, miner_fn = get_loss_and_miner(config.mining_strategy, config.margin)

    # Optimizer with differential LR
    optimizer = torch.optim.Adam([
        {"params": model.get_backbone_params(), "lr": config.lr * config.lr_backbone_factor},
        {"params": model.get_head_params(), "lr": config.lr},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # Trainer
    trainer = Trainer(
        model, loss_fn, miner_fn, optimizer, scheduler,
        device=config.device, patience=config.patience,
    )

    # The trainer.validate uses the train_eval_loader for prototype fitting
    # Override validate to use separate loaders
    original_validate = trainer.validate

    def custom_validate(train_loader_unused, val_loader_arg):
        return original_validate(train_eval_loader, val_loader_arg)

    trainer.validate = custom_validate

    history = trainer.fit(
        train_loader, val_loader,
        epochs=config.epochs, fold=fold,
        save_dir=config.models_dir,
    )

    return history


def main():
    config = Config()
    set_seed(config.seed)

    print("Plant Disease Classification — 5-Fold CV Training")
    print(f"Device: {config.device}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Mining: {config.mining_strategy}, Margin: {config.margin}")
    print(f"LR: {config.lr}, Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}, Patience: {config.patience}")

    # Get stratified folds
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)

    all_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        set_seed(config.seed + fold_idx)
        history = train_fold(config, fold_idx, train_idx, val_idx)
        all_histories.append(history)

        # Plot training curves
        plot_training_curves(
            history, fold_idx,
            config.plots_dir / f"training_curves_fold{fold_idx}.png"
        )

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    f1_scores = [h["best_f1"] for h in all_histories]
    for i, f1 in enumerate(f1_scores):
        print(f"  Fold {i}: F1 = {f1:.4f}")
    print(f"  Mean F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print(f"  Baseline: 0.76")
    print(f"  {'PASSED' if np.mean(f1_scores) > 0.76 else 'BELOW BASELINE'}")

    # Plot CV results
    plot_cv_results(all_histories, config.plots_dir / "cv_results.png")

    # Save results table
    df = pd.DataFrame({
        "Fold": range(config.n_folds),
        "Best_F1": f1_scores,
        "Best_Epoch": [h["best_epoch"] for h in all_histories],
    })
    df.loc[len(df)] = ["Mean", np.mean(f1_scores), ""]
    df.loc[len(df)] = ["Std", np.std(f1_scores), ""]
    df.to_csv(config.tables_dir / "cv_results.csv", index=False)
    print(f"Saved: {config.tables_dir / 'cv_results.csv'}")

    # Save training log
    log_path = config.logs_dir / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(
            [{"fold": i, "train_loss": h["train_loss"],
              "val_f1_macro": h["val_f1_macro"],
              "best_f1": h["best_f1"], "best_epoch": h["best_epoch"]}
             for i, h in enumerate(all_histories)],
            f, indent=2
        )
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
