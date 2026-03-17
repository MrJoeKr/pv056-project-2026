"""
run_experiment.py — Fast single-fold experiment runner for autoresearch.

Trains fold 0 only with reduced epochs for quick iteration.
Prints structured metrics at the end for automated parsing.

Usage:
    python autoresearch/run_experiment.py > autoresearch/run.log 2>&1
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def main():
    config = Config(
        epochs=10,
        patience=3,
    )
    set_seed(config.seed)

    print(f"=== Autoresearch Experiment ===")
    print(f"Device:        {config.device}")
    print(f"Backbone:      {config.backbone}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Image size:    {config.img_size}")
    print(f"Batch size:    {config.batch_size}")
    print(f"LR:            {config.lr}")
    print(f"Epochs:        {config.epochs}")
    print(f"Patience:      {config.patience}")
    print(f"Mining:        {config.mining_strategy}")
    print(f"Margin:        {config.margin}")
    print()

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Get fold 0 only
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)
    train_idx, val_idx = folds[0]

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
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Sampler
    train_sampler = create_samplers(train_dataset.labels, config.samples_per_class)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    # Eval loader for prototype fitting (no augmentation)
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
        optimizer, mode="max", patience=2, factor=0.5
    )

    # Trainer
    trainer = Trainer(
        model, loss_fn, miner_fn, optimizer, scheduler,
        device=config.device, patience=config.patience,
    )

    # Override validate to use eval loader for prototypes
    original_validate = trainer.validate

    def custom_validate(train_loader_unused, val_loader_arg):
        return original_validate(train_eval_loader, val_loader_arg)

    trainer.validate = custom_validate

    # Train with timing
    t_start = time.time()
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config.epochs, fold=0,
        save_dir=None,  # Don't save checkpoints during autoresearch
    )
    t_end = time.time()
    training_seconds = t_end - t_start

    # VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Results
    best_f1 = history["best_f1"]
    best_epoch = history["best_epoch"]
    final_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")

    # Structured output block (parsed by the agent)
    print()
    print("---")
    print(f"val_f1_macro:     {best_f1:.4f}")
    print(f"train_loss:       {final_loss:.4f}")
    print(f"best_epoch:       {best_epoch}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
