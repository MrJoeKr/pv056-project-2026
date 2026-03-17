"""
03_hpo.py — Hyperparameter Optimization with Optuna (R2a)

Tunes on fold 0 for speed. Hyperparameters:
  - embedding_dim: {128, 256, 512}
  - margin: [0.05, 0.5]
  - lr: [1e-5, 1e-3] (log scale)
  - mining_strategy: {batch_hard, semihard, multi_similarity}
  - batch_size: {32, 64, 128}
  - samples_per_class: {2, 4, 8}

Generates:
  - results/plots/hpo_history.png
  - results/tables/hpo_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import optuna
from optuna.pruners import MedianPruner
import pandas as pd
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
from src.utils import set_seed, compute_all_embeddings
from src.prototype import PrototypeClassifier
from src.visualization import plot_hpo_history
from sklearn.metrics import f1_score


def objective(trial: optuna.Trial, base_config: Config, folds):
    """Optuna objective: train on fold 0, return best val F1."""
    # Sample hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    margin = trial.suggest_float("margin", 0.05, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    mining_strategy = trial.suggest_categorical(
        "mining_strategy", ["batch_hard", "semihard", "multi_similarity"]
    )
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    samples_per_class = trial.suggest_categorical("samples_per_class", [2, 4, 8])

    set_seed(base_config.seed)
    train_idx, val_idx = folds[0]

    # Create datasets
    train_dataset = PlantVillageDataset(
        base_config.data_dir, base_config.classes,
        transform=get_train_transform(base_config.img_size),
        indices=train_idx,
    )
    val_dataset = PlantVillageDataset(
        base_config.data_dir, base_config.classes,
        transform=get_val_transform(base_config.img_size),
        indices=val_idx,
    )

    # Samplers and loaders
    try:
        train_sampler = create_samplers(train_dataset.labels, samples_per_class)
    except Exception:
        return 0.0  # Invalid config

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=base_config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=base_config.num_workers, pin_memory=True,
    )
    train_eval_loader = DataLoader(
        PlantVillageDataset(
            base_config.data_dir, base_config.classes,
            transform=get_val_transform(base_config.img_size),
            indices=train_idx,
        ),
        batch_size=batch_size,
        shuffle=False, num_workers=base_config.num_workers, pin_memory=True,
    )

    # Model, loss, optimizer
    model = EmbeddingNet(embedding_dim, base_config.pretrained)
    loss_fn, miner_fn = get_loss_and_miner(mining_strategy, margin)

    optimizer = torch.optim.Adam([
        {"params": model.get_backbone_params(), "lr": lr * 0.1},
        {"params": model.get_head_params(), "lr": lr},
    ], weight_decay=base_config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # Train with pruning
    device = base_config.device
    model = model.to(device)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(base_config.hpo_epochs):
        # Train one epoch
        model.train()
        total_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            hard_pairs = miner_fn(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_pairs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validate
        train_emb, train_lab = compute_all_embeddings(model, train_eval_loader, device)
        val_emb, val_lab = compute_all_embeddings(model, val_loader, device)

        proto = PrototypeClassifier()
        proto.fit(train_emb, train_lab)
        preds = proto.predict(val_emb)
        f1_macro = f1_score(val_lab.numpy(), preds.numpy(), average="macro")

        scheduler.step(f1_macro)

        # Report to Optuna for pruning
        trial.report(f1_macro, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if f1_macro > best_f1:
            best_f1 = f1_macro
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= base_config.patience:
                break

    return best_f1


def main():
    config = Config()
    set_seed(config.seed)

    print("HPO — Optuna Hyperparameter Optimization")
    print(f"Device: {config.device}")
    print(f"Trials: {config.hpo_n_trials}, Timeout: {config.hpo_timeout}s")

    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="plant_disease_hpo",
    )

    study.optimize(
        lambda trial: objective(trial, config, folds),
        n_trials=config.hpo_n_trials,
        timeout=config.hpo_timeout,
        show_progress_bar=True,
    )

    # Results
    print(f"\n{'='*60}")
    print("  HPO RESULTS")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(config.tables_dir / "hpo_results.csv", index=False)
    print(f"Saved: {config.tables_dir / 'hpo_results.csv'}")

    # Save best params
    import json
    with open(config.tables_dir / "best_hpo_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved: {config.tables_dir / 'best_hpo_params.json'}")

    # Plot
    plot_hpo_history(study, config.plots_dir / "hpo_history.png")

    print("\n=== HPO Complete ===")


if __name__ == "__main__":
    main()
