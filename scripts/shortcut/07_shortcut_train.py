"""
07_shortcut_train.py — Train on masked images (leaf-only or bg-only)

Trains a fresh model on either leaf-only or background-only images (fold 1),
then evaluates on all three variants (original, leaf, background) to measure
how much class-discriminative information lives in each region.

Masks are produced by rembg and cached under results/masks/.

Usage:
    python scripts/shortcut/07_shortcut_train.py --mode leaf
    python scripts/shortcut/07_shortcut_train.py --mode background
    python scripts/shortcut/07_shortcut_train.py --mode leaf --fill random_color
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.config import Config, load_config_overrides
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
from src.prototype import PrototypeClassifier
from src.utils import set_seed, compute_all_embeddings
from src.visualization import plot_training_curves
from src.masking import FILL_MODES, MaskedDataset, ensure_masks


def build_datasets(samples, masks_dir, mode, fill, transform, seed=None):
    """Create a MaskedDataset from raw (path, label) samples."""
    return MaskedDataset(samples, masks_dir, transform, mode=mode, fill=fill, seed=seed)


def evaluate_variant(model, samples, masks_dir, mode, fill, proto_clf, config):
    """Evaluate model on one masking variant. Returns (f1_macro, f1_per_class)."""
    ds = MaskedDataset(samples, masks_dir, get_val_transform(config.img_size),
                       mode=mode, fill=fill, seed=config.seed)  # deterministic for eval
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)
    emb, labels = compute_all_embeddings(model, loader, config.device)
    preds = proto_clf.predict(emb).numpy()
    y_true = labels.numpy()
    return f1_score(y_true, preds, average="macro"), f1_score(y_true, preds, average=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["leaf", "background"],
                        help="which region to keep during training")
    parser.add_argument("--fill", choices=FILL_MODES, default="random_color",
                        help="how to fill the masked-out region (default: random_color)")
    parser.add_argument("--fold", type=int, default=1,
                        help="CV fold to train (default: 1, matches eval scripts)")
    parser.add_argument("--rembg-model", default="u2netp")
    args = parser.parse_args()

    config = Config()
    config = load_config_overrides(config)
    set_seed(config.seed)

    # Apply HPO params (same as 02_train / 04_evaluate)
    hpo_path = config.tables_dir / "best_hpo_params.json"
    if hpo_path.exists():
        best = json.loads(hpo_path.read_text())
        for k in ("embedding_dim", "margin", "lr", "mining_strategy", "batch_size",
                   "samples_per_class", "weight_decay", "lr_backbone_factor"):
            if k in best:
                setattr(config, k, best[k])
        print(f"Applied HPO params from {hpo_path.name}")

    tag = f"{args.mode}_fill-{args.fill}"
    save_dir = config.models_dir / "shortcut"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir = config.results_dir / "shortcut"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Shortcut training — mode={args.mode}, fill={args.fill}, fold={args.fold}")
    print(f"Device: {config.device} | Backbone: {config.backbone}")
    print(f"Embedding dim: {config.embedding_dim} | LR: {config.lr}")
    print(f"Epochs: {config.epochs} | Patience: {config.patience}")

    # Fold split
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)
    train_idx, val_idx = folds[args.fold]

    # Resolve raw samples
    full = PlantVillageDataset(config.data_dir, config.classes)
    train_samples = [full.samples[i] for i in train_idx]
    val_samples = [full.samples[i] for i in val_idx]

    # Ensure masks for train + val
    masks_dir = config.results_dir / "masks"
    print("Ensuring masks for train + val...")
    ensure_masks(train_samples + val_samples, masks_dir, model_name=args.rembg_model)

    # ── Datasets ──
    # Training: masked with random fill each epoch (seed=None)
    train_ds = build_datasets(
        train_samples, masks_dir, args.mode, args.fill,
        get_train_transform(config.img_size), seed=None,
    )
    # Val: masked with deterministic fill
    val_ds = build_datasets(
        val_samples, masks_dir, args.mode, args.fill,
        get_val_transform(config.img_size), seed=config.seed,
    )
    # Train-eval: same masking as training but val_transform (for prototype fitting)
    train_eval_ds = build_datasets(
        train_samples, masks_dir, args.mode, args.fill,
        get_val_transform(config.img_size), seed=config.seed,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Sampler
    train_sampler = create_samplers(train_ds.labels, config.samples_per_class)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # ── Model ──
    model = EmbeddingNet(config.embedding_dim, config.pretrained, backbone=config.backbone)
    loss_fn, miner_fn = get_loss_and_miner(config.mining_strategy, config.margin)

    optimizer = torch.optim.Adam([
        {"params": model.get_backbone_params(), "lr": config.lr * config.lr_backbone_factor},
        {"params": model.get_head_params(), "lr": config.lr},
    ], weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5,
    )

    trainer = Trainer(
        model, loss_fn, miner_fn, optimizer, scheduler,
        device=config.device, patience=config.patience,
        use_amp=config.use_amp,
    )

    # Override validate to use train_eval_loader for prototypes
    original_validate = trainer.validate

    def custom_validate(train_loader_unused, val_loader_arg):
        return original_validate(train_eval_loader, val_loader_arg)

    trainer.validate = custom_validate

    # ── Train ──
    set_seed(config.seed + args.fold)
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config.epochs, fold=args.fold,
        save_dir=save_dir,
    )

    # Rename checkpoint to include tag
    src_ckpt = save_dir / f"best_model_fold{args.fold}.pt"
    dst_ckpt = save_dir / f"best_model_fold{args.fold}_{tag}.pt"
    if src_ckpt.exists():
        src_ckpt.rename(dst_ckpt)
        print(f"Saved: {dst_ckpt}")

    # Training curves
    plot_training_curves(
        history, args.fold,
        out_dir / f"training_curves_fold{args.fold}_{tag}.png",
    )

    # ── Post-training evaluation on all 3 variants ──
    print("\n=== Post-training evaluation ===")
    model.eval()

    # Fit prototypes from masked train set (matches training distribution)
    train_emb, train_lbl = compute_all_embeddings(model, train_eval_loader, config.device)
    proto_clf = PrototypeClassifier()
    proto_clf.fit(train_emb, train_lbl)

    chance = 1.0 / len(config.classes)
    results = {}
    for eval_mode in ("none", "leaf", "background"):
        f1m, f1_per = evaluate_variant(
            model, val_samples, masks_dir, eval_mode, args.fill, proto_clf, config,
        )
        print(f"  {eval_mode:15s}  F1macro = {f1m:.4f}  (chance = {chance:.4f})")
        results[eval_mode] = (f1m, f1_per)

    # ── Summary CSV ──
    df = pd.DataFrame({
        "class": config.classes,
        "f1_original": results["none"][1],
        "f1_leaf_only": results["leaf"][1],
        "f1_background_only": results["background"][1],
    })
    df.loc[len(df)] = [
        "MACRO_AVERAGE",
        results["none"][0], results["leaf"][0], results["background"][0],
    ]
    csv_path = out_dir / f"shortcut_train_{tag}_fold{args.fold}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"\nTrained on: {args.mode} (fill={args.fill})")
    print(f"Best val F1 (in-distribution): {history['best_f1']:.4f} at epoch {history['best_epoch']}")
    print(f"Original-image F1: {results['none'][0]:.4f}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
