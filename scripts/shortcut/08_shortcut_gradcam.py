"""
08_shortcut_gradcam.py — Baseline vs leaf-only Grad-CAM comparison

For the same fold-1 validation samples, computes Grad-CAM from two models:
  - baseline: full-image model (results/models/train/best_model_fold1.pt)
  - leaf-only: trained on leaf-only inputs with random_color fill
               (results/models/shortcut/best_model_fold1_leaf_fill-random_color.pt)

Each model sees the input distribution it was trained on (baseline on unmasked,
leaf-only on leaf-masked). Output is a two-row grid
(results/plots/shortcut/shortcut_gradcam_comparison.png).

Usage:
    python scripts/shortcut/08_shortcut_gradcam.py [--fold 1] [--n-samples 6]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config_overrides
from src.dataset import (
    PlantVillageDataset,
    get_val_transform,
    get_stratified_folds,
)
from src.gradcam import compute_gradcam_cams
from src.masking import MaskedDataset, ensure_masks
from src.model import EmbeddingNet
from src.prototype import PrototypeClassifier
from src.utils import compute_all_embeddings, load_checkpoint, set_seed
from src.visualization import plot_gradcam_comparison


def apply_hpo(config: Config):
    hpo_path = config.tables_dir / "best_hpo_params.json"
    if hpo_path.exists():
        best = json.loads(hpo_path.read_text())
        for k in ("embedding_dim", "margin", "lr", "mining_strategy", "batch_size",
                  "samples_per_class", "weight_decay", "lr_backbone_factor"):
            if k in best:
                setattr(config, k, best[k])
        print(f"Applied HPO params from {hpo_path.name}")


def make_dataset(idx, samples, masks_dir, mode, fill, config):
    """Build a val-transform dataset for the given split. mode='none' → unmasked."""
    if mode == "none":
        return PlantVillageDataset(
            config.data_dir, config.classes,
            transform=get_val_transform(config.img_size),
            indices=idx,
        )
    split_samples = [samples[i] for i in idx]
    return MaskedDataset(
        split_samples, masks_dir, get_val_transform(config.img_size),
        mode=mode, fill=fill, seed=config.seed,
    )


def load_model(checkpoint_path: Path, config: Config) -> EmbeddingNet:
    model = EmbeddingNet(config.embedding_dim, pretrained=False, backbone=config.backbone)
    load_checkpoint(checkpoint_path, model, device=config.device)
    model = model.to(config.device)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


def fit_prototypes(model, train_ds, config):
    loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    emb, lbl = compute_all_embeddings(model, loader, config.device)
    proto = PrototypeClassifier()
    proto.fit(emb, lbl)
    return proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--fill", default="random_color")
    args = parser.parse_args()

    config = Config()
    config = load_config_overrides(config)
    set_seed(config.seed)
    apply_hpo(config)

    baseline_ckpt = config.models_dir / "train" / f"best_model_fold{args.fold}.pt"
    leaf_ckpt = (config.models_dir / "shortcut"
                 / f"best_model_fold{args.fold}_leaf_fill-{args.fill}.pt")
    for p in (baseline_ckpt, leaf_ckpt):
        if not p.exists():
            raise FileNotFoundError(p)

    # Fold splits
    folds = get_stratified_folds(config.data_dir, config.classes,
                                 config.n_folds, config.seed)
    train_idx, val_idx = folds[args.fold]
    full_samples = PlantVillageDataset(config.data_dir, config.classes).samples

    # Masks for leaf-only pipeline
    masks_dir = config.results_dir / "masks"
    train_samples_raw = [full_samples[i] for i in train_idx]
    val_samples_raw = [full_samples[i] for i in val_idx]
    ensure_masks(train_samples_raw + val_samples_raw, masks_dir)

    # ── Baseline (unmasked) ──
    print("\n=== Baseline (unmasked) ===")
    baseline_model = load_model(baseline_ckpt, config)
    baseline_train = make_dataset(train_idx, full_samples, masks_dir,
                                  mode="none", fill=args.fill, config=config)
    baseline_val = make_dataset(val_idx, full_samples, masks_dir,
                                mode="none", fill=args.fill, config=config)
    baseline_proto = fit_prototypes(baseline_model, baseline_train, config)

    # ── Leaf-only (masked) ──
    print("\n=== Leaf-only (masked) ===")
    leaf_model = load_model(leaf_ckpt, config)
    leaf_train = make_dataset(train_idx, full_samples, masks_dir,
                              mode="leaf", fill=args.fill, config=config)
    leaf_val = make_dataset(val_idx, full_samples, masks_dir,
                            mode="leaf", fill=args.fill, config=config)
    leaf_proto = fit_prototypes(leaf_model, leaf_train, config)

    # Same dataset indices for both models
    indices = np.linspace(0, len(val_idx) - 1, args.n_samples, dtype=int).tolist()
    titles = [config.classes[full_samples[val_idx[i]][1]] for i in indices]
    print(f"\nSample indices: {indices}")
    print(f"Classes:        {titles}")

    print("\n=== Grad-CAM: baseline ===")
    _, cam_baseline, _ = compute_gradcam_cams(
        baseline_model, baseline_val, baseline_proto, config.device, indices,
    )

    print("=== Grad-CAM: leaf-only ===")
    _, cam_leaf, _ = compute_gradcam_cams(
        leaf_model, leaf_val, leaf_proto, config.device, indices,
    )

    out_dir = config.plots_dir / "shortcut"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "shortcut_gradcam_comparison.png"
    plot_gradcam_comparison(
        cam_rows=[cam_baseline, cam_leaf],
        titles=titles,
        row_labels=["Baseline\n(full image)", "Leaf-only\n(bg replaced)"],
        save_path=save_path,
    )
    print(f"\nDone: {save_path}")


if __name__ == "__main__":
    main()
