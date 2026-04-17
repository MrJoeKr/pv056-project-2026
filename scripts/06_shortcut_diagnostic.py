"""
06_shortcut_diagnostic.py — Background-shortcut diagnostic (R3b)

Tests whether the trained model relies on background cues rather than the leaf.
Reuses the fold-1 model and prototypes from training on unmodified images, then
evaluates three variants of the validation set:

  1. original          — sanity check, should match 04_evaluate results
  2. leaf_only         — background zeroed out (model sees only the leaf)
  3. background_only   — leaf zeroed out (model sees only the background)

If (3) scores >> chance (1/15 ≈ 6.7% F1macro), the model has latched onto
background cues. Gap between (1) and (2) shows how much the leaf alone carries.

Masks are produced by `rembg` (U2Net) and cached under results/masks/.

Usage:
    pip install rembg onnxruntime
    python scripts/06_shortcut_diagnostic.py [--fold 1] [--max-per-class N]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.config import Config, load_config_overrides
from src.dataset import PlantVillageDataset, get_val_transform, get_stratified_folds, get_stratified_subset
from src.model import EmbeddingNet
from src.prototype import PrototypeClassifier
from src.utils import set_seed, compute_all_embeddings, load_checkpoint
from src.masking import (
    MASK_THRESHOLD, FILL_MODES, MaskedDataset,
    ensure_masks, _fill_region,
)


def evaluate(model, samples, masks_dir, mode, proto_clf, config, fill: str = "black",
             seed: int = 42) -> tuple[float, np.ndarray]:
    ds = MaskedDataset(samples, masks_dir, get_val_transform(config.img_size), mode,
                       fill=fill, seed=seed)  # deterministic for eval
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)
    emb, labels = compute_all_embeddings(model, loader, config.device)
    preds = proto_clf.predict(emb).numpy()
    y_true = labels.numpy()
    return f1_score(y_true, preds, average="macro"), f1_score(y_true, preds, average=None)


def save_example_plot(samples, masks_dir, classes, save_path: Path, fill: str = "black",
                      n: int = 4, seed: int = 0):
    """Side-by-side original / leaf-only / background-only for a few samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(samples), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, i in enumerate(idxs):
        path, label = samples[i]
        p = Path(path)
        img = np.array(Image.open(path).convert("RGB"))
        mask = np.array(Image.open(masks_dir / p.parent.name / (p.stem + ".png")).convert("L"))
        m = mask >= MASK_THRESHOLD

        leaf = img.copy()
        _fill_region(leaf, ~m, fill, np.random.default_rng((seed, int(i), 0)))
        bg = img.copy()
        _fill_region(bg, m, fill, np.random.default_rng((seed, int(i), 1)))

        axes[row, 0].imshow(img); axes[row, 0].set_title(f"orig — {classes[label]}", fontsize=8)
        axes[row, 1].imshow(leaf); axes[row, 1].set_title(f"leaf only (fill={fill})", fontsize=8)
        axes[row, 2].imshow(bg); axes[row, 2].set_title(f"background only (fill={fill})", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, help="validation fold (default 1, matches 04_evaluate)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="stratified subsample for val set (default: use full val fold)")
    parser.add_argument("--rembg-model", default="u2netp", help="rembg model (u2netp is lightweight, u2net is heavier)")
    parser.add_argument("--fill", choices=FILL_MODES, default="black",
                        help="how to fill the masked-out region (default: black)")
    args = parser.parse_args()

    config = Config()
    config = load_config_overrides(config)
    set_seed(config.seed)

    print("Shortcut diagnostic — background vs leaf contribution")
    print(f"Device: {config.device} | Backbone: {config.backbone} | Fold: {args.fold}")

    # Apply HPO overrides (same pattern as 04_evaluate)
    hpo_path = config.tables_dir / "best_hpo_params.json"
    if hpo_path.exists():
        best = json.loads(hpo_path.read_text())
        for k in ("embedding_dim", "margin", "lr", "mining_strategy", "batch_size", "samples_per_class"):
            if k in best:
                setattr(config, k, best[k])
        print(f"Applied HPO params from {hpo_path.name}")

    # Load checkpoint
    ckpt = config.models_dir / "train" / f"best_model_fold{args.fold}.pt"
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}. Run 02_train.py first.")
        return
    model = EmbeddingNet(config.embedding_dim, pretrained=False, backbone=config.backbone)
    load_checkpoint(ckpt, model, device=config.device)
    model = model.to(config.device).eval()
    print(f"Loaded: {ckpt}")

    # Fold split
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)
    train_idx, val_idx = folds[args.fold]

    # Build full dataset once to resolve samples
    full = PlantVillageDataset(config.data_dir, config.classes)
    train_samples = [full.samples[i] for i in train_idx]
    val_samples = [full.samples[i] for i in val_idx]

    # Optional subsample for val (train stays full — prototypes should be well-estimated)
    if args.max_per_class:
        labels = np.array([lbl for _, lbl in val_samples])
        sub = get_stratified_subset(np.arange(len(val_samples)), labels, args.max_per_class, seed=config.seed)
        val_samples = [val_samples[i] for i in sub]
        print(f"Subsampled val to {len(val_samples)} images ({args.max_per_class}/class)")

    # Prototypes from unmasked train
    print("Fitting prototypes on unmasked train set...")
    train_ds = PlantVillageDataset(
        config.data_dir, config.classes, transform=get_val_transform(config.img_size), indices=train_idx,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)
    train_emb, train_lbl = compute_all_embeddings(model, train_loader, config.device)
    proto_clf = PrototypeClassifier()
    proto_clf.fit(train_emb, train_lbl)

    # Ensure masks exist for val set
    masks_dir = config.results_dir / "masks"
    ensure_masks(val_samples, masks_dir, model_name=args.rembg_model)

    # Three evaluation modes
    print(f"\nFill mode for masked region: {args.fill}")
    results = {}
    for mode in ("none", "leaf", "background"):
        print(f"\n-- mode: {mode} --")
        f1m, f1_per = evaluate(model, val_samples, masks_dir, mode, proto_clf, config,
                               fill=args.fill, seed=config.seed)
        chance = 1.0 / len(config.classes)
        print(f"  F1 macro = {f1m:.4f}   (chance ≈ {chance:.4f})")
        results[mode] = (f1m, f1_per)

    # Report
    chance = 1.0 / len(config.classes)
    f1_orig = results["none"][0]
    f1_leaf = results["leaf"][0]
    f1_bg = results["background"][0]
    print("\n=== Summary ===")
    print(f"  original         F1macro = {f1_orig:.4f}")
    print(f"  leaf only        F1macro = {f1_leaf:.4f}   (drop vs orig: {f1_orig - f1_leaf:+.4f})")
    print(f"  background only  F1macro = {f1_bg:.4f}   (above chance {chance:.4f}: {f1_bg - chance:+.4f})")
    print(f"  shortcut ratio   bg / orig = {f1_bg / f1_orig:.2%}")

    # CSV
    df = pd.DataFrame({
        "class": config.classes,
        "f1_original": results["none"][1],
        "f1_leaf_only": results["leaf"][1],
        "f1_background_only": results["background"][1],
    })
    df.loc[len(df)] = ["MACRO_AVERAGE", f1_orig, f1_leaf, f1_bg]
    suffix = f"fold{args.fold}_fill-{args.fill}"
    out_csv = config.tables_dir / f"shortcut_diagnostic_{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Example plot
    save_example_plot(val_samples, masks_dir, config.classes,
                      config.plots_dir / f"shortcut_diagnostic_examples_{suffix}.png",
                      fill=args.fill)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
