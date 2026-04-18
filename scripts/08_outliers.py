"""
08_outliers.py — Deep Feature-Based Outlier Detection (R1b)

Uses the trained EmbeddingNet to extract per-image embeddings, then detects
outliers via KNN distance in the embedding space.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import PlantVillageDataset, get_val_transform
from src.model import EmbeddingNet
from src.outlier_detection import detect_outliers_knn
from src.utils import (
    compute_all_embeddings,
    load_checkpoint,
    load_embeddings_checkpoint,
    save_embeddings_checkpoint,
    set_seed,
)
from src.visualization import (
    plot_distance_distribution,
    plot_outlier_gallery_comparison,
    plot_per_class_outlier_counts,
    plot_umap_outliers,
)


def _slugify_class_name(name: str) -> str:
    """Create a filesystem-safe class name for plot filenames."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _get_embeddings_checkpoint_path(config: Config) -> Path:
    """Return deterministic embeddings checkpoint path for current config."""
    out_dir = config.models_dir / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"embeddings_{config.backbone}_dim{config.embedding_dim}.pt"


def get_embeddings(config: Config):
    """Get embeddings by loading cached checkpoint or computing from model checkpoint.

    Returns:
        embeddings_np: numpy array of shape (N, D)
        labels_np: numpy array of shape (N,)
        all_paths: list of image paths
        idx_to_class: dataset class-index mapping
        embeddings_ckpt_path: path to embeddings checkpoint
    """
    embeddings_ckpt_path = _get_embeddings_checkpoint_path(config)

    if embeddings_ckpt_path.exists():
        embeddings_np, labels_np, all_paths, idx_to_class = load_embeddings_checkpoint(
            embeddings_ckpt_path
        )
        print(f"Loaded embeddings checkpoint: {embeddings_ckpt_path}")
        print(f"Embeddings shape: {embeddings_np.shape}")
        return embeddings_np, labels_np, all_paths, idx_to_class, embeddings_ckpt_path

    checkpoint_path = config.models_dir / "train" / "best_model_fold0.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No model checkpoint at {checkpoint_path}. Run 02_train.py first."
        )

    model = EmbeddingNet(
        config.embedding_dim, pretrained=False, backbone=config.backbone
    )

    load_checkpoint(checkpoint_path, model, device=config.device)
    model = model.to(config.device)
    model.eval()
    print(f"Loaded model checkpoint: {checkpoint_path}")

    full_dataset = PlantVillageDataset(
        config.data_dir,
        config.classes,
        transform=get_val_transform(config.img_size),
    )
    all_paths = [s[0] for s in full_dataset.samples]
    idx_to_class = full_dataset.idx_to_class
    print(f"Total images: {len(full_dataset)}")

    full_loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    print("Extracting embeddings...")
    embeddings, labels_tensor = compute_all_embeddings(
        model,
        full_loader,
        config.device,
    )
    embeddings_np = embeddings.numpy()
    labels_np = labels_tensor.numpy()
    print(f"Embeddings shape: {embeddings_np.shape}")

    embeddings_ckpt_path = save_embeddings_checkpoint(
        out_path=_get_embeddings_checkpoint_path(config),
        embeddings=embeddings_np,
        labels=labels_np,
        paths=all_paths,
        idx_to_class=idx_to_class,
        config=config,
        source_checkpoint=checkpoint_path,
    )
    print(f"Saved embeddings checkpoint: {embeddings_ckpt_path}")

    return embeddings_np, labels_np, all_paths, idx_to_class, embeddings_ckpt_path


def _run_outlier_detection(config: Config, features: np.ndarray, labels: np.ndarray):
    if config.outlier_detection == "knn_distance":
        details = detect_outliers_knn(
            stats={"features": features},
            k=5,
        )
        return details["outlier_indices"], details
    raise ValueError(f"Invalid outlier_detection method: {config.outlier_detection}")


def main():
    config = Config()
    set_seed(config.seed)

    print("Deep Feature Outlier Detection — KNN Distance")
    print(f"Device: {config.device}")

    # ── Apply HPO params if available (keeps embedding_dim consistent with trained model) ──
    hpo_params_path = config.tables_dir / "best_hpo_params.json"
    if hpo_params_path.exists():
        with open(hpo_params_path) as f:
            best_params = json.load(f)
        config.embedding_dim = best_params.get("embedding_dim", config.embedding_dim)

    try:
        embeddings_np, labels_np, all_paths, idx_to_class, _ = get_embeddings(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    print(
        f"\n--- Per-Class {config.outlier_detection.capitalize()} Outlier Detection ---"
    )
    perclass_outlier_flags = np.zeros(len(embeddings_np), dtype=bool)
    per_class_counts = {}
    perclass_gallery_dir = (
        config.plots_dir / f"outliers_{config.outlier_detection}_perclass"
    )

    for cls_idx, cls_name in idx_to_class.items():
        mask = labels_np == cls_idx
        cls_positions = np.where(mask)[0]
        cls_features = embeddings_np[mask]

        cls_outlier_local, cls_details = _run_outlier_detection(
            config,
            cls_features,
            labels_np[mask],
        )
        # Map local indices back to full-dataset positions
        cls_outlier_global = cls_positions[cls_outlier_local].tolist()
        perclass_outlier_flags[cls_positions] = cls_details["is_outlier"]
        per_class_counts[cls_name] = len(cls_outlier_global)

        print(f"  {cls_name}: {len(cls_outlier_global)}/{len(cls_features)} outliers")

        cls_outlier_set = set(cls_outlier_global)
        plot_outlier_gallery_comparison(
            [all_paths[i] for i in cls_positions if i not in cls_outlier_set],
            [all_paths[i] for i in cls_outlier_global],
            perclass_gallery_dir / f"{_slugify_class_name(cls_name)}.png",
            title=cls_name,
        )

        if "distances" in cls_details:
            plot_distance_distribution(
                distances=cls_details["distances"],
                threshold=cls_details["threshold"],
                save_path=perclass_gallery_dir
                / f"{_slugify_class_name(cls_name)}_distance_distribution.png",
                title=f"{cls_name} Mahalanobis Distance Distribution",
            )

    print("Generating UMAP with outlier highlighting...")
    plot_umap_outliers(
        embeddings=embeddings_np,
        labels=labels_np,
        is_outlier=perclass_outlier_flags,
        class_names=config.classes,
        save_path=config.plots_dir / f"umap_outliers_{config.outlier_detection}.png",
    )

    plot_per_class_outlier_counts(
        per_class_counts,
        config.plots_dir / f"outlier_per_class_counts_{config.outlier_detection}.png",
    )

    # ── Save full report ──
    df = pd.DataFrame(
        {
            "path": all_paths,
            "class": [idx_to_class[l] for l in labels_np],
            f"perclass_outlier": perclass_outlier_flags,
        }
    )
    csv_path = config.tables_dir / f"outlier_{config.outlier_detection}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} outliers)")

    # ── Summary ──
    print("\n=== Outlier Detection Summary ===")
    print(f"Total images:               {len(embeddings_np)}")
    print(
        f"Per-class {config.outlier_detection} outliers: {int(perclass_outlier_flags.sum())}"
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()
