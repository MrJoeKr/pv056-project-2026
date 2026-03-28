from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ──────────────────── EDA Plots ────────────────────


def plot_class_distribution(class_counts: Dict[str, int], save_path: Path):
    """Bar chart of class distribution (R1a)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    short_names = [c.replace("__", "_").replace("Tomato_", "T_").replace("Pepper__bell___", "P_")
                   .replace("Potato___", "Pt_").replace("Spider_mites_Two_spotted_spider_mite", "Spider_mites")
                   for c in classes]

    bars = ax.bar(range(len(classes)), counts, color=sns.color_palette("husl", len(classes)))
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution — PlantVillage Dataset")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_sample_images(data_dir: Path, classes: List[str], save_path: Path,
                       n_samples: int = 3):
    """Grid of sample images per class."""
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 2.5))

    for i, cls_name in enumerate(classes):
        cls_dir = data_dir / cls_name
        imgs = sorted(cls_dir.iterdir())[:n_samples]
        for j in range(n_samples):
            ax = axes[i, j] if n_classes > 1 else axes[j]
            if j < len(imgs):
                img = Image.open(imgs[j]).convert("RGB")
                ax.imshow(img)
            ax.axis("off")
            if j == 0:
                short = cls_name.replace("__", "_")[:25]
                ax.set_title(short, fontsize=8, loc="left")

    plt.suptitle("Sample Images per Class", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_outlier_gallery(paths: List[str], save_path: Path, max_images: int = 20):
    """Gallery of detected outlier images (R1b)."""
    n = min(len(paths), max_images)
    if n == 0:
        print("No outliers to display.")
        return

    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < n:
            img = Image.open(paths[i]).convert("RGB")
            axes[i].imshow(img)
            name = Path(paths[i]).stem[:20]
            axes[i].set_title(name, fontsize=7)
        axes[i].axis("off")

    plt.suptitle(f"Detected Outliers ({n} images)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_pixel_histograms(stats: Dict, save_path: Path):
    """Histograms of per-image mean and std pixel values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    channel_names = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

    for i, (name, color) in enumerate(zip(channel_names, colors)):
        axes[0, i].hist(stats["means"][:, i], bins=50, color=color, alpha=0.7)
        axes[0, i].set_title(f"{name} Channel Mean")
        axes[0, i].set_xlabel("Mean Pixel Value")

        axes[1, i].hist(stats["stds"][:, i], bins=50, color=color, alpha=0.7)
        axes[1, i].set_title(f"{name} Channel Std")
        axes[1, i].set_xlabel("Std Pixel Value")

    plt.suptitle("Pixel Statistics Distribution", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────── Training Plots ────────────────────


def plot_training_curves(history: Dict, fold: int, save_path: Path):
    """Plot training loss and validation F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Triplet Loss")
    ax1.set_title(f"Fold {fold} — Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_f1_macro"], "r-", label="Val F1 Macro")
    ax2.axhline(y=0.76, color="gray", linestyle="--", label="Baseline (0.76)")
    best_epoch = history["best_epoch"]
    best_f1 = history["best_f1"]
    ax2.scatter([best_epoch], [best_f1], color="gold", s=100, zorder=5, label=f"Best: {best_f1:.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title(f"Fold {fold} — Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cv_results(fold_results: List[Dict], save_path: Path):
    """Box plot of cross-validation F1 scores (R3b)."""
    f1_scores = [r["best_f1"] for r in fold_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(range(len(f1_scores)), f1_scores, color=sns.color_palette("husl", len(f1_scores)),
           alpha=0.8, edgecolor="black")
    ax.axhline(y=mean_f1, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean_f1:.4f} +/- {std_f1:.4f}")
    ax.axhline(y=0.76, color="gray", linestyle=":", linewidth=2, label="Baseline: 0.76")

    ax.set_xticks(range(len(f1_scores)))
    ax.set_xticklabels([f"Fold {i}" for i in range(len(f1_scores))])
    ax.set_ylabel("F1 Macro")
    ax.set_title("5-Fold Cross-Validation Results")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────── Evaluation Plots ────────────────────


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], save_path: Path):
    """Confusion matrix heatmap (R3a)."""
    cm = confusion_matrix(y_true, y_pred)
    short_names = [c.replace("__", "_").replace("Tomato_", "T_").replace("Pepper__bell___", "P_")
                   .replace("Potato___", "Pt_").replace("Spider_mites_Two_spotted_spider_mite", "Spider_mites")
                   for c in class_names]

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm, display_labels=short_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_umap_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                         class_names: List[str], save_path: Path):
    """UMAP visualization of learned embeddings."""
    from umap import UMAP

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if label < len(class_names) else f"Class {label}"
        short = name.replace("__", "_")[:20]
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[palette[i]], label=short,
                   alpha=0.6, s=10)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, markerscale=3)
    ax.set_title("UMAP Embedding Visualization")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_gradcam(images: List, cam_images: List, titles: List[str], save_path: Path):
    """Grad-CAM visualization grid (R3a)."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f"Original: {titles[i]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(cam_images[i])
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle("Grad-CAM Explainability", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_class_f1(f1_scores: List[float], class_names: List[str], save_path: Path):
    """Horizontal bar chart of per-class F1 scores."""
    short_names = [c.replace("__", "_").replace("Tomato_", "T_").replace("Pepper__bell___", "P_")
                   .replace("Potato___", "Pt_").replace("Spider_mites_Two_spotted_spider_mite", "Spider_mites")
                   for c in class_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(f1_scores))
    colors = ["green" if f >= 0.76 else "red" for f in f1_scores]

    ax.barh(y_pos, f1_scores, color=colors, alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    ax.axvline(x=0.76, color="gray", linestyle="--", label="Baseline")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────── Unknown Detection Plots ────────────────────


def plot_distance_histograms(known_distances: np.ndarray, unknown_distances: np.ndarray,
                             threshold: float, save_path: Path):
    """Overlapping histograms of Mahalanobis distances for known vs unknown."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(known_distances, bins=50, alpha=0.6, label="Known Classes", color="blue", density=True)
    ax.hist(unknown_distances, bins=50, alpha=0.6, label="Unknown (Tomato Bacterial Spot)", color="red", density=True)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.2f}")

    ax.set_xlabel("Mahalanobis Distance to Nearest Prototype")
    ax.set_ylabel("Density")
    ax.set_title("Distance Distribution: Known vs Unknown Samples")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curve(fpr, tpr, auroc: float, save_path: Path):
    """ROC curve for unknown detection."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="blue")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Unknown Disease Detection")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_umap_unknown(
    known_embeddings: np.ndarray,
    known_labels: np.ndarray,
    unknown_embeddings: np.ndarray,
    known_class_names: List[str],
    save_path: Path,
):
    """UMAP of known class embeddings + unknown class overlay."""
    from umap import UMAP

    all_embeddings = np.concatenate([known_embeddings, unknown_embeddings], axis=0)
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(all_embeddings)

    known_coords = coords[:len(known_embeddings)]
    unknown_coords = coords[len(known_embeddings):]

    fig, ax = plt.subplots(figsize=(12, 10))
    palette = sns.color_palette("husl", len(known_class_names))

    for i, name in enumerate(known_class_names):
        mask = known_labels == i
        short = name.replace("__", "_")[:20]
        ax.scatter(known_coords[mask, 0], known_coords[mask, 1],
                   c=[palette[i]], label=short, alpha=0.6, s=10)

    ax.scatter(unknown_coords[:, 0], unknown_coords[:, 1],
               c="black", marker="x", s=20, alpha=0.7,
               label="Unknown (Tomato Bacterial Spot)", zorder=5)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=2)
    ax.set_title("UMAP: Known vs Unknown Embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_unknown_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
):
    """Binary confusion matrix (Known vs Unknown) at optimal threshold."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Known", "Unknown"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Binary Confusion Matrix — Known vs Unknown")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ──────────────────── HPO Plots ────────────────────


def plot_hpo_history(study, save_path: Path):
    """Plot HPO optimization history."""
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_values = [max(values[:i+1]) for i in range(len(values))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(range(len(values)), values, alpha=0.6, c="blue", s=20)
    ax1.plot(range(len(best_values)), best_values, "r-", linewidth=2, label="Best so far")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("F1 Macro")
    ax1.set_title("HPO Optimization History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Parameter importance (simple: show best trial params)
    best_trial = study.best_trial
    params = best_trial.params
    names = list(params.keys())
    vals_str = [str(v)[:10] for v in params.values()]

    ax2.barh(names, [best_trial.value] * len(names), alpha=0.3)
    for i, (n, v) in enumerate(zip(names, vals_str)):
        ax2.text(0.01, i, f"{n} = {v}", va="center", fontsize=9)
    ax2.set_title(f"Best Trial (F1 = {best_trial.value:.4f})")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
