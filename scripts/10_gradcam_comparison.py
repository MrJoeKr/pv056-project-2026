"""
07_gradcam_comparison.py — Grad-CAM Comparison for Background Bias Detection

Compares Grad-CAM visualizations from two models:
  - Model 1: Trained on NON-preprocessed images (raw) → background bias expected
  - Model 2: Trained on preprocessed images (background removed) → disease-focused

Generates 3-column side-by-side compositions (Original | Model1 CAM | Model2 CAM)
to visually demonstrate whether the non-preprocessed model has background bias.

Outputs to: results/plots/gradcam/comparisons/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import Config
from src.dataset import (
    PlantVillageDataset,
    get_val_transform,
    get_stratified_folds,
)
from src.model import EmbeddingNet
from src.prototype import PrototypeClassifier
from src.utils import (
    set_seed,
    compute_all_embeddings,
    load_checkpoint,
)


def save_embeddings_checkpoint(embeddings, labels, checkpoint_path: Path):
    """Save computed embeddings to a checkpoint file.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: numpy array of shape (n_samples,)
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        checkpoint_path,
    )
    print(f"  ✓ Saved embeddings checkpoint: {checkpoint_path}")


def load_embeddings_checkpoint(checkpoint_path: Path):
    """Load embeddings from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple of (embeddings, labels) or (None, None) if not found
    """
    if not checkpoint_path.exists():
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        embeddings = checkpoint["embeddings"]
        labels = checkpoint["labels"]
        print(f"  ✓ Loaded embeddings checkpoint: {checkpoint_path}")
        return embeddings, labels
    except Exception as e:
        print(f"  ✗ Failed to load embeddings checkpoint: {e}")
        return None, None


def load_samples_from_json(json_path: Path):
    """Load sample image paths from JSON file.

    Returns:
        List of relative paths to images (e.g., "data/PlantVillage/Pepper__bell___Bacterial_spot/image.jpg")
    """
    if not json_path.exists():
        print(f"WARNING: {json_path} not found. Will use default sampling.")
        return None

    with open(json_path) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {json_path}")
    return samples


def get_image_index_in_dataset(dataset, image_path: str):
    """Find the index of an image in the dataset by its relative path.

    Args:
        dataset: PlantVillageDataset instance
        image_path: Relative path like "Pepper__bell___Bacterial_spot/image.jpg"

    Returns:
        Index in dataset.samples, or None if not found
    """
    target_path = str(dataset.data_dir / image_path.replace("data/PlantVillage/", ""))

    for idx, (sample_path, _) in enumerate(dataset.samples):
        if sample_path == target_path:
            return idx

    return None


def generate_gradcam_for_dataset(
    model, dataset, proto_clf, class_names, device, indices, model_name="Model"
):
    """Generate Grad-CAM visualizations for specified sample indices.

    Args:
        model: EmbeddingNet instance
        dataset: PlantVillageDataset instance
        proto_clf: PrototypeClassifier instance
        class_names: List of class names
        device: torch device
        indices: List of indices to process
        model_name: Name of model (for logging)

    Returns:
        Tuple of (images, cams, labels) lists
    """
    centroids = proto_clf.prototypes.to(device)  # (num_classes, embedding_dim)

    class ModelWithPrototypeTarget(torch.nn.Module):
        """Wrapper that outputs negative distance to prototype for a given label."""

        def __init__(self, backbone_model, centroids, target_label):
            super().__init__()
            self.model = backbone_model
            self.centroids = centroids
            self.target_label = target_label

        def forward(self, x):
            embedding = self.model(x)  # [batch, embedding_dim]
            centroid = self.centroids[self.target_label]
            distance = torch.norm(embedding - centroid, dim=-1)  # [batch]
            return -distance

    class MaximizeScalarTarget:
        """Target that maximizes the model's scalar output."""

        def __call__(self, model_output):
            return model_output

    # Get the last conv layer of the backbone for visualization
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print(f"[{model_name}] Could not find target layer for Grad-CAM")
        return [], [], []

    images = []
    cam_images = []
    labels = []

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in indices:
        if idx >= len(dataset):
            print(
                f"[{model_name}] Warning: Index {idx} out of range (dataset size: {len(dataset)})"
            )
            continue

        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Create wrapper model for this specific label
        wrapper = ModelWithPrototypeTarget(model, centroids, label).to(device)
        wrapper.eval()

        # Create GradCAM for the wrapper model
        cam = GradCAM(model=wrapper, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[MaximizeScalarTarget()])
        grayscale_cam = grayscale_cam[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)

        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        images.append(img_np)
        cam_images.append(cam_img)
        labels.append(label)

    print(f"[{model_name}] Generated Grad-CAM for {len(images)} samples")
    return images, cam_images, labels


def compose_comparison_grid(
    original_images, cam_images_1, cam_images_2, labels, class_names, save_dir: Path
):
    """Compose 3-column side-by-side Grad-CAM comparisons.

    Columns: Original | Model1 (non-preprocessed) | Model2 (preprocessed)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (orig, cam1, cam2, label) in enumerate(
        zip(original_images, cam_images_1, cam_images_2, labels)
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Column 1: Original image
        axes[0].imshow(orig)
        axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Column 2: Model 1 (non-preprocessed) Grad-CAM
        axes[1].imshow(cam1)
        axes[1].set_title(
            "Model 1 (No Preprocessing)\nGrad-CAM", fontsize=11, fontweight="bold"
        )
        axes[1].axis("off")

        # Column 3: Model 2 (preprocessed) Grad-CAM
        axes[2].imshow(cam2)
        axes[2].set_title(
            "Model 2 (BG Removed)\nGrad-CAM", fontsize=11, fontweight="bold"
        )
        axes[2].axis("off")

        class_name = class_names[label]
        plt.suptitle(
            f"{class_name} — Background Bias Comparison", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        # Save per-image
        output_path = save_dir / f"comparison_{i:03d}_{class_name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")


def compose_summary_grid(
    original_images, cam_images_1, cam_images_2, labels, class_names, save_path: Path
):
    """Compose a summary grid showing all comparisons in a multi-panel layout.

    Each row shows: Original | Model1 CAM | Model2 CAM
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = len(original_images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    for i, (orig, cam1, cam2, label) in enumerate(
        zip(original_images, cam_images_1, cam_images_2, labels)
    ):
        class_name = class_names[label]

        # Column 1: Original
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original\n{class_name}", fontsize=9)
        axes[i, 0].axis("off")

        # Column 2: Model 1
        axes[i, 1].imshow(cam1)
        axes[i, 1].set_title("Model 1 (No Preprocessing)", fontsize=9)
        axes[i, 1].axis("off")

        # Column 3: Model 2
        axes[i, 2].imshow(cam2)
        axes[i, 2].set_title("Model 2 (BG Removed)", fontsize=9)
        axes[i, 2].axis("off")

    plt.suptitle(
        "Grad-CAM Comparison: Background Bias Analysis", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summary: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM Comparison for Background Bias Detection"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to JSON file with sample image paths. If not provided, uses all validation images.",
    )
    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)
    device = config.device

    print("\n" + "=" * 80)
    print("Grad-CAM Comparison: Background Bias Detection")
    print("=" * 80)
    print(f"Device: {device}\n")

    # ── Load sample image paths ──
    sample_paths = None
    if args.json:
        sample_paths = load_samples_from_json(args.json)

    if sample_paths is None or len(sample_paths) == 0:
        print("No sample JSON provided or JSON is empty.")
        print("Will create comparisons for ALL validation images.\n")
        sample_paths = []
    else:
        print(
            f"Using {len(sample_paths)} samples from {args.json} for Grad-CAM comparison\n"
        )

    # ── Get fold 0 for datasets ──
    print("Loading fold 0 validation set...")
    folds = get_stratified_folds(
        config.data_dir, config.classes, config.n_folds, config.seed
    )
    train_idx, val_idx = folds[0]

    # ── Create datasets ──
    print("\nCreating datasets...")
    print("  Dataset A (for Model 1, no preprocessing): raw images")
    dataset_no_prep = PlantVillageDataset(
        config.data_dir,
        config.classes,
        transform=get_val_transform(config.img_size),
        apply_preprocessing=False,
        use_preprocessed_cache=False,
        indices=val_idx,
    )

    print("  Dataset B (for Model 2, with preprocessing): preprocessed images")
    dataset_with_prep = PlantVillageDataset(
        config.data_dir,
        config.classes,
        transform=get_val_transform(config.img_size),
        apply_preprocessing=True,
        use_preprocessed_cache=True,
        indices=val_idx,
    )

    # ── Find sample indices in datasets ──
    sample_indices = []

    if sample_paths:
        # Use samples from JSON
        for sample_path in sample_paths:
            # Extract relative path
            relative = sample_path.replace("data/PlantVillage/", "")
            idx = get_image_index_in_dataset(dataset_no_prep, relative)
            if idx is not None:
                sample_indices.append(idx)
                print(f"  Found sample: {relative} (index: {idx})")
            else:
                print(f"  WARNING: Sample not found in validation set: {relative}")
    else:
        # Use all validation images
        sample_indices = list(range(len(dataset_no_prep)))
        print(
            f"No JSON samples provided. Using ALL {len(sample_indices)} validation images."
        )

    if not sample_indices:
        print("ERROR: No valid samples found. Using first 2 images as fallback.")
        sample_indices = [0, 1] if len(dataset_no_prep) >= 2 else [0]

    # ── Load models ──
    print("\n" + "-" * 80)
    print("Loading models...")
    print("-" * 80)

    # Model 1: Trained on non-preprocessed images
    print("Loading Model 1 (trained on non-preprocessed images)...")
    model1 = EmbeddingNet(config.embedding_dim, pretrained=False)
    model1_path = (
        config.models_dir / "train-no-preprocessing" / "best_model_fold0_resnet50.pt"
    )
    if model1_path.exists():
        load_checkpoint(model1_path, model1, device=device)
        print(f"  ✓ Loaded: {model1_path}")
    else:
        print(f"  ✗ ERROR: Model not found at {model1_path}")
        return
    model1 = model1.to(device)
    model1.eval()

    # Model 2: Trained on preprocessed images
    print("Loading Model 2 (trained on preprocessed images with background removal)...")
    model2 = EmbeddingNet(config.embedding_dim, pretrained=False)
    model2_path = config.models_dir / "train" / "best_model_fold0.pt"
    if model2_path.exists():
        load_checkpoint(model2_path, model2, device=device)
        print(f"  ✓ Loaded: {model2_path}")
    else:
        print(f"  ✗ ERROR: Model not found at {model2_path}")
        return
    model2 = model2.to(device)
    model2.eval()

    # ── Compute embeddings and fit prototypes ──
    print("\n" + "-" * 80)
    print("Computing embeddings and fitting prototypes...")
    print("-" * 80)

    train_dataset_no_prep = PlantVillageDataset(
        config.data_dir,
        config.classes,
        transform=get_val_transform(config.img_size),
        apply_preprocessing=False,
        use_preprocessed_cache=False,
        indices=train_idx,
    )

    train_dataset_with_prep = PlantVillageDataset(
        config.data_dir,
        config.classes,
        transform=get_val_transform(config.img_size),
        apply_preprocessing=True,
        use_preprocessed_cache=True,
        indices=train_idx,
    )

    # Use smaller batch size for memory efficiency on CPU
    embedding_batch_size = max(8, config.batch_size // 4)

    train_loader_no_prep = DataLoader(
        train_dataset_no_prep,
        batch_size=embedding_batch_size,
        shuffle=False,
        num_workers=0,
    )

    train_loader_with_prep = DataLoader(
        train_dataset_with_prep,
        batch_size=embedding_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Model 1 embeddings and prototypes ──
    checkpoint_1_path = (
        config.models_dir / "train-no-preprocessing" / "embeddings_fold0.pt"
    )
    print("  Model 1: Computing embeddings on training set (no preprocessing)...")

    train_embeddings_1, train_labels_1 = load_embeddings_checkpoint(checkpoint_1_path)
    if train_embeddings_1 is None:
        print("    (No checkpoint found, computing from scratch...)")
        train_embeddings_1, train_labels_1 = compute_all_embeddings(
            model1, train_loader_no_prep, device
        )
        save_embeddings_checkpoint(
            train_embeddings_1, train_labels_1, checkpoint_1_path
        )

    proto_clf_1 = PrototypeClassifier()
    proto_clf_1.fit(train_embeddings_1, train_labels_1)
    print(f"  ✓ Model 1 prototypes: {proto_clf_1.prototypes.shape}")

    # Free memory
    del train_embeddings_1, train_labels_1
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Model 2 embeddings and prototypes ──
    checkpoint_2_path = config.models_dir / "train" / "embeddings_fold0.pt"
    print("  Model 2: Computing embeddings on training set (with preprocessing)...")

    train_embeddings_2, train_labels_2 = load_embeddings_checkpoint(checkpoint_2_path)
    if train_embeddings_2 is None:
        print("    (No checkpoint found, computing from scratch...)")
        train_embeddings_2, train_labels_2 = compute_all_embeddings(
            model2, train_loader_with_prep, device
        )
        save_embeddings_checkpoint(
            train_embeddings_2, train_labels_2, checkpoint_2_path
        )

    proto_clf_2 = PrototypeClassifier()
    proto_clf_2.fit(train_embeddings_2, train_labels_2)
    print(f"  ✓ Model 2 prototypes: {proto_clf_2.prototypes.shape}")

    # Free memory
    del train_embeddings_2, train_labels_2
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Generate Grad-CAM visualizations ──
    print("\n" + "-" * 80)
    print("Generating Grad-CAM visualizations...")
    print("-" * 80)

    images_1, cams_1, labels_1 = generate_gradcam_for_dataset(
        model1,
        dataset_no_prep,
        proto_clf_1,
        config.classes,
        device,
        sample_indices,
        "Model1",
    )

    images_2, cams_2, labels_2 = generate_gradcam_for_dataset(
        model2,
        dataset_with_prep,
        proto_clf_2,
        config.classes,
        device,
        sample_indices,
        "Model2",
    )

    if not images_1 or not images_2:
        print("ERROR: Failed to generate Grad-CAM visualizations")
        return

    # ── Compose comparisons ──
    print("\n" + "-" * 80)
    print("Composing side-by-side comparisons...")
    print("-" * 80)

    comparison_dir = config.plots_dir / "gradcam" / "comparisons"

    # Generate individual comparison images
    compose_comparison_grid(
        images_1, cams_1, cams_2, labels_1, config.classes, comparison_dir
    )

    # Generate summary grid
    summary_path = comparison_dir / "comparison_summary.png"
    compose_summary_grid(
        images_1, cams_1, cams_2, labels_1, config.classes, summary_path
    )

    print("\n" + "=" * 80)
    print(f"✓ Comparison complete!")
    print(f"  Output directory: {comparison_dir}")
    print(f"  Generated {len(images_1)} comparison images")
    print("=" * 80)
    print("\nInterpretation Guide:")
    print("  - Model 1 (left CAM): Trained on raw images → May show background bias")
    print(
        "  - Model 2 (right CAM): Trained on preprocessed images → Focuses on disease features"
    )
    print("  - If Model 1 highlights green leaves, soil → BACKGROUND BIAS detected")
    print(
        "  - If both focus on disease regions → NO background bias (preprocessing not critical)"
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
