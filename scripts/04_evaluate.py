"""
04_evaluate.py — Evaluation with Confusion Matrix, Grad-CAM, UMAP (R3a + R3b)

Loads the best model from fold 0, evaluates on the validation set.

Generates:
  - results/plots/confusion_matrix.png
  - results/plots/umap_embeddings.png
  - results/plots/gradcam_samples.png
  - results/plots/per_class_f1.png
  - results/tables/results_summary.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from src.config import Config
from src.dataset import (
    PlantVillageDataset,
    get_val_transform,
    get_stratified_folds,
)
from src.model import EmbeddingNet
from src.prototype import PrototypeClassifier
from src.utils import set_seed, compute_all_embeddings, load_checkpoint
from src.visualization import (
    plot_confusion_matrix,
    plot_umap_embeddings,
    plot_gradcam,
    plot_per_class_f1,
)


def generate_gradcam(model, dataset, class_names, device, n_samples=6, save_path=None):
    """Generate Grad-CAM visualizations for sample images."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("pytorch-grad-cam not installed, skipping Grad-CAM")
        return

    # For Grad-CAM, we need a wrapper that outputs class logits
    # We'll use distances to prototypes as "logits" (negated, since smaller = better)
    # Instead, use the backbone's last conv layer directly

    # Get the last conv layer of the backbone
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        print("Could not find target layer for Grad-CAM")
        return

    # Wrapper model that outputs embeddings as logits
    class GradCAMWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    wrapper = GradCAMWrapper(model)
    cam = GradCAM(model=wrapper, target_layers=[target_layer])

    images = []
    cam_images = []
    titles = []

    # Pick diverse samples
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in indices:
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Use first embedding dimension as target
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

        # Denormalize image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean).clip(0, 1)

        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        images.append(img_np)
        cam_images.append(cam_img)
        titles.append(class_names[label])

    if save_path:
        plot_gradcam(images, cam_images, titles, save_path)


def main():
    config = Config()
    set_seed(config.seed)

    print("Evaluation — Confusion Matrix, Grad-CAM, UMAP")
    print(f"Device: {config.device}")

    # Check for best HPO params
    hpo_params_path = config.tables_dir / "best_hpo_params.json"
    if hpo_params_path.exists():
        with open(hpo_params_path) as f:
            best_params = json.load(f)
        print(f"Using HPO best params: {best_params}")
        config.embedding_dim = best_params.get("embedding_dim", config.embedding_dim)
        config.margin = best_params.get("margin", config.margin)
        config.lr = best_params.get("lr", config.lr)
        config.mining_strategy = best_params.get("mining_strategy", config.mining_strategy)
        config.batch_size = best_params.get("batch_size", config.batch_size)
        config.samples_per_class = best_params.get("samples_per_class", config.samples_per_class)

    # Load model
    model = EmbeddingNet(config.embedding_dim, pretrained=False)
    checkpoint_path = config.models_dir / "best_model_fold0.pt"
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, device=config.device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Run 02_train.py first!")
        return

    model = model.to(config.device)
    model.eval()

    # Get fold 0 validation set
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)
    train_idx, val_idx = folds[0]

    train_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_val_transform(config.img_size),
        indices=train_idx,
    )
    val_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_val_transform(config.img_size),
        indices=val_idx,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # Compute embeddings
    print("Computing embeddings...")
    train_embeddings, train_labels = compute_all_embeddings(model, train_loader, config.device)
    val_embeddings, val_labels = compute_all_embeddings(model, val_loader, config.device)

    # Fit prototypes
    proto_clf = PrototypeClassifier()
    proto_clf.fit(train_embeddings, train_labels)

    # Predict
    predictions = proto_clf.predict(val_embeddings)
    y_true = val_labels.numpy()
    y_pred = predictions.numpy()

    # ── Metrics ──
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per_class = f1_score(y_true, y_pred, average=None)
    print(f"\nF1 Macro: {f1_macro:.4f}")
    print(f"Baseline: 0.76 — {'PASSED' if f1_macro > 0.76 else 'BELOW'}")

    report = classification_report(y_true, y_pred, target_names=config.classes)
    print(f"\n{report}")

    # ── Confusion Matrix (R3a) ──
    plot_confusion_matrix(y_true, y_pred, config.classes,
                          config.plots_dir / "confusion_matrix.png")

    # ── Per-class F1 ──
    plot_per_class_f1(f1_per_class.tolist(), config.classes,
                      config.plots_dir / "per_class_f1.png")

    # ── UMAP ──
    print("Generating UMAP visualization...")
    all_embeddings = torch.cat([train_embeddings, val_embeddings]).numpy()
    all_labels = torch.cat([train_labels, val_labels]).numpy()
    plot_umap_embeddings(all_embeddings, all_labels, config.classes,
                         config.plots_dir / "umap_embeddings.png")

    # ── Grad-CAM (R3a) ──
    print("Generating Grad-CAM visualizations...")
    generate_gradcam(model, val_dataset, config.classes, config.device,
                     n_samples=6, save_path=config.plots_dir / "gradcam_samples.png")

    # ── Results summary ──
    results_df = pd.DataFrame({
        "Class": config.classes,
        "F1_Score": f1_per_class,
    })
    results_df.loc[len(results_df)] = ["MACRO_AVERAGE", f1_macro]
    results_df.to_csv(config.tables_dir / "results_summary.csv", index=False)
    print(f"Saved: {config.tables_dir / 'results_summary.csv'}")

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
