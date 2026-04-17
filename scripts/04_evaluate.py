"""
04_evaluate.py — Evaluation with Confusion Matrix, Grad-CAM, UMAP (R3a + R3b)

Loads the best model from fold 0, evaluates on the validation set.

Generates:
  - results/plots/evaluation/confusion_matrix.png
  - results/plots/evaluation/umap_embeddings.png
  - results/plots/evaluation/gradcam_samples.png
  - results/plots/evaluation/per_class_f1.png
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

from src.config import Config, load_config_overrides
from src.gradcam import compute_gradcam_cams
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


def generate_gradcam(model, dataset, proto_clf, class_names, device, n_samples=6, save_path=None):
    """Generate Grad-CAM visualizations using negative prototype distance as target."""
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int).tolist()
    images, cam_images, labels = compute_gradcam_cams(
        model, dataset, proto_clf, device, indices,
    )
    titles = [class_names[label] for label in labels]
    if save_path:
        plot_gradcam(images, cam_images, titles, save_path)


def main():
    config = Config()
    config = load_config_overrides(config)
    set_seed(config.seed)

    eval_plots_dir = config.plots_dir / "evaluation"
    eval_plots_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluation — Confusion Matrix, Grad-CAM, UMAP")
    print(f"Device: {config.device} | Backbone: {config.backbone}")
    print(f"Plots dir: {eval_plots_dir}")

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
    # Use fold 1 (not fold 0) — HPO was tuned on fold 0's validation set,
    # so reporting on fold 0 would have optimistic bias from hyperparameter selection.
    eval_fold = min(1, config.n_folds - 1)
    model = EmbeddingNet(config.embedding_dim, pretrained=False, backbone=config.backbone)
    checkpoint_path = config.models_dir / "train" / f"best_model_fold{eval_fold}.pt"
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model, device=config.device)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Run 02_train.py first!")
        return

    model = model.to(config.device)
    model.eval()

    # Get validation set for the eval fold
    folds = get_stratified_folds(config.data_dir, config.classes, config.n_folds, config.seed)
    train_idx, val_idx = folds[eval_fold]

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
                          eval_plots_dir / "confusion_matrix.png")

    # ── Per-class F1 ──
    plot_per_class_f1(f1_per_class.tolist(), config.classes,
                      eval_plots_dir / "per_class_f1.png")

    # ── UMAP ──
    print("Generating UMAP visualization...")
    all_embeddings = torch.cat([train_embeddings, val_embeddings]).numpy()
    all_labels = torch.cat([train_labels, val_labels]).numpy()
    plot_umap_embeddings(all_embeddings, all_labels, config.classes,
                         eval_plots_dir / "umap_embeddings.png")

    # ── Grad-CAM (R3a) ──
    print("Generating Grad-CAM visualizations...")
    generate_gradcam(model, val_dataset, proto_clf, config.classes, config.device,
                     n_samples=6, save_path=eval_plots_dir / "gradcam_samples.png")

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
