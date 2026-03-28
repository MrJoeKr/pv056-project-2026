from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, roc_curve,
)
from scipy.stats import mannwhitneyu

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
from src.prototype import PrototypeClassifier
from src.trainer import Trainer
from src.utils import set_seed, compute_all_embeddings, load_checkpoint


def train_known_classes(
    config: Config,
    fold: int = 0,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[EmbeddingNet, PrototypeClassifier]:
    """Train model on known classes only (excluding unknown_class).

    If checkpoint_path is provided, loads the model and skips training.

    Returns:
        model: trained EmbeddingNet
        proto_clf: fitted PrototypeClassifier
    """
    set_seed(config.seed)

    # Get folds for known classes only (needed for prototype fitting in both paths)
    folds = get_stratified_folds(
        config.data_dir, config.classes, config.n_folds, config.seed,
        exclude_classes=[config.unknown_class],
    )
    train_idx, val_idx = folds[fold]

    # Create datasets
    train_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_train_transform(config.img_size),
        indices=train_idx,
        exclude_classes=[config.unknown_class],
    )
    val_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_val_transform(config.img_size),
        indices=val_idx,
        exclude_classes=[config.unknown_class],
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    model = EmbeddingNet(config.embedding_dim, config.pretrained)

    if checkpoint_path is not None:
        print(f"Loading model from {checkpoint_path} (skipping training)")
        load_checkpoint(checkpoint_path, model, device=config.device)
        model = model.to(config.device)
    else:
        train_sampler = create_samplers(train_dataset.labels, config.samples_per_class)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            sampler=train_sampler, num_workers=config.num_workers,
            pin_memory=True, drop_last=True,
        )
        loss_fn, miner_fn = get_loss_and_miner(config.mining_strategy, config.margin)
        optimizer = torch.optim.Adam([
            {"params": model.get_backbone_params(), "lr": config.lr * config.lr_backbone_factor},
            {"params": model.get_head_params(), "lr": config.lr},
        ], weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
        trainer = Trainer(
            model, loss_fn, miner_fn, optimizer, scheduler,
            device=config.device, patience=config.patience,
        )
        trainer.fit(
            train_loader, val_loader,
            epochs=config.epochs, fold=fold,
            save_dir=config.models_dir / "unknown",
        )

    # Compute final prototypes using eval-transform loader (no augmentation, no drop_last)
    proto_loader = DataLoader(
        PlantVillageDataset(
            config.data_dir, config.classes,
            transform=get_val_transform(config.img_size),
            indices=train_idx,
            exclude_classes=[config.unknown_class],
        ),
        batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    train_embeddings, train_labels = compute_all_embeddings(model, proto_loader, config.device)
    proto_clf = PrototypeClassifier()
    proto_clf.fit(train_embeddings, train_labels)

    return model, proto_clf


def evaluate_unknown_detection(
    model: EmbeddingNet,
    proto_clf: PrototypeClassifier,
    config: Config,
) -> Dict:
    """Evaluate unknown detection on held-out unknown class.

    Creates a test set with:
    - Known: validation samples from known classes
    - Unknown: all samples from the unknown class (Tomato_Bacterial_spot)

    Computes Mahalanobis distances and evaluates AUROC, PR-AUC,
    precision/recall/F1 at optimal threshold, and Mann-Whitney U test.

    Returns:
        results: dict with metrics, distances, embeddings, and labels
    """
    device = config.device
    model.eval()

    # Known class validation set
    folds = get_stratified_folds(
        config.data_dir, config.classes, config.n_folds, config.seed,
        exclude_classes=[config.unknown_class],
    )
    _, val_idx = folds[0]

    known_dataset = PlantVillageDataset(
        config.data_dir, config.classes,
        transform=get_val_transform(config.img_size),
        indices=val_idx,
        exclude_classes=[config.unknown_class],
    )

    # Unknown class dataset (all samples)
    unknown_dataset = PlantVillageDataset(
        config.data_dir, [config.unknown_class],
        transform=get_val_transform(config.img_size),
    )

    known_loader = DataLoader(
        known_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )
    unknown_loader = DataLoader(
        unknown_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # Compute embeddings and capture labels for UMAP
    known_embeddings, known_class_labels = compute_all_embeddings(model, known_loader, device)
    unknown_embeddings, _ = compute_all_embeddings(model, unknown_loader, device)

    # Compute Mahalanobis distances (all on CPU, proto_clf stays on CPU)
    known_distances = proto_clf.mahalanobis_distances(known_embeddings).numpy()
    unknown_distances = proto_clf.mahalanobis_distances(unknown_embeddings).numpy()

    # Binary labels: 0 = known, 1 = unknown
    y_true = np.concatenate([
        np.zeros(len(known_distances)),
        np.ones(len(unknown_distances)),
    ])
    distances = np.concatenate([known_distances, unknown_distances])

    # AUROC
    auroc = roc_auc_score(y_true, distances)

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, distances)
    pr_auc = auc(recall_curve, precision_curve)

    # ROC curve + optimal threshold (Youden's J statistic)
    fpr, tpr, roc_thresholds = roc_curve(y_true, distances)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]

    # Precision / Recall / F1 at optimal threshold
    y_pred = (distances >= optimal_threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Mann-Whitney U test: unknown distances stochastically greater than known
    mw_stat, mw_pvalue = mannwhitneyu(unknown_distances, known_distances, alternative="greater")

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "optimal_threshold": optimal_threshold,
        "known_distances": known_distances,
        "unknown_distances": unknown_distances,
        "y_true": y_true,
        "y_pred": y_pred,
        "distances": distances,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mw_stat": float(mw_stat),
        "mw_pvalue": float(mw_pvalue),
        "known_embeddings": known_embeddings.numpy(),
        "known_class_labels": known_class_labels.numpy(),
        "unknown_embeddings": unknown_embeddings.numpy(),
    }
