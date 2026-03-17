import torch
import numpy as np
from typing import Dict, Tuple, Optional


class PrototypeClassifier:
    """Prototype-based classifier using class mean embeddings.

    For each class, compute the mean embedding (prototype).
    Classify by finding the nearest prototype using Euclidean distance.
    Also supports Mahalanobis distance for unknown detection.
    """

    def __init__(self) -> None:
        self.prototypes: Optional[torch.Tensor] = None  # (num_classes, embedding_dim)
        self.class_labels: Optional[list] = None
        self.covariance_inv: Optional[Dict[int, torch.Tensor]] = None  # Per-class inverse covariance
        self.shared_cov_inv: Optional[torch.Tensor] = None  # Shared inverse covariance

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Compute prototypes and covariance matrices from training embeddings.

        Args:
            embeddings: (N, D) tensor of embeddings
            labels: (N,) tensor of integer labels
        """
        unique_labels = sorted(labels.unique().tolist())
        self.class_labels = unique_labels

        prototypes = []
        cov_matrices = []

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

            # Compute class covariance
            if class_embeddings.size(0) > 1:
                centered = class_embeddings - prototype.unsqueeze(0)
                cov = (centered.T @ centered) / (centered.size(0) - 1)
                cov_matrices.append(cov)

        self.prototypes = torch.stack(prototypes)

        # Compute shared (pooled) covariance and its inverse
        if cov_matrices:
            shared_cov = torch.stack(cov_matrices).mean(dim=0)
            # Add regularization for numerical stability
            reg = 1e-5 * torch.eye(shared_cov.size(0))
            shared_cov += reg
            self.shared_cov_inv = torch.linalg.inv(shared_cov)

            # Per-class inverse covariance
            self.covariance_inv = {}
            for i, (label, cov) in enumerate(zip(unique_labels, cov_matrices)):
                cov_reg = cov + reg
                try:
                    self.covariance_inv[label] = torch.linalg.inv(cov_reg)
                except torch.linalg.LinAlgError:
                    self.covariance_inv[label] = self.shared_cov_inv

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict class labels using nearest prototype (Euclidean).

        Args:
            embeddings: (N, D) tensor

        Returns:
            predictions: (N,) tensor of predicted class indices
        """
        # Compute distances to all prototypes: (N, num_classes)
        dists = torch.cdist(embeddings, self.prototypes)
        pred_indices = dists.argmin(dim=1)
        # Map back to actual class labels
        predictions = torch.tensor([self.class_labels[i] for i in pred_indices])
        return predictions

    def euclidean_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance to nearest prototype.

        Args:
            embeddings: (N, D) tensor

        Returns:
            min_distances: (N,) tensor of minimum distances
        """
        dists = torch.cdist(embeddings, self.prototypes)
        return dists.min(dim=1).values

    def mahalanobis_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance to nearest class prototype.

        Uses per-class covariance matrices for more accurate distance estimation.

        Args:
            embeddings: (N, D) tensor

        Returns:
            min_distances: (N,) tensor of minimum Mahalanobis distances
        """
        all_distances = []

        for i, label in enumerate(self.class_labels):
            diff = embeddings - self.prototypes[i].unsqueeze(0)  # (N, D)
            cov_inv = self.covariance_inv.get(label, self.shared_cov_inv)
            # Mahalanobis: sqrt(diff @ cov_inv @ diff.T)
            mahal = torch.sqrt(torch.sum((diff @ cov_inv) * diff, dim=1).clamp(min=0))
            all_distances.append(mahal)

        all_distances = torch.stack(all_distances, dim=1)  # (N, num_classes)
        return all_distances.min(dim=1).values

    def to(self, device):
        """Move prototypes and covariance matrices to device."""
        if self.prototypes is not None:
            self.prototypes = self.prototypes.to(device)
        if self.shared_cov_inv is not None:
            self.shared_cov_inv = self.shared_cov_inv.to(device)
        if self.covariance_inv is not None:
            self.covariance_inv = {k: v.to(device) for k, v in self.covariance_inv.items()}
        return self
