import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import EmbeddingNet
from src.prototype import PrototypeClassifier
from src.utils import compute_all_embeddings, save_checkpoint
from sklearn.metrics import f1_score


class Trainer:
    """Training loop for triplet-loss based metric learning."""

    def __init__(
        self,
        model: EmbeddingNet,
        loss_fn,
        miner_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        patience: int = 7,
    ):
        self.model = model.to(device)
        # Compile model for faster inference (PyTorch 2.0+)
        if device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass  # torch.compile not available or failed
        self.loss_fn = loss_fn
        self.miner_fn = miner_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.prototype_clf = PrototypeClassifier()
        self.scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    embeddings = self.model(images)
                    hard_pairs = self.miner_fn(embeddings, labels)
                    loss = self.loss_fn(embeddings, labels, hard_pairs)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model(images)
                hard_pairs = self.miner_fn(embeddings, labels)
                loss = self.loss_fn(embeddings, labels, hard_pairs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Tuple[float, Dict]:
        """Validate using prototype classifier.

        1. Compute embeddings on train set -> fit prototypes
        2. Predict on val set -> compute F1macro

        Returns:
            f1_macro: macro F1 score
            metrics: dict with per-class metrics
        """
        # Fit prototypes on training embeddings
        train_embeddings, train_labels = compute_all_embeddings(
            self.model, train_loader, self.device
        )
        self.prototype_clf.fit(train_embeddings, train_labels)

        # Predict on validation set
        val_embeddings, val_labels = compute_all_embeddings(
            self.model, val_loader, self.device
        )
        predictions = self.prototype_clf.predict(val_embeddings)

        # Compute metrics
        f1_macro = f1_score(
            val_labels.numpy(), predictions.numpy(), average="macro"
        )
        f1_per_class = f1_score(
            val_labels.numpy(), predictions.numpy(), average=None
        )

        metrics = {
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class.tolist(),
        }
        return f1_macro, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        fold: int = 0,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """Full training loop with early stopping.

        Returns:
            history: dict with training curves
        """
        history = {
            "train_loss": [],
            "val_f1_macro": [],
            "best_f1": 0.0,
            "best_epoch": 0,
        }

        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            start = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            f1_macro, metrics = self.validate(train_loader, val_loader)
            history["val_f1_macro"].append(f1_macro)

            elapsed = time.time() - start

            print(
                f"Fold {fold} | Epoch {epoch+1}/{epochs} | "
                f"Loss: {train_loss:.4f} | F1: {f1_macro:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(f1_macro)
                else:
                    self.scheduler.step()

            # Early stopping
            if f1_macro > best_f1:
                best_f1 = f1_macro
                history["best_f1"] = best_f1
                history["best_epoch"] = epoch + 1
                patience_counter = 0

                # Save best checkpoint
                if save_dir is not None:
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        fold,
                        metrics,
                        save_dir / f"best_model_fold{fold}.pt",
                    )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history
