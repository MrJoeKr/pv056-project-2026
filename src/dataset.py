import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from pytorch_metric_learning.samplers import MPerClassSampler


class PlantVillageDataset(Dataset):
    """PlantVillage dataset for plant disease classification."""

    def __init__(
        self,
        data_dir: Path,
        classes: List[str],
        transform=None,
        indices: Optional[np.ndarray] = None,
        exclude_classes: Optional[List[str]] = None,
        preprocessed_data_dir: Optional[Path] = None,
        apply_preprocessing: bool = True,
        use_preprocessed_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.preprocessed_data_dir = Path(preprocessed_data_dir) if preprocessed_data_dir else None
        self.apply_preprocessing = apply_preprocessing
        self.use_preprocessed_cache = use_preprocessed_cache
        self.transform = transform
        self.class_to_idx = {}
        self.samples: List[Tuple[str, int]] = []  # (raw_image_path, label)
        self.cache_paths: List[Optional[str]] = []  # (preprocessed_image_path or None, ...)

        # Filter classes
        active_classes = [c for c in classes if exclude_classes is None or c not in exclude_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(active_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Collect all image paths
        all_samples = []
        all_cache_paths = []
        for cls_name in active_classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    raw_path = str(cls_dir / img_name)
                    all_samples.append((raw_path, self.class_to_idx[cls_name]))
                    # Compute cache path if preprocessing is enabled
                    if apply_preprocessing and preprocessed_data_dir:
                        cache_path = Path(preprocessed_data_dir) / cls_name / img_name
                        all_cache_paths.append(str(cache_path))
                    else:
                        all_cache_paths.append(None)

        # Apply index subset if provided (for CV splits)
        if indices is not None:
            self.samples = [all_samples[i] for i in indices]
            self.cache_paths = [all_cache_paths[i] for i in indices]
        else:
            self.samples = all_samples
            self.cache_paths = all_cache_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_path, label = self.samples[idx]
        cache_path = self.cache_paths[idx]
        
        # Try to load from cache first
        if cache_path and self.use_preprocessed_cache:
            cache_file = Path(cache_path)
            if cache_file.exists():
                try:
                    image = Image.open(cache_file).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    return image, label
                except Exception as e:
                    print(f"[Dataset] Warning: Failed to load cache {cache_path}: {e}. Falling back to raw image.")
        
        # Load raw image
        image = Image.open(raw_path).convert("RGB")
        
        # Apply preprocessing and cache if enabled
        if self.apply_preprocessing and cache_path and self.preprocessed_data_dir:
            try:
                preprocessed = PreprocessingBgRemoval()(image)
                # Create parent directories if needed
                cache_file = Path(cache_path)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                # Save preprocessed image
                preprocessed.save(cache_path, quality=95)
                image = preprocessed
            except Exception as e:
                print(f"[Dataset] Warning: Failed to preprocess/cache {raw_path}: {e}. Using raw image.")
        
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_stratified_folds(data_dir: Path, classes: List[str], n_folds: int = 5, seed: int = 42,
                          exclude_classes: Optional[List[str]] = None,
                          preprocessed_data_dir: Optional[Path] = None,
                          apply_preprocessing: bool = True,
                          use_preprocessed_cache: bool = True):
    """Create stratified k-fold splits.

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Build full dataset to get labels
    dataset = PlantVillageDataset(
        data_dir, 
        classes, 
        exclude_classes=exclude_classes,
        preprocessed_data_dir=preprocessed_data_dir,
        apply_preprocessing=apply_preprocessing,
        use_preprocessed_cache=use_preprocessed_cache,
    )
    labels = np.array(dataset.labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        folds.append((train_idx, val_idx))
    return folds


def create_samplers(labels: List[int], samples_per_class: int):
    """Create MPerClassSampler for effective triplet mining."""
    return MPerClassSampler(labels, m=samples_per_class, length_before_new_iter=len(labels))


def get_stratified_subset(
    indices: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Return a stratified subset of fold indices with at most max_per_class per class.

    Args:
        indices:       fold indices into the full dataset (e.g. train_idx from folds)
        labels:        class label for each index in `indices` (same length)
        max_per_class: cap on samples per class; classes with fewer samples keep all
        seed:          RNG seed for reproducibility

    Returns:
        Subset of `indices`, stratified by class.
    """
    rng = np.random.default_rng(seed)
    subset = []
    for cls in np.unique(labels):
        cls_indices = indices[labels == cls]
        n = min(max_per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=n, replace=False)
        subset.append(chosen)
    return np.concatenate(subset)
