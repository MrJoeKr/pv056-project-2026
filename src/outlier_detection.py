from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_pixel_statistics(data_dir: Path, classes: List[str]) -> Dict:
    """Compute per-image pixel statistics for outlier detection.

    Returns dict with:
        - paths: list of image paths
        - labels: list of class names
        - means: (N,3) array of per-channel means
        - stds: (N,3) array of per-channel stds
        - sizes: list of (width, height) tuples
    """
    paths = []
    labels = []
    means = []
    stds = []
    sizes = []

    for cls_name in classes:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_name in sorted(cls_dir.iterdir()):
            if img_name.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                img = Image.open(img_name).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                paths.append(str(img_name))
                labels.append(cls_name)
                means.append(arr.mean(axis=(0, 1)))
                stds.append(arr.std(axis=(0, 1)))
                sizes.append(img.size)
            except Exception:
                continue

    return {
        "paths": paths,
        "labels": labels,
        "means": np.array(means),
        "stds": np.array(stds),
        "sizes": sizes,
    }


def detect_outliers_zscore(
    stats: Dict, threshold: float = 3.0
) -> Tuple[List[int], Dict]:
    """Detect outliers using z-score on pixel statistics.

    An image is flagged as an outlier if any of its channel-wise
    mean or std values is more than `threshold` standard deviations
    from the global mean.

    Returns:
        outlier_indices: list of indices of outlier images
        details: dict with z-scores and flags
    """
    means = stats["means"]  # (N, 3)
    stds_arr = stats["stds"]  # (N, 3)

    # Combine features
    features = np.concatenate([means, stds_arr], axis=1)  # (N, 6)

    # Compute z-scores
    global_mean = features.mean(axis=0)
    global_std = features.std(axis=0)
    global_std[global_std == 0] = 1.0  # avoid division by zero
    z_scores = np.abs((features - global_mean) / global_std)

    # Flag outliers: any feature exceeds threshold
    is_outlier = (z_scores > threshold).any(axis=1)
    outlier_indices = np.where(is_outlier)[0].tolist()

    return outlier_indices, {
        "z_scores": z_scores,
        "is_outlier": is_outlier,
        "threshold": threshold,
    }
