from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row vector."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _detect_elbow_in_scores(
    scores: np.ndarray, min_samples: int = 5
) -> Tuple[int, float, float]:
    """Detect elbow point in sorted 1D score array via max distance to curve.

    Returns:
        elbow_idx: index where elbow is detected (0-based)
        elbow_score: score value at elbow
        elbow_strength: [0, 1] measure of elbow prominence (1 = strong knee)
    """
    if len(scores) < min_samples:
        return len(scores) - 1, float(scores[-1]), 0.0

    sorted_scores = np.sort(scores)
    x = np.arange(len(sorted_scores), dtype=np.float64)
    y = sorted_scores.astype(np.float64)

    # Normalize both axes to [0, 1] so elbow strength is scale-independent.
    # Without this, x has range N-1 (hundreds) while y has range ~0.1 (kNN
    # distances on L2-normalized embeddings), causing bbox_diag ≈ N and
    # elbow_strength ≈ 0 always — the elbow condition is never triggered.
    y_range = y[-1] - y[0]
    if y_range < 1e-12:
        # All scores nearly identical; no meaningful elbow
        return len(sorted_scores) - 1, float(sorted_scores[-1]), 0.0
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y_norm = (y - y[0]) / y_range

    # Line runs from (0, 0) to (1, 1) in normalized space — always valid
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    line_unit = np.array([1.0, 1.0], dtype=np.float64) / np.sqrt(2.0)
    points = np.column_stack((x_norm, y_norm))
    vec_from_p1 = points - p1
    projection = np.outer(vec_from_p1 @ line_unit, line_unit)
    perpendicular = vec_from_p1 - projection
    dist_to_line = np.linalg.norm(perpendicular, axis=1)

    elbow_idx = int(np.argmax(dist_to_line))
    elbow_score = float(sorted_scores[elbow_idx])

    # Strength: max perpendicular distance normalized by bbox diagonal.
    # In [0,1]x[0,1] space the diagonal is always sqrt(2), so strength is
    # comparable across classes regardless of N or score magnitude.
    max_dist = dist_to_line[elbow_idx]
    elbow_strength = float(min(1.0, max_dist / (np.sqrt(2.0) / 2.0)))

    return elbow_idx, elbow_score, elbow_strength


def _compute_mad_threshold(
    scores: np.ndarray, multiplier: float = 3.5, use_log: bool = False
) -> float:
    """Compute robust threshold using Median Absolute Deviation.

    Args:
        scores: 1D array of distance scores
        multiplier: threshold = median + multiplier * MAD (default: 3.5)
        use_log: if True, compute MAD on log1p-transformed scores

    Returns:
        threshold: scalar threshold value in original score space.
        Returns max(scores)+1 (flags nothing) when MAD is near-zero relative
        to the score range — indicating the distribution is too uniform to
        identify outliers via spread (common for well-trained embeddings).
    """
    if use_log:
        log_scores = np.log1p(scores)
        median_log = float(np.median(log_scores))
        mad_log = float(np.median(np.abs(log_scores - median_log)))
        # Guard: near-zero MAD means flat distribution, no identifiable outliers
        if mad_log < 1e-8:
            return float(scores.max()) + 1.0
        # Convert threshold back to original space
        threshold = np.expm1(median_log + multiplier * mad_log)
    else:
        median_score = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median_score)))
        score_range = float(scores.max() - scores.min())
        # Guard: near-zero MAD relative to scale means distribution is too flat/uniform
        # to reliably identify outliers. Return a threshold above all scores.
        if mad < 1e-8 * max(score_range, median_score, 1e-12):
            return float(scores.max()) + 1.0
        threshold = median_score + multiplier * mad

    return max(0.0, float(threshold))


def detect_outliers_knn(
    stats: Dict,
    k: int = 5,
    metric: str = "euclidean",
    l2_normalize: bool = True,
) -> Dict[str, Any]:
    """kNN outlier detection with adaptive thresholding for a single class.

    Scores each sample by mean distance to its k nearest neighbors, then
    detects outliers via automatic thresholding using elbow detection on the
    sorted score curve, with MAD-based fallback for small classes or flat
    distributions.

    Args:
        stats: dict with 'features' key containing embeddings (N, D)
        k: number of nearest neighbors
        metric: 'euclidean' or 'cosine'
        l2_normalize: if True, L2-normalize embeddings before scoring

    Returns:
        Dict with keys:
            distances: (n_samples,) array of kNN distances
            threshold: scalar threshold value
            is_outlier: (n_samples,) boolean mask
            outlier_indices: list of outlier indices
            method_used: 'elbow' or 'mad'
            elbow_idx: elbow index (-1 if MAD was used)
            elbow_strength: elbow prominence score (0.0 if MAD was used)
            score_spread: max - min of kNN scores
    """
    if "features" not in stats:
        raise ValueError("stats must contain 'features' key with embedding array.")
    embeddings = np.asarray(stats["features"], dtype=np.float64)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must be non-empty")

    metric = metric.lower()
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'euclidean' or 'cosine'.")
    if k < 1:
        raise ValueError("k must be >= 1")

    emb = _l2_normalize_rows(embeddings) if l2_normalize else embeddings
    n_samples = emb.shape[0]

    if n_samples == 1:
        scores = np.array([0.0], dtype=np.float64)
    else:
        effective_k = min(k, n_samples - 1)
        nn = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric="euclidean" if metric == "euclidean" else "cosine",
            algorithm="auto",
        )
        emb32 = np.asarray(emb, dtype=np.float32)
        nn.fit(emb32)
        dists, _ = nn.kneighbors(emb32)
        # First neighbor is the sample itself (distance 0), so drop it.
        scores = dists[:, 1:].mean(axis=1).astype(np.float64)

    score_spread = float(scores.max() - scores.min())

    elbow_idx, elbow_score, elbow_strength = _detect_elbow_in_scores(scores, min_samples=5)

    if elbow_strength > 0.15 and elbow_idx >= int(0.90 * len(scores)):
        threshold = elbow_score
        method_used = "elbow"
    else:
        threshold = _compute_mad_threshold(scores, multiplier=4.5, use_log=False)
        elbow_idx = -1
        elbow_strength = 0.0
        method_used = "mad"

    is_outlier = scores >= threshold

    return {
        "distances": scores,
        "threshold": float(threshold),
        "is_outlier": is_outlier,
        "outlier_indices": np.where(is_outlier)[0].tolist(),
        "method_used": method_used,
        "elbow_idx": int(elbow_idx),
        "elbow_strength": float(elbow_strength),
        "score_spread": float(score_spread),
    }


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


def detect_outliers_convautoencoder(
    image_paths: List[str],
    img_size: int = 64,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    percentile: float = 95.0,
    device: Optional[torch.device] = None,
    hidden_dims: Optional[List[int]] = None,
    checkpoint_dir: Optional[Path] = None,
    save_interval: int = 5,
    resume_from: Optional[Path] = None,
) -> Dict[str, Any]:
    """Detect outliers using Convolutional Autoencoder reconstruction error.

    The method trains a ConvAE on the provided images and uses a percentile
    threshold on reconstruction errors to flag outliers.

    Args:
        image_paths: List of image file paths (single class)
        img_size: Target image size (64 or 96, default: 64)
        batch_size: Batch size for training (default: 64)
        epochs: Number of training epochs (default: 20)
        learning_rate: Adam learning rate
        percentile: Percentile threshold for outlier flagging (default: 95)
        device: PyTorch device (cuda or cpu)
        hidden_dims: Channel dimensions for encoder [32, 64, 128, 256]
        checkpoint_dir: Directory to save/load checkpoints (optional)
        save_interval: Save checkpoint every N epochs (default: 5)
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Dict with keys:
            distances: (n_samples,) reconstruction errors (MSE per image)
            is_outlier: (n_samples,) boolean mask
            outlier_indices: list of outlier sample indices
            threshold: scalar threshold value
            model: trained ConvAE model
            losses: training losses per epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = list(image_paths)

    if len(image_paths) == 0:
        raise ValueError("image_paths must not be empty")

    # Train ConvAE with checkpointing support
    model, losses = train_convautoencoder(
        image_paths,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        hidden_dims=hidden_dims,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval,
        resume_from=resume_from,
    )

    # Compute reconstruction errors on all images
    model.eval()
    dataset = ImageDataset(image_paths, img_size=img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    reconstruction_errors = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            x_recon = model(x)
            # MSE per image (average over channels and spatial dims)
            mse = torch.mean((x - x_recon) ** 2, dim=(1, 2, 3))
            reconstruction_errors.extend(mse.cpu().numpy().tolist())

    distances = np.array(reconstruction_errors, dtype=np.float64)

    threshold = float(np.percentile(distances, percentile))
    is_outlier = distances >= threshold
    outlier_indices = np.where(is_outlier)[0].tolist()

    return {
        "distances": distances,
        "is_outlier": is_outlier,
        "outlier_indices": outlier_indices,
        "threshold": threshold,
        "model": model,
        "losses": losses,
        "img_size": img_size,
        "n_outliers": int(is_outlier.sum()),
    }
