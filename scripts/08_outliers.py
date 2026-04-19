"""
08_outliers.py — Multi-Method Deep Feature Outlier Detection (R1b)

Runs three outlier detection methods on all dataset images:
  1. knn_distance     — kNN mean distance in embedding space
  2. zscore           — z-score on per-channel pixel statistics
  3. convautoencoder  — reconstruction error from a ConvAE (trained once globally)

For each class, prints a pairwise NxN match table showing how many images each
pair of methods agrees on, plus the all-methods consensus count.  After all
classes, prints the same aggregated table globally.

Saves:
  results/tables/outlier_consensus_results.json   — match tables as JSON
  results/images/outlier_consensus_gallery.png     — 20 random consensus outliers (seed 42)
  results/images/umap_outliers_consensus.png       — UMAP with consensus outliers highlighted
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import PlantVillageDataset, get_val_transform
from src.model import EmbeddingNet
from src.outlier_detection import (
    compute_pixel_statistics,
    detect_outliers_convautoencoder,
    detect_outliers_knn,
    detect_outliers_zscore,
)
from src.utils import (
    compute_all_embeddings,
    load_checkpoint,
    load_embeddings_checkpoint,
    save_embeddings_checkpoint,
    set_seed,
)
from src.visualization import (
    plot_outlier_gallery_comparison,
    plot_umap_outliers,
)

METHODS = ["knn_distance", "zscore", "convautoencoder"]


# ── Method wrappers ──────────────────────────────────────────────────────────

def run_knn(cls_embeddings: np.ndarray) -> dict:
    """kNN mean-distance outlier detection in embedding space."""
    return detect_outliers_knn({"features": cls_embeddings}, k=5)


def run_zscore(cls_means: np.ndarray, cls_stds: np.ndarray) -> dict:
    """Per-channel pixel z-score outlier detection."""
    _, details = detect_outliers_zscore({"means": cls_means, "stds": cls_stds})
    return details  # contains "is_outlier" key


def run_convautoencoder_inference(cls_errors: np.ndarray, percentile: float = 95.0) -> dict:
    """Apply per-class percentile threshold to pre-computed ConvAE reconstruction errors."""
    threshold = float(np.percentile(cls_errors, percentile))
    is_outlier = cls_errors >= threshold
    return {
        "is_outlier": is_outlier,
        "outlier_indices": np.where(is_outlier)[0].tolist(),
        "threshold": threshold,
        "distances": cls_errors,
    }


# ── Data helpers ─────────────────────────────────────────────────────────────

def _get_embeddings_checkpoint_path(config: Config) -> Path:
    out_dir = config.models_dir / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"embeddings_{config.backbone}_dim{config.embedding_dim}.pt"


def get_embeddings(config: Config):
    """Load cached embeddings or compute them from the fold-0 model checkpoint.

    Returns:
        embeddings_np: (N, D) float32 array
        labels_np:     (N,) int array
        all_paths:     list[str] of image paths in dataset order
        idx_to_class:  dict[int, str]
        ckpt_path:     Path to the embeddings .pt file
    """
    ckpt_path = _get_embeddings_checkpoint_path(config)

    if ckpt_path.exists():
        embeddings_np, labels_np, all_paths, idx_to_class = load_embeddings_checkpoint(ckpt_path)
        print(f"Loaded embeddings checkpoint: {ckpt_path}")
        print(f"Embeddings shape: {embeddings_np.shape}")
        return embeddings_np, labels_np, all_paths, idx_to_class, ckpt_path

    model_ckpt = config.models_dir / "train" / "best_model_fold0.pt"
    if not model_ckpt.exists():
        raise FileNotFoundError(
            f"No model checkpoint at {model_ckpt}. Run 02_train.py first."
        )

    model = EmbeddingNet(config.embedding_dim, pretrained=False, backbone=config.backbone)
    load_checkpoint(model_ckpt, model, device=config.device)
    model = model.to(config.device)
    model.eval()
    print(f"Loaded model checkpoint: {model_ckpt}")

    full_dataset = PlantVillageDataset(
        config.data_dir, config.classes, transform=get_val_transform(config.img_size)
    )
    all_paths = [s[0] for s in full_dataset.samples]
    idx_to_class = full_dataset.idx_to_class
    print(f"Total images: {len(full_dataset)}")

    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=False)
    print("Extracting embeddings...")
    embeddings, labels_tensor = compute_all_embeddings(model, full_loader, config.device)
    embeddings_np = embeddings.numpy()
    labels_np = labels_tensor.numpy()
    print(f"Embeddings shape: {embeddings_np.shape}")

    save_embeddings_checkpoint(
        out_path=ckpt_path,
        embeddings=embeddings_np,
        labels=labels_np,
        paths=all_paths,
        idx_to_class=idx_to_class,
        config=config,
        source_checkpoint=model_ckpt,
    )
    print(f"Saved embeddings checkpoint: {ckpt_path}")
    return embeddings_np, labels_np, all_paths, idx_to_class, ckpt_path


def get_pixel_stats(config: Config, all_paths: list) -> tuple:
    """Return (means, stds) arrays aligned to all_paths order, with disk cache.

    Uses compute_pixel_statistics internally and re-aligns results to the
    embedding dataset order via a path lookup.

    Returns:
        means: (N, 3) float32 per-channel means
        stds:  (N, 3) float32 per-channel stds
    """
    cache_path = config.models_dir / "outliers" / "pixel_stats.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded pixel stats cache: {cache_path}")
        return data["means"], data["stds"]

    print("Computing pixel statistics...")
    stats = compute_pixel_statistics(config.data_dir, config.classes)

    # compute_pixel_statistics iterates directories independently; re-align to
    # the embedding dataset order (all_paths) so indices stay consistent.
    path_to_idx = {p: i for i, p in enumerate(stats["paths"])}
    means_aligned = np.zeros((len(all_paths), 3), dtype=np.float32)
    stds_aligned = np.zeros((len(all_paths), 3), dtype=np.float32)
    for dst_i, p in enumerate(all_paths):
        src_i = path_to_idx.get(p)
        if src_i is not None:
            means_aligned[dst_i] = stats["means"][src_i]
            stds_aligned[dst_i] = stats["stds"][src_i]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, means=means_aligned, stds=stds_aligned)
    print(f"Saved pixel stats cache: {cache_path}")
    return means_aligned, stds_aligned


def get_convae_errors(config: Config, all_paths: list) -> np.ndarray:
    """Train ConvAE once on all dataset images and return per-image reconstruction errors.

    Uses a .npy cache — on re-runs loads errors directly and skips training.

    Returns:
        all_errors: (N,) float64 array of MSE reconstruction errors aligned to all_paths
    """
    errors_path = config.models_dir / "outliers" / "convautoencoder" / "global" / "errors.npy"
    if errors_path.exists():
        all_errors = np.load(errors_path)
        print(f"Loaded ConvAE errors cache: {errors_path}  (shape: {all_errors.shape})")
        return all_errors

    print("Training global ConvAE on full dataset (this may take a while)...")
    ckpt_dir = errors_path.parent
    resume_from = ckpt_dir / "convaae_best.pt"
    result = detect_outliers_convautoencoder(
        image_paths=all_paths,
        device=torch.device(config.device),
        checkpoint_dir=ckpt_dir,
        resume_from=resume_from,
    )
    all_errors = result["distances"]
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(errors_path, all_errors)
    print(f"Saved ConvAE errors cache: {errors_path}")
    return all_errors


# ── Table printing ───────────────────────────────────────────────────────────

def _print_match_table(match_table: dict, methods: list, header: str) -> None:
    """Print a symmetric NxN match-count table to stdout."""
    col_w = max(len(m) for m in methods) + 2
    label_w = col_w
    print(f"\n{header}")
    header_row = " " * label_w + "".join(m.center(col_w) for m in methods)
    print(header_row)
    print("-" * len(header_row))
    for m1 in methods:
        row = m1.ljust(label_w)
        for m2 in methods:
            row += str(match_table[m1][m2]).center(col_w)
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    set_seed(config.seed)

    print("Multi-Method Outlier Detection")
    print(f"Device:  {config.device}")
    print(f"Methods: {METHODS}")

    # Apply HPO params if available (keeps embedding_dim consistent with trained model)
    hpo_params_path = config.tables_dir / "best_hpo_params.json"
    if hpo_params_path.exists():
        with open(hpo_params_path) as f:
            best_params = json.load(f)
        config.embedding_dim = best_params.get("embedding_dim", config.embedding_dim)

    # ── Phase 1: Embeddings ──────────────────────────────────────────────────
    try:
        embeddings_np, labels_np, all_paths, idx_to_class, _ = get_embeddings(config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    # ── Phase 1b: Pixel statistics ───────────────────────────────────────────
    stats = compute_pixel_statistics(config.data_dir, config.classes) 
    all_means = stats["means"]
    all_stds = stats["stds"]

    # ── Phase 2: Train ConvAE globally (once) ────────────────────────────────
    all_convae_errors = get_convae_errors(config, all_paths)

    # ── Phase 3: Per-class loop ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-CLASS OUTLIER DETECTION")
    print("=" * 70)

    global_match_matrix = {m1: {m2: 0 for m2 in METHODS} for m1 in METHODS}
    global_matched_paths = {m1: {m2: [] for m2 in METHODS} for m1 in METHODS}
    global_all_count = 0
    global_consensus_paths: list = []
    global_consensus_flags = np.zeros(len(embeddings_np), dtype=bool)
    per_class_results: dict = {}

    for cls_idx, cls_name in idx_to_class.items():
        mask = labels_np == cls_idx
        cls_positions = np.where(mask)[0]
        cls_embeddings = embeddings_np[mask]
        cls_means = all_means[mask]
        cls_stds = all_stds[mask]
        cls_errors = all_convae_errors[mask]
        cls_paths = [all_paths[i] for i in cls_positions]

        print(f"\n[{cls_name}]  ({len(cls_paths)} images)")

        method_results = {
            "knn_distance":    run_knn(cls_embeddings),
            "zscore":          run_zscore(cls_means, cls_stds),
            "convautoencoder": run_convautoencoder_inference(cls_errors),
        }

        for m, res in method_results.items():
            print(f"  {m}: {int(res['is_outlier'].sum())} outliers")

        # NxN pairwise match table — counts and matched path lists
        match_table = {m1: {} for m1 in METHODS}
        cls_matched_paths = {m1: {} for m1 in METHODS}
        for m1 in METHODS:
            for m2 in METHODS:
                combined = method_results[m1]["is_outlier"] & method_results[m2]["is_outlier"]
                match_table[m1][m2] = int(combined.sum())
                cls_matched_paths[m1][m2] = [cls_paths[i] for i, flag in enumerate(combined) if flag]
        _print_match_table(match_table, METHODS, "  Match table:")

        # Consensus — images flagged by ALL methods
        cls_consensus = np.ones(len(cls_paths), dtype=bool)
        for res in method_results.values():
            cls_consensus &= res["is_outlier"]
        cls_all_count = int(cls_consensus.sum())
        print(f"  All methods intersection: {cls_all_count}")

        # Accumulate global state
        for m1 in METHODS:
            for m2 in METHODS:
                global_match_matrix[m1][m2] += match_table[m1][m2]
                global_matched_paths[m1][m2].extend(cls_matched_paths[m1][m2])
        global_all_count += cls_all_count

        consensus_global_positions = cls_positions[cls_consensus]
        global_consensus_flags[consensus_global_positions] = True
        global_consensus_paths.extend(all_paths[i] for i in consensus_global_positions)

        consensus_paths_cls = [cls_paths[i] for i, flag in enumerate(cls_consensus) if flag]
        per_class_results[cls_name] = {
            "match_table": {m1: {m2: match_table[m1][m2] for m2 in METHODS} for m1 in METHODS},
            "matched_paths": {m1: {m2: cls_matched_paths[m1][m2] for m2 in METHODS} for m1 in METHODS},
            "all_methods_intersection": cls_all_count,
            "all_methods_paths": consensus_paths_cls,
        }

    # ── Phase 4: Global table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GLOBAL OUTLIER DETECTION SUMMARY")
    print("=" * 70)
    _print_match_table(global_match_matrix, METHODS, "Global match table:")
    print(f"\nAll methods intersection (total): {global_all_count}")

    # ── Phase 5: Per-pair galleries ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    pairs_dir = config.images_dir / "outlier_pairs"
    for i, m1 in enumerate(METHODS):
        for m2 in METHODS[i + 1:]:
            pair_paths = global_matched_paths[m1][m2]
            n_sample = min(20, len(pair_paths))
            if n_sample == 0:
                print(f"No matched outliers for {m1} ∩ {m2} — skipping gallery.")
                continue
            sampled = rng.choice(pair_paths, size=n_sample, replace=False).tolist()
            plot_outlier_gallery_comparison(
                normal_paths=[],
                outlier_paths=sampled,
                save_path=pairs_dir / f"gallery_{m1}_vs_{m2}.png",
                title=f"{m1} ∩ {m2}",
            )

    # ── Phase 5b: All-methods consensus gallery ──────────────────────────────
    if global_consensus_paths:
        n_sample = min(20, len(global_consensus_paths))
        sampled = rng.choice(global_consensus_paths, size=n_sample, replace=False).tolist()
        plot_outlier_gallery_comparison(
            normal_paths=[],
            outlier_paths=sampled,
            save_path=config.images_dir / "outlier_consensus_gallery.png",
            title="Consensus Outliers (all methods agree)",
        )
    else:
        print("No consensus outliers found — skipping gallery.")

    # ── Phase 5b: UMAP with consensus outliers highlighted ───────────────────
    print("Generating UMAP with consensus outliers highlighted...")
    plot_umap_outliers(
        embeddings=embeddings_np,
        labels=labels_np,
        is_outlier=global_consensus_flags,
        class_names=config.classes,
        save_path=config.images_dir / "umap_outliers_consensus.png",
    )

    # ── Phase 6: JSON output ─────────────────────────────────────────────────
    json_payload = {
        "per_class": per_class_results,
        "total": {
            "match_table": {m1: {m2: global_match_matrix[m1][m2] for m2 in METHODS} for m1 in METHODS},
            "matched_paths": {m1: {m2: global_matched_paths[m1][m2] for m2 in METHODS} for m1 in METHODS},
            "all_methods_intersection": global_all_count,
            "all_methods_paths": global_consensus_paths,
        },
    }
    json_path = config.tables_dir / "outlier_consensus_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\nSaved: {json_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
