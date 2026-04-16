"""
05_unknown.py — Unknown Disease Detection (Subtask b)

Excludes Tomato_Bacterial_spot, retrains on 14 known classes,
evaluates with Mahalanobis distance thresholding.

Generates:
  - results/plots/unknown_distance_histogram.png
  - results/plots/unknown_roc_curve.png
  - results/plots/unknown_confusion_matrix.png
  - results/plots/unknown_umap_known_only.png
  - results/plots/unknown_umap.png
  - results/tables/unknown_detection_results.csv

Usage:
  python scripts/05_unknown.py                          # train from scratch
  python scripts/05_unknown.py --skip-training          # load best_model_fold0.pt
  python scripts/05_unknown.py --skip-training --checkpoint path/to/model.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import pandas as pd

from src.config import Config, load_config_overrides
from src.unknown_detection import train_known_classes, evaluate_unknown_detection
from src.utils import set_seed
from src.visualization import (
    plot_distance_histograms,
    plot_roc_curve,
    plot_umap_embeddings,
    plot_umap_unknown,
    plot_unknown_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unknown Disease Detection — Subtask b")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and load an existing model checkpoint",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to checkpoint .pt file (default: results/models/unknown/best_model_fold0.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config = load_config_overrides(config)
    set_seed(config.seed)

    print("Unknown Disease Detection — Subtask b")
    print(f"Device: {config.device} | Backbone: {config.backbone}")
    print(f"Unknown class: {config.unknown_class}")
    print(f"Known classes ({len(config.get_known_classes())}): {config.get_known_classes()}")

    # Use fold 1 (not fold 0) — HPO was tuned on fold 0's validation set,
    # so reporting on fold 0 would have optimistic bias from hyperparameter selection.
    # Clamp to n_folds-1 for small-fold configs (e.g. fast-mode 2-fold).
    eval_fold = min(1, config.n_folds - 1)

    # Resolve checkpoint path
    checkpoint_path = None
    if args.skip_training:
        checkpoint_path = args.checkpoint or config.models_dir / "unknown" / f"best_model_fold{eval_fold}.pt"
        print(f"\nSkipping training — loading checkpoint: {checkpoint_path}")

    # ── Train or load model ──
    section = "Loading model" if checkpoint_path else f"Training on known classes (fold {eval_fold})"
    print(f"\n=== {section} ===")
    model, proto_clf = train_known_classes(config, fold=eval_fold, checkpoint_path=checkpoint_path)

    # ── Evaluate unknown detection ──
    print("\n=== Evaluating unknown detection ===")
    results = evaluate_unknown_detection(model, proto_clf, config)

    print(f"\nResults:")
    print(f"  AUROC: {results['auroc']:.4f}")
    print(f"  PR-AUC: {results['pr_auc']:.4f}")
    print(f"  Optimal threshold: {results['optimal_threshold']:.4f}")
    print(f"  Known samples: {len(results['known_distances'])}")
    print(f"  Unknown samples: {len(results['unknown_distances'])}")

    print(f"\n  Distance statistics:")
    print(f"    Known:   mean={results['known_distances'].mean():.4f}, std={results['known_distances'].std():.4f}")
    print(f"    Unknown: mean={results['unknown_distances'].mean():.4f}, std={results['unknown_distances'].std():.4f}")

    print(f"\n  At optimal threshold ({results['optimal_threshold']:.4f}):")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    F1 Score:  {results['f1']:.4f}")

    print(f"\n  Mann-Whitney U test (unknown distances > known):")
    print(f"    U statistic: {results['mw_stat']:.0f}")
    print(f"    p-value:     {results['mw_pvalue']:.2e}")

    # ── Plots ──
    plot_distance_histograms(
        results["known_distances"],
        results["unknown_distances"],
        results["optimal_threshold"],
        config.plots_dir / "unknown_distance_histogram.png",
    )

    plot_roc_curve(
        results["fpr"], results["tpr"], results["auroc"],
        config.plots_dir / "unknown_roc_curve.png",
    )

    plot_unknown_confusion_matrix(
        results["y_true"].astype(int),
        results["y_pred"],
        config.plots_dir / "unknown_confusion_matrix.png",
    )

    plot_umap_embeddings(
        results["known_embeddings"],
        results["known_class_labels"],
        config.get_known_classes(),
        config.plots_dir / "unknown_umap_known_only.png",
    )

    plot_umap_unknown(
        results["known_embeddings"],
        results["known_class_labels"],
        results["unknown_embeddings"],
        config.get_known_classes(),
        config.plots_dir / "unknown_umap.png",
    )

    # ── Save results ──
    results_df = pd.DataFrame({
        "Metric": [
            "AUROC", "PR-AUC", "Optimal Threshold",
            "Precision @ Threshold", "Recall @ Threshold", "F1 @ Threshold",
            "Known Mean Distance", "Known Std Distance",
            "Unknown Mean Distance", "Unknown Std Distance",
            "Mann-Whitney U Stat", "Mann-Whitney P-Value",
            "N Known", "N Unknown",
        ],
        "Value": [
            results["auroc"],
            results["pr_auc"],
            results["optimal_threshold"],
            results["precision"],
            results["recall"],
            results["f1"],
            results["known_distances"].mean(),
            results["known_distances"].std(),
            results["unknown_distances"].mean(),
            results["unknown_distances"].std(),
            results["mw_stat"],
            results["mw_pvalue"],
            len(results["known_distances"]),
            len(results["unknown_distances"]),
        ],
    })
    results_df.to_csv(config.tables_dir / "unknown_detection_results.csv", index=False)
    print(f"Saved: {config.tables_dir / 'unknown_detection_results.csv'}")

    print("\n=== Unknown Detection Complete ===")


if __name__ == "__main__":
    main()
