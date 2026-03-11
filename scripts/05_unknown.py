"""
05_unknown.py — Unknown Disease Detection (Subtask b)

Excludes Tomato_Bacterial_spot, retrains on 11 classes,
evaluates with Mahalanobis distance thresholding.

Generates:
  - results/plots/unknown_distance_histogram.png
  - results/plots/unknown_roc_curve.png
  - results/tables/unknown_detection_results.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.config import Config
from src.unknown_detection import train_known_classes, evaluate_unknown_detection
from src.utils import set_seed
from src.visualization import plot_distance_histograms, plot_roc_curve


def main():
    config = Config()
    set_seed(config.seed)

    print("Unknown Disease Detection — Subtask b")
    print(f"Device: {config.device}")
    print(f"Unknown class: {config.unknown_class}")
    print(f"Known classes ({len(config.get_known_classes())}): {config.get_known_classes()}")

    # ── Train on known classes ──
    print("\n=== Training on known classes (fold 0) ===")
    model, proto_clf = train_known_classes(config, fold=0)

    # ── Evaluate unknown detection ──
    print("\n=== Evaluating unknown detection ===")
    results = evaluate_unknown_detection(model, proto_clf, config)

    print(f"\nResults:")
    print(f"  AUROC: {results['auroc']:.4f}")
    print(f"  PR-AUC: {results['pr_auc']:.4f}")
    print(f"  Optimal threshold: {results['optimal_threshold']:.4f}")
    print(f"  Known samples: {len(results['known_distances'])}")
    print(f"  Unknown samples: {len(results['unknown_distances'])}")

    # Distance statistics
    print(f"\n  Known distances:  mean={results['known_distances'].mean():.4f}, "
          f"std={results['known_distances'].std():.4f}")
    print(f"  Unknown distances: mean={results['unknown_distances'].mean():.4f}, "
          f"std={results['unknown_distances'].std():.4f}")

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

    # ── Save results ──
    results_df = pd.DataFrame({
        "Metric": ["AUROC", "PR-AUC", "Optimal Threshold",
                    "Known Mean Distance", "Known Std Distance",
                    "Unknown Mean Distance", "Unknown Std Distance",
                    "N Known", "N Unknown"],
        "Value": [
            results["auroc"],
            results["pr_auc"],
            results["optimal_threshold"],
            results["known_distances"].mean(),
            results["known_distances"].std(),
            results["unknown_distances"].mean(),
            results["unknown_distances"].std(),
            len(results["known_distances"]),
            len(results["unknown_distances"]),
        ],
    })
    results_df.to_csv(config.tables_dir / "unknown_detection_results.csv", index=False)
    print(f"Saved: {config.tables_dir / 'unknown_detection_results.csv'}")

    print("\n=== Unknown Detection Complete ===")


if __name__ == "__main__":
    main()
