"""
01_eda.py — Exploratory Data Analysis (R1a + R1b)

Generates:
  - results/plots/eda/class_distribution.png
  - results/plots/eda/sample_images.png
  - results/plots/eda/outliers.png
  - results/plots/eda/pixel_histograms.png
  - results/tables/class_counts.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.config import Config
from src.outlier_detection import compute_pixel_statistics, detect_outliers_zscore
from src.visualization import (
    plot_class_distribution,
    plot_sample_images,
    plot_outlier_gallery,
    plot_pixel_histograms,
)


def main():
    config = Config()
    eda_dir = config.plots_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {config.data_dir}")
    print(f"Classes: {config.num_classes}")
    print(f"Plots dir: {eda_dir}")

    # ── R1a: Class distribution ──
    print("\n=== R1a: Class Distribution ===")
    class_counts = {}
    for cls_name in config.classes:
        cls_dir = config.data_dir / cls_name
        if cls_dir.exists():
            count = len([f for f in cls_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            class_counts[cls_name] = count
            print(f"  {cls_name}: {count}")

    total = sum(class_counts.values())
    print(f"  Total: {total}")

    plot_class_distribution(class_counts, eda_dir / "class_distribution.png")

    # Save as CSV
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    df["Percentage"] = (df["Count"] / total * 100).round(2)
    df.to_csv(config.tables_dir / "class_counts.csv", index=False)
    print(f"Saved: {config.tables_dir / 'class_counts.csv'}")

    # ── Sample images ──
    print("\n=== Sample Images ===")
    plot_sample_images(config.data_dir, config.classes,
                       eda_dir / "sample_images.png", n_samples=3)

    # ── R1b: Outlier Detection ──
    print("\n=== R1b: Outlier Detection ===")
    stats = compute_pixel_statistics(config.data_dir, config.classes)
    print(f"Computed statistics for {len(stats['paths'])} images")

    outlier_indices, details = detect_outliers_zscore(stats, threshold=3.0)
    print(f"Detected {len(outlier_indices)} outliers (z-score > 3.0)")

    if outlier_indices:
        outlier_paths = [stats["paths"][i] for i in outlier_indices]
        outlier_classes = [stats["labels"][i] for i in outlier_indices]

        # Print outlier distribution
        from collections import Counter
        outlier_counts = Counter(outlier_classes)
        for cls, cnt in sorted(outlier_counts.items()):
            print(f"  {cls}: {cnt} outliers")

        plot_outlier_gallery(outlier_paths, eda_dir / "outliers.png")

    # Pixel histograms
    plot_pixel_histograms(stats, eda_dir / "pixel_histograms.png")

    print("\n=== EDA Complete ===")


if __name__ == "__main__":
    main()
