# Progress Notes

## EDA (01_eda.py)
- 15,466 images across 12 classes
- Potato_healthy has only 152 images (significant imbalance)
- 258 outliers detected via z-score > 3.0
  - Most outliers in Tomato_Late_blight (173)
- Plots generated: class_distribution, sample_images, outliers, pixel_histograms

## Training (02_train.py)
- TODO: Run 5-fold CV training

### Observations from CV runs
- **Epochs**: Some folds were still improving at epoch 30 → try increasing max_epochs to 50–60.
- **Spiky curves**: Training curves were zig-zaggy / high variance between epochs. Likely because
  the HPO-found LR is ~6× larger than the old default. ReduceLROnPlateau is active so it may
  self-correct, but worth monitoring. If spikiness persists even after the scheduler kicks in,
  consider adding a warmup schedule or capping the LR.

## HPO (03_hpo.py)
- TODO: Run Optuna optimization

## Evaluation (04_evaluate.py)
- TODO: Generate confusion matrix, Grad-CAM, UMAP

## Unknown Detection (05_unknown.py)
- TODO: Run subtask b pipeline
