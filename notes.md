# Progress Notes

## Google Colab Deliverable

No need to convert the project to `.ipynb`. The Colab notebook (`colab.ipynb`) is a thin orchestration layer that:
1. Clones the GitHub repo
2. Installs dependencies
3. Runs scripts via `!python scripts/...`
4. Displays results/plots inline with `IPython.display`

The repo acts as a package for the grader — structure stays intact.

**TODO before submission**: replace `YOUR_USERNAME` placeholder in the `git clone` cell of `colab.ipynb` with the actual GitHub repo URL. Also set `DRIVE_DATASET_PATH` to wherever PlantVillage lives on Drive.

## EDA (01_eda.py)
- 15,466 images across 12 classes
- Potato_healthy has only 152 images (significant imbalance)
- 258 outliers detected via z-score > 3.0
  - Most outliers in Tomato_Late_blight (173)
- Plots generated: class_distribution, sample_images, outliers, pixel_histograms

## Training (02_train.py)

### Run 1 — 5-fold CV (2026-03-28, nymfe GPU)
**Config**: embedding_dim=256, margin=0.1484, LR=0.000652, batch_size=32, epochs=30, patience=7, mining=batch_hard

| Fold | Best F1   | Early stop? |
|------|-----------|-------------|
| 0    | 0.9972    | No (ep 30)  |
| 1    | 0.9983    | No (ep 30)  |
| 2    | 0.9963    | Yes (ep 21) |
| 3    | 0.9967    | No (ep 30)  |
| 4    | 0.9983    | Yes (ep 30) |
| **Mean** | **0.9974 ± 0.0008** | Baseline: 0.76 — **PASSED** |

**Loss trajectory**: All folds show a characteristic two-phase drop — gradual from ~0.16 to ~0.11 over epochs 1–14, then a sudden ~50% drop to ~0.06 around epoch 12–16. This likely corresponds to ReduceLROnPlateau firing. After the step-down, loss continues falling slowly to ~0.01–0.03 by epoch 30.

**LR spike observation resolved**: Previous spiky curves came from a too-high LR; the scheduler appears to have handled it well — final F1 across all folds is >0.996.

**Convergence**: Folds 0, 1, 3 hit epoch 30 with still-declining loss, suggesting they hadn't fully converged. Fold 2 early-stopped at epoch 21. Fold 4 technically "early stopped at epoch 30" (patience triggered on the last epoch).

**Epoch 6, Fold 4 anomaly**: Time jumped to 830s (vs normal ~114s) — likely a server hiccup or preemption on nymfe. Did not affect results.

**Observation**: Increasing max_epochs to 50 may squeeze out a bit more (folds 0, 1, 3 were still improving), but gains are likely marginal given F1 is already 0.997+.

## HPO (03_hpo.py)

### Run 1 — 9 trials, 2h timeout (2026-03-17, nymfe GPU)
**Config**: hpo_n_trials=30, hpo_timeout=7200s, hpo_epochs=10, fold 0 only

| Trial | F1       | embedding_dim | margin | lr       | mining_strategy | batch_size | Notes |
|-------|----------|---------------|--------|----------|-----------------|------------|-------|
| 0     | 0.9813   | 128           | 0.0847 | 4.11e-05 | multi_similarity | 128       |       |
| 1     | 0.9681   | 128           | 0.4408 | 2.64e-05 | semihard        | 32         |       |
| 2     | 0.9827   | 128           | 0.4178 | 4.06e-04 | batch_hard      | 64         |       |
| 3     | 0.9845   | 512           | 0.1748 | 6.20e-05 | semihard        | 32         |       |
| 4     | 0.9256   | 512           | 0.2187 | 1.21e-05 | multi_similarity | 128       |       |
| 5     | pruned   | —             | —      | —        | —               | —          |       |
| 6     | **0.9893** | 128         | 0.1219 | 5.75e-04 | batch_hard      | 32         |       |
| 7     | pruned   | —             | —      | —        | —               | —          |       |
| 8     | pruned   | —             | —      | —        | —               | —          |       |
| 9     | **0.9911** | 256         | 0.1484 | 6.52e-04 | batch_hard      | 32         | Best  |

**Best params (trial 9)**:
- embedding_dim: 256, margin: 0.1484, lr: 6.52e-04, weight_decay: 2.80e-04
- lr_backbone_factor: 0.1826, mining_strategy: batch_hard, batch_size: 32

**Observations**:
- Only 9 trials completed before 2h timeout (trials ran ~14min each on fold 0)
- 3 trials pruned early (5, 7, 8) — MedianPruner working as expected
- batch_hard + small batch_size=32 + lr~6e-4 consistently outperforms other combos
- multi_similarity with large batch_size=128 underperforms (trials 0, 4)
- These params were used directly for the 5-fold CV training run (Run 1 above), achieving 0.9974 mean F1

**Run 2** — 26 trials completed/pruned, 7h timeout (nymfe89, 2026-03-29, `logs_out/hpo_20260329.log`)
**Config**: hpo_n_trials=30, hpo_timeout=25200s, hpo_epochs=10, fold 0 only

| Trial | F1       | embedding_dim | margin | lr       | mining_strategy  | batch_size | Notes |
|-------|----------|---------------|--------|----------|------------------|------------|-------|
| 0     | 0.9315   | 256           | 0.469  | 4.33e-5  | semihard         | 64         |       |
| 1     | 0.9931   | 256           | 0.356  | 8.01e-5  | batch_hard       | 64         |       |
| 2     | 0.9205   | 256           | 0.058  | 1.46e-5  | semihard         | 128        |       |
| 3     | 0.9907   | 128           | 0.446  | 7.40e-5  | semihard         | 128        |       |
| 4     | 0.9949   | 512           | 0.169  | 2.59e-4  | semihard         | 128        |       |
| 5–9   | pruned   | —             | —      | —        | —                | —          |       |
| 10    | 0.9956   | 512           | 0.327  | 7.46e-4  | semihard         | 32         |       |
| 11    | 0.9955   | 512           | 0.318  | 8.03e-4  | semihard         | 32         |       |
| 12    | pruned   | —             | —      | —        | —                | —          |       |
| 13    | 0.9949   | 512           | 0.364  | 9.78e-4  | semihard         | 32         |       |
| 14    | 0.9964   | 512           | 0.308  | 4.57e-4  | semihard         | 32         |       |
| 15    | 0.9964   | 512           | 0.260  | 3.80e-4  | multi_similarity | 32         |       |
| 16    | 0.9973   | 512           | 0.256  | 2.94e-4  | multi_similarity | 32         |       |
| 17    | 0.9964   | 512           | 0.249  | 1.98e-4  | multi_similarity | 32         |       |
| 18–19 | pruned   | —             | —      | —        | —                | —          |       |
| 20    | 0.9940   | 512           | 0.216  | 1.33e-4  | multi_similarity | 32         |       |
| 21    | 0.9958   | 512           | 0.261  | 2.71e-4  | multi_similarity | 32         |       |
| 22    | 0.9964   | 512           | 0.246  | 4.85e-4  | multi_similarity | 32         |       |
| 23    | 0.9973   | 512           | 0.166  | 2.26e-4  | multi_similarity | 32         |       |
| **24** | **0.9976** | **512**   | **0.178** | **2.81e-4** | **multi_similarity** | **32** | **Best** |
| 25    | pruned   | —             | —      | —        | —                | —          |       |
| 26    | —        | —             | —      | —        | —                | —          | cut off mid-trial |

**Best params (trial 24)**:
- embedding_dim: 512, margin: 0.1776, lr: 2.808e-4, weight_decay: 5.372e-5
- lr_backbone_factor: 0.3088, mining_strategy: multi_similarity, batch_size: 32

**Observations**:
- 26 trials completed/pruned; log cuts off mid-trial 26 (20,181 of 25,200s elapsed)
- Strong convergence on **embedding_dim=512 + multi_similarity + batch_size=32** — dominant pattern in trials 14–24
- Unlike Run 1, batch_hard barely appears; multi_similarity wins here
- embedding_dim=512 consistently outperforms 128/256 in this run — opposite of wrong file I read earlier
- HPO F1 of best trial (0.9976) slightly exceeds the current training run mean (0.9974) but comparison is not apples-to-apples (10ep vs 30ep, fold 0 vs 5-fold)
- **Decision**: Potentially worth re-running train.py — dim=512 is better for Mahalanobis/subtask b, and multi_similarity is untested in full CV. See tradeoff discussion below.

### Run 2 — 5-fold CV (2026-03-31, nymfe89)
**Config**: embedding_dim=512, margin=0.1776, LR=2.808e-4, weight_decay=5.372e-5, lr_backbone_factor=0.3088, batch_size=32, epochs=50, patience=7, mining=multi_similarity
Log: `logs_out/train_emb512.log`

| Fold | Best F1   | Best Epoch | Epochs run | Early stop? |
|------|-----------|------------|------------|-------------|
| 0    | 0.9994    | 33         | 40         | Yes         |
| 1    | 0.9980    | 23         | 30         | Yes         |
| 2    | 0.9982    | 26         | 33         | Yes         |
| 3    | 0.9984    | 31         | 38         | Yes         |
| 4    | 0.9985    | 22         | 29         | Yes         |
| **Mean** | **0.9985 ± 0.0005** | | | Baseline: 0.76 — **PASSED** |

**Comparison with Run 1** (emb256, batch_hard):
- Mean F1: 0.9985 vs 0.9974 (+0.0011)
- Std: 0.0005 vs 0.0008 (more consistent)
- All folds early-stopped (patience=7 was sufficient with 50 epochs)
- Best fold (0) reached 0.9994 — near-perfect

**Loss trajectory**: Similar two-phase pattern to Run 1 — gradual descent to ~0.08 over epochs 1–8, then a sharp drop (likely ReduceLROnPlateau firing) to ~0.03–0.04, continuing to ~0.005–0.01 by convergence.

**Convergence**: All 5 folds early-stopped well before epoch 50, confirming that epochs=50 with patience=7 was sufficient. Best epochs ranged from 22 to 33.

## Evaluation (04_evaluate.py)

### emb256 (Run 1, 2026-03-28, nymfe01)
- F1 Macro: **0.9972** (baseline 0.76, PASSED)
- Outputs: confusion_matrix.png, per_class_f1.png, umap_embeddings.png, gradcam_samples.png, results_summary.csv

### Grad-CAM
- **Fixed**: now uses negative L2 distance to the predicted class prototype as the Grad-CAM target.
  Gradients highlight regions that push the embedding toward the correct prototype — semantically
  meaningful for metric learning.
- Previous version used `ClassifierOutputTarget(0)` (embedding dim 0), which was arbitrary and
  produced random-looking heatmaps despite the model being excellent.
- Per-class results after fix:
  - **Pepper Bacterial spot**: nearly entirely cold/blue — almost no activation. The bacterial
    spots are very small relative to the leaf area so the gradient signal is too diffuse to localise.
    Weakest result.
  - **Potato Late blight**: strong, sharply focused red hotspot directly on the large brown
    necrotic lesion in the upper-centre of the leaf. Rest of the leaf is cold. Best/clearest result.
  - **Tomato Late blight**: broad warm activation (orange/red) covering the central and right
    body of the leaf, including the visibly wilted and dark tissue. Diffuse rather than pinpointed,
    which makes sense — late blight tends to affect large contiguous areas.
  - **Tomato Septoria leaf spot**: concentrated warm activation in the lower-centre of the leaf,
    corresponding to the densest cluster of small circular Septoria lesions. Upper part of the leaf
    is cooler. Reasonably localised.
  - **Tomato YellowLeaf Curl Virus**: strong red/orange activation on the upper-centre of the
    leaf where the curling and yellowing is most pronounced. Clear and meaningful.
  - **Tomato healthy**: warm activation along the right leaf edge and into the leaf body. Unexpected
    for a healthy leaf; likely the model focuses on the uniform, undamaged leaf margin/texture as
    a "confirmation of health" signal rather than any lesion.
- pytorch-grad-cam imports moved to top of file (were inside the function).

### emb512 — fold 0 (Run 2, 2026-04-13, nymfe89) — SUPERSEDED
- F1 Macro: **0.9994** (baseline 0.76, PASSED)
- Fold 0 validation: **perfect** 1.00 precision/recall/F1 across all 15 classes (4,128 samples)
- Log: `logs_out/evaluate_emb512.log`
- **Note**: This used fold 0, same as HPO — results have optimistic bias from hyperparameter selection. Superseded by fold 1 run below.

### emb512 — fold 1 (Run 3, 2026-04-14, nymfe89)
- **Fold changed to 1** to avoid HPO bias: HPO was tuned on fold 0's validation set, so evaluating
  on fold 0 double-dips into the same data. Fold 1 is a clean held-out split.
- F1 Macro: **0.9980** (baseline 0.76, PASSED) — vs 0.9994 on fold 0 (overfit) and 0.9972 for emb256
- Fold 1 val set: 4,128 samples. Most classes 1.00; minor imperfections:
  - `Potato___healthy`: P=0.97, R=1.00, F1=0.98 (smallest class, only 31 samples)
  - `Potato___Late_blight`: P=1.00, R=0.99 (one missed sample)
  - `Tomato_Early_blight`: P=1.00, R=0.99
  - `Tomato_Late_blight`: P=0.99, R=1.00
- Outputs: confusion_matrix.png, per_class_f1.png, umap_embeddings.png, gradcam_samples.png, results_summary.csv
- Log: `logs_out/evaluate_emb512.log`

**Interpretation**: The ~0.0014 drop (0.9994 → 0.9980) from fold 0 to fold 1 quantifies the
optimistic bias introduced by HPO on fold 0. The fold 1 number is the honest estimate of
generalization performance for this hyperparameter configuration.

### Evaluation methodology — no separate test set
- Entire dataset used for 5-fold stratified CV; no held-out test set.
- **Defensible** because smallest class is 152 images — a further split would critically reduce
  minority-class training data. Cite: Kohavi (1995), IJCAI.
- **Caveat to acknowledge**: validation folds are also used for early stopping, introducing mild
  optimistic bias. Nested CV would eliminate this but is impractical given class sizes.
- Suggested report wording: "We report F1 on held-out validation folds of a 5-fold stratified CV.
  Given the smallest class contains only 152 samples [Kohavi, 1995], a further train/test split
  would critically reduce training data for minority classes. Note that validation folds were also
  used for early stopping, which may introduce a minor optimistic bias."

## Unknown Detection (05_unknown.py)

### emb256 — fold 0 (2026-03-28)
- Results (fold 0, checkpoint from training run):
  - AUROC: **0.9795**, PR-AUC: **0.9627**
  - Optimal threshold: 6.49 (Youden's J)
  - Precision: 0.8874 / Recall: 0.9262 / F1: 0.9064 at threshold
  - Known distances: mean=3.78, std=2.24
  - Unknown distances: mean=18.51, std=9.05
  - Mann-Whitney U p-value: ~0.00 (unknown distances stochastically greater, highly significant)

### UMAP visualizations
- `unknown_umap_known_only.png`: 14 known classes form tight, well-separated clusters —
  confirms the metric learning is working and embeddings are discriminative.
- `unknown_umap.png`: `Tomato_Bacterial_spot` (black X) forms a large dense mass that
  visually overlaps several known Tomato clusters in the 2D projection.
- **Important caveat**: UMAP is a lossy 2D projection — the visual overlap does NOT contradict
  the AUROC of 0.98. Mahalanobis distance operates in the full 256-D space where separation is
  clean (unknown mean 18.51 vs known mean 3.78).
- **Report point**: use this contrast to argue that 2D visualization alone is misleading and
  that high-dimensional distance metrics (Mahalanobis) are necessary for reliable detection.
- The unknown class landing near known Tomato clusters also makes biological sense — it's still
  a tomato disease sharing visual features with known classes, just never seen during training.

### Prototype fitting matters for reported distances
- First run (prototype fit from augmented `train_loader` with `drop_last=True`) gave
  unknown mean distance 13.73 and AUROC 0.9772.
- After fix (eval-transform loader, all samples, no augmentation), unknown mean distance
  jumped to 18.51 and AUROC improved to 0.9795.
- **Same model weights, better prototypes.** Augmentation adds noise to centroid estimates;
  dropping the last batch silently excludes some training samples.
- The reported numbers above use the corrected eval-transform prototype fitting.
- **Important for the report**: mention that prototype quality (how class centroids are computed)
  directly affects Mahalanobis distance scale and threshold — not just model training.

### emb512 — fold 1 (Run 3, 2026-04-14, nymfe89)
- **Fold changed to 1** (same reasoning as eval — avoid HPO bias on fold 0).
- Trained from scratch on 14 known classes, fold 1 split. Early stopped at epoch 37.
- Results:
  - AUROC: **0.9933** (vs 0.9795 for emb256 fold 0)
  - PR-AUC: **0.9899** (vs 0.9627 for emb256 fold 0)
  - Optimal threshold: **8.41** (Youden's J)
  - Precision: 0.9489 / Recall: 0.9520 / **F1: 0.9505** at threshold
  - Known distances (n=3,703): mean=**4.67**, std=1.57
  - Unknown distances (n=2,127): mean=**21.74**, std=9.40
  - Mann-Whitney U p-value: ~0 (unknown distances stochastically greater, highly significant)
- Outputs: unknown_distance_histogram.png, unknown_roc_curve.png, unknown_confusion_matrix.png,
  unknown_umap_known_only.png, unknown_umap.png, unknown_detection_results.csv
- Log: `logs_out/unknown_emb512.log`

**Comparison with emb256 fold 0** (previous unknown result):
- AUROC +0.0138 (0.9795 → 0.9933)
- F1 @ threshold +0.044 (0.9064 → 0.9505)
- Unknown/known distance ratio improved: was 18.51/3.78 = 4.9x; now 21.74/4.67 = 4.7x
  (slightly lower ratio but both distributions shifted up due to emb512's higher-dim space —
   what matters is the lower std of known distances: 1.57 vs 2.24 = tighter clusters)
- **Root cause of improvement**: emb512 dimensionality plus multi_similarity mining produces
  more discriminative embeddings, making known classes cluster tighter (lower known-distance std)
  and pushing unknowns further out in the Mahalanobis metric space.
