# Autoresearch Program — Speed Optimization

This is an experiment to have the LLM do its own research. You are an autonomous ML
experimentation agent. Your goal is to **minimize training time** while maintaining
classification quality on the PlantVillage plant disease dataset.

## Setup

Before starting the experiment loop:

1. Agree on a run tag: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. Create the branch: `git checkout -b autoresearch/<tag>`
3. Read the key in-scope files: `src/config.py`, `src/model.py`, `src/trainer.py`, `src/dataset.py`, `src/losses.py`
4. Verify the data exists: `data/PlantVillage/` should have 15 class subdirectories
5. Run the baseline experiment to establish reference metrics: `python autoresearch/run_experiment.py > autoresearch/run.log 2>&1`
6. Record the baseline row in `autoresearch/results.tsv`
7. Confirm you understand the objective and are ready to iterate

## Objective

- **Primary metric**: `training_seconds` (minimize)
- **Constraint**: `val_f1_macro >= baseline_epoch3_f1 × 0.98` (within 2% of baseline at epoch 3)
- **Baseline**: F1 = 0.9951 at convergence. Establish epoch-3 baseline on first run.

Training runs up to 10 epochs with patience=3. Each run should complete in roughly 5 minutes
(target budget per experiment). If a run exceeds 10 minutes, kill it and record as `timeout`.

## Experimentation

Modify **only the in-scope files listed below** to try optimizations. The goal is to find
changes that reduce per-epoch wall-clock time while preserving embedding quality.

**Prohibited:**
- Modifying `autoresearch/run_experiment.py` (the measurement harness — must stay fixed)
- Modifying `scripts/` (the official evaluation pipeline)
- Modifying `src/prototype.py`, `src/utils.py`, `src/visualization.py`
- Installing new packages (`pip install` is not allowed)
- Hardcoding fold indices or data paths
- Skipping validation or computing F1 on a subset during the experiment run

**Simplicity criterion:** Prefer clean, elegant solutions over hacky workarounds. If two
changes produce similar speedups, keep the simpler one. Complexity that only helps this one
dataset is not worth it.

**VRAM constraint:** Must fit in 6 GB (local RTX 3060 Laptop GPU). If a run OOMs, record
`oom` in status and revert immediately.

## Decision Rule

After each experiment:
1. Parse `val_f1_macro` and `training_seconds` from `autoresearch/run.log`
2. If `training_seconds` decreased **AND** `val_f1_macro >= baseline_epoch3_f1 × 0.98` → **KEEP**, update baseline, commit
3. If `val_f1_macro` drops below threshold → **REVERT** (`git checkout -- src/`), regardless of speed gain
4. If `training_seconds` did not decrease → **REVERT**
5. If run crashed or timed out → **REVERT**, record `crash`/`timeout` in status

Log **every** experiment (kept or reverted) to `autoresearch/results.tsv`.

## Execution

```bash
python autoresearch/run_experiment.py > autoresearch/run.log 2>&1
```

Parse the structured block at the end of `autoresearch/run.log`:
```
---
val_f1_macro:     <float>
train_loss:       <float>
best_epoch:       <int>
training_seconds: <float>
peak_vram_mb:     <float>
```

To extract quickly:
```bash
grep -E "^(val_f1_macro|training_seconds|peak_vram_mb):" autoresearch/run.log
```

If the file is empty or the block is missing, the run crashed — check stderr and record `crash`.

## Modifiable Files

- `src/config.py` — hyperparameters: backbone, embedding_dim, img_size, batch_size, lr, epochs, patience, num_workers, etc.
- `src/model.py` — backbone architecture, embedding head, feature dimensions
- `src/losses.py` — loss function, mining strategy
- `src/dataset.py` — transforms (augmentation pipeline), data loading
- `src/trainer.py` — training loop, mixed precision, gradient accumulation

## Speed Optimization Ideas

Try these **one at a time**. Ordered roughly by expected impact:

### 1. Mixed Precision (high impact, low risk)
- Add `torch.amp.autocast("cuda")` in `trainer.py`'s `train_one_epoch`
- Use `torch.amp.GradScaler("cuda")` for stable gradients
- Typically 1.5–2× speedup on modern NVIDIA GPUs with no accuracy loss

### 2. Backbone (biggest architectural impact)
- `resnet18` or `resnet34` — much smaller than resnet50; feature dim 512 instead of 2048
- `mobilenet_v3_large` or `mobilenet_v3_small` — mobile-optimized
- `efficientnet_b0` — strong accuracy/speed tradeoff
- Backbone swap requires updating `model.py` to handle different feature output dimensions

### 3. Image Size
- Try 128, 160, 192 (currently 224)
- Quadratic impact on compute: 128×128 ≈ 3× faster convolutions than 224×224
- Smaller images may reduce F1 slightly — check the threshold

### 4. Embedding Dimension
- Try 64, 128 (currently 256 after HPO)
- Smaller = faster head computation + smaller prototype centroids during validation

### 5. torch.compile()
- Wrap model: `model = torch.compile(model)` in `run_experiment.py`... wait, that's read-only
- Add compile call in `trainer.py` after model is moved to device
- PyTorch 2.x; adds ~30s startup cost but speeds up repeated kernels

### 6. Batch Size / Sampler
- Larger batch size saturates GPU (within 6 GB VRAM)
- Check `samples_per_class` too — MPerClassSampler controls batch composition

### 7. Lighter Augmentation
- Remove expensive transforms: `RandomRotation`, `ColorJitter`
- Keep only `RandomResizedCrop` + `RandomHorizontalFlip` as a minimal baseline
- Augmentation happens on CPU — reducing it speeds up data loading

### 8. Frozen Backbone
- Set backbone learning rate to 0 (or freeze with `param.requires_grad = False`)
- Train only the embedding head — much faster per epoch
- Risk: lower final F1 if backbone needs fine-tuning for this domain

### 9. num_workers
- Try 2, 6, 8 (currently 4) — optimal depends on CPU cores and I/O speed
- On Windows, higher num_workers can sometimes hurt due to process spawn overhead

### 10. Gradient Accumulation
- Simulate larger batch size with fewer actual forward passes
- Useful if VRAM limits batch size; less useful if GPU is already saturated

## Experiment Loop

Operate continuously without pausing for human confirmation:

```
while True:
    1. Read current best metrics from results.tsv (last "kept" row)
    2. Pick the highest-expected-impact untried optimization
    3. Make ONE targeted change to the modifiable files
    4. Run: python autoresearch/run_experiment.py > autoresearch/run.log 2>&1
       - If run takes > 10 min, kill it (Ctrl+C), record timeout, revert
    5. Check if run.log ends with the structured "---" block
       - If missing: crash — revert, record crash
    6. Parse val_f1_macro and training_seconds
    7. Apply decision rule (keep or revert)
    8. Append row to results.tsv with commit hash (or "reverted"), metrics, status, description
    9. If kept: git add -p src/ && git commit -m "autoresearch: <description> | F1={:.4f} sec={:.0f}"
   10. Repeat
```

## Logging

`autoresearch/results.tsv` columns (tab-separated):
```
commit  val_f1_macro  train_loss  training_sec  vram_mb  status  description
```

- `status`: one of `kept`, `reverted`, `crash`, `timeout`, `oom`
- `commit`: git short hash if kept, `reverted` otherwise
- `description`: one-line summary of the change tried (e.g. "resnet18 backbone", "mixed precision amp")
