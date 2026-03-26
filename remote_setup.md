# Remote Run Guide (nymfe) — Agent Instructions

This file is written for an AI agent running remote experiments on MUNI nymfe workstations.

## Prerequisites

Before proceeding, collect from the user:
- **xlogin** — their MUNI login (e.g. `xkraus1`). Ask if not provided.
- **Workstation number** — ask if not provided. Available: nymfe**01–02** and nymfe**87–104**.

Replace `xlogin` and `NN` with the actual values throughout this guide.

> Faculty VPN must be active on the user's machine to reach nymfe. Remind them if SSH fails to connect.

---

## 1. Check GPU Availability

Before running anything, verify the chosen workstation is free:

```bash
ssh xlogin@nymfeNN.fi.muni.cz "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

- If GPU utilization is high or memory is nearly full, tell the user and ask whether to try a different machine.
- Only one user should occupy a GPU at a time.

> **Note:** Workstations restart daily at 5 AM. The user should reserve their slot in the scheduling sheet.

---

## 2. Remote is Already Set Up

The repository, virtualenv, and dataset are already present. No installation needed.

| Path | Description |
|------|-------------|
| `~/data/pv056-project-2026/` | Repository root |
| `~/data/pv056-project-2026/.venv/` | Python virtual environment |
| `~/data/pv056-project-2026/src/` | Source code |
| `~/data/pv056-project-2026/scripts/` | Runnable scripts |
| `~/data/pv056-project-2026/data/PlantVillage/` | Dataset (15 class subdirs) |
| `~/data/pv056-project-2026/results/` | Output: plots, models |
| `~/data/pv056-project-2026/results/logs/` | Script logs |

Activate the venv at the start of each session:

```bash
source ~/data/pv056-project-2026/.venv/bin/activate
```

---

## 3. tmux (Use for Long Runs)

All long-running commands should be launched as **detached tmux sessions** so they survive SSH disconnects and clean up automatically when done:

```bash
tmux new-session -s <session-name> -d "<command>; tmux kill-session -t <session-name>"
```

- `-d` starts the session detached (no interactive terminal needed)
- `; tmux kill-session -t <session-name>` at the end auto-cleans the session on completion

Attach to a running session to see live output:

```bash
ssh xlogin@nymfeNN.fi.muni.cz "tmux attach -t <session-name>"
```

List active sessions:

```bash
ssh xlogin@nymfeNN.fi.muni.cz "tmux ls"
```

---

## 4. Run Scripts

All scripts must be run from the **project root**. Output goes to `results/`.

### Pull latest code and run a script (one-liner)

```bash
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s run -d 'cd ~/data/pv056-project-2026 && git fetch && git checkout <branch> && git pull && source .venv/bin/activate && python scripts/<script>.py > results/logs/<script>.log 2>&1; tmux kill-session -t run'"
```

### Available scripts

```bash
# EDA: class distribution + outlier detection
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s eda -d 'cd ~/data/pv056-project-2026 && source .venv/bin/activate && python scripts/01_eda.py > results/logs/eda.log 2>&1; tmux kill-session -t eda'"

# 5-fold CV training with early stopping
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s train -d 'cd ~/data/pv056-project-2026 && source .venv/bin/activate && python scripts/02_train.py > results/logs/train.log 2>&1; tmux kill-session -t train'"

# HPO (Optuna, 30 trials, fold 0)
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s hpo -d 'cd ~/data/pv056-project-2026 && source .venv/bin/activate && python scripts/03_hpo.py > results/logs/hpo.log 2>&1; tmux kill-session -t hpo'"

# Evaluation: confusion matrix, Grad-CAM, UMAP, per-class F1
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s evaluate -d 'cd ~/data/pv056-project-2026 && source .venv/bin/activate && python scripts/04_evaluate.py > results/logs/evaluate.log 2>&1; tmux kill-session -t evaluate'"

# Unknown disease detection (subtask b, Mahalanobis)
ssh xlogin@nymfeNN.fi.muni.cz "tmux new-session -s unknown -d 'cd ~/data/pv056-project-2026 && source .venv/bin/activate && python scripts/05_unknown.py > results/logs/unknown.log 2>&1; tmux kill-session -t unknown'"
```

### Fetch a log after the run

```bash
ssh xlogin@nymfeNN.fi.muni.cz "cat ~/data/pv056-project-2026/results/logs/<script>.log"
```

---

## 5. Monitor a Running Job

```bash
# Tail a live log
ssh xlogin@nymfeNN.fi.muni.cz "tail -f ~/data/pv056-project-2026/results/logs/train.log"

# GPU utilization
ssh xlogin@nymfeNN.fi.muni.cz "nvidia-smi"
```

---

## 6. Sync Code Changes

```bash
# 1. Make changes locally, commit and push
git add src/ scripts/
git commit -m "<description>"
git push origin <branch>

# 2. Pull on remote (already included in the run one-liners above)
ssh xlogin@nymfeNN.fi.muni.cz "cd ~/data/pv056-project-2026 && git pull"
```

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| SSH connection refused | Confirm VPN is active; try a different nymfe number |
| GPU memory full (OOM) | Ask user to pick a different machine |
| SSH connection dropped mid-run | Reconnect and run `tmux attach -t pv056` |
| `torch.cuda.is_available()` → False | Verify machine has a GPU (`nvidia-smi`); re-activate venv |
| Log file empty after run | Run crashed before writing output — check stderr via `cat results/logs/<script>.log` |

---

## Additional Resources

- Workstation rules: https://www.fi.muni.cz/tech/unix/nymfe.html.cs
