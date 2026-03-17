# Autoresearch — Autonomous Speed Optimization

Inspired by Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern: an AI agent autonomously runs ML experiments in a loop, making one change at a time, measuring impact, and keeping only improvements.

## Goal

Minimize **training time** while maintaining **F1 macro >= 0.99** on the PlantVillage plant disease classification task.

## How It Works

1. The agent reads `program.md` for instructions and constraints
2. It makes **one change** to the source code (e.g., smaller backbone, mixed precision, reduced image size)
3. It runs `run_experiment.py`, which trains fold 0 for up to 10 epochs and reports structured metrics
4. If training got faster AND F1 stayed above 0.99 → keep the change and commit
5. Otherwise → revert and try something else
6. Every experiment is logged to `results.tsv`
7. Loop indefinitely until interrupted

## Quick Start

```bash
# 1. Establish baseline
python autoresearch/run_experiment.py

# 2. Paste the prompt below into Claude Code to start the agent loop
```

## Agent Prompt

Copy and paste this into Claude Code to start an autoresearch session:

```
Read autoresearch/program.md and follow the instructions exactly. Begin with the Setup section, then enter the experiment loop. Do not stop or ask for confirmation between experiments.
```

## Files

| File | Purpose |
|---|---|
| `program.md` | Agent instructions — objective, rules, optimization ideas |
| `run_experiment.py` | Fast single-fold trainer (fold 0, 10 epochs, patience 3) |
| `results.tsv` | Experiment log (gitignored) |
| `run.log` | Last experiment output (gitignored) |
