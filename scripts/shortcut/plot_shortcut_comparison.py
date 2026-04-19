"""Generate combined training-curve figure for leaf-only and bg-only ablation.

Reads the Fold-1 epoch summary lines from both training logs and produces a
side-by-side comparison plot for the report.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"Fold \d+ \| Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| F1: ([\d.]+)"
)


def parse_history(log_path: Path):
    epochs, losses, f1s = [], [], []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r", "\n")
    for m in EPOCH_RE.finditer(text):
        epochs.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        f1s.append(float(m.group(3)))
    return epochs, losses, f1s


def plot_side_by_side(leaf_log: Path, bg_log: Path, out_path: Path,
                      leaf_eval: dict, bg_eval: dict,
                      leaf_best_f1: float, leaf_best_epoch: int,
                      bg_best_f1: float, bg_best_epoch: int):
    l_epoch, l_loss, l_f1 = parse_history(leaf_log)
    b_epoch, b_loss, b_f1 = parse_history(bg_log)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=False)

    for ax, (title, epoch, loss, f1, eval_dict, best_f1, best_epoch) in zip(
        axes,
        [
            ("Trained on leaf-only", l_epoch, l_loss, l_f1, leaf_eval,
             leaf_best_f1, leaf_best_epoch),
            ("Trained on background-only", b_epoch, b_loss, b_f1, bg_eval,
             bg_best_f1, bg_best_epoch),
        ],
    ):
        color_f1 = "tab:blue"
        color_loss = "tab:red"
        ax.plot(epoch, f1, color=color_f1, lw=1.4, label="val F1-macro")
        ax.set_xlabel("epoch")
        ax.set_ylabel("F1-macro", color=color_f1)
        ax.tick_params(axis="y", labelcolor=color_f1)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(epoch, loss, color=color_loss, lw=1.0, ls="--",
                 label="train loss", alpha=0.8)
        ax2.set_ylabel("train loss", color=color_loss)
        ax2.tick_params(axis="y", labelcolor=color_loss)

        ax.axvline(best_epoch, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{title} — best val F1 = {best_f1:.4f}", fontsize=10)

        # eval-on-variants annotation box
        annot = "\n".join([
            f"eval on full:    {eval_dict['none']:.3f}",
            f"eval on leaf:    {eval_dict['leaf']:.3f}",
            f"eval on bg:      {eval_dict['background']:.3f}",
        ])
        ax.text(0.55, 0.05, annot, transform=ax.transAxes, fontsize=8,
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray",
                          boxstyle="round,pad=0.3"))

    fig.suptitle("Shortcut ablation — training on masked images (fold 1, chance=0.067)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-log", type=Path,
                        default=Path("results/logs/shortcut_train_leaf.log"))
    parser.add_argument("--bg-log", type=Path,
                        default=Path("results/logs/shortcut_train_bg.log"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/shortcut/shortcut_training_comparison.png"))
    args = parser.parse_args()

    # Cross-eval values and best val F1 (from run output)
    leaf_eval = {"none": 0.9955, "leaf": 0.9973, "background": 0.0845}
    bg_eval = {"none": 0.7368, "leaf": 0.0328, "background": 0.9664}

    plot_side_by_side(
        args.leaf_log, args.bg_log, args.out, leaf_eval, bg_eval,
        leaf_best_f1=0.9980, leaf_best_epoch=37,
        bg_best_f1=0.9700, bg_best_epoch=31,
    )


if __name__ == "__main__":
    main()
