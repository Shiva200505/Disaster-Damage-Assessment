"""
member4_visualization/static_plots.py
────────────────────────────────────────────────────────────────────────────
Member 4 – Matplotlib / Seaborn Static Plots

Functions
─────────
plot_training_curves  – loss + F1 over epochs (train vs val)
plot_confusion_matrix – Seaborn heatmap normalised by row
plot_sample_predictions – 4-panel pre/post/GT mask/pred mask viewer
plot_class_distribution – bar chart of building counts per damage class
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


DAMAGE_LABELS = ["No Damage", "Minor", "Major", "Destroyed"]
DAMAGE_COLORS = ["#2ECC71",   "#F1C40F", "#E67E22", "#E74C3C"]


# ── Training Curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot loss and F1 curves for train and validation splits.

    Parameters
    ----------
    history   : dict with keys "train_loss", "val_loss", "train_f1", "val_f1"
                Each value is a list of per-epoch floats.
    save_path : if given, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="#3498DB", linewidth=1.8)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#E74C3C", linewidth=1.8, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # F1
    ax2.plot(epochs, history["train_f1"], label="Train", color="#3498DB", linewidth=1.8)
    ax2.plot(epochs, history["val_f1"],   label="Val",   color="#E74C3C", linewidth=1.8, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm          : (C, C) integer confusion matrix
    class_names : list of class label strings (default DAMAGE_LABELS)
    normalize   : if True, normalize each row to sum to 1
    """
    if class_names is None:
        class_names = DAMAGE_LABELS

    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.where(row_sums > 0, data / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", None

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data,
        annot=True, fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0, vmax=vmax,
        linewidths=0.5, linecolor="#dddddd",
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Sample Prediction Grid ────────────────────────────────────────────────────

def plot_sample_predictions(
    pre_imgs:   List[np.ndarray],
    post_imgs:  List[np.ndarray],
    gt_masks:   List[np.ndarray],
    pred_masks: List[np.ndarray],
    n_samples:  int = 4,
    save_path:  Optional[str | Path] = None,
    title:      str = "Sample Predictions",
) -> plt.Figure:
    """
    Display an N×4 grid: Pre | Post | Ground-truth mask | Predicted mask.

    All images should be (H, W, 3) uint8 or float32; masks should be (H, W).
    """
    n = min(n_samples, len(pre_imgs))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]   # ensure 2D

    headers = ["Pre-Disaster", "Post-Disaster", "GT Change Mask", "Predicted Mask"]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=10, fontweight="bold")

    for row in range(n):
        for col, img in enumerate([pre_imgs[row], post_imgs[row],
                                   gt_masks[row],  pred_masks[row]]):
            ax = axes[row, col]
            if col < 2:
                disp = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
                ax.imshow(disp[:, :, :3] if disp.ndim == 3 else disp, cmap=None)
            else:
                ax.imshow(img, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Class Distribution Bar Chart ──────────────────────────────────────────────

def plot_class_distribution(
    counts: Dict[str, int],
    save_path: Optional[str | Path] = None,
    title: str = "Damage Class Distribution",
) -> plt.Figure:
    """
    Bar chart showing number of buildings per damage class.

    Parameters
    ----------
    counts : dict mapping class name → count  (keys from DAMAGE_LABELS)
    """
    labels = [k for k in DAMAGE_LABELS if k.lower().replace(" ", "_") in counts
              or k in counts]
    values = [counts.get(k, counts.get(k.lower().replace(" ", "_"), 0)) for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=DAMAGE_COLORS[:len(labels)], edgecolor="white",
                  linewidth=0.8, width=0.55)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Buildings")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
