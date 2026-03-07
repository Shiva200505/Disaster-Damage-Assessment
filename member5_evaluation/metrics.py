"""
member5_evaluation/metrics.py
────────────────────────────────────────────────────────────────────────────
Member 5 – Evaluation Metrics

Implements:
  compute_iou          – pixel-wise Intersection over Union (binary)
  compute_f1           – pixel-wise F1 / Dice coefficient (binary)
  per_class_f1         – per-class F1 for multi-class classification
  compute_xbd_score    – xBD challenge harmonic mean (loc_f1, cls_f1)
  evaluate_segmentation– full evaluation pass over a DataLoader
  evaluate_classifier  – full evaluation pass for classifier DataLoader
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    f1_score as sklearn_f1,
    confusion_matrix as sklearn_cm,
    classification_report,
)


# ── Binary segmentation metrics ───────────────────────────────────────────────

def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Pixel-wise Intersection over Union for binary change masks.

    Parameters
    ----------
    pred    : (B, 1, H, W) float tensor.  If values are logits, pass through
              sigmoid first; if already probabilities set threshold accordingly.
    target  : (B, 1, H, W) binary float tensor {0, 1}
    threshold: binarisation threshold applied to pred

    Returns
    -------
    Scalar IoU value averaged over the batch.
    """
    with torch.no_grad():
        p = (pred > threshold).float().view(-1)
        t = target.float().view(-1)
        intersection = (p * t).sum().item()
        union = (p + t - p * t).sum().item()
    return (intersection + smooth) / (union + smooth)


def compute_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Pixel-wise F1 (Dice coefficient) for binary change masks.

    Returns
    -------
    Scalar F1 value averaged over the batch.
    """
    with torch.no_grad():
        p = (pred > threshold).float().view(-1)
        t = target.float().view(-1)
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)


# ── Multi-class classification metrics ────────────────────────────────────────

def per_class_f1(
    preds: List[int],
    targets: List[int],
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class and macro-averaged F1 for damage classification.

    Parameters
    ----------
    preds, targets : flat lists of integer class predictions / ground-truth
    num_classes    : total number of classes
    class_names    : optional class label strings

    Returns
    -------
    Dict with keys "<class_name>_f1" and "macro_f1".
    """
    names  = class_names or [f"class_{i}" for i in range(num_classes)]
    scores = sklearn_f1(targets, preds, labels=list(range(num_classes)),
                        average=None, zero_division=0)
    result = {f"{names[i]}_f1": float(scores[i]) for i in range(len(scores))}
    result["macro_f1"] = float(np.mean(scores))
    return result


def confusion_matrix(
    preds: List[int],
    targets: List[int],
    num_classes: int = 4,
) -> np.ndarray:
    """Return (num_classes × num_classes) integer confusion matrix."""
    return sklearn_cm(targets, preds, labels=list(range(num_classes)))


# ── xBD Challenge Metric ──────────────────────────────────────────────────────

def compute_xbd_score(localization_f1: float, classification_f1: float) -> float:
    """
    xBD challenge overall score = harmonic mean of localisation F1 and
    classification F1 (macro-averaged over 4 damage levels).

    Parameters
    ----------
    localization_f1   : pixel-level F1 from Stage 1 (change detection)
    classification_f1 : macro F1 from Stage 2 (damage severity)

    Returns
    -------
    Scalar harmonic mean in [0, 1].
    """
    if localization_f1 + classification_f1 < 1e-9:
        return 0.0
    return 2 * (localization_f1 * classification_f1) / (localization_f1 + classification_f1)


# ── End-to-end evaluation helpers ────────────────────────────────────────────

@torch.no_grad()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run the Siamese network over all validation/test batches and
    aggregate IoU and F1.

    Returns
    -------
    Dict with "mean_iou" and "mean_f1".
    """
    model.eval()
    total_iou, total_f1, n = 0.0, 0.0, 0

    for batch in loader:
        pre  = batch["pre"].to(device)
        post = batch["post"].to(device)
        mask = batch["change_mask"].to(device)

        logits = model(pre, post)
        preds  = torch.sigmoid(logits)

        total_iou += compute_iou(preds, mask, threshold=threshold)
        total_f1  += compute_f1(preds,  mask, threshold=threshold)
        n += 1

    return {
        "mean_iou": total_iou / max(n, 1),
        "mean_f1":  total_f1  / max(n, 1),
    }


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    num_classes: int = 4,
) -> Dict:
    """
    Run the damage classifier over all batches and compute per-class F1
    and the confusion matrix.

    Returns
    -------
    Dict with "macro_f1", per-class F1s, and "confusion_matrix".
    """
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

    f1s = per_class_f1(all_preds, all_labels, num_classes=num_classes)
    cm  = confusion_matrix(all_preds, all_labels, num_classes=num_classes)
    return {**f1s, "confusion_matrix": cm}
