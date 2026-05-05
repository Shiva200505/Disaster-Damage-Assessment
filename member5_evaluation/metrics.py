import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def compute_iou(pred, target, eps: float = 1e-7) -> float:
    pred, target = _to_numpy(pred) > 0.5, _to_numpy(target) > 0.5
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / (union + eps))

def compute_f1(pred, target, eps: float = 1e-7) -> float:
    pred, target = _to_numpy(pred) > 0.5, _to_numpy(target) > 0.5
    intersection = np.logical_and(pred, target).sum()
    return float(2 * intersection / (pred.sum() + target.sum() + eps))

def compute_precision_recall(pred, target, eps: float = 1e-7):
    pred, target = _to_numpy(pred) > 0.5, _to_numpy(target) > 0.5
    intersection = np.logical_and(pred, target).sum()
    precision = intersection / (pred.sum() + eps)
    recall = intersection / (target.sum() + eps)
    return float(precision), float(recall)

def compute_ap(pred_prob, target) -> float:
    """Area under the precision-recall curve using sklearn."""
    pred_prob, target = _to_numpy(pred_prob), _to_numpy(target)
    if target.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(target.flatten(), pred_prob.flatten())
    return float(auc(recall, precision))

def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error for binary targets."""
    probs, labels = _to_numpy(probs), _to_numpy(labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    probs_flat = probs.flatten()
    labels_flat = labels.flatten()

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs_flat > bin_lower, probs_flat <= bin_upper)
        prob_in_bin = in_bin.mean()

        if prob_in_bin > 0:
            accuracy_in_bin = labels_flat[in_bin].mean()
            avg_confidence_in_bin = probs_flat[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return float(ece)

def compute_per_class_iou(pred, target, num_classes: int = 4):
    """Computes IoU for each class (0 to num_classes-1)."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    ious = []
    for cls_idx in range(num_classes):
        pred_cls = (pred == cls_idx)
        target_cls = (target == cls_idx)
        ious.append(compute_iou(pred_cls, target_cls))
    return np.array(ious)

def xbd_localization_score(pred_cls, target_cls) -> float:
    """
    Harmonic mean of F1 for "no_damage" class vs all other classes.
    Assuming our mapping: 0=no_damage, 1=minor, 2=major, 3=destroyed.
    """
    pred_cls, target_cls = _to_numpy(pred_cls), _to_numpy(target_cls)
    f1_no_damage = compute_f1(pred_cls == 0, target_cls == 0)
    
    f1_damage = compute_f1(pred_cls > 0, target_cls > 0)
    
    if (f1_no_damage + f1_damage) == 0:
        return 0.0
    return float(2 * (f1_no_damage * f1_damage) / (f1_no_damage + f1_damage))
