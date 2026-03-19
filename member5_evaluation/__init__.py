"""member5_evaluation – metrics, inference pipeline, and evaluation utilities."""
from member5_evaluation.metrics import (
    compute_iou, compute_f1, per_class_f1, build_confusion_matrix,
    compute_xbd_score, evaluate_segmentation, evaluate_classifier,
)
from member5_evaluation.inference_pipeline import run_inference

__all__ = [
    "compute_iou", "compute_f1", "per_class_f1", "build_confusion_matrix",
    "compute_xbd_score", "evaluate_segmentation", "evaluate_classifier",
    "run_inference",
]
