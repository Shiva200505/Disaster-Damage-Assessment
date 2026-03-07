"""
member3_classifier/focal_loss.py
────────────────────────────────────────────────────────────────────────────
Member 3 – Focal Loss

Focal Loss (Lin et al., 2017) down-weights easy examples and focuses
learning on rare, hard-to-classify samples — ideal for the xBD dataset
where the "destroyed" class is significantly underrepresented.

FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

Parameters
----------
alpha : class balance factor (scalar or per-class weight tensor)
gamma : focusing parameter — higher γ ⟹ more focus on hard examples
        (γ=0 reduces Focal Loss to standard cross-entropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Accepts raw logits and integer class labels (same API as nn.CrossEntropyLoss).

    Parameters
    ----------
    alpha : float or list[float]
        If float: uniform scalar applied to all classes.
        If list:  per-class weight (length = num_classes).
        Default 0.25.
    gamma : float
        Focusing parameter. Default 2.0.
    reduction : "mean" | "sum" | "none"
    """

    def __init__(
        self,
        alpha: float | list = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        if isinstance(alpha, (int, float)):
            self.alpha = alpha
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw class logits
        targets : (B,)   integer class labels in [0, C)

        Returns
        -------
        Scalar focal loss (or per-sample tensor if reduction="none").
        """
        # Standard cross-entropy per sample (natural log, not log2)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # p_t = probability of the correct class
        p_t = torch.exp(-ce_loss)                         # (B,)

        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * ((1.0 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
