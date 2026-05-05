"""
member2_siamese/loss.py
────────────────────────────────────────────────────────────────────────────
Member 2 – Segmentation Loss Functions

Provides:
  DiceLoss         – smooth Dice loss for binary segmentation
  FocalLoss        – binary focal loss (γ=2) for extreme class imbalance
  CombinedSegLoss  – weighted Dice + Focal + (optional) BCE for stable training

Key design choices:
  - smooth=1e-4 (not 1.0) in Dice — prevents zero-gradient on all-negative patches
  - FocalLoss(gamma=2) down-weights easy negatives, amplifies hard positives
  - At 0.16% positive pixels, Focal is critical; plain BCE with low pos_weight
    produces near-zero gradients and the model predicts all-zero masks

These are the SAFE losses for Stages 1 and 2.
For Stage 3 fine-tuning, see member2_siamese/losses_advanced.py (LovaszLoss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Smooth Dice Loss for binary segmentation.

    Dice = 2·|P∩G| / (|P| + |G| + smooth)
    Loss = 1 - Dice

    Works on raw logits (applies sigmoid internally).

    NOTE: smooth=1e-4 (not 1.0) is intentional.
    With smooth=1.0 and sparse masks (0.16% positives), Dice = (0+1)/(0+0+1) = 1.0
    → DiceLoss = 0.0 → zero gradient on ~99% of batches. 1e-4 avoids this.

    Parameters
    ----------
    smooth : smoothing constant to avoid division by zero (keep small: 1e-4)
    """

    def __init__(self, smooth: float = 1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw network output
        targets : (B, 1, H, W) binary float mask {0, 1}

        Returns
        -------
        Scalar Dice loss tensor.
        """
        preds   = torch.sigmoid(logits)
        preds   = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Works on raw logits. At gamma=2, easy negatives (p≈0) get weight (1-0)^2=1
    but hard-to-classify positives (p≈0.1) get amplified: (1-0.1)^2=0.81 weight
    while trivially-easy predictions at high confidence contribute little.

    This is critical for xBD which has only ~0.16% positive pixels:
    without focal loss, the model learns to predict all-zero masks because
    that achieves near-zero loss with standard BCE.

    Parameters
    ----------
    gamma      : focusing parameter (2 is standard; higher = more focus on hard examples)
    alpha      : positive class weight scalar (analogous to pos_weight in BCE)
    reduction  : 'mean' or 'sum'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) binary float mask {0.0, 1.0}

        Returns
        -------
        Scalar focal loss tensor.
        """
        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1.0 - probs) * (1.0 - targets)   # prob of correct class
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_w = alpha_t * (1.0 - p_t) ** self.gamma

        loss = focal_w * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedSegLoss(nn.Module):
    """
    Weighted sum of Dice Loss + Focal Loss + (optional) BCE.

    Recommended weights for xBD (0.16% positive pixels):
      dice_weight  = 0.5
      focal_weight = 0.3
      bce_weight   = 0.2
      pos_weight   = 25.0  (in BCE; Focal uses its own alpha internally)

    Dice  — handles class imbalance at the region level
    Focal — focuses gradient on hard positives (most important for sparse masks)
    BCE   — provides stable, well-understood gradient baseline

    This is the RECOMMENDED loss for Stage 1 (baseline) and Stage 2 (V2 warm-start).
    Do NOT replace with Lovász until Val F1 > 0.65 (Stage 1 target).

    Parameters
    ----------
    dice_weight  : relative weight of Dice term (default 0.5)
    bce_weight   : relative weight of BCE term  (default 0.2)
    focal_weight : relative weight of Focal term (default 0.3)
    smooth       : smoothing for Dice (keep at 1e-4)
    pos_weight   : optional positive-class weight for BCEWithLogitsLoss
    focal_gamma  : gamma for FocalLoss (default 2.0)
    focal_alpha  : alpha for FocalLoss (default 0.75)
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.2,
        focal_weight: float = 0.3,
        smooth: float = 1e-4,
        pos_weight: float | None = None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        super().__init__()
        self.dice_weight  = dice_weight
        self.bce_weight   = bce_weight
        self.focal_weight = focal_weight

        self.dice  = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) binary float mask {0.0, 1.0}

        Returns
        -------
        Combined scalar loss tensor.
        """
        loss = 0.0
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        return loss
