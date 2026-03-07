"""
member2_siamese/loss.py
────────────────────────────────────────────────────────────────────────────
Member 2 – Segmentation Loss Functions

Provides:
  DiceLoss         – smooth Dice loss for binary segmentation
  CombinedSegLoss  – weighted Dice + BCE for stable training
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

    Parameters
    ----------
    smooth : smoothing constant to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
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
        preds = torch.sigmoid(logits)
        preds   = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedSegLoss(nn.Module):
    """
    Weighted sum of Dice Loss + Binary Cross-Entropy for segmentation.

    Dice Loss handles class imbalance between changed/unchanged pixels.
    BCE provides stable gradient flow throughout training.

    Parameters
    ----------
    dice_weight : relative weight of Dice term (default 0.6)
    bce_weight  : relative weight of BCE term  (default 0.4)
    smooth      : smoothing for Dice
    pos_weight  : optional positive-class weight for BCEWithLogitsLoss
    """

    def __init__(
        self,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice = DiceLoss(smooth=smooth)

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
        dice_loss = self.dice(logits, targets)
        bce_loss  = self.bce(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
