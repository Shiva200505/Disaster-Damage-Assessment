"""
member2_siamese/losses_advanced.py
Lovász Loss for Stage 3 fine-tuning.
Only use AFTER a stable model exists (Val F1 > 0.65).
Using Lovász from scratch causes degenerate all-zero predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient for binary case."""
    p = gt_sorted.shape[0]
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:p - 1]
    return jaccard


def _binary_lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovász hinge loss for one image."""
    if labels.sum() == 0:
        return logits.sum() * 0.0   # no positive pixels — zero loss

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Binary Lovász Hinge Loss.
    Directly optimizes IoU — better than Dice for final fine-tuning.

    WARNING: Do NOT use this from scratch (random weights).
    The model must already produce non-trivial predictions.
    Use CombinedSegLoss for Stages 1 and 2.
    Use this only for Stage 3.

    Paper: Berman et al., "The Lovász-Softmax loss", CVPR 2018.
    """
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) binary float {0, 1}
        """
        B = logits.shape[0]
        if self.per_image:
            losses = []
            for i in range(B):
                l = logits[i].view(-1)
                t = targets[i].view(-1).long()
                losses.append(_binary_lovasz_hinge(l, t))
            return torch.stack(losses).mean()
        else:
            return _binary_lovasz_hinge(logits.view(-1), targets.view(-1).long())


# ── Deep Supervision Loss ─────────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Weighted sum of losses at multiple decoder scales.
    Main output gets weight 1.0, auxiliary outputs get decreasing weights (e.g. 0.4, 0.2).
    """
    def __init__(self, base_loss: nn.Module, weights: tuple = (1.0, 0.4, 0.2)):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, logits_list, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits_list, (list, tuple)):
            # Fallback if deep supervision is disabled
            return self.base_loss(logits_list, targets)

        loss = 0.0
        for i, logits in enumerate(logits_list):
            w = self.weights[i] if i < len(self.weights) else 0.0
            if w > 0:
                # Resize targets to match current scale logits if needed
                if logits.shape[-2:] != targets.shape[-2:]:
                    tgt = targets
                    if tgt.dim() == 3:
                        tgt = tgt.unsqueeze(1)
                    tgt = F.interpolate(tgt.float(), size=logits.shape[-2:], mode='nearest')
                    if targets.dim() == 3:
                        tgt = tgt.squeeze(1)
                else:
                    tgt = targets
                loss += w * self.base_loss(logits, tgt)
        return loss


# ── Soft Dice Loss with OHEM ──────────────────────────────────────────────────

class SoftDiceLossOHEM(nn.Module):
    """Dice loss with OHEM — focuses on the hardest X% of pixels each batch."""
    def __init__(self, ohem_ratio: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.ohem_ratio = ohem_ratio
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()

        # Hardest pixels have the highest BCE loss
        with torch.no_grad():
            pixel_losses = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')
            k = int(self.ohem_ratio * pixel_losses.numel())
            if k == 0:
                k = pixel_losses.numel()
            _, topk_idx = torch.topk(pixel_losses, k)

        probs_hard = probs[topk_idx]
        targets_hard = targets[topk_idx]

        intersection = (probs_hard * targets_hard).sum()
        union = probs_hard.sum() + targets_hard.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice
