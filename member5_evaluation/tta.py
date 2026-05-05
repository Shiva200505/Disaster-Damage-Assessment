"""
member5_evaluation/tta.py
────────────────────────────────────────────────────────────────────────────
Test Time Augmentation for change detection inference.
Applies N augmentations, averages the sigmoid outputs.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class TTAWrapper:
    """
    Wraps a Siamese model for TTA inference.
    
    Augmentations applied (and their inverses):
      0: Original (no transform)
      1: HorizontalFlip
      2: VerticalFlip  
      3: Rotate90
      4: Rotate180
      5: Rotate270
      6: HorizontalFlip + VerticalFlip
    """
    def __init__(self, model: nn.Module, n_augments: int = 4):
        # n_augments: 1=original only, 4=orig+hflip+vflip+rot90, 7=all
        self.model = model
        self.n_augments = min(n_augments, 7)
    
    @torch.no_grad()
    def predict(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Returns averaged sigmoid probability map (B, 1, H, W)."""
        probs_sum = 0.0
        
        for aug_idx in range(self.n_augments):
            pre_aug, post_aug = self._apply_aug(pre, post, aug_idx)
            
            # Predict
            logits = self.model(pre_aug, post_aug)
            # Handle Deep Supervision logic: only use main output
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
                
            probs = torch.sigmoid(logits)
            
            # Invert
            probs = self._invert_aug(probs, aug_idx)
            probs_sum += probs
            
        return probs_sum / self.n_augments
    
    def _apply_aug(self, pre: torch.Tensor, post: torch.Tensor, aug_idx: int):
        """Apply augmentation by index."""
        if aug_idx == 0:
            return pre, post
        elif aug_idx == 1:
            return F.hflip(pre), F.hflip(post)
        elif aug_idx == 2:
            return F.vflip(pre), F.vflip(post)
        elif aug_idx == 3:
            return torch.rot90(pre, 1, [-2, -1]), torch.rot90(post, 1, [-2, -1])
        elif aug_idx == 4:
            return torch.rot90(pre, 2, [-2, -1]), torch.rot90(post, 2, [-2, -1])
        elif aug_idx == 5:
            return torch.rot90(pre, 3, [-2, -1]), torch.rot90(post, 3, [-2, -1])
        elif aug_idx == 6:
            return F.vflip(F.hflip(pre)), F.vflip(F.hflip(post))
        else:
            return pre, post
    
    def _invert_aug(self, pred: torch.Tensor, aug_idx: int):
        """Invert augmentation on prediction map."""
        if aug_idx == 0:
            return pred
        elif aug_idx == 1:
            return F.hflip(pred)
        elif aug_idx == 2:
            return F.vflip(pred)
        elif aug_idx == 3:
            return torch.rot90(pred, -1, [-2, -1])
        elif aug_idx == 4:
            return torch.rot90(pred, -2, [-2, -1])
        elif aug_idx == 5:
            return torch.rot90(pred, -3, [-2, -1])
        elif aug_idx == 6:
            return F.hflip(F.vflip(pred))
        else:
            return pred
