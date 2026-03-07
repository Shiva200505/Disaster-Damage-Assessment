"""
member3_classifier/efficientnet_classifier.py
────────────────────────────────────────────────────────────────────────────
Member 3 – EfficientNet-B3 Damage Severity Classifier

For every building polygon detected by the Siamese network, this module:
  1. Extracts a fixed-size crop of the post-disaster image using the polygon
     bounding box from xBD GeoJSON annotations.
  2. Runs the crop through an EfficientNet-B3 fine-tuned on xBD.
  3. Returns a 4-class prediction: no_damage / minor / major / destroyed.

Uses timm (https://github.com/huggingface/pytorch-image-models) for the
EfficientNet-B3 backbone — freely available and pip-installable.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import timm


class DamageClassifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for 4-class building damage severity.

    Parameters
    ----------
    num_classes : number of output classes (default 4)
    pretrained  : load ImageNet-pretrained weights (default True)
    dropout     : dropout rate before the classification head
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,          # remove default head → bare feature extractor
            global_pool="avg",
        )
        in_features = self.backbone.num_features   # 1536 for B3

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) float tensor, values in [0, 1]

        Returns
        -------
        logits : (B, num_classes) — apply softmax for class probabilities
        """
        features = self.backbone(x)  # (B, 1536)
        return self.head(features)   # (B, 4)


# ── Building Crop Extraction ──────────────────────────────────────────────────

def extract_building_crops(
    post_image: np.ndarray,
    polygons: List[List[Tuple[float, float]]],
    crop_size: int = 64,
    padding: int = 8,
) -> List[np.ndarray]:
    """
    Extract fixed-size crops centred on each building polygon bounding box.

    Parameters
    ----------
    post_image : H×W×3 uint8 or float32 post-disaster image
    polygons   : list of polygon coordinate lists [[x,y], ...] in pixel space
    crop_size  : output crop side length (pixels)
    padding    : extra pixels added around the bounding box

    Returns
    -------
    List of (crop_size × crop_size × 3) float32 crops, normalised to [0, 1].
    """
    import cv2

    H, W = post_image.shape[:2]
    crops = []

    for poly in polygons:
        pts = np.array(poly, dtype=np.float32)
        x_min = max(int(pts[:, 0].min()) - padding, 0)
        y_min = max(int(pts[:, 1].min()) - padding, 0)
        x_max = min(int(pts[:, 0].max()) + padding, W)
        y_max = min(int(pts[:, 1].max()) + padding, H)

        crop = post_image[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
        else:
            crop = cv2.resize(crop, (crop_size, crop_size))

        if crop.dtype == np.uint8:
            crop = crop.astype(np.float32) / 255.0

        crops.append(crop)

    return crops


def crops_to_tensor(crops: List[np.ndarray], device: str = "cpu") -> torch.Tensor:
    """
    Stack a list of H×W×C numpy crops into a (N, C, H, W) float tensor.
    """
    arr = np.stack(crops, axis=0)                   # (N, H, W, C)
    t   = torch.from_numpy(arr).permute(0, 3, 1, 2) # (N, C, H, W)
    return t.float().to(device)


@torch.no_grad()
def classify_buildings(
    model: DamageClassifier,
    crops: List[np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> List[int]:
    """
    Run the damage classifier over building crops in mini-batches.

    Parameters
    ----------
    model      : trained DamageClassifier (eval mode)
    crops      : list of (crop_size × crop_size × 3) float32 crops
    batch_size : inference batch size
    device     : torch device string

    Returns
    -------
    List of predicted integer class labels (0–3 per building).
    """
    model.eval()
    labels = []

    for start in range(0, len(crops), batch_size):
        batch = crops_to_tensor(crops[start : start + batch_size], device=device)
        logits = model(batch)
        preds  = logits.argmax(dim=1).cpu().tolist()
        labels.extend(preds)

    return labels
