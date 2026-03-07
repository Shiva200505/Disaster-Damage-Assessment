"""
tests/test_sanity.py
────────────────────────────────────────────────────────────────────────────
Sanity checks that verify every module can be imported and forward-passes
run without errors using purely synthetic (random) data — no xBD dataset
required.

Run with:
    python -m pytest tests/test_sanity.py -v
"""

import numpy as np
import pytest
import torch


# ───────────────────────── Member 1 ──────────────────────────────────────────

def test_spectral_indices_rgb():
    """add_spectral_channels should return H×W×5 from H×W×3 input."""
    from member1_preprocessing.spectral_indices import add_spectral_channels
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = add_spectral_channels(img)
    assert out.shape == (64, 64, 5), f"Expected (64,64,5), got {out.shape}"
    assert out.dtype == np.float32


def test_normalize_uint8():
    from member1_preprocessing.preprocessing import normalize
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = normalize(img)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_extract_patches():
    from member1_preprocessing.preprocessing import extract_patches
    tile = np.zeros((512, 512, 3), dtype=np.uint8)
    patches = extract_patches(tile, patch_size=256, stride=256)
    assert len(patches) == 4   # 2×2 grid
    assert patches[0].shape == (256, 256, 3)


def test_align_images_same_image():
    """Aligning an image to itself should return something of the same shape."""
    from member1_preprocessing.preprocessing import align_images
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    aligned = align_images(img, img.copy())
    assert aligned.shape == img.shape


def test_augmentation_pipelines():
    from member1_preprocessing.augmentation import get_train_transforms, get_val_transforms
    tf = get_train_transforms(image_size=64)
    pre  = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    post = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2,   (128, 128),    dtype=np.uint8)
    out  = tf(image=pre, image0=post, mask=mask)
    assert out["image"].shape[:2]  == (64, 64)
    assert out["image0"].shape[:2] == (64, 64)
    assert out["mask"].shape       == (64, 64)


# ───────────────────────── Member 2 ──────────────────────────────────────────

def test_siamese_forward_pass():
    """SiameseUNet should output (1, 1, H, W) logits."""
    from member2_siamese.siamese_net import SiameseUNet
    model = SiameseUNet(in_channels=3, pretrained=False)
    model.eval()
    pre  = torch.randn(1, 3, 256, 256)
    post = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(pre, post)
    assert out.shape == (1, 1, 256, 256), f"Unexpected shape: {out.shape}"


def test_dice_loss():
    from member2_siamese.loss import DiceLoss
    criterion = DiceLoss()
    logits  = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0          # scalar
    assert 0.0 <= loss.item() <= 2.0


def test_combined_seg_loss():
    from member2_siamese.loss import CombinedSegLoss
    criterion = CombinedSegLoss()
    logits  = torch.randn(2, 1, 64, 64)
    targets = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0


# ───────────────────────── Member 3 ──────────────────────────────────────────

def test_classifier_forward():
    """DamageClassifier should output (B, 4) logits."""
    from member3_classifier.efficientnet_classifier import DamageClassifier
    model = DamageClassifier(num_classes=4, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 4), f"Unexpected shape: {out.shape}"


def test_focal_loss():
    from member3_classifier.focal_loss import FocalLoss
    criterion = FocalLoss(gamma=2.0)
    logits  = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_extract_building_crops():
    from member3_classifier.efficientnet_classifier import extract_building_crops
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    polygons = [
        [(100, 100), (200, 100), (200, 200), (100, 200)],
        [(300, 300), (400, 300), (400, 400), (300, 400)],
    ]
    crops = extract_building_crops(img, polygons, crop_size=64)
    assert len(crops) == 2
    assert crops[0].shape == (64, 64, 3)


# ───────────────────────── Member 4 ──────────────────────────────────────────

def test_predictions_to_geojson():
    from member4_visualization.geojson_utils import predictions_to_geojson
    polygons = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        [(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0)],
        [(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)],
    ]
    labels = [0, 2, 3]
    gj = predictions_to_geojson(polygons, labels)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 3
    assert gj["features"][1]["properties"]["damage_label"] == 2
    assert gj["features"][2]["properties"]["damage_class"] == "destroyed"


# ───────────────────────── Member 5 ──────────────────────────────────────────

def test_compute_iou_perfect():
    """Perfect prediction → IoU = 1.0."""
    from member5_evaluation.metrics import compute_iou
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    iou  = compute_iou(mask, mask)
    assert abs(iou - 1.0) < 1e-4


def test_compute_f1_perfect():
    from member5_evaluation.metrics import compute_f1
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    f1   = compute_f1(mask, mask)
    assert abs(f1 - 1.0) < 1e-4


def test_compute_iou_zeros():
    """All-zero pred vs all-one target should give near-zero IoU."""
    from member5_evaluation.metrics import compute_iou
    pred   = torch.zeros(1, 1, 32, 32)
    target = torch.ones(1,  1, 32, 32)
    iou = compute_iou(pred, target)
    assert iou < 0.01


def test_per_class_f1():
    from member5_evaluation.metrics import per_class_f1
    preds   = [0, 1, 2, 3, 0, 1]
    targets = [0, 1, 2, 3, 0, 1]
    result  = per_class_f1(preds, targets, num_classes=4)
    assert abs(result["macro_f1"] - 1.0) < 1e-4


def test_xbd_score():
    from member5_evaluation.metrics import compute_xbd_score
    score = compute_xbd_score(localization_f1=0.8, classification_f1=0.6)
    expected = 2 * 0.8 * 0.6 / (0.8 + 0.6)
    assert abs(score - expected) < 1e-6


def test_xbd_score_zeros():
    from member5_evaluation.metrics import compute_xbd_score
    assert compute_xbd_score(0.0, 0.0) == 0.0
