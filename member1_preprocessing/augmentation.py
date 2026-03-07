"""
member1_preprocessing/augmentation.py
────────────────────────────────────────────────────────────────────────────
Member 1 – Albumentations Augmentation Pipelines

Provides consistent train/val transform factories that apply identical
spatial augmentations to the pre-image, post-image, and change mask.

The Albumentations `additional_targets` mechanism ensures that "image0"
(the post-disaster image) and "mask" receive identical spatial transforms.

Usage
-----
    from member1_preprocessing.augmentation import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms(image_size=256)
    result   = train_tf(image=pre_img, image0=post_img, mask=change_mask)
    pre_aug, post_aug, mask_aug = result["image"], result["image0"], result["mask"]
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline.

    Spatial transforms (applied identically to pre, post, mask):
      • Random horizontal + vertical flip
      • Random 90° rotation
      • Random crop to patch_size

    Pixel-level transforms (applied to images only, NOT mask):
      • Random brightness / contrast
      • Gaussian blur (mild)
      • Random Gaussian noise

    Parameters
    ----------
    image_size : target H×W size after cropping

    Returns
    -------
    Albumentations Compose pipeline configured for dual-image + mask input.
    """
    return A.Compose(
        [
            # ── Spatial (applied to pre, post, AND mask) ──
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),

            # ── Pixel-level (applied to images only) ──────
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        ],
        additional_targets={"image0": "image"},   # post-disaster image treated as "image"
        is_check_shapes=False,
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """
    Validation / test augmentation pipeline.

    Only a CenterCrop is applied — no random transforms.

    Parameters
    ----------
    image_size : target H×W size for cropping

    Returns
    -------
    Albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
        ],
        additional_targets={"image0": "image"},
        is_check_shapes=False,
    )


def get_inference_transforms(image_size: int = 256) -> A.Compose:
    """
    Inference pipeline — just resize to the expected input shape.
    Used when running the full tile through the model at test time.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, p=1.0),
        ],
        additional_targets={"image0": "image"},
        is_check_shapes=False,
    )
