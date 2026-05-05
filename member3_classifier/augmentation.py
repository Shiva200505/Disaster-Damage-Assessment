"""
augmentation.py — Albumentations pipelines for EfficientNet-B3 classifier crops.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_crop_train_transforms(crop_size=64):
    """
    Strong augmentation pipeline for training the building damage classifier.
    Returns an Albumentations Compose object.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(4, 8), hole_width_range=(4, 8), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_crop_val_transforms(crop_size=64):
    """
    Validation pipeline for building crops. Only non-destructive transforms.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
