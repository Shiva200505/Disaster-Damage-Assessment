"""
utils/config.py
────────────────────────────────────────────────────────────────────────────
Central configuration for the Disaster Damage Assessment project.
All hyperparameters, paths, and constants are defined here so every module
imports from a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Dataset Paths ─────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    # Root of the xBD dataset directory (update after downloading from xview2.org)
    xbd_root: Path = Path("data/xbd")

    # Subdirectories inside xbd_root
    train_dir: Path = field(init=False)
    val_dir:   Path = field(init=False)
    test_dir:  Path = field(init=False)

    # Output directories
    checkpoint_dir: Path = Path("checkpoints")
    output_dir:     Path = Path("outputs")
    geojson_dir:    Path = Path("outputs/geojson")
    maps_dir:       Path = Path("outputs/maps")

    def __post_init__(self):
        self.train_dir = self.xbd_root / "train"
        self.val_dir   = self.xbd_root / "test"   # use test split as val (no separate val folder)
        self.test_dir  = self.xbd_root / "test"


# ── Preprocessing Config ──────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    patch_size: int   = 256    # Crop size fed to the network
    stride:     int   = 128    # Sliding-window stride during patch extraction
    image_size: int   = 1024   # Original xBD tile size
    mean: tuple       = (0.485, 0.456, 0.406)   # ImageNet mean
    std:  tuple       = (0.229, 0.224, 0.225)   # ImageNet std
    add_spectral: bool = False  # False = 3-ch RGB (faster, less memory)


# ── Siamese Network Config (Member 2) ─────────────────────────────────────────

@dataclass
class SiameseConfig:
    encoder_name:   str   = "resnet50"
    encoder_weights: str  = "imagenet"
    in_channels:    int   = 3      # Increase to 5 if using NDVI+NDWI channels
    num_classes:    int   = 1      # Binary change mask
    decoder_channels: tuple = (256, 128, 64, 32, 16)

    # Training
    batch_size:     int   = 4    # 4 fits most 4-8 GB VRAM laptops
    learning_rate:  float = 1e-4
    weight_decay:   float = 1e-4
    epochs:         int   = 100
    patience:       int   = 10     # Early stopping patience
    dice_weight:    float = 0.6    # Weight for Dice in combined loss
    bce_weight:     float = 0.4

    # Scheduler
    scheduler: str = "cosine"      # "cosine" | "step"


# ── Classifier Config (Member 3) ──────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    model_name:     str   = "efficientnet_b3"
    pretrained:     bool  = True
    num_classes:    int   = 4      # no_damage / minor / major / destroyed
    crop_size:      int   = 64     # Building crop size fed to classifier
    in_channels:    int   = 3

    # Training
    batch_size:     int   = 32
    learning_rate:  float = 1e-4
    weight_decay:   float = 1e-4
    epochs:         int   = 50
    patience:       int   = 10

    # Focal Loss
    focal_alpha:    float = 0.25
    focal_gamma:    float = 2.0

    # Class labels (index → name)
    class_names: tuple = ("no_damage", "minor_damage", "major_damage", "destroyed")
    # Color hex for visualization
    class_colors: tuple = ("#2ECC71", "#F1C40F", "#E67E22", "#E74C3C")


# ── Experiment Tracking ───────────────────────────────────────────────────────

@dataclass
class WandbConfig:
    project:  str  = "disaster-damage-assessment"
    entity:   str  = ""           # Your W&B username / team
    enabled:  bool = False        # Set True to enable W&B logging
    log_freq: int  = 10           # Log every N batches


# ── Master Config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    paths:      PathConfig      = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    siamese:    SiameseConfig   = field(default_factory=SiameseConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    wandb:      WandbConfig     = field(default_factory=WandbConfig)

    seed: int = 42
    num_workers: int = 0   # 0 = main process only (required on Windows)
    device: str = "cuda"   # "cuda" | "cpu"

    def make_dirs(self):
        """Create all output directories if they don't exist."""
        for attr in ("checkpoint_dir", "output_dir", "geojson_dir", "maps_dir"):
            getattr(self.paths, attr).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
cfg = Config()
