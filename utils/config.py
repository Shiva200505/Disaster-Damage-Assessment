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
        val_path = self.xbd_root / "val"
        self.val_dir   = val_path if val_path.exists() else self.xbd_root / "test"
        self.test_dir  = self.xbd_root / "test"


# ── Preprocessing Config ──────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    patch_size: int   = 256    # Crop size fed to the network
    stride:     int   = 128    # Sliding-window stride during patch extraction
    image_size: int   = 1024   # Original xBD tile size
    mean: tuple       = (0.485, 0.456, 0.406)   # ImageNet mean
    std:  tuple       = (0.229, 0.224, 0.225)   # ImageNet std
    add_spectral: bool = False  # False = 3-ch RGB (faster, less memory); keep False for Stage 1


# ── Siamese Network Config (Member 2) ─────────────────────────────────────────

@dataclass
class SiameseConfig:
    model_version:  str   = "v1"     # "v1" | "v2" | "v3" — use v1 for Stage 1
    encoder_name:   str   = "resnet50"
    encoder_weights: str  = "imagenet"
    in_channels:    int   = 3      # 3 for Stage 1 (RGB only); 5 if using NDVI+NDWI
    num_classes:    int   = 1      # Binary change mask
    decoder_channels: tuple = (256, 128, 64, 32, 16)
    map_threshold:  float = 0.35   # Prediction threshold for F1/IoU computation

    # Training
    batch_size:     int   = 4      # Reduced from 8 for stability
    learning_rate:  float = 8e-5   # Reduced from 1e-4 — lower LR is the most critical fix
    weight_decay:   float = 2e-4   # Increased from 1e-4 — more regularization
    epochs:         int   = 100    # Total epochs per stage
    patience:       int   = 20     # Early stopping patience (increased for noisy training)
    dice_weight:    float = 0.7    # Increased from 0.6
    bce_weight:     float = 0.3    # Decreased from 0.4

    # Scheduler
    scheduler: str = "cosine"      # "cosine" | "step"

    # Inference improvements
    use_tta: bool = True
    tta_n_augments: int = 4
    sliding_window_stride: int = 128


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
    epochs:         int   = 1
    patience:       int   = 10

    # Focal Loss
    focal_alpha:    float = 0.25
    focal_gamma:    float = 2.0

    # Class labels (index → name)
    class_names: tuple = ("no_damage", "minor_damage", "major_damage", "destroyed")
    # Color hex for visualization
    class_colors: tuple = ("#2ECC71", "#F1C40F", "#E67E22", "#E74C3C")


# ── Data Pipeline Config ──────────────────────────────────────────────────────

@dataclass
class DataConfig:
    crop_min_area: int = 50      # min contour area for polygon extraction
    overlap_threshold: float = 0.3 # IoU threshold for dedup


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
    data:       DataConfig      = field(default_factory=DataConfig)
    wandb:      WandbConfig     = field(default_factory=WandbConfig)

    seed: int = 42
    num_workers: int = 0   # 0 = main process only (required on Windows)
    device: str = "cuda"   # "cuda" | "cpu"

    # Stage control — 1, 2, or 3 — mirrors TRAINING_STAGE in train_optimized.py
    training_stage: int = 1

    # Prediction threshold — lower than 0.5 catches predictions before the model is confident
    prediction_threshold: float = 0.35

    def make_dirs(self):
        """Create all output directories if they don't exist."""
        for attr in ("checkpoint_dir", "output_dir", "geojson_dir", "maps_dir"):
            getattr(self.paths, attr).mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
cfg = Config()
