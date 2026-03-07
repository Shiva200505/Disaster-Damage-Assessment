"""
member1_preprocessing/dataset.py
────────────────────────────────────────────────────────────────────────────
Member 1 – xBD Dataset Loader

Loads paired pre-/post-disaster satellite image tiles from the xBD dataset
directory structure and returns:
    pre_img    : (C, H, W) float32 tensor  [pre-disaster image]
    post_img   : (C, H, W) float32 tensor  [post-disaster image]
    change_mask: (1, H, W) float32 tensor  [binary change mask; 1 = changed]
    damage_label: int                      [0=no_damage … 3=destroyed]

xBD Directory Structure (after download from xview2.org)
─────────────────────────────────────────────────────────
data/xbd/
├── train/
│   ├── images/
│   │   ├── <disaster>_<id>_pre_disaster.png
│   │   └── <disaster>_<id>_post_disaster.png
│   └── labels/
│       ├── <disaster>_<id>_pre_disaster.json   (GeoJSON polygons)
│       └── <disaster>_<id>_post_disaster.json
├── val/  (same structure)
└── test/ (same structure)

Damage label encoding (from xBD JSON "subtype" field):
    0 → no-damage
    1 → minor-damage
    2 → major-damage
    3 → destroyed
    4 → un-classified  (skipped / mapped to 0)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from member1_preprocessing.spectral_indices import add_spectral_channels
from member1_preprocessing.preprocessing import normalize, extract_patches, align_images
from utils.config import cfg
from utils.logger import get_logger

log = get_logger(__name__)

# xBD damage subtype → integer label
DAMAGE_MAP = {
    "no-damage":      0,
    "minor-damage":   1,
    "major-damage":   2,
    "destroyed":      3,
    "un-classified":  0,   # treat as no-damage
}


class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD building damage assessment benchmark.

    Parameters
    ----------
    root_dir   : path to train/ val/ or test/ split directory
    patch_size : size of square patches to extract from tiles (default 256)
    split      : "train" | "val" | "test"
    transform  : optional Albumentations transform applied to (pre, post, mask)
    add_spectral: if True, append NDVI + NDWI channels to images
    use_patches: if True extract multiple patches per tile; else return full tile
    """

    def __init__(
        self,
        root_dir: str | Path,
        patch_size: int = 256,
        split: str = "train",
        transform=None,
        add_spectral: bool = True,
        use_patches: bool = True,
    ):
        self.root_dir     = Path(root_dir)
        self.images_dir   = self.root_dir / "images"
        self.labels_dir   = self.root_dir / "labels"
        self.patch_size   = patch_size
        self.split        = split
        self.transform    = transform
        self.add_spectral = add_spectral
        self.use_patches  = use_patches

        self.samples: List[Tuple[Path, Path, Path, Path]] = []
        self._build_sample_list()

        log.info(
            f"[{split}] XBDDataset loaded — {len(self.samples)} tile pairs "
            f"from {self.root_dir}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_sample_list(self) -> None:
        """Collect all (pre_img, post_img, pre_json, post_json) tuples."""
        if not self.images_dir.exists():
            log.warning(
                f"Images directory not found: {self.images_dir}. "
                "Dataset will be empty. Download xBD from xview2.org first."
            )
            return

        for pre_path in sorted(self.images_dir.glob("*_pre_disaster.png")):
            stem       = pre_path.stem.replace("_pre_disaster", "")
            post_path  = self.images_dir / f"{stem}_post_disaster.png"
            pre_json   = self.labels_dir / f"{stem}_pre_disaster.json"
            post_json  = self.labels_dir / f"{stem}_post_disaster.json"

            if post_path.exists() and post_json.exists():
                self.samples.append((pre_path, post_path, pre_json, post_json))
            else:
                log.warning(f"Missing post image or label for: {stem}")

    def _load_image(self, path: Path) -> np.ndarray:
        """Load PNG as H×W×3 uint8 array (BGR→RGB)."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _parse_change_mask(self, post_json: Path, shape: Tuple[int, int]) -> np.ndarray:
        """
        Build a binary change mask from the post-disaster GeoJSON.
        Pixels inside a building polygon with damage > no-damage are set to 1.
        """
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)

        try:
            with open(post_json, "r") as f:
                geojson = json.load(f)
        except Exception as e:
            log.warning(f"Could not parse {post_json}: {e}")
            return mask

        features = geojson.get("features", {}).get("xy", [])
        for feat in features:
            props   = feat.get("properties", {})
            subtype = props.get("subtype", "no-damage")
            label   = DAMAGE_MAP.get(subtype, 0)
            if label == 0:
                continue  # no change

            coords = feat.get("wkt", "")
            poly   = self._wkt_to_polygon(coords, H, W)
            if poly is not None:
                cv2.fillPoly(mask, [poly], 1)

        return mask

    def _wkt_to_polygon(
        self, wkt: str, H: int, W: int
    ) -> Optional[np.ndarray]:
        """Parse a POLYGON WKT string into pixel coordinates."""
        try:
            coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
            pts = []
            for pair in coords_str.split(","):
                x, y = pair.strip().split()
                pts.append([float(x), float(y)])
            arr = np.array(pts, dtype=np.float32)
            # Clip to image bounds
            arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
            return arr.astype(np.int32)
        except Exception:
            return None

    def _parse_damage_label(self, post_json: Path) -> int:
        """Return the *worst* damage label present in this tile."""
        try:
            with open(post_json, "r") as f:
                geojson = json.load(f)
            worst = 0
            for feat in geojson.get("features", {}).get("xy", []):
                subtype = feat.get("properties", {}).get("subtype", "no-damage")
                worst = max(worst, DAMAGE_MAP.get(subtype, 0))
            return worst
        except Exception:
            return 0

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        pre_path, post_path, pre_json, post_json = self.samples[idx]

        # Load raw images
        pre_img  = self._load_image(pre_path)
        post_img = self._load_image(post_path)

        # Align post to pre using homography
        post_img = align_images(pre_img, post_img)

        H, W = pre_img.shape[:2]

        # Build masks & label
        change_mask  = self._parse_change_mask(post_json, (H, W))
        damage_label = self._parse_damage_label(post_json)

        # Optional spectral channels
        if self.add_spectral:
            pre_img  = add_spectral_channels(pre_img)
            post_img = add_spectral_channels(post_img)

        # Augmentation (applied identically to pre, post, mask)
        if self.transform is not None:
            augmented   = self.transform(
                image=pre_img, image0=post_img,
                mask=change_mask,
            )
            pre_img     = augmented["image"]
            post_img    = augmented["image0"]
            change_mask = augmented["mask"]

        # Normalize to [0,1]
        pre_img   = normalize(pre_img)
        post_img  = normalize(post_img)

        # → torch tensors
        pre_t    = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        post_t   = torch.from_numpy(post_img).permute(2, 0, 1).float()
        mask_t   = torch.from_numpy(change_mask).unsqueeze(0).float()

        return {
            "pre":          pre_t,
            "post":         post_t,
            "change_mask":  mask_t,
            "damage_label": torch.tensor(damage_label, dtype=torch.long),
            "stem":         pre_path.stem.replace("_pre_disaster", ""),
        }


# ── Convenience factory functions ─────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = 8,
    num_workers: int = 4,
    patch_size: int = 256,
    add_spectral: bool = True,
    train_transform=None,
    val_transform=None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Return (train_loader, val_loader) for the xBD dataset.
    Paths are read from utils.cfg.
    """
    from torch.utils.data import DataLoader

    train_ds = XBDDataset(
        root_dir=cfg.paths.train_dir,
        patch_size=patch_size,
        split="train",
        transform=train_transform,
        add_spectral=add_spectral,
    )
    val_ds = XBDDataset(
        root_dir=cfg.paths.val_dir,
        patch_size=patch_size,
        split="val",
        transform=val_transform,
        add_spectral=add_spectral,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
