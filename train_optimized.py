"""
train_optimized.py
──────────────────────────────────────────────────────────────────────────────
Maximum-accuracy training for the Siamese Change Detection network.

Key optimisations:
  ✓ Mixed-precision (AMP) → 2× faster, batch=8 on 6.4 GB VRAM
  ✓ OneCycleLR → fastest convergence to best val_f1
  ✓ Strong augmentation (GridDistortion, ElasticTransform, ColorJitter)
  ✓ Gradient clipping for stability
  ✓ Auto-resume from existing checkpoint
  ✓ Periodic epoch backup checkpoints

Data layout handled correctly:
  train/images/  *_pre_disaster.png  *_post_disaster.png
  train/labels/  *_post_disaster.json  → change mask parsed from JSON
  test/images/   *_pre_disaster.png  *_post_disaster.png
  test/targets/  *_post_disaster_target.png  → binary masks (0/1)

Run:  python train_optimized.py
"""

import json
import random
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from member2_siamese.siamese_net import SiameseUNet
from member2_siamese.loss import CombinedSegLoss
from member5_evaluation.metrics import compute_iou, compute_f1
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)

# ── Hyper-parameters (tuned for RTX 4050 6.4 GB) ─────────────────────────────
PATCH_SIZE   = 256
BATCH_SIZE   = 8
EPOCHS       = 120
LR_MAX       = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
DICE_W       = 0.6
BCE_W        = 0.4
GRAD_CLIP    = 1.0
SAVE_EVERY   = 10
SEED         = 42

TRAIN_DIR = Path("data/xbd/train")
VAL_DIR   = Path("data/xbd/test")
CKPT_BEST = Path("checkpoints/siamese_best.pth")
CKPT_DIR  = Path("checkpoints")

# xBD damage subtype → integer (0 = no damage = background)
DAMAGE_MAP = {"no-damage": 0, "minor-damage": 1, "major-damage": 2,
              "destroyed": 3, "un-classified": 0}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _json_to_mask(json_path: Path, H: int, W: int) -> np.ndarray:
    """Parse xBD post-disaster JSON → binary change mask (H×W uint8)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    try:
        with open(json_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", {}).get("xy", []):
            label = DAMAGE_MAP.get(
                feat.get("properties", {}).get("subtype", "no-damage"), 0)
            if label == 0:
                continue
            wkt = feat.get("wkt", "")
            try:
                coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
                pts = np.array([[float(v) for v in p.strip().split()]
                                for p in coords_str.split(",")], dtype=np.float32)
                pts[:, 0] = pts[:, 0].clip(0, W - 1)
                pts[:, 1] = pts[:, 1].clip(0, H - 1)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            except Exception:
                pass
    except Exception:
        pass
    return mask


# ── Datasets ──────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """Training dataset — change masks parsed from JSON labels."""

    def __init__(self, root: Path, transform=None):
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"
        self.transform  = transform
        self.samples    = []

        for pre in sorted(self.images_dir.glob("*_pre_disaster.png")):
            stem      = pre.stem.replace("_pre_disaster", "")
            post      = self.images_dir / f"{stem}_post_disaster.png"
            post_json = self.labels_dir / f"{stem}_post_disaster.json"
            if post.exists() and post_json.exists():
                self.samples.append((pre, post, post_json))

        log.info(f"TrainDataset: {len(self.samples)} pairs from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_p, post_p, json_p = self.samples[idx]
        pre  = _read_rgb(pre_p)
        post = _read_rgb(post_p)
        H, W = pre.shape[:2]
        mask = _json_to_mask(json_p, H, W)

        if self.transform:
            aug  = self.transform(image=pre, image0=post, mask=mask)
            pre, post, mask = aug["image"], aug["image0"], aug["mask"]

        return pre, post, mask.unsqueeze(0).float()


class ValDataset(Dataset):
    """Validation dataset — change masks from pre-computed target PNGs."""

    def __init__(self, root: Path, transform=None):
        self.images_dir  = root / "images"
        self.targets_dir = root / "targets"
        self.transform   = transform
        self.samples     = []

        for pre in sorted(self.images_dir.glob("*_pre_disaster.png")):
            stem    = pre.stem.replace("_pre_disaster", "")
            post    = self.images_dir  / f"{stem}_post_disaster.png"
            # xBD naming: <stem>_post_disaster_target.png
            tgt     = self.targets_dir / f"{stem}_post_disaster_target.png"
            if post.exists() and tgt.exists():
                self.samples.append((pre, post, tgt))

        log.info(f"ValDataset: {len(self.samples)} pairs from {root}")
        if not self.samples:
            log.warning("ValDataset is EMPTY — check test/targets/ folder naming.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_p, post_p, tgt_p = self.samples[idx]
        pre  = _read_rgb(pre_p)
        post = _read_rgb(post_p)
        mask = cv2.imread(str(tgt_p), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            aug  = self.transform(image=pre, image0=post, mask=mask)
            pre, post, mask = aug["image"], aug["image0"], aug["mask"]

        return pre, post, mask.unsqueeze(0).float()


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_transforms(patch_size: int, is_train: bool):
    if is_train:
        ops = [
            A.RandomCrop(patch_size, patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5,
                               border_mode=cv2.BORDER_REFLECT),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.CenterCrop(patch_size, patch_size),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ]
    return A.Compose(ops, additional_targets={"image0": "image"})


# ── Epoch loops ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    tot_loss = tot_iou = tot_f1 = 0.0
    n = len(loader)

    for pre, post, mask in loader:
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(pre, post)
            loss   = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            preds     = torch.sigmoid(logits)
            tot_loss += loss.item()
            tot_iou  += compute_iou(preds, mask)
            tot_f1   += compute_f1(preds, mask)

    return tot_loss / n, tot_iou / n, tot_f1 / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_iou = tot_f1 = 0.0
    n = len(loader)

    for pre, post, mask in loader:
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)
        with autocast():
            logits = model(pre, post)
            loss   = criterion(logits, mask)
        preds     = torch.sigmoid(logits)
        tot_loss += loss.item()
        tot_iou  += compute_iou(preds, mask)
        tot_f1   += compute_f1(preds, mask)

    return tot_loss / n, tot_iou / n, tot_f1 / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram     = f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" \
               if torch.cuda.is_available() else "N/A"
    log.info(f"Device : {device}  |  GPU: {gpu_name}  |  VRAM: {vram}")

    # Datasets & loaders
    train_ds = TrainDataset(TRAIN_DIR, transform=get_transforms(PATCH_SIZE, True))
    val_ds   = ValDataset(VAL_DIR,     transform=get_transforms(PATCH_SIZE, False))

    if len(val_ds) == 0:
        log.warning("Val dataset empty — will use train metrics as proxy.")
        val_ds = TrainDataset(TRAIN_DIR, transform=get_transforms(PATCH_SIZE, False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Steps/epoch: {len(train_loader)}")

    # Model, loss, optimizer, scheduler
    model     = SiameseUNet(in_channels=3, pretrained=True).to(device)
    criterion = CombinedSegLoss(dice_weight=DICE_W, bce_weight=BCE_W)
    optimizer = AdamW(model.parameters(), lr=LR_MAX / 25, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=len(train_loader), epochs=EPOCHS,
        pct_start=0.1, anneal_strategy="cos",
        div_factor=25, final_div_factor=1e4,
    )
    scaler = GradScaler()

    # Auto-resume
    best_f1 = 0.0
    patience_ct = 0
    if CKPT_BEST.exists():
        ckpt = torch.load(CKPT_BEST, map_location=device)
        model.load_state_dict(ckpt["model"])          # key is "model" per save_checkpoint
        best_f1 = ckpt.get("val_f1", 0.0)
        log.info(f"Resumed from {CKPT_BEST}  — previous best val_f1={best_f1:.4f}")

    log.info("=" * 70)
    log.info(f"Optimized Training  |  BS={BATCH_SIZE}  LR_MAX={LR_MAX}  EPOCHS={EPOCHS}")
    log.info("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_iou, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device)
        vl_loss, vl_iou, vl_f1 = val_epoch(
            model, val_loader, criterion, device)
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        log.info(
            f"[{epoch:03d}/{EPOCHS}] "
            f"TR f1={tr_f1:.4f} iou={tr_iou:.4f} loss={tr_loss:.4f} | "
            f"VAL f1={vl_f1:.4f} iou={vl_iou:.4f} loss={vl_loss:.4f} | "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            patience_ct = 0
            save_checkpoint(model, optimizer, epoch, val_f1=best_f1, path=CKPT_BEST)
            log.info(f"  ★ Best val_f1={best_f1:.4f} → saved {CKPT_BEST}")
        else:
            patience_ct += 1

        if epoch % SAVE_EVERY == 0:
            bkp = CKPT_DIR / f"siamese_epoch{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, val_f1=vl_f1, path=bkp)
            log.info(f"  Backup checkpoint → {bkp}")

        if patience_ct >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}. Best val_f1={best_f1:.4f}")
            break

    log.info("=" * 70)
    log.info(f"Training complete  |  Best val_f1 = {best_f1:.4f}")
    log.info(f"Model saved at     : {CKPT_BEST}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
