"""
train_optimized.py
──────────────────────────────────────────────────────────────────────────────
Three-stage training for the Siamese Change Detection network.

Stage 1: SiameseUNet (V1) + CombinedSegLoss (Dice+BCE) — stable baseline
         Target: Val F1 > 0.65

Stage 2: SiameseUNetV2 warm-started from Stage 1 checkpoint
         Target: Val F1 > 0.75

Stage 3: SiameseUNetV2 + LovászLoss fine-tuning from Stage 2 checkpoint
         Target: Val F1 > 0.80

Scheduler: CosineAnnealingLR
    - Resumes correctly from checkpoints (scheduler state is saved/restored)
    - No warmup phase — starts at LR_MAX immediately
    - Smooth monotonic decay — no spikes or restarts
    - Stepped ONCE per epoch in main(), NOT inside train_epoch

Run:
    # Stage 1 (first run — do NOT change TRAINING_STAGE)
    python train_optimized.py

    # Stage 2 (after Stage 1 converges)
    # Edit TRAINING_STAGE = 2, then:
    python train_optimized.py

    # Stage 3 (after Stage 2 converges)
    # Edit TRAINING_STAGE = 3, then:
    python train_optimized.py
"""

import platform
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
import torch.optim.lr_scheduler as lr_sched
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2
from member2_siamese.loss import CombinedSegLoss
from member5_evaluation.metrics import compute_iou, compute_f1
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.config import cfg

log = get_logger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
PATCH_SIZE     = 256
BATCH_SIZE     = 8
EPOCHS         = 40        # Stage 3 gentle fine-tuning
LR_MAX         = 1e-4
WEIGHT_DECAY   = 2e-4
PATIENCE       = 20        # tight patience — stop fast if not improving
DICE_W         = 0.5
BCE_W          = 0.2
FOCAL_W        = 0.3
GRAD_CLIP      = 0.5
SAVE_EVERY     = 10
SEED           = 42
# THRESHOLD=0.10: PNG targets have ~2% positives (vs 0.16% in JSON).
# With consistent train/val distributions the model produces stronger sigmoid
# activations for positives, so a lower threshold catches them better.
THRESHOLD      = 0.10
WARMUP_EPOCHS  = 0        # Stage 3: no warmup — model is pre-trained, start at LR_MAX/4 directly
# POS_WEIGHT tuned to PNG positive ratio (~2%):
#   neg/pos = (100-2)/2 = 49 — use 15 as a stable, slightly conservative value.
# Old JSON-based value of 25 was correct for 0.16% ratio; now that we use PNG
# targets (~2%), a lower pos_weight prevents over-penalizing negatives.
POS_WEIGHT     = 15.0
TRAINING_STAGE = 3        # Stage 3: Lovász fine-tuning from Stage 2 best

# ── Checkpoint Paths (stage-specific) ────────────────────────────────────────
CKPT_BEST = Path(f"checkpoints/siamese_stage{TRAINING_STAGE}_best.pth")
CKPT_DIR  = Path("checkpoints")

# ── Data Paths ────────────────────────────────────────────────────────────────
TRAIN_DIR = Path("data/xbd/train")
VAL_DIR   = Path("data/xbd/test")

# xBD damage subtype → integer (kept for reference; no longer used in TrainDataset)
# TrainDataset now uses pre-computed PNG targets (same as ValDataset) for label consistency.
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
    """
    Training dataset — change masks from pre-computed target PNGs.

    IMPORTANT: We use PNG targets (data/xbd/train/targets/) instead of parsing
    JSON polygon labels for two critical reasons:

    1. LABEL CONSISTENCY: JSON parsing yields only ~0.16% positive pixels while
       PNG targets yield ~2.49%. ValDataset uses PNG targets (~1.96% positive).
       Training on JSON but evaluating on PNG causes a 15x distribution mismatch,
       which explains why the model plateaued at val_F1=0.51.

    2. LABEL QUALITY: PNG targets are pre-computed from the official xBD scoring
       script and include full building footprints + damage labels — they are the
       ground truth used by the xBD benchmark, not a noisy polygon re-parse.
    """

    def __init__(self, root: Path, transform=None):
        self.images_dir  = root / "images"
        self.targets_dir = root / "targets"
        self.transform   = transform
        self.samples     = []

        for pre in sorted(self.images_dir.glob("*_pre_disaster.png")):
            stem = pre.stem.replace("_pre_disaster", "")
            post = self.images_dir  / f"{stem}_post_disaster.png"
            tgt  = self.targets_dir / f"{stem}_post_disaster_target.png"
            if post.exists() and tgt.exists():
                self.samples.append((pre, post, tgt))

        log.info(f"TrainDataset: {len(self.samples)} pairs from {root}")
        if not self.samples:
            log.error("TrainDataset is EMPTY — check data/xbd/train/targets/ folder.")

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

# ImageNet normalization stats — required for pretrained ResNet50.
# The encoder was trained on ImageNet with these exact stats.
# Using A.ToFloat(max_value=255) without Normalize gives inputs in [0,1] but
# with wrong mean/std — the pretrained weights effectively can't do transfer learning.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(patch_size: int, is_train: bool):
    if is_train:
        ops = [
            A.RandomCrop(patch_size, patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(noise_scale_factor=0.1, p=0.3),
            # Normalize THEN ToTensorV2 — this is the correct order for pretrained encoders.
            # ToFloat is NOT used; Normalize handles uint8→float + mean/std in one step.
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        ops = [
            A.CenterCrop(patch_size, patch_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    return A.Compose(ops, additional_targets={"image0": "image"})


# ── Epoch loops ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    # NOTE: no scheduler parameter — scheduler is stepped in main() after val
    model.train()
    tot_loss = tot_iou = tot_f1 = 0.0
    n = len(loader)

    for pre, post, mask in loader:
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(pre, post)
            logits = output[0] if isinstance(output, list) else output
            loss   = criterion(logits, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # Apply THRESHOLD before metrics (metrics use hardcoded > 0.5 internally)
            preds     = (torch.sigmoid(logits) > THRESHOLD).float()
            tot_loss += loss.item()
            tot_iou  += compute_iou(preds, mask)
            tot_f1   += compute_f1(preds,  mask)

    return tot_loss / n, tot_iou / n, tot_f1 / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_iou = tot_f1 = 0.0
    n = len(loader)

    for pre, post, mask in loader:
        pre, post, mask = pre.to(device), post.to(device), mask.to(device)
        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(pre, post)
            logits = output[0] if isinstance(output, list) else output
            loss   = criterion(logits, mask)
        # Apply THRESHOLD before metrics (metrics use hardcoded > 0.5 internally)
        preds     = (torch.sigmoid(logits) > THRESHOLD).float()
        tot_loss += loss.item()
        tot_iou  += compute_iou(preds, mask)
        tot_f1   += compute_f1(preds,  mask)

    return tot_loss / n, tot_iou / n, tot_f1 / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram     = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" \
               if torch.cuda.is_available() else "N/A"
    log.info(f"Device : {device}  |  GPU: {gpu_name}  |  VRAM: {vram}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = TrainDataset(TRAIN_DIR, transform=get_transforms(PATCH_SIZE, True))
    val_ds   = ValDataset(VAL_DIR,     transform=get_transforms(PATCH_SIZE, False))

    if len(val_ds) == 0:
        log.error("ValDataset is empty! Exiting to prevent silent fallback to train data.")
        import sys
        sys.exit(1)

    # Windows-safe num_workers — multiprocessing not reliable on Windows
    safe_workers = 0 if platform.system() == "Windows" else 2

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=safe_workers, pin_memory=True, drop_last=True,
        persistent_workers=(safe_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=safe_workers, pin_memory=True,
        persistent_workers=(safe_workers > 0)
    )

    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Steps/epoch: {len(train_loader)}")

    # ── Model instantiation by stage ──────────────────────────────────────────
    if TRAINING_STAGE == 1:
        model = SiameseUNet(in_channels=3, pretrained=True).to(device)
        # pos_weight penalises missing a damage pixel POS_WEIGHT× harder than a FP
        # Critical fix for xBD class imbalance (most pixels are no-damage)
        criterion = CombinedSegLoss(
            dice_weight=DICE_W, bce_weight=BCE_W,
            focal_weight=FOCAL_W, pos_weight=POS_WEIGHT,
        ).to(device)
        current_lr = LR_MAX

    elif TRAINING_STAGE == 2:
        model = SiameseUNetV2(in_channels=3, pretrained=True).to(device)
        criterion = CombinedSegLoss(
            dice_weight=DICE_W, bce_weight=BCE_W,
            focal_weight=FOCAL_W, pos_weight=POS_WEIGHT,
        ).to(device)
        current_lr = LR_MAX / 2   # 5e-5
        # Load Stage 1 encoder weights — skip mismatched layers
        stage1_ckpt = Path("checkpoints/siamese_stage1_best.pth")
        if stage1_ckpt.exists():
            state = torch.load(stage1_ckpt, map_location=device, weights_only=False)
            model_state = model.state_dict()
            filtered = {k: v for k, v in state["model"].items()
                        if k in model_state and model_state[k].shape == v.shape}
            model_state.update(filtered)
            model.load_state_dict(model_state)
            log.info(f"Stage 2: loaded {len(filtered)} matching layers from Stage 1")
        else:
            log.warning("Stage 1 checkpoint not found. Training Stage 2 from scratch.")

    elif TRAINING_STAGE == 3:
        model = SiameseUNetV2(in_channels=3, pretrained=True).to(device)
        # Very gentle Lovász nudge: 10% Lovász + 90% CombinedSegLoss.
        # 50/50 split was too aggressive — Lovász gradients overwhelmed learned features.
        # At 10%, Lovász slightly nudges IoU optimisation without disrupting weights.
        from member2_siamese.losses_advanced import LovaszLoss
        _lovasz   = LovaszLoss().to(device)
        _combined = CombinedSegLoss(
            dice_weight=DICE_W, bce_weight=BCE_W,
            focal_weight=FOCAL_W, pos_weight=POS_WEIGHT,
        ).to(device)
        class _HybridLoss(nn.Module):
            def forward(self, logits, targets):
                return 0.1 * _lovasz(logits, targets) + 0.9 * _combined(logits, targets)
        criterion = _HybridLoss().to(device)
        current_lr = LR_MAX / 20   # 5e-6 — very small LR for safe fine-tuning
        stage2_ckpt = Path("checkpoints/siamese_stage2_best.pth")
        if stage2_ckpt.exists():
            state = torch.load(stage2_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            log.info("Stage 3: loaded full Stage 2 model")
        else:
            log.warning("Stage 2 checkpoint not found.")

    else:
        raise ValueError(f"TRAINING_STAGE must be 1, 2, or 3. Got: {TRAINING_STAGE}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=current_lr,
        weight_decay=WEIGHT_DECAY
    )

    # ── Scheduler — Linear warmup → CosineAnnealingLR ────────────────────────
    # Linear warmup: LR rises from LR_MAX*0.1 to LR_MAX over WARMUP_EPOCHS
    # Then CosineAnnealingLR decays smoothly for the remaining epochs.
    # Both resume correctly from checkpoints (SequentialLR saves sub-scheduler states).
    # Stepped ONCE per epoch in main(), NOT inside train_epoch.
    _warmup = lr_sched.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    _cosine = lr_sched.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6,
    )
    scheduler = lr_sched.SequentialLR(
        optimizer,
        schedulers=[_warmup, _cosine],
        milestones=[WARMUP_EPOCHS],
    )

    # Handle CUDA/CPU GradScaler correctly
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ── Auto-resume logic ─────────────────────────────────────────────────────
    # Restores model, optimizer, AND scheduler state so LR continues correctly.
    best_f1     = 0.0
    patience_ct = 0
    start_epoch = 1

    if CKPT_BEST.exists():
        try:
            ckpt = torch.load(CKPT_BEST, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            # CRITICAL: restore scheduler state so LR continues from where it left off
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
                log.info("Scheduler state restored — LR will continue from checkpoint.")
            else:
                log.warning("No scheduler state in checkpoint — LR restarting from epoch 1.")
            best_f1     = ckpt.get("val_f1", 0.0)
            start_epoch = ckpt.get("epoch", 0) + 1
            log.info(f"Resumed from {CKPT_BEST} "
                     f"(epoch {start_epoch-1}, best_f1={best_f1:.4f})")
        except Exception as e:
            log.warning(f"Could not resume: {e}. Starting from scratch.")
            best_f1     = 0.0
            start_epoch = 1
    else:
        log.info("No checkpoint found. Starting from scratch.")

    log.info("=" * 70)
    log.info(
        f"STAGE {TRAINING_STAGE} | "
        f"BS={BATCH_SIZE} | "
        f"LR_MAX={current_lr:.1e} | "
        f"EPOCHS={EPOCHS} | "
        f"THRESHOLD={THRESHOLD} | "
        f"SCHEDULER=Warmup{WARMUP_EPOCHS}+CosineAnnealing | POS_WEIGHT={POS_WEIGHT}"
    )
    log.info("=" * 70)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_iou, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        vl_loss, vl_iou, vl_f1 = val_epoch(
            model, val_loader, criterion, device
        )

        # Step scheduler ONCE per epoch — after validation
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        log.info(
            f"[{epoch:03d}/{EPOCHS}] "
            f"TR  f1={tr_f1:.4f} iou={tr_iou:.4f} loss={tr_loss:.4f} | "
            f"VAL f1={vl_f1:.4f} iou={vl_iou:.4f} loss={vl_loss:.4f} | "
            f"lr={lr_now:.2e} gap={tr_f1 - vl_f1:+.3f} {elapsed:.0f}s"
        )

        # Overfitting warning
        if epoch > 20 and (tr_f1 - vl_f1) > 0.20:
            log.warning(
                f"  ⚠ Overfitting: train_f1 - val_f1 = {tr_f1 - vl_f1:.3f}"
            )

        # Save best checkpoint — INCLUDE scheduler state
        if vl_f1 > best_f1:
            best_f1     = vl_f1
            patience_ct = 0
            save_checkpoint(
                model, optimizer, epoch,
                val_f1=best_f1,
                path=CKPT_BEST,
                extra={"scheduler": scheduler.state_dict()}
            )
            log.info(
                f"  ★ Best val_f1={best_f1:.4f} "
                f"(Stage {TRAINING_STAGE}) → saved {CKPT_BEST}"
            )
        else:
            patience_ct += 1

        # Periodic backup — also include scheduler state
        if epoch % SAVE_EVERY == 0:
            bkp = CKPT_DIR / f"siamese_stage{TRAINING_STAGE}_epoch{epoch:03d}.pth"
            save_checkpoint(
                model, optimizer, epoch,
                val_f1=vl_f1,
                path=bkp,
                extra={"scheduler": scheduler.state_dict()}
            )
            log.info(f"  Backup checkpoint → {bkp}")

        # Early stopping
        if patience_ct >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch}. Best val_f1={best_f1:.4f}")
            break

    # Training complete
    log.info("=" * 70)
    log.info(f"Stage {TRAINING_STAGE} complete  |  Best val_f1 = {best_f1:.4f}")
    log.info(f"Model saved at : {CKPT_BEST}")
    log.info("=" * 70)

    if best_f1 < 0.65 and TRAINING_STAGE == 1:
        log.warning(
            f"Stage 1 ended with val_f1={best_f1:.4f} < 0.65. "
            f"Do NOT proceed to Stage 2 yet. "
            f"Increase EPOCHS or check data quality."
        )
    elif TRAINING_STAGE == 1:
        log.info(
            f"Stage 1 complete. "
            f"Change TRAINING_STAGE = 2 and run again for Stage 2."
        )


if __name__ == "__main__":
    main()
