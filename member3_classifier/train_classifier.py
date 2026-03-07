"""
member3_classifier/train_classifier.py
────────────────────────────────────────────────────────────────────────────
Member 3 – Damage Classifier Training Loop

Trains the EfficientNet-B3 model to classify individual building crops
into 4 damage severity levels using Focal Loss.

Training data: building crops extracted from xBD images using GeoJSON
polygon coordinates. Each crop is labelled with the polygon's "subtype"
damage category (no-damage, minor-damage, major-damage, destroyed).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix

from member3_classifier.efficientnet_classifier import DamageClassifier
from member3_classifier.focal_loss import FocalLoss
from utils.checkpoint import save_checkpoint
from utils.config import cfg
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb

log = get_logger(__name__)

CLASS_NAMES = list(cfg.classifier.class_names)


# ── Building Crop Dataset ─────────────────────────────────────────────────────

class BuildingCropDataset(Dataset):
    """
    Loads pre-extracted building crop images and their damage labels.

    Expects a directory structure built by the preprocessing pipeline:
        data/building_crops/
            ├── 0_no_damage/    *.png  ...
            ├── 1_minor/        *.png  ...
            ├── 2_major/        *.png  ...
            └── 3_destroyed/    *.png  ...

    Parameters
    ----------
    root_dir  : path to the building_crops/ directory
    transform : optional Albumentations transform
    crop_size : resize target (default 64)
    """

    import json

    def __init__(self, root_dir: str | Path, transform=None, crop_size: int = 64):
        import cv2
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.crop_size = crop_size
        self.samples: list = []   # (image_path, label)
        self._scan()

    def _scan(self):
        label_dirs = {
            "0_no_damage":  0,
            "1_minor":      1,
            "2_major":      2,
            "3_destroyed":  3,
        }
        for dir_name, label in label_dirs.items():
            d = self.root_dir / dir_name
            if not d.exists():
                continue
            for img_path in d.glob("*.png"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.crop_size, self.crop_size))

        if self.transform:
            img = self.transform(image=img)["image"]

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img, torch.tensor(label, dtype=torch.long)


# ── Training / Validation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": total_loss / len(loader), "macro_f1": macro_f1}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm        = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    return {
        "loss":       total_loss / len(loader),
        "macro_f1":   macro_f1,
        "per_class_f1": per_class.tolist(),
        "confusion_matrix": cm,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Training classifier on: {device}")
    cfg.make_dirs()

    if cfg.wandb.enabled:
        init_wandb(cfg.wandb.project, cfg.wandb.entity,
                   config={"model": "EfficientNetB3", **vars(args)},
                   run_name="classifier_run")

    crop_root = Path("data/building_crops")
    train_ds  = BuildingCropDataset(crop_root / "train", crop_size=cfg.classifier.crop_size)
    val_ds    = BuildingCropDataset(crop_root / "val",   crop_size=cfg.classifier.crop_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model     = DamageClassifier(num_classes=4, pretrained=True).to(device)
    criterion = FocalLoss(alpha=cfg.classifier.focal_alpha, gamma=cfg.classifier.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.classifier.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, patience_counter = 0.0, 0
    ckpt_path = cfg.paths.checkpoint_dir / "classifier_best.pth"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={tr['loss']:.4f}  "
            f"val_loss={va['loss']:.4f}  "
            f"val_macro_f1={va['macro_f1']:.4f}"
        )
        for i, name in enumerate(CLASS_NAMES):
            log.info(f"  {name}: F1={va['per_class_f1'][i]:.4f}")

        log_metrics({
            "train/loss": tr["loss"], "train/macro_f1": tr["macro_f1"],
            "val/loss": va["loss"],   "val/macro_f1": va["macro_f1"],
        }, step=epoch)

        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1=best_f1, path=ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    log.info(f"Training complete. Best macro F1 = {best_f1:.4f}")
    finish_wandb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B3 Damage Classifier")
    parser.add_argument("--epochs",      type=int,   default=cfg.classifier.epochs)
    parser.add_argument("--batch-size",  type=int,   default=cfg.classifier.batch_size)
    parser.add_argument("--lr",          type=float, default=cfg.classifier.learning_rate)
    parser.add_argument("--patience",    type=int,   default=cfg.classifier.patience)
    parser.add_argument("--device",      type=str,   default=cfg.device)
    parser.add_argument("--num-workers", type=int,   default=cfg.num_workers)
    args = parser.parse_args()
    train(args)
