"""
member2_siamese/train_siamese.py
────────────────────────────────────────────────────────────────────────────
Member 2 – Siamese Network Training Loop

Features:
  • AdamW optimiser with cosine annealing LR schedule
  • Early stopping on validation F1 score (patience = cfg.siamese.patience)
  • Per-epoch logging of loss, IoU, and F1
  • Best checkpoint saved via utils.checkpoint
  • Optional Weights & Biases run via utils.logger
"""

import argparse
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from member1_preprocessing.augmentation import get_train_transforms, get_val_transforms
from member1_preprocessing.dataset import get_dataloaders
from member2_siamese.siamese_net import SiameseUNet
from member2_siamese.loss import CombinedSegLoss
from member5_evaluation.metrics import compute_iou, compute_f1
from utils.checkpoint import save_checkpoint
from utils.config import cfg
from utils.logger import get_logger, init_wandb, log_metrics, finish_wandb

log = get_logger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training / validation step ────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss, total_iou, total_f1 = 0.0, 0.0, 0.0
    n = len(loader)

    for batch in loader:
        pre   = batch["pre"].to(device)
        post  = batch["post"].to(device)
        mask  = batch["change_mask"].to(device)

        optimizer.zero_grad()
        logits = model(pre, post)
        loss   = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            iou   = compute_iou(preds, mask)
            f1    = compute_f1(preds, mask)

        total_loss += loss.item()
        total_iou  += iou
        total_f1   += f1

    return {"loss": total_loss / n, "iou": total_iou / n, "f1": total_f1 / n}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, total_iou, total_f1 = 0.0, 0.0, 0.0
    n = len(loader)

    for batch in loader:
        pre   = batch["pre"].to(device)
        post  = batch["post"].to(device)
        mask  = batch["change_mask"].to(device)

        logits = model(pre, post)
        loss   = criterion(logits, mask)
        preds  = (torch.sigmoid(logits) > 0.5).float()

        total_loss += loss.item()
        total_iou  += compute_iou(preds, mask)
        total_f1   += compute_f1(preds, mask)

    return {"loss": total_loss / n, "iou": total_iou / n, "f1": total_f1 / n}


# ── Main training loop ────────────────────────────────────────────────────────

def train(args) -> None:
    set_seed(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")
    cfg.make_dirs()

    # W&B
    if cfg.wandb.enabled:
        init_wandb(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config={"model": "SiameseUNet", **vars(args)},
            run_name="siamese_run",
        )

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=cfg.preprocess.patch_size,
        add_spectral=cfg.preprocess.add_spectral,
        train_transform=get_train_transforms(cfg.preprocess.patch_size),
        val_transform=get_val_transforms(cfg.preprocess.patch_size),
    )

    # Model
    in_ch = 5 if cfg.preprocess.add_spectral else 3
    model = SiameseUNet(in_channels=in_ch, pretrained=True).to(device)

    criterion = CombinedSegLoss(
        dice_weight=cfg.siamese.dice_weight,
        bce_weight=cfg.siamese.bce_weight,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.siamese.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, patience_counter = 0.0, 0
    ckpt_path = cfg.paths.checkpoint_dir / "siamese_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        # Log to W&B
        log_metrics(
            {
                "train/loss": train_metrics["loss"],
                "train/iou":  train_metrics["iou"],
                "train/f1":   train_metrics["f1"],
                "val/loss":   val_metrics["loss"],
                "val/iou":    val_metrics["iou"],
                "val/f1":     val_metrics["f1"],
                "lr":         scheduler.get_last_lr()[0],
            },
            step=epoch,
        )

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1=best_f1, path=ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    log.info(f"Training complete. Best val F1 = {best_f1:.4f}")
    finish_wandb()


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Change Detection Network")
    parser.add_argument("--epochs",      type=int,   default=cfg.siamese.epochs)
    parser.add_argument("--batch-size",  type=int,   default=cfg.siamese.batch_size)
    parser.add_argument("--lr",          type=float, default=cfg.siamese.learning_rate)
    parser.add_argument("--patience",    type=int,   default=cfg.siamese.patience)
    parser.add_argument("--device",      type=str,   default=cfg.device)
    parser.add_argument("--num-workers", type=int,   default=cfg.num_workers)
    args = parser.parse_args()
    train(args)
