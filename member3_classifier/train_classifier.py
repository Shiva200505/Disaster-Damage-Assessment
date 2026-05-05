"""
train_classifier.py — Train EfficientNet-B3 on per-building crops.
Run: python train_classifier.py --data-dir data/building_crops --epochs 30
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import timm
from tqdm import tqdm

import torchvision.transforms as T
from member3_classifier.augmentation import get_crop_train_transforms, get_crop_val_transforms
from utils.logger import get_logger

log = get_logger(__name__)

# --- Optional Custom Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# --- Dataset ---
class CropDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Fast existence check: build a set of files in the directory once
        existing_files = set(os.listdir(str(self.img_dir)))
        self.df["exists"] = self.df["filename"].isin(existing_files)
        if not self.df["exists"].all():
            missing = (~self.df["exists"]).sum()
            log.warning(f"Found {missing} missing images in {img_dir}. Filtering them out.")
            self.df = self.df[self.df["exists"]]
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(self.img_dir / row["filename"])
        label = int(row["label"])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            if hasattr(self.transform, "transforms"): # Check if torchvision vs albumentation
                try: # Albumentations
                    res = self.transform(image=img)
                    img = res["image"]
                except TypeError: # Torchvision
                    from PIL import Image
                    img = Image.fromarray(img)
                    img = self.transform(img)
            else:
                 res = self.transform(image=img)
                 img = res["image"]
            
        return img, label
        
    def get_labels(self):
        return self.df["label"].values

# --- Training loop ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Training", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device.type):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for imgs, targets in tqdm(loader, desc="Validating", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        
        with torch.amp.autocast(device.type):
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/building_crops", help="Base dir with train/val subfolders")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from", default=None)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone for first N epochs")
    parser.add_argument("--freeze-epochs", type=int, default=5, help="Number of epochs to freeze backbone")
    parser.add_argument("--use-torchvision", action="store_true", help="Use standard torchvision transforms instead of Albumentations")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Transforms (ImageNet normalization via Albumentations or torchvision)
    if args.use_torchvision:
        # Proper ImageNet normalization via torchvision.transforms
        log.info("Using torchvision.transforms with proper ImageNet normalization.")
        train_tfms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        log.info("Using Albumentations pipelines with proper ImageNet normalization.")
        train_tfms = get_crop_train_transforms(crop_size=64)
        val_tfms = get_crop_val_transforms(crop_size=64)
    
    # Datasets
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    
    train_csv = train_dir / "metadata.csv"
    val_csv = val_dir / "metadata.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        log.error("Missing metadata.csv! Ensure extract_crops.py has been run for --split train and --out-split val")
        return
        
    train_ds = CropDataset(train_csv, train_dir, transform=train_tfms)
    val_ds = CropDataset(val_csv, val_dir, transform=val_tfms)
    log.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    # Weighted Random Sampler for Class Imbalance
    class_counts = np.bincount(train_ds.get_labels())
    class_weights = np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0)
    sample_weights = [class_weights[lbl] for lbl in train_ds.get_labels()]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Model
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=4)
    model.to(device)
    
    # Disabled torch.compile due to Triton dependency on Windows
    pass

    # Freeze Backbone mechanism
    if args.freeze_backbone and args.freeze_epochs > 0:
        log.info(f"Freezing backbone for the first {args.freeze_epochs} epochs")
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model._orig_mod if hasattr(model, "_orig_mod") else model, "get_classifier"):
            classifier = (model._orig_mod if hasattr(model, "_orig_mod") else model).get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True
    
    criterion = FocalLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    
    # CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        if Path(args.resume).exists():
            log.info(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_f1 = ckpt.get("best_f1", 0.0)
        else:
            log.warning(f"Resume checkpoint {args.resume} not found. Starting fresh.")
    
    Path("outputs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        
        # Unfreeze backbone check
        if args.freeze_backbone and epoch == args.freeze_epochs:
            log.info("Unfreezing backbone for fine-tuning")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, preds, targets = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        t1 = time.time()
        
        # Sklearn classification report
        target_names = ["no_damage", "minor_damage", "major_damage", "destroyed"]
        report = classification_report(targets, preds, target_names=target_names, output_dict=True, zero_division=0)
        report_str = classification_report(targets, preds, target_names=target_names, zero_division=0)
        
        macro_f1 = report["macro avg"]["f1-score"]
        log.info(f"Epoch {epoch}/{args.epochs-1} [{t1-t0:.1f}s] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro F1: {macro_f1:.4f}")
        
        with open("outputs/classifier_eval.txt", "a") as f:
            f.write(f"\n--- Epoch {epoch} ---\n")
            f.write(report_str)
            
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": state_dict,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_f1": best_f1
            }, "checkpoints/classifier_best.pth")
            log.info("Saved new best checkpoint.")

if __name__ == "__main__":
    main()
