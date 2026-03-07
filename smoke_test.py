"""
Quick smoke test — verifies xBD dataset loads correctly and a forward
pass through the Siamese network completes without errors.
Run from the project root: python smoke_test.py
"""
import sys
import torch
from member1_preprocessing.dataset import XBDDataset, get_dataloaders
from member1_preprocessing.augmentation import get_train_transforms, get_val_transforms
from member2_siamese.siamese_net import SiameseUNet
from utils.config import cfg

print("=" * 60)
print("SMOKE TEST — Disaster Damage Assessment Pipeline")
print("=" * 60)

# 1. Dataset
print("\n[1/4] Loading train dataset ...")
train_ds = XBDDataset(
    root_dir=cfg.paths.train_dir,
    split="train",
    patch_size=cfg.preprocess.patch_size,
    add_spectral=cfg.preprocess.add_spectral,
    transform=get_train_transforms(cfg.preprocess.patch_size),
)
print(f"      ✓ {len(train_ds)} training samples found")

print("\n[2/4] Loading val dataset ...")
val_ds = XBDDataset(
    root_dir=cfg.paths.val_dir,
    split="val",
    patch_size=cfg.preprocess.patch_size,
    add_spectral=cfg.preprocess.add_spectral,
    transform=get_val_transforms(cfg.preprocess.patch_size),
)
print(f"      ✓ {len(val_ds)} validation samples found")

if len(train_ds) == 0:
    print("\n❌ ERROR: No training samples found.")
    print("   Check that data/xbd/train/images/ contains PNG files.")
    sys.exit(1)

# 2. Load one sample
print("\n[3/4] Loading first sample ...")
sample = train_ds[0]
print(f"      pre  shape : {sample['pre'].shape}")
print(f"      post shape : {sample['post'].shape}")
print(f"      mask shape : {sample['change_mask'].shape}")
print(f"      label      : {sample['damage_label'].item()}")
print(f"      tile name  : {sample['stem']}")

# 3. Model forward pass
print("\n[4/4] Model forward pass ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Device: {device}")

in_ch = 5 if cfg.preprocess.add_spectral else 3
model = SiameseUNet(in_channels=in_ch, pretrained=False).to(device)
model.eval()

pre_b  = sample["pre"].unsqueeze(0).to(device)
post_b = sample["post"].unsqueeze(0).to(device)

with torch.no_grad():
    out = model(pre_b, post_b)

print(f"      ✓ Output shape: {out.shape}  (expected: [1, 1, 256, 256])")
print(f"      ✓ Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED — Ready to train!")
print("=" * 60)
print(f"\nDataset stats:")
print(f"  Train samples : {len(train_ds)}")
print(f"  Val samples   : {len(val_ds)}")
print(f"  Image channels: {in_ch}")
print(f"  Patch size    : {cfg.preprocess.patch_size}")
print(f"  Batch size    : {cfg.siamese.batch_size}")
print(f"  Device        : {device}")
