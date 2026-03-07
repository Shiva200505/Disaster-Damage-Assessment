# 🛰️ Disaster Damage Assessment from Satellite Imagery

> Academic Deep Learning project — fully free, open-source, reproducible.

---

## Team Structure

| Member | Workstream | Module |
|--------|-----------|--------|
| **Member 1** | Data Preprocessing | `member1_preprocessing/` |
| **Member 2** | Siamese Change Detection | `member2_siamese/` |
| **Member 3** | Damage Severity Classifier | `member3_classifier/` |
| **Member 4** | Geospatial Visualization | `member4_visualization/` |
| **Member 5** | Evaluation & Demo Notebook | `member5_evaluation/` |

---

## Project Structure

```
DL_CP/
├── requirements.txt
├── README.md
├── utils/                         # Shared: config, logger, checkpoint
├── member1_preprocessing/         # xBD dataset loader, augmentation, spectral indices
├── member2_siamese/               # Siamese U-Net, Dice+BCE loss, training loop
├── member3_classifier/            # EfficientNet-B3, Focal Loss, training loop
├── member4_visualization/         # GeoJSON, Folium maps, Matplotlib/Seaborn plots
├── member5_evaluation/            # Metrics, inference pipeline, demo notebook
└── tests/                         # Sanity tests (no dataset required)
```

---

## Dataset

Download the **xBD dataset** (free after registration):

1. Go to [xview2.org](https://xview2.org) and register.
2. Download the dataset and extract to `data/xbd/`:

```
data/xbd/
├── train/   images/ + labels/
├── val/     images/ + labels/
└── test/    images/ + labels/
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure paths and hyperparameters

Edit `utils/config.py` to update `xbd_root`, `checkpoint_dir`, and W&B settings.

---

## Training

### Stage 1 – Siamese Change Detection (Member 2)

```bash
python -m member2_siamese.train_siamese \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --device cuda
```

Saves best checkpoint to `checkpoints/siamese_best.pth`.

### Stage 2 – Damage Severity Classifier (Member 3)

> First extract building crops from xBD tiles using the preprocessing pipeline,
> then run training:

```bash
python -m member3_classifier.train_classifier \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --device cuda
```

Saves best checkpoint to `checkpoints/classifier_best.pth`.

---

## Inference (End-to-End)

```python
from member5_evaluation.inference_pipeline import run_inference

result = run_inference(
    pre_img_path    = "data/xbd/test/images/hurricane-harvey_00000001_pre_disaster.png",
    post_img_path   = "data/xbd/test/images/hurricane-harvey_00000001_post_disaster.png",
    geojson_path    = "data/xbd/test/labels/hurricane-harvey_00000001_post_disaster.json",
    siamese_ckpt    = "checkpoints/siamese_best.pth",
    classifier_ckpt = "checkpoints/classifier_best.pth",
    output_dir      = "outputs/demo",
    disaster_name   = "Hurricane Harvey",
)

print("Map saved →", result["map_html_path"])
```

Open `outputs/demo/damage_map.html` in any browser for the interactive map.

---

## Demo Notebook

```bash
jupyter notebook member5_evaluation/demo_notebook.ipynb
```

The notebook provides a fully self-contained walkthrough using synthetic data
(no download required) and shows the real inference flow when checkpoints exist.

---

## Running Sanity Tests

```bash
python -m pytest tests/test_sanity.py -v
```

All 23 tests use **synthetic random data** — no xBD download needed.

---

## Evaluation Metrics

| Metric | Stage | Description |
|--------|-------|-------------|
| **IoU** | Stage 1 | Intersection over Union (pixel-level change mask) |
| **F1** (binary) | Stage 1 | Pixel-wise F1 / Dice score |
| **Per-class F1** | Stage 2 | F1 per damage level (no/minor/major/destroyed) |
| **Macro F1** | Stage 2 | Average F1 across 4 classes |
| **xBD Score** | Both | Harmonic mean of localisation F1 + classification F1 |

---

## Free Resources Used

| Resource | Purpose |
|----------|---------|
| [xBD Dataset](https://xview2.org) | Training & evaluation data |
| PyTorch + torchvision + timm | Models & pretrained weights |
| [Kaggle Notebooks](https://kaggle.com) | Free 30 GPU hrs/week (T4) |
| [Google Colab](https://colab.research.google.com) | Free T4/V100 for experiments |
| [W&B Free Tier](https://wandb.ai) | Experiment tracking |
| Folium + OpenStreetMap | Interactive map rendering |
| Albumentations | Image augmentation |

**Total cost: ₹0**

---

## Architecture Overview

```
Pre-disaster image ──┐
                     ├── Shared ResNet-50 Encoder ──┐
Post-disaster image ─┘   (2 parallel branches)      │
                                                     ▼
                                         Sub-Cat Fusion at 4 scales
                                                     │
                                                     ▼
                                         U-Net Decoder → Change Mask
                                                     │
                                    ┌────────────────┘
                                    ▼
                         Building polygon crops (from xBD GeoJSON)
                                    │
                                    ▼
                         EfficientNet-B3 → 4-class damage label
                                    │
                                    ▼
                         GeoJSON + Folium Interactive Map
```
