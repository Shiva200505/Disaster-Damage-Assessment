"""
member5_evaluation/inference_pipeline.py
────────────────────────────────────────────────────────────────────────────
Member 5 – End-to-End Inference Pipeline

Loads a pre-trained Siamese network and EfficientNet-B3 classifier,
runs them over a pre/post image pair, and produces:
  • A binary change mask (H × W)
  • Per-building damage labels
  • A GeoJSON FeatureCollection
  • An interactive Folium HTML damage map

Usage
-----
    from member5_evaluation.inference_pipeline import run_inference

    result = run_inference(
        pre_img_path   = "data/test/pre.png",
        post_img_path  = "data/test/post.png",
        geojson_path   = "data/test/buildings.json",   # optional xBD annotation
        siamese_ckpt   = "checkpoints/siamese_best.pth",
        classifier_ckpt= "checkpoints/classifier_best.pth",
        output_dir     = "outputs/inference",
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from member1_preprocessing.preprocessing import align_images, normalize
from member1_preprocessing.spectral_indices import add_spectral_channels
from member2_siamese.siamese_net import SiameseUNet
from member3_classifier.efficientnet_classifier import (
    DamageClassifier, extract_building_crops, classify_buildings,
)
from member4_visualization.geojson_utils import (
    predictions_to_geojson, save_geojson,
)
from member4_visualization.folium_map import render_damage_map
from utils.checkpoint import load_checkpoint
from utils.config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ── Image Loading ─────────────────────────────────────────────────────────────

def _load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as (H, W, 3) uint8 RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """H×W×C float32 → (1, C, H, W) tensor on device."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t.float().to(device)


# ── Polygon parsing from xBD GeoJSON ─────────────────────────────────────────

def _load_polygons_from_geojson(geojson_path: str | Path) -> List[List[Tuple[float, float]]]:
    """
    Extract pixel-space polygon coordinates from an xBD label JSON.
    Returns list of polygon coord lists.
    """
    import json

    with open(geojson_path) as f:
        gj = json.load(f)

    polygons = []
    for feat in gj.get("features", {}).get("xy", []):
        wkt = feat.get("wkt", "")
        try:
            coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
            pts = []
            for pair in coords_str.split(","):
                x, y = pair.strip().split()
                pts.append((float(x), float(y)))
            polygons.append(pts)
        except Exception:
            pass
    return polygons


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_inference(
    pre_img_path:    str | Path,
    post_img_path:   str | Path,
    siamese_ckpt:    str | Path,
    classifier_ckpt: str | Path,
    geojson_path:    Optional[str | Path] = None,
    output_dir:      str | Path = "outputs/inference",
    device:          str = "cuda",
    add_spectral:    bool = True,
    map_threshold:   float = 0.5,
    disaster_name:   str = "Disaster Zone",
) -> Dict:
    """
    Full end-to-end inference: pre+post images → damage map + GeoJSON.

    Parameters
    ----------
    pre_img_path    : path to pre-disaster image (PNG/TIFF)
    post_img_path   : path to post-disaster image (PNG/TIFF)
    siamese_ckpt    : path to Siamese network .pth checkpoint
    classifier_ckpt : path to EfficientNet .pth checkpoint
    geojson_path    : optional xBD label JSON for polygon coordinates.
                      If None, building outlines are approximated from
                      connected components of the change mask.
    output_dir      : directory to write outputs
    device          : "cuda" | "cpu"
    add_spectral    : append NDVI/NDWI channels before inference
    map_threshold   : sigmoid threshold for binarising the change mask
    disaster_name   : label used in the Folium map legend

    Returns
    -------
    dict with keys:
      change_mask   – (H, W) uint8 binary mask
      damage_labels – list of integers per building
      geojson       – GeoJSON FeatureCollection dict
      geojson_path  – saved .geojson file path
      map_html_path – saved Folium HTML path
    """
    device = device if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    log.info("Loading images …")
    pre_img  = _load_rgb(pre_img_path)
    post_img = _load_rgb(post_img_path)
    post_img = align_images(pre_img, post_img)

    H, W = pre_img.shape[:2]

    # ── 2. Spectral channels ──────────────────────────────────────────────────
    if add_spectral:
        pre_in  = add_spectral_channels(normalize(pre_img))
        post_in = add_spectral_channels(normalize(post_img))
    else:
        pre_in  = normalize(pre_img)
        post_in = normalize(post_img)

    in_ch = pre_in.shape[-1]   # 5 or 3

    # ── 3. Siamese forward pass ───────────────────────────────────────────────
    log.info("Running Siamese change detection …")
    siamese = SiameseUNet(in_channels=in_ch, pretrained=False).to(device)
    load_checkpoint(siamese_ckpt, siamese, device=device)
    siamese.eval()

    pre_t  = _to_tensor(pre_in,  device)
    post_t = _to_tensor(post_in, device)

    with torch.no_grad():
        logits = siamese(pre_t, post_t)   # (1, 1, H, W)

    change_prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize to original if needed
    if change_prob.shape != (H, W):
        change_prob = cv2.resize(change_prob, (W, H))

    change_mask = (change_prob > map_threshold).astype(np.uint8)

    # ── 4. Extract building polygons ──────────────────────────────────────────
    if geojson_path is not None and Path(geojson_path).exists():
        log.info("Loading polygons from xBD annotation …")
        polygons = _load_polygons_from_geojson(geojson_path)
    else:
        log.info("Approximating building outlines from change mask contours …")
        polygons = _mask_to_polygons(change_mask)

    log.info(f"Found {len(polygons)} building polygons.")

    # ── 5. Damage classification ──────────────────────────────────────────────
    log.info("Classifying damage severity …")
    classifier = DamageClassifier(num_classes=4, pretrained=False).to(device)
    load_checkpoint(classifier_ckpt, classifier, device=device)

    crops  = extract_building_crops(post_img, polygons,
                                    crop_size=cfg.classifier.crop_size)
    labels = classify_buildings(classifier, crops, device=device) if crops else []

    # ── 6. GeoJSON ────────────────────────────────────────────────────────────
    # Convert pixel coords to pseudo lon/lat (identity for local CRS)
    geojson = predictions_to_geojson(polygons=polygons, damage_labels=labels)
    gj_path = out / "predictions.geojson"
    save_geojson(geojson, gj_path)
    log.info(f"GeoJSON saved → {gj_path}")

    # ── 7. Folium map ─────────────────────────────────────────────────────────
    map_path = out / "damage_map.html"
    render_damage_map(
        geojson_path=gj_path,
        output_html=map_path,
        disaster_name=disaster_name,
    )
    log.info(f"Damage map saved → {map_path}")

    # ── 8. Save change mask ───────────────────────────────────────────────────
    mask_path = out / "change_mask.png"
    cv2.imwrite(str(mask_path), change_mask * 255)

    return {
        "change_mask":   change_mask,
        "damage_labels": labels,
        "geojson":       geojson,
        "geojson_path":  str(gj_path),
        "map_html_path": str(map_path),
        "mask_path":     str(mask_path),
    }


# ── Mask → polygon fallback ───────────────────────────────────────────────────

def _mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 50,
) -> List[List[Tuple[float, float]]]:
    """
    Extract polygon outlines of connected components in a binary mask.
    Used when xBD GeoJSON annotations are unavailable at inference time.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        poly = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
        polys.append(poly)
    return polys
