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
        output_dir     = "outputs/inference",
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags

from member1_preprocessing.preprocessing import align_images, normalize
from member1_preprocessing.spectral_indices import add_spectral_channels
from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2, SiameseUNetV3
from member4_visualization.geojson_utils import (
    parse_xbd_geojson, predictions_to_geojson, save_geojson,
)
from member4_visualization.folium_map import render_damage_map
from member5_evaluation.sliding_window import sliding_window_inference
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


def _extract_gps_info(image_path: str | Path) -> Optional[dict]:
    """Extract EXIF GPS metadata if available using Pillow."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        for tag, val in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_info = {}
                for t in val:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    # Convert byte/tuple structures for JSON compatibility
                    if isinstance(val[t], tuple):
                        gps_info[sub_decoded] = list(val[t])
                    elif isinstance(val[t], bytes):
                        gps_info[sub_decoded] = val[t].decode(errors="ignore")
                    else:
                        gps_info[sub_decoded] = val[t]
                return gps_info
    except Exception:
        pass
    return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_inference(
    pre_img_path:    str | Path,
    post_img_path:   str | Path,
    siamese_ckpt:    str | Path,
    geojson_path:    Optional[str | Path] = None,
    output_dir:      str | Path = "outputs/inference",
    device:          str = "cuda",
    add_spectral:    bool = cfg.preprocess.add_spectral,
    map_threshold:   float = 0.5,
    disaster_name:   str = "Disaster Zone",
    ensemble_ckpts:  Optional[List[str | Path]] = None,
    use_onnx:        bool = False,
    classifier_ckpt: Optional[str | Path] = None,
    model_version:   str = cfg.siamese.model_version,
) -> Dict:
    """
    Full end-to-end inference: pre+post images → damage map + GeoJSON.

    Parameters
    ----------
    pre_img_path    : path to pre-disaster image (PNG/TIFF)
    post_img_path   : path to post-disaster image (PNG/TIFF)
    siamese_ckpt    : path to Siamese network .pth checkpoint
    geojson_path    : optional xBD label JSON for polygon coordinates.
    output_dir      : directory to write outputs
    device          : "cuda" | "cpu"
    add_spectral    : append NDVI/NDWI channels before inference
    map_threshold   : sigmoid threshold for binarising the change mask
    disaster_name   : label used in the Folium map legend
    ensemble_ckpts  : optional list of model checkpoint paths for ensembling
    """
    device = device if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load images ────────────────────────────────────────────────────────
    log.info("Loading images …")
    pre_img       = _load_rgb(pre_img_path)
    post_img_orig = _load_rgb(post_img_path)   # keep original for overlay (no warp artifacts)
    post_img      = align_images(pre_img, post_img_orig)

    H, W = pre_img.shape[:2]
    
    # EXIF GPS Fallback extraction
    exif_gps = _extract_gps_info(post_img_path)

    # ── 2. Spectral channels ──────────────────────────────────────────────────
    if add_spectral:
        pre_in  = add_spectral_channels(normalize(pre_img))
        post_in = add_spectral_channels(normalize(post_img))
    else:
        pre_in  = normalize(pre_img)
        post_in = normalize(post_img)

    in_ch = pre_in.shape[-1]   # 5 or 3

    # ── 3. Siamese forward pass (Sliding Window + Ensembling or ONNX) ─────────────────
    from member5_evaluation.onnx_inference import ONNXSiamese
    import time
    
    log.info("Running Siamese change detection …")
    
    if use_onnx:
        log.info(f"Using ONNX Inference (TTA disabled) -> {siamese_ckpt}")
        try:
            onnx_siamese = ONNXSiamese(str(siamese_ckpt), device=device)
            t0 = time.time()
            # Direct prediction with ONNX (dynamic axes allows full resolution processing)
            change_prob = onnx_siamese.predict(pre_in, post_in)
            log.info(f"ONNX Inference Time: {time.time()-t0:.3f}s")
            
            if change_prob.shape != (H, W):
                change_prob = cv2.resize(change_prob, (W, H))
                
        except Exception as e:
            raise RuntimeError(f"ONNX Inference failed: {e}")
            
    else:
        ensemble_probs = []
        ckpts_to_run = ensemble_ckpts if ensemble_ckpts else [siamese_ckpt]
        
        MODEL_MAP = {"v1": SiameseUNet, "v2": SiameseUNetV2, "v3": SiameseUNetV3}
        ModelClass = MODEL_MAP.get(model_version, SiameseUNet)

        for ckpt in ckpts_to_run:
            log.info(f"Loading checkpoint -> {ckpt}")
            try:
                if model_version in ["v2", "v3"]:
                    siamese = ModelClass(in_channels=in_ch, pretrained=False, deep_supervision=False).to(device)
                else:
                    siamese = ModelClass(in_channels=in_ch, pretrained=False).to(device)
                load_checkpoint(ckpt, siamese, device=device)
            except Exception as e:
                log.error(f"Failed to load checkpoint {ckpt}: {e}")
                continue
                
            siamese.eval()

            # Sliding window with TTA ensures edge artifacts are managed
            change_prob = sliding_window_inference(
                model=siamese,
                pre_img=pre_in,
                post_img=post_in,
                patch_size=cfg.preprocess.patch_size,
                stride=cfg.siamese.sliding_window_stride,
                device=device,
                use_tta=cfg.siamese.use_tta,
                n_augments=cfg.siamese.tta_n_augments,
            )
            ensemble_probs.append(change_prob)
            
        if not ensemble_probs:
            raise ValueError("Failed to evaluate masks from models. Check checkpoint paths.")

        # Average probabilities across the ensemble
        change_prob = np.mean(ensemble_probs, axis=0)

        if change_prob.shape != (H, W):
            change_prob = cv2.resize(change_prob, (W, H))

    change_mask = (change_prob > map_threshold).astype(np.uint8)

    # ── 4. Extract building polygons (Pixel + Geo) ────────────────────────────
    geo_polygons = None
    if geojson_path is not None and Path(geojson_path).exists():
        log.info("Loading polygons from xBD annotation …")
        parsed = parse_xbd_geojson(geojson_path)
        polygons = parsed.get("pixel_polygons", [])
        geo_polygons = parsed.get("geo_polygons", [])
        if not polygons:
             log.warning("GeoJSON empty or invalid, fallback to mask contours.")
             polygons = _mask_to_polygons(change_mask)
             geo_polygons = None
    else:
        log.info("Approximating building outlines from change mask contours …")
        polygons = _mask_to_polygons(change_mask)

    log.info(f"Found {len(polygons)} building polygons.")

    # ── 5. Damage classification ──────────────────────────────────────────────
    log.info("Classifying damage severity ...")
    if classifier_ckpt is not None and Path(classifier_ckpt).exists():
        log.info(f"Using trained EfficientNet classifier -> {classifier_ckpt}")
        labels = _classify_with_model(polygons, post_img_orig, classifier_ckpt, device=device)
    else:
        log.info("Fallback: Classifying damage severity from change probability map …")
        labels = _classify_by_change_prob(polygons, change_prob)

    # ── 6. GeoJSON ────────────────────────────────────────────────────────────
    geojson = predictions_to_geojson(
        polygons=polygons,
        damage_labels=labels,
        image_shape=(H, W),
        geo_polygons=geo_polygons
    )
    
    # Attach EXIF GPS if available as fallback data
    if exif_gps:
        geojson["properties"]["exif_gps"] = exif_gps

    gj_path = out / "predictions.geojson"
    save_geojson(geojson, gj_path)
    log.info(f"GeoJSON saved → {gj_path}")

    # ── 7. Folium map ─────────────────────────────────────────────────────────
    post_img_path_out = out / "post_img.png"
    overlay_rgb = post_img_orig   # original, before homography warp
    rgba = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2RGBA)
    black_mask = np.all(overlay_rgb < 10, axis=-1)
    rgba[black_mask, 3] = 0       # make residual black pixels fully transparent
    cv2.imwrite(str(post_img_path_out), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    H_orig, W_orig = overlay_rgb.shape[:2]
    map_path = out / "damage_map.html"
    render_damage_map(
        geojson_path=gj_path,
        output_html=map_path,
        image_path=str(post_img_path_out),
        image_shape=(H_orig, W_orig),
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


# ── Change-probability → damage label classifier ──────────────────────────────

def _classify_by_change_prob(
    polygons: List[List[Tuple[float, float]]],
    change_prob: np.ndarray,
    thresholds: Tuple[float, float, float] = (0.20, 0.45, 0.55),
) -> List[int]:
    """Assign damage labels based on Siamese change probability."""
    H, W = change_prob.shape[:2] if change_prob.ndim == 2 else change_prob.shape
    t_no, t_minor, t_major = thresholds
    labels = []

    for poly in polygons:
        pts = np.array(poly, dtype=np.float32)
        x_min = int(max(pts[:, 0].min(), 0))
        y_min = int(max(pts[:, 1].min(), 0))
        x_max = int(min(pts[:, 0].max(), W - 1))
        y_max = int(min(pts[:, 1].max(), H - 1))

        if x_max <= x_min or y_max <= y_min:
            labels.append(0)
            continue

        region = change_prob[y_min:y_max, x_min:x_max]
        mean_p = float(region.mean()) if region.size > 0 else 0.0

        if mean_p < t_no:
            labels.append(0)
        elif mean_p < t_minor:
            labels.append(1)
        elif mean_p < t_major:
            labels.append(2)
        else:
            labels.append(3)

    return labels


# ── Mask → polygon fallback ───────────────────────────────────────────────────

def _mask_to_polygons(
    mask: np.ndarray,
    min_area: int = 20,
) -> List[List[Tuple[float, float]]]:
    """Extract polygon outlines of connected components in a binary mask."""
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

# ── Neural Network Classifier ────────────────────────────────────────────────

def _classify_with_model(
    polygons: List[List[Tuple[float, float]]],
    post_img: np.ndarray,
    classifier_ckpt: str | Path,
    device: str = "cuda",
) -> List[int]:
    import timm
    from member3_classifier.augmentation import get_crop_val_transforms
    
    # Load model
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=4)
    load_checkpoint(classifier_ckpt, model, device=device)
    model.to(device)
    model.eval()
    
    transform = get_crop_val_transforms(crop_size=64)
    H, W = post_img.shape[:2]
    
    labels = []
    
    with torch.no_grad():
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32)
            if len(pts) == 0:
                labels.append(0)
                continue
                
            x_min = int(max(pts[:, 0].min(), 0))
            y_min = int(max(pts[:, 1].min(), 0))
            x_max = int(min(pts[:, 0].max(), W - 1))
            y_max = int(min(pts[:, 1].max(), H - 1))
            
            # Add padding
            pad = 8
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(W - 1, x_max + pad)
            y_max = min(H - 1, y_max + pad)

            if x_max <= x_min or y_max <= y_min:
                labels.append(0)
                continue
                
            crop = post_img[y_min:y_max, x_min:x_max]
            # Resize to expected 64x64
            crop = cv2.resize(crop, (64, 64))
            
            # Apply transforms
            res = transform(image=crop)
            img_t = res["image"].unsqueeze(0).to(device)
            
            with torch.amp.autocast(device):
                logits = model(img_t)
                pred = logits.argmax(dim=1).item()
                
            labels.append(pred)
            
    return labels

