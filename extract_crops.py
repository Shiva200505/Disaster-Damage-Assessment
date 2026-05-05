"""
extract_crops.py — Extract per-building crops from xBD for classifier training.
Run ONCE after downloading xBD, before training the classifier.
Usage: python extract_crops.py --split train
       python extract_crops.py --split test --out-split val
"""

import argparse
import json
import logging
import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    if not poly1.intersects(poly2):
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def process_image(args_dict):
    """
    Process a single post-disaster image and its JSON label.
    Extract building crops and return a list of metadata dictionaries.
    """
    post_img_path = args_dict['post_img_path']
    json_path = args_dict['json_path']
    out_dir = args_dict['out_dir']
    crop_size = args_dict['crop_size']
    padding = args_dict['padding']
    min_area = args_dict['min_area']

    img = cv2.imread(str(post_img_path))
    if img is None:
        return []
    
    H, W = img.shape[:2]

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return []

    # Extract disaster name and tile_id from filename. Example: "palu-tsunami_00000001_post_disaster.png"
    stem = post_img_path.stem
    parts = stem.split("_")
    disaster = parts[0] if len(parts) > 0 else "unknown"
    tile_id  = parts[1] if len(parts) > 1 else "unknown"

    features = data.get("features", {}).get("xy", [])
    
    # Parse polygons and metadata
    buildings = []
    for feat in features:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "un-classified")
        if subtype == "un-classified":
            continue
            
        wkt = feat.get("wkt", "")
        if not wkt.startswith("POLYGON"):
            continue
            
        try:
            coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
            pts = []
            for pair in coords_str.split(","):
                x, y = pair.strip().split()
                pts.append((float(x), float(y)))
            if len(pts) < 3:
                continue
            poly = Polygon(pts)
            if poly.area < min_area:
                continue
            
            # Bounding box
            minx, miny, maxx, maxy = poly.bounds
            
            # Validate: box entirely outside
            if minx >= W or maxx < 0 or miny >= H or maxy < 0:
                continue
                
            buildings.append({
                "poly": poly,
                "subtype": subtype,
                "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
            })
        except Exception:
            continue

    # Deduplicate overlapping polygons (IoU > 0.9)
    keep_indices = []
    for i, b1 in enumerate(buildings):
        is_dup = False
        for j in keep_indices:
            b2 = buildings[j]
            if compute_iou(b1["poly"], b2["poly"]) > 0.9:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)
            
    filtered_buildings = [buildings[i] for i in keep_indices]

    results = []
    for idx, bldg in enumerate(filtered_buildings):
        minx, miny, maxx, maxy = bldg["minx"], bldg["miny"], bldg["maxx"], bldg["maxy"]
        subtype = bldg["subtype"]
        
        # Apply padding
        x1 = int(max(0, minx - padding))
        y1 = int(max(0, miny - padding))
        x2 = int(min(W, maxx + padding))
        y2 = int(min(H, maxy + padding))
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Resize to crop_size
        crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        
        out_filename = f"{disaster}_{tile_id}_bldg{idx}_{subtype}.png"
        out_filepath = out_dir / out_filename
        
        # Save crop
        cv2.imwrite(str(out_filepath), crop_resized)
        
        label_int = DAMAGE_MAP.get(subtype, 0)
        
        results.append({
            "filename": out_filename,
            "label": label_int,
            "subtype": subtype,
            "tile_id": tile_id,
            "disaster": disaster,
            "bbox_x": int(minx),
            "bbox_y": int(miny),
            "bbox_w": int(maxx - minx),
            "bbox_h": int(maxy - miny)
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract crops for classifier training.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Dataset split in xBD to read from.")
    parser.add_argument("--out-split", type=str, default=None, help="Output folder name (e.g., train, val, test). Defaults to --split value.")
    parser.add_argument("--crop-size", type=int, default=64, help="Resize extracted crops to this size.")
    parser.add_argument("--padding", type=int, default=8, help="Padding in pixels around bounding box.")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum polygon area to extract.")
    args = parser.parse_args()

    out_split = args.out_split if args.out_split else args.split
    img_dir = Path(f"data/xbd/{args.split}/images")
    lbl_dir = Path(f"data/xbd/{args.split}/labels")

    if not img_dir.exists():
        logging.error(f"Image directory not found: {img_dir}")
        return
        
    if not lbl_dir.exists():
        logging.error(f"Labels directory not found: {lbl_dir}")
        return

    out_dir = Path(f"data/building_crops/{out_split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    post_images = sorted(list(img_dir.glob("*_post_disaster.png")))
    
    tasks = []
    for img_path in post_images:
        json_name = img_path.name.replace(".png", ".json")
        json_path = lbl_dir / json_name
        if json_path.exists():
            tasks.append({
                'post_img_path': img_path,
                'json_path': json_path,
                'out_dir': out_dir,
                'crop_size': args.crop_size,
                'padding': args.padding,
                'min_area': args.min_area
            })

    logging.info(f"Found {len(tasks)} post_disaster images with JSON labels in {args.split}.")

    workers = multiprocessing.cpu_count()
    all_results = []
    
    with multiprocessing.Pool(processes=workers) as pool:
        for res in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Extracting crops"):
            if res:
                all_results.extend(res)

    if len(all_results) == 0:
        logging.warning("No crops extracted. Please check your data paths and parameters.")
        return

    df = pd.DataFrame(all_results)
    
    # Save metadata
    meta_path = out_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    logging.info(f"Saved metadata with {len(df)} crops to {meta_path}")

    # Print class distribution
    dist = df["subtype"].value_counts()
    print("\n--- Class Distribution ---")
    print(dist.to_string())
    print("--------------------------\n")

if __name__ == "__main__":
    main()
