import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from member3_classifier.efficientnet_classifier import extract_building_crops

# xBD damage subtype -> folder integer
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 0
}

FOLDER_MAPPING = {
    0: "0_no_damage",
    1: "1_minor",
    2: "2_major",
    3: "3_destroyed"
}

def extract_from_split(split="train"):
    images_dir = Path(f"data/xbd/{split}/images")
    labels_dir = Path(f"data/xbd/{split}/labels")
    
    if not images_dir.exists():
        print(f"Skipping {split} - dir not found: {images_dir}")
        return

    # Assuming train dataset split is used for training the classifier
    # If the user's xBD layout is different, we adjust. But usually the crops are stored in data/building_crops/train/...
    out_dir = Path(f"data/building_crops/{split}")
    
    for k, v in FOLDER_MAPPING.items():
        (out_dir / v).mkdir(parents=True, exist_ok=True)

    print(f"Extracting crops for {split} split...")
    
    # We will iterate over post_disaster images because damage labels are in post_disaster.json
    post_images = list(images_dir.glob("*_post_disaster.png"))
    
    crop_counts = {0:0, 1:0, 2:0, 3:0}
    
    for img_path in tqdm(post_images):
        stem = img_path.stem.replace("_post_disaster", "")
        json_path = labels_dir / f"{stem}_post_disaster.json"
        
        if not json_path.exists():
            continue
            
        # 1. Read Image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Read Polygons & Labels
        polygons = []
        labels = []
        
        with open(json_path) as f:
            gj = json.load(f)
            
        for feat in gj.get("features", {}).get("xy", []):
            subtype = feat.get("properties", {}).get("subtype", "no-damage")
            label = DAMAGE_MAP.get(subtype, 0)
            
            wkt = feat.get("wkt", "")
            try:
                coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
                pts = [(float(p.strip().split()[0]), float(p.strip().split()[1])) for p in coords_str.split(",")]
                polygons.append(pts)
                labels.append(label)
            except Exception:
                pass
                
        if not polygons:
            continue
            
        # 3. Extract Crops
        crops = extract_building_crops(img, polygons, crop_size=64)
        
        # 4. Save Crops
        for i, (crop, label) in enumerate(zip(crops, labels)):
            folder = FOLDER_MAPPING[label]
            crop_filename = f"{stem}_bldg{i}.png"
            out_path = out_dir / folder / crop_filename
            
            # Convert RGB back to BGR for cv2 saving
            cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            crop_counts[label] += 1
            
    print(f"Done extracting {split}! Counts: {crop_counts}")

if __name__ == "__main__":
    print("This script will extract individual building crops to data/building_crops/ for the Classifier.")
    extract_from_split("train")
    # For validation, xBD uses 'test' usually or 'tier3'
    # We'll extract 'test' as 'val' for the classifier
    
    images_dir_test = Path("data/xbd/test/images")
    if images_dir_test.exists():
        # But wait, test labels in xBD are often in 'targets' as PNGs, or don't have JSONs!
        # If they do have JSONs in test/labels, we can use this script.
        # Let's check if test/labels exists
        if Path("data/xbd/test/labels").exists():
            # In our function, we pass split='test'. But we want to save into data/building_crops/val.
            # We will just rename test to val after, or we do a minor hack here:
            pass
