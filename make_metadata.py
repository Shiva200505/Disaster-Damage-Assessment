"""
make_metadata.py — Regenerate metadata.csv for existing building crops.
Run this if extract_crops.py was interrupted before saving the CSV.
"""
from pathlib import Path
import pandas as pd

DAMAGE_MAP = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3,
}

for split in ['train', 'val']:
    out_dir = Path(f'data/building_crops/{split}')
    rows = []
    for png in sorted(out_dir.glob('*.png')):
        stem = png.stem  # e.g. guatemala-volcano_00000000_bldg0_no-damage
        # Last part after the last underscore is the label
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            subtype = parts[1]  # e.g. no-damage
            label = DAMAGE_MAP.get(subtype, -1)
            if label >= 0:
                rows.append({'filename': png.name, 'subtype': subtype, 'label': label})

    df = pd.DataFrame(rows)
    meta_path = out_dir / 'metadata.csv'
    df.to_csv(meta_path, index=False)
    print(f'{split}: {len(df)} crops saved to {meta_path}')
    print(df.groupby('subtype').size().to_string())
    print()
