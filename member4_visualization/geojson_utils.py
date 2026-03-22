"""
member4_visualization/geojson_utils.py
────────────────────────────────────────────────────────────────────────────
Member 4 – GeoJSON Generation from Model Predictions

Converts per-building damage predictions (integer labels 0–3) into a
standard GeoJSON FeatureCollection that can be opened in QGIS, rendered
by Folium, or shared with rescue teams.

The damage color schema follows the humanitarian standard:
    0  no_damage    → #2ECC71  (green)
    1  minor_damage → #F1C40F  (yellow)
    2  major_damage → #E67E22  (orange)
    3  destroyed    → #E74C3C  (red)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Color / label mapping ─────────────────────────────────────────────────────

DAMAGE_LABELS = ["no_damage", "minor_damage", "major_damage", "destroyed"]
DAMAGE_COLORS = ["#2ECC71",   "#F1C40F",      "#E67E22",      "#E74C3C"]


def _make_feature(
    polygon_coords: List[Tuple[float, float]],
    damage_label: int,
    building_id: str = "",
    confidence: Optional[float] = None,
    extra_props: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Create one GeoJSON Feature for a single building polygon.

    Parameters
    ----------
    polygon_coords : list of (longitude, latitude) or (x, y) pixel coordinates
    damage_label   : integer 0–3
    building_id    : unique identifier string for this polygon
    confidence     : softmax probability of the predicted class (optional)
    extra_props    : any additional key/value pairs to store in properties

    Returns
    -------
    GeoJSON Feature dict.
    """
    label_name = DAMAGE_LABELS[damage_label] if 0 <= damage_label < 4 else "unknown"
    color      = DAMAGE_COLORS[damage_label] if 0 <= damage_label < 4 else "#808080"

    props: Dict[str, Any] = {
        "building_id":    building_id,
        "damage_label":   damage_label,
        "damage_class":   label_name,
        "marker_color":   color,
    }
    if confidence is not None:
        props["confidence"] = round(float(confidence), 4)
    if extra_props:
        props.update(extra_props)

    # Close the polygon ring
    coords = list(polygon_coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[list(pt) for pt in coords]],
        },
        "properties": props,
    }


def predictions_to_geojson(
    polygons: List[List[Tuple[float, float]]],
    damage_labels: List[int],
    confidences: Optional[List[float]] = None,
    building_ids: Optional[List[str]] = None,
    extra_props: Optional[List[Dict[str, Any]]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Build a GeoJSON FeatureCollection from lists of polygons and labels.

    Parameters
    ----------
    polygons      : list of polygon coordinate lists, each [[lon, lat], ...]
    damage_labels : list of integer class predictions (0–3) per polygon
    confidences   : optional list of float confidence scores
    building_ids  : optional list of string IDs
    extra_props   : optional list of per-building property dicts
    image_shape   : optional (H, W) to invert Y-axis for local image plotting

    Returns
    -------
    Python dict representing a valid GeoJSON FeatureCollection.
    """
    assert len(polygons) == len(damage_labels), \
        "polygons and damage_labels must have the same length"

    n = len(polygons)
    ids   = building_ids or [f"building_{i:05d}" for i in range(n)]
    confs = confidences  or [None] * n
    props = extra_props  or [None] * n

    H = image_shape[0] if image_shape else 0
    
    transformed_polys = []
    for poly in polygons:
        if image_shape:
            # Map pixel (x, y) to (x, H - y) for Folium
            transformed_polys.append([(pt[0], H - pt[1]) for pt in poly])
        else:
            transformed_polys.append(poly)

    features = [
        _make_feature(
            polygon_coords=transformed_polys[i],
            damage_label=damage_labels[i],
            building_id=ids[i],
            confidence=confs[i],
            extra_props=props[i],
        )
        for i in range(n)
    ]

    # Summary statistics in the collection-level properties
    from collections import Counter
    label_counts = Counter(damage_labels)
    summary = {label: label_counts.get(i, 0) for i, label in enumerate(DAMAGE_LABELS)}

    return {
        "type": "FeatureCollection",
        "properties": {
            "total_buildings": n,
            **summary,
        },
        "features": features,
    }


def save_geojson(geojson: dict, path: str | Path) -> None:
    """Write a GeoJSON dict to a .geojson file (UTF-8, pretty-printed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


def load_geojson(path: str | Path) -> dict:
    """Load a GeoJSON file from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
