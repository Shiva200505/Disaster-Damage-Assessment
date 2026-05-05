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

def parse_xbd_geojson(json_path: str | Path) -> dict:
    """
    Parse an xBD label JSON and return a dict with:
      - 'pixel_polygons': list of pixel-space polygon coords (from features.xy)
      - 'geo_polygons':   list of (lon, lat) polygon coords (from features.lng_lat)
      - 'subtypes':       list of damage subtype strings per polygon
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        
    pixel_polygons = []
    geo_polygons = []
    subtypes = []
    
    for feat in data.get("features", {}).get("xy", []):
        wkt = feat.get("wkt", "")
        pix_coords = []
        try:
            coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
            for pair in coords_str.split(","):
                x, y = pair.strip().split()
                pix_coords.append((float(x), float(y)))
            pixel_polygons.append(pix_coords)
            subtypes.append(feat.get("properties", {}).get("subtype", "un-classified"))
        except Exception:
            pass

    for feat in data.get("features", {}).get("lng_lat", []):
        wkt = feat.get("wkt", "")
        geo_coords = []
        try:
            coords_str = wkt.replace("POLYGON ((", "").replace("))", "").strip()
            for pair in coords_str.split(","):
                x, y = pair.strip().split()
                geo_coords.append((float(x), float(y)))
            geo_polygons.append(geo_coords)
        except Exception:
            pass
            
    return {
        "pixel_polygons": pixel_polygons,
        "geo_polygons": geo_polygons,
        "subtypes": subtypes
    }


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
    geo_polygons: Optional[List[List[Tuple[float, float]]]] = None,
) -> dict:
    """
    Build a GeoJSON FeatureCollection from lists of polygons and labels.

    Parameters
    ----------
    polygons      : list of pixel polygon coordinate lists, each [[x, y], ...]
    damage_labels : list of integer class predictions (0–3) per polygon
    confidences   : optional list of float confidence scores
    building_ids  : optional list of string IDs
    extra_props   : optional list of per-building property dicts
    image_shape   : optional (H, W) to invert Y-axis for local image plotting
    geo_polygons  : optional list of real WGS84 (lon, lat) polygons

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

    transformed_polys = []
    is_pixel = False

    # If true geo polygons are passed, bypass pixel Y-flip logic
    if geo_polygons and len(geo_polygons) == n:
        transformed_polys = geo_polygons
    else:
        is_pixel = True
        H = image_shape[0] if image_shape else 0
        for poly in polygons:
            if image_shape:
                # Map pixel (x, y) to (x, H - y) for Folium CRS Simple
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

    from collections import Counter
    label_counts = Counter(damage_labels)
    summary = {label: label_counts.get(i, 0) for i, label in enumerate(DAMAGE_LABELS)}

    collection_props = {
        "total_buildings": n,
        **summary,
    }
    
    if is_pixel:
        collection_props["crs"] = "pixel"

    return {
        "type": "FeatureCollection",
        "properties": collection_props,
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
