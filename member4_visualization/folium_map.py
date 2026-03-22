"""
member4_visualization/folium_map.py
────────────────────────────────────────────────────────────────────────────
Member 4 – Interactive Damage Map via Folium

Renders building polygons colour-coded by damage severity onto an
OpenStreetMap base layer using Folium (Python / Leaflet.js).

All folium imports are lazy (inside render_damage_map) so this module
can be imported even without folium installed (e.g. during unit tests).

Usage
-----
    from member4_visualization.folium_map import render_damage_map

    render_damage_map(
        geojson_path="outputs/geojson/predictions.geojson",
        output_html="outputs/maps/damage_map.html",
        disaster_name="Hurricane Harvey",
    )
"""

from pathlib import Path
from typing import Optional, Tuple

from member4_visualization.geojson_utils import (
    load_geojson, DAMAGE_LABELS, DAMAGE_COLORS,
)


# ── Pure-Python helpers (no folium dependency) ────────────────────────────────

def _style_fn(feature: dict) -> dict:
    label = feature["properties"].get("damage_label", 0)
    color = DAMAGE_COLORS[label] if 0 <= label < 4 else "#808080"
    return {"fillColor": color, "color": "#333333",
            "weight": 1, "fillOpacity": 0.65}


def _highlight_fn(feature: dict) -> dict:
    return {"weight": 3, "color": "#ffffff", "fillOpacity": 0.85}


def _popup_html(props: dict) -> str:
    label  = props.get("damage_label", 0)
    damage = props.get("damage_class", "unknown")
    bid    = props.get("building_id", "")
    conf   = props.get("confidence")
    color  = DAMAGE_COLORS[label] if 0 <= label < 4 else "#808080"
    conf_row = (f"<tr><td><b>Confidence</b></td><td>{conf:.1%}</td></tr>"
                if conf else "")
    return (
        f'<div style="font-family:Arial,sans-serif;min-width:160px;">'
        f'<div style="background:{color};color:#fff;padding:6px 10px;'
        f'border-radius:4px 4px 0 0;"><b>&#127968; '
        f'{damage.replace("_", " ").title()}</b></div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        f"<tr><td><b>Building ID</b></td><td>{bid}</td></tr>"
        f"{conf_row}</table></div>"
    )


def _compute_center(geojson: dict) -> Tuple[float, float]:
    all_coords = []
    for feat in geojson.get("features", []):
        for ring in feat["geometry"]["coordinates"]:
            all_coords.extend(ring)
    if not all_coords:
        return (0.0, 0.0)
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def _legend_html(disaster_name: str) -> str:
    items = "".join(
        f'<div><span style="background:{c};width:14px;height:14px;'
        f'display:inline-block;border-radius:3px;margin-right:6px;">'
        f'</span>{lbl.replace("_", " ").title()}</div>'
        for lbl, c in zip(DAMAGE_LABELS, DAMAGE_COLORS)
    )
    return (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:rgba(255,255,255,0.92);border:1px solid #ccc;'
        'border-radius:8px;padding:12px 16px;font-family:Arial,sans-serif;'
        'font-size:13px;line-height:1.8;'
        'box-shadow:2px 2px 8px rgba(0,0,0,0.15);">'
        f'<b>&#128752; {disaster_name}</b><br/>Damage Severity'
        f'<hr style="margin:4px 0"/>{items}</div>'
    )


# ── Main map renderer ─────────────────────────────────────────────────────────

def render_damage_map(
    geojson_path: str | Path,
    output_html: str | Path = "outputs/maps/damage_map.html",
    disaster_name: str = "Disaster Zone",
    zoom_start: int = 15,
    tiles: Optional[str] = "CartoDB positron",
    image_path: Optional[str | Path] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Render an interactive damage map from a GeoJSON predictions file.

    Parameters
    ----------
    geojson_path  : path to the GeoJSON FeatureCollection
    output_html   : path to write the self-contained HTML file
    disaster_name : display name shown in the legend
    zoom_start    : initial map zoom level
    tiles         : Folium tile provider name (or None for Simple CRS)
    image_path    : optional bounding image to overlay (e.g. disaster image)
    image_shape   : (H, W) needed if image_path is provided

    Returns
    -------
    Absolute path of the written HTML file as a string.
    """
    # Lazy folium import — only required when actually rendering
    try:
        import folium
        from folium.plugins import MiniMap, Fullscreen
    except ImportError as e:
        raise ImportError(
            "folium is required for map rendering. "
            "Install it with: pip install folium"
        ) from e

    geojson = load_geojson(geojson_path)
    center  = _compute_center(geojson)

    # Use Simple CRS and ImageOverlay if an image is provided
    if image_path and image_shape:
        H, W = image_shape
        bounds = [[0, 0], [H, W]]
        center = [H / 2, W / 2]
        m = folium.Map(location=center, zoom_start=1, crs="Simple", tiles=None)
        
        folium.raster_layers.ImageOverlay(
            image=str(image_path),
            bounds=bounds,
        ).add_to(m)
        m.fit_bounds(bounds)
    else:
        m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)

    # Building polygons GeoJSON layer
    folium.GeoJson(
        geojson,
        name="Building Damage",
        style_function=_style_fn,
        highlight_function=_highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["damage_class", "building_id"],
            aliases=["Damage:", "ID:"],
            sticky=False,
        ),
    ).add_to(m)

    # Per-feature popups with rich HTML
    for feature in geojson.get("features", []):
        props = feature["properties"]
        ring  = feature["geometry"]["coordinates"][0]
        lat_c = sum(pt[1] for pt in ring) / len(ring)
        lon_c = sum(pt[0] for pt in ring) / len(ring)
        folium.Popup(_popup_html(props), max_width=220).add_to(
            folium.Marker(
                location=[lat_c, lon_c],
                icon=folium.DivIcon(html="", icon_size=(1, 1)),
            ).add_to(m)
        )

    # Controls & legend
    folium.LayerControl().add_to(m)
    if not image_path:
        MiniMap(toggle_display=True).add_to(m)
    Fullscreen().add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html(disaster_name)))

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    return str(output_html.absolute())
