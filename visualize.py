"""
Interactive shadow map visualization.

Generates an HTML map with shadow overlay on OpenStreetMap tiles.
Shows shadows for multiple times of day to demonstrate the system.
"""

import sys
import json
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import folium
import rasterio
import rasterio.features
from pyproj import Transformer
from shapely.geometry import shape, mapping

from shadow_engine import compute_shadows_for_location, load_dsm, compute_shadows, get_sun_position


DEFAULT_LAT = 40.4401
DEFAULT_LON = -3.7110


def shadow_to_geojson(shadow_mask, dsm_path):
    """Convert shadow mask to GeoJSON polygons in WGS84."""
    with rasterio.open(dsm_path) as src:
        transform = src.transform
        crs = str(src.crs)

    shapes = list(rasterio.features.shapes(
        shadow_mask.astype(np.uint8),
        mask=shadow_mask,
        transform=transform,
    ))

    proj = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    polygons = []

    for geom, value in shapes:
        if value == 1:
            s = shape(geom)
            coords = []
            for ring in [s.exterior] + list(s.interiors):
                transformed = [proj.transform(x, y) for x, y in ring.coords]
                coords.append(transformed)
            from shapely.geometry import Polygon
            poly = Polygon(coords[0], coords[1:])
            polygons.append(poly)

    return polygons


def create_shadow_map(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    dsm_path: Path | None = None,
    output_path: Path | None = None,
):
    """Create an interactive HTML map with shadow overlays for multiple times."""

    # Find DSM
    if dsm_path is None:
        dsm_dir = Path(__file__).parent / "data" / "dsm"
        dsm_files = list(dsm_dir.glob("*.tif"))
        if not dsm_files:
            print("No DSM files found. Run fetch_dsm.py first.")
            sys.exit(1)
        dsm_path = dsm_files[0]

    if output_path is None:
        output_path = Path(__file__).parent / "shadow_map.html"

    print(f"DSM: {dsm_path}")
    print(f"Location: {lat}, {lon}")

    # Create base map
    m = folium.Map(
        location=[lat, lon],
        zoom_start=17,
        tiles="OpenStreetMap",
    )

    # Times to visualize (UTC — Madrid is UTC+1 in winter, UTC+2 in summer)
    # March 9 2026 in Madrid local time = UTC+1
    times = [
        ("09:00 (morning)", datetime(2026, 3, 9, 8, 0, tzinfo=timezone.utc)),
        ("12:00 (noon)", datetime(2026, 3, 9, 11, 0, tzinfo=timezone.utc)),
        ("14:00 (afternoon)", datetime(2026, 3, 9, 13, 0, tzinfo=timezone.utc)),
        ("17:00 (evening)", datetime(2026, 3, 9, 16, 0, tzinfo=timezone.utc)),
    ]

    colors = ["#1a237e", "#4a148c", "#b71c1c", "#e65100"]

    dsm, meta = load_dsm(dsm_path)

    for (label, dt), color in zip(times, colors):
        sun = get_sun_position(lat, lon, dt)

        if sun.altitude <= 0:
            print(f"  {label}: Sun below horizon, skipping")
            continue

        shadow = compute_shadows(dsm, sun.altitude, sun.azimuth, meta["pixel_size"])
        polygons = shadow_to_geojson(shadow, dsm_path)

        total = shadow.size
        shadowed = int(np.sum(shadow))
        sunny_pct = (total - shadowed) / total * 100

        print(f"  {label}: sun alt={sun.altitude:.1f} az={sun.azimuth:.1f}"
              f" | {sunny_pct:.0f}% sunny | {len(polygons)} shadow polygons")

        # Create feature group for this time
        fg = folium.FeatureGroup(name=f"{label} (sun {sun.altitude:.0f}°, {sunny_pct:.0f}% sunny)")

        for poly in polygons:
            coords = [(y, x) for x, y in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                weight=0,
                popup=f"{label}<br>Sun: {sun.altitude:.1f}° alt, {sun.azimuth:.1f}° az",
            ).add_to(fg)

        fg.add_to(m)

    # Add marker for Cea Bermudez 38
    folium.Marker(
        [lat, lon],
        popup="Cea Bermudez 38 (Alcaravea)",
        icon=folium.Icon(color="orange", icon="cutlery", prefix="fa"),
    ).add_to(m)

    # Layer control to toggle times
    folium.LayerControl(collapsed=False).add_to(m)

    # Save
    m.save(str(output_path))
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LAT
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LON

    path = create_shadow_map(lat, lon)

    # Open in browser
    print("Opening in browser...")
    webbrowser.open(f"file://{path.resolve()}")
