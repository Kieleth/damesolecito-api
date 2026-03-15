"""
Fetch DSM data from Spain's IGN WCS API.

Three coverages available at wcs-mds.idee.es/mds:
  - mds05:       5m full DSM (elevation + buildings + trees)
  - mdsn_e025:   2.5m normalized building heights (height above ground)
  - mdsn_v025:   2.5m normalized vegetation heights

No authentication required. Works for any location in Spain.
"""

import sys
import numpy as np
from pathlib import Path

import httpx
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyproj import Transformer


WCS_URL = "https://wcs-mds.idee.es/mds"

# Default: Cea Bermudez 38, Madrid
DEFAULT_LAT = 40.4401
DEFAULT_LON = -3.7110


def latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to UTM Zone 30N (EPSG:25830)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def _fetch_coverage(
    coverage: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    resolution: float,
) -> np.ndarray:
    """Fetch a single WCS coverage and return as numpy array."""
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    params = {
        "SERVICE": "WCS",
        "VERSION": "1.0.0",
        "REQUEST": "GetCoverage",
        "COVERAGE": coverage,
        "CRS": "EPSG:25830",
        "BBOX": f"{x_min},{y_min},{x_max},{y_max}",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/tiff",
    }

    response = httpx.get(WCS_URL, params=params, timeout=30.0)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "xml" in content_type or "html" in content_type:
        raise RuntimeError(f"WCS returned {content_type}: {response.text[:300]}")

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            return src.read(1).astype(np.float64)


def fetch_dsm(
    lat: float,
    lon: float,
    radius: float = 200.0,
    output_path: Path | None = None,
) -> Path:
    """Fetch best available DSM for shadow computation.

    Strategy: fetch 2.5m building heights (mdsn_e025) as the primary layer,
    plus 2.5m vegetation heights (mdsn_v025). Combine into a single obstacle
    height map. This gives the height of shadow-casting objects above ground
    at 2.5m resolution.

    Args:
        lat, lon: Center point in WGS84.
        radius: Half-width of the area in meters.
        output_path: Where to save the GeoTIFF.

    Returns:
        Path to the saved GeoTIFF.
    """
    cx, cy = latlon_to_utm(lat, lon)

    x_min = cx - radius
    x_max = cx + radius
    y_min = cy - radius
    y_max = cy + radius

    resolution = 2.5
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)

    print(f"Fetching 2.5m DSM from IGN WCS...")
    print(f"  Center: {lat:.4f}, {lon:.4f} (UTM: {cx:.0f}, {cy:.0f})")
    print(f"  Area: {radius*2:.0f}x{radius*2:.0f}m")
    print(f"  Grid: {width}x{height} at {resolution}m")

    # Fetch building heights (2.5m)
    print(f"  Fetching building heights (mdsn_e025)...")
    buildings = _fetch_coverage("mdsn_e025", x_min, y_min, x_max, y_max, resolution)
    bld_cells = np.count_nonzero(buildings > 0)
    print(f"    Buildings: {bld_cells} cells, max {np.max(buildings):.0f}m")

    # Fetch vegetation heights (2.5m)
    print(f"  Fetching vegetation heights (mdsn_v025)...")
    vegetation = _fetch_coverage("mdsn_v025", x_min, y_min, x_max, y_max, resolution)
    veg_cells = np.count_nonzero(vegetation > 0)
    print(f"    Vegetation: {veg_cells} cells, max {np.max(vegetation):.0f}m")

    # Combine: take the maximum of buildings and vegetation per cell.
    # This gives us a complete obstacle map for shadow casting.
    dsm = np.maximum(buildings, vegetation)
    total_obstacle = np.count_nonzero(dsm > 0)
    print(f"  Combined: {total_obstacle} obstacle cells, height range 0-{np.max(dsm):.0f}m")

    # Output path
    if output_path is None:
        output_dir = Path(__file__).parent / "data" / "dsm"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"dsm_{lat:.4f}_{lon:.4f}_{int(radius*2)}m.tif"

    # Write GeoTIFF
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float64",
        crs=CRS.from_string("EPSG:25830"),
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(dsm, 1)

    print(f"  Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LAT
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LON
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0

    path = fetch_dsm(lat, lon, radius)
    print(f"\nDSM ready at: {path}")
    print(f"Test it: python shadow_engine.py {path}")
