"""
Process LAZ (LiDAR) point clouds into 1m DSM GeoTIFF files.

Takes raw .laz files from data/raw/ and outputs .tif DSM files to data/dsm/.
Uses proper CRS handling (EPSG:25830 UTM Zone 30N for Spain).
"""

import sys
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata

import laspy
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


def process_laz_to_dsm(
    laz_path: Path,
    output_path: Path,
    resolution: float = 1.0,
    crs: str = "EPSG:25830",
) -> dict:
    """Convert a LAZ point cloud to a DSM GeoTIFF.

    Reads all points, grids them at the specified resolution using
    maximum height per cell (DSM = Digital Surface Model, includes buildings).

    Args:
        laz_path: Path to input .laz file.
        output_path: Path for output .tif file.
        resolution: Grid cell size in meters (default 1.0).
        crs: Coordinate reference system of the LAZ data.

    Returns:
        Dict with processing stats.
    """
    print(f"Reading {laz_path}...")
    las = laspy.read(str(laz_path))

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    print(f"  Points: {len(x):,}")
    print(f"  X range: {x.min():.2f} - {x.max():.2f}")
    print(f"  Y range: {y.min():.2f} - {y.max():.2f}")
    print(f"  Z range: {z.min():.2f} - {z.max():.2f}")

    # Check for classification (LAS standard: 2=ground, 6=building)
    has_classification = hasattr(las, "classification")
    if has_classification:
        classes = np.array(las.classification)
        unique_classes = np.unique(classes)
        print(f"  Classifications: {unique_classes}")

    # Build DSM grid: maximum height per cell
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Grid dimensions
    cols = int(np.ceil((x_max - x_min) / resolution))
    rows = int(np.ceil((y_max - y_min) / resolution))

    print(f"  Grid: {rows}x{cols} at {resolution}m resolution")

    # Assign each point to a grid cell
    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(int), 0, rows - 1)  # flip Y

    # Maximum height per cell (DSM behavior)
    dsm = np.full((rows, cols), np.nan, dtype=np.float64)

    # Use np.maximum.at for efficient max aggregation
    # First pass: initialize cells
    linear_idx = row_idx * cols + col_idx
    flat_dsm = dsm.ravel()
    np.maximum.at(flat_dsm, linear_idx, z)
    dsm = flat_dsm.reshape(rows, cols)

    # Fill NaN gaps with interpolation (small gaps from sparse point coverage)
    nan_mask = np.isnan(dsm)
    nan_count = int(nan_mask.sum())
    if nan_count > 0 and nan_count < dsm.size:
        print(f"  Filling {nan_count} NaN cells ({nan_count/dsm.size*100:.1f}%)...")
        valid = ~nan_mask
        valid_coords = np.array(np.where(valid)).T
        valid_values = dsm[valid]
        nan_coords = np.array(np.where(nan_mask)).T
        filled = griddata(valid_coords, valid_values, nan_coords, method="nearest")
        dsm[nan_mask] = filled

    # Write GeoTIFF
    # Rasterio transform: from_bounds(west, south, east, north, width, height)
    # For UTM: x_min=west, y_min=south, x_max+(remainder)=east, y_max=north
    east = x_min + cols * resolution
    north = y_max
    south = y_max - rows * resolution
    west = x_min

    transform = from_bounds(west, south, east, north, cols, rows)

    print(f"  Writing {output_path}...")
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float64",
        crs=CRS.from_string(crs),
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(dsm, 1)

    stats = {
        "points": len(x),
        "grid_rows": rows,
        "grid_cols": cols,
        "resolution": resolution,
        "z_min": float(np.nanmin(dsm)),
        "z_max": float(np.nanmax(dsm)),
        "nan_filled": nan_count,
        "crs": crs,
        "output": str(output_path),
    }

    print(f"  Done. Height range: {stats['z_min']:.1f} - {stats['z_max']:.1f}m")
    return stats


def process_all(raw_dir: Path, dsm_dir: Path, resolution: float = 1.0):
    """Process all LAZ files in raw_dir, output DSMs to dsm_dir."""
    laz_files = list(raw_dir.glob("*.laz")) + list(raw_dir.glob("*.LAZ"))
    if not laz_files:
        print(f"No LAZ files found in {raw_dir}")
        return

    dsm_dir.mkdir(parents=True, exist_ok=True)

    for laz_path in sorted(laz_files):
        output_path = dsm_dir / laz_path.with_suffix(".tif").name
        try:
            process_laz_to_dsm(laz_path, output_path, resolution)
        except Exception as e:
            print(f"  ERROR processing {laz_path.name}: {e}")


if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    raw_dir = base / "raw"
    dsm_dir = base / "dsm"

    if len(sys.argv) > 1:
        # Process a specific LAZ file
        laz_path = Path(sys.argv[1])
        output_path = dsm_dir / laz_path.with_suffix(".tif").name
        dsm_dir.mkdir(parents=True, exist_ok=True)
        process_laz_to_dsm(laz_path, output_path)
    else:
        # Process all LAZ files in data/raw/
        process_all(raw_dir, dsm_dir)
