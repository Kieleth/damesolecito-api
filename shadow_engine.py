"""
Shadow engine: raster shadow casting on a DSM grid.

Given a Digital Surface Model (heights on a regular grid) and a sun position,
computes which cells are in shadow. All coordinate work uses pyproj with
EPSG:25830 (UTM Zone 30N) for metric calculations.

No Blender, no rotation hacks, no equirectangular approximations.
"""

import numpy as np
import rasterio
from pyproj import Transformer
from pysolar.solar import get_altitude, get_azimuth
from scipy import ndimage
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


class SunPosition(NamedTuple):
    altitude: float  # degrees above horizon
    azimuth: float   # degrees from north, clockwise


class ShadowResult(NamedTuple):
    shadow_mask: np.ndarray   # bool array, True = in shadow
    sun: SunPosition
    dsm_bounds: tuple         # (west, south, east, north) in EPSG:4326
    pixel_size: float         # meters
    crs: str                  # source CRS of DSM


def get_sun_position(lat: float, lon: float, dt: datetime) -> SunPosition:
    """Get sun altitude and azimuth for a location and time.

    Uses pysolar. Returns altitude in degrees above horizon (negative = below)
    and azimuth in degrees from north clockwise.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    altitude = get_altitude(lat, lon, dt)
    azimuth = get_azimuth(lat, lon, dt)

    return SunPosition(altitude=altitude, azimuth=azimuth)


def remove_isolated_pixels(
    dsm: np.ndarray,
    min_component_pixels: int = 3,
) -> np.ndarray:
    """Remove isolated small components from DSM.

    Single-pixel and tiny clusters are LiDAR noise (especially from the
    vegetation layer which has ~200 single-pixel artifacts per tile).
    Real trees and buildings are always multi-pixel at 2.5m resolution.

    Args:
        dsm: 2D height array.
        min_component_pixels: Components smaller than this are zeroed.
            3 pixels at 2.5m = 7.5m minimum footprint.

    Returns:
        Cleaned DSM array (copy).
    """
    cleaned = dsm.copy()
    obstacle_mask = cleaned > 0
    labeled, num_features = ndimage.label(obstacle_mask)
    if num_features > 0:
        component_sizes = np.array(
            ndimage.sum(obstacle_mask, labeled, range(1, num_features + 1))
        )
        small_ids = np.where(component_sizes < min_component_pixels)[0] + 1
        if len(small_ids) > 0:
            cleaned[np.isin(labeled, small_ids)] = 0
    return cleaned


def load_dsm(
    dsm_path: Path,
    target_resolution: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Load a DSM GeoTIFF, optionally upscaling to finer resolution.

    The IGN WCS API provides 2.5m data. Upscaling to 1m via bilinear
    interpolation dramatically reduces the staircase artifact on diagonal
    shadow edges (steps go from 2.5m to 1m — 2.5x less visible).

    This doesn't add real height information, but the interpolated surface
    gives the shadow ray marcher smoother boundaries to work with.

    Args:
        dsm_path: Path to GeoTIFF.
        target_resolution: Desired pixel size in meters.
            Set to 0 or negative to skip upscaling.
    """
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            dsm[dsm == nodata] = np.nan

        native_pixel_size = abs(src.transform.a)
        bounds = src.bounds
        meta = {
            "transform": src.transform,
            "crs": str(src.crs),
            "bounds": bounds,
            "pixel_size": native_pixel_size,
            "shape": dsm.shape,
        }

    # Remove isolated pixel noise at native resolution (before upscaling)
    dsm = remove_isolated_pixels(dsm)

    # Upscale to target resolution if finer than native
    if target_resolution > 0 and target_resolution < native_pixel_size:
        scale = native_pixel_size / target_resolution
        dsm = ndimage.zoom(dsm, scale, order=1)  # bilinear interpolation

        # Update metadata for the new grid
        meta["pixel_size"] = target_resolution
        meta["shape"] = dsm.shape
        from rasterio.transform import from_bounds
        meta["transform"] = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            dsm.shape[1], dsm.shape[0],
        )

    return dsm, meta


def clean_shadow_mask(
    shadow: np.ndarray,
    min_shadow_pixels: int = 4,
    min_hole_pixels: int = 4,
) -> np.ndarray:
    """Remove small shadow islands and fill small sun holes.

    At 2.5m resolution, a 4-pixel threshold = 10m. This removes:
      - Tiny shadow fragments from vegetation noise (< 10m patches)
      - Small sun holes inside building shadows (< 10m gaps)

    Result: much cleaner polygons, fewer artifacts, same overall pattern.
    """
    cleaned = shadow.copy()

    # Remove small shadow islands
    labeled, n = ndimage.label(cleaned)
    if n > 0:
        sizes = np.array(ndimage.sum(cleaned, labeled, range(1, n + 1)))
        small_ids = np.where(sizes < min_shadow_pixels)[0] + 1
        if len(small_ids) > 0:
            cleaned[np.isin(labeled, small_ids)] = False

    # Fill small sun holes inside shadows
    inverted = ~cleaned
    labeled_h, n_h = ndimage.label(inverted)
    if n_h > 0:
        hole_sizes = np.array(ndimage.sum(inverted, labeled_h, range(1, n_h + 1)))
        small_hole_ids = np.where(hole_sizes < min_hole_pixels)[0] + 1
        if len(small_hole_ids) > 0:
            cleaned[np.isin(labeled_h, small_hole_ids)] = True

    return cleaned


def compute_shadows(
    dsm: np.ndarray,
    sun_altitude: float,
    sun_azimuth: float,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Compute shadow mask from a DSM and sun position.

    Algorithm: vectorized raster ray marching. For each distance step k
    (in the direction of the sun), check if any cell k steps toward the sun
    is tall enough to cast a shadow on the current cell.

    Args:
        dsm: 2D array of heights in meters. Row 0 = north, row increases south.
             Column 0 = west, column increases east.
        sun_altitude: degrees above horizon (must be > 0).
        sun_azimuth: degrees from north, clockwise.
        pixel_size: grid cell size in meters.

    Returns:
        Boolean 2D array. True = cell is in shadow.
    """
    if sun_altitude <= 0:
        # Sun below horizon — everything is in shadow
        return np.ones(dsm.shape, dtype=bool)

    if sun_altitude >= 90:
        # Sun directly overhead — no shadows
        return np.zeros(dsm.shape, dtype=bool)

    alt_rad = np.radians(sun_altitude)
    azi_rad = np.radians(sun_azimuth)

    # Step direction toward the sun in grid coordinates.
    # Azimuth: 0=N, 90=E, 180=S, 270=W
    # Grid: row 0 = north (row increases southward), col increases eastward.
    raw_dc = np.sin(azi_rad)   # east component toward sun
    raw_dr = -np.cos(azi_rad)  # north component (negative = toward north = lower row index)

    # Normalize so the larger component is 1 pixel per step (DDA-style).
    step_len = max(abs(raw_dc), abs(raw_dr), 1e-10)
    step_dc = raw_dc / step_len
    step_dr = raw_dr / step_len

    # Height gain per step along the ray toward the sun.
    step_dist = pixel_size * np.sqrt(step_dc**2 + step_dr**2)
    dz_per_step = step_dist * np.tan(alt_rad)

    rows, cols = dsm.shape
    shadow = np.zeros((rows, cols), dtype=bool)

    # Max ray distance: tallest feature's shadow length in pixels.
    height_range = np.nanmax(dsm) - np.nanmin(dsm)
    max_shadow_pixels = int(height_range / np.tan(alt_rad) / pixel_size) + 2
    max_steps = min(max_shadow_pixels, max(rows, cols))

    for k in range(1, max_steps + 1):
        # Integer offset for this step
        offset_r = int(round(k * step_dr))
        offset_c = int(round(k * step_dc))

        # Skip if this offset is the same as previous step (can happen at shallow angles)
        if k > 1:
            prev_r = int(round((k - 1) * step_dr))
            prev_c = int(round((k - 1) * step_dc))
            if offset_r == prev_r and offset_c == prev_c:
                continue

        # Bail if offset exceeds grid
        if abs(offset_r) >= rows or abs(offset_c) >= cols:
            break

        # Slice ranges: target cells (might be shadowed) and blocker cells (toward sun).
        # Blocker is at (target_r + offset_r, target_c + offset_c).
        if offset_r >= 0:
            tgt_r = slice(0, rows - offset_r)
            blk_r = slice(offset_r, rows)
        else:
            tgt_r = slice(-offset_r, rows)
            blk_r = slice(0, rows + offset_r)

        if offset_c >= 0:
            tgt_c = slice(0, cols - offset_c)
            blk_c = slice(offset_c, cols)
        else:
            tgt_c = slice(-offset_c, cols)
            blk_c = slice(0, cols + offset_c)

        # Height difference: how much taller the blocker is than the target.
        height_diff = dsm[blk_r, blk_c] - dsm[tgt_r, tgt_c]

        # Threshold: minimum height difference to cast shadow at this distance.
        threshold = k * dz_per_step

        # Mark shadowed cells.
        shadow[tgt_r, tgt_c] |= (height_diff > threshold)

    return shadow


def compute_shadows_for_location(
    dsm_path: Path,
    lat: float,
    lon: float,
    dt: datetime,
    radius: float = 200.0,
) -> ShadowResult:
    """Full pipeline: load DSM, compute sun position, cast shadows.

    Args:
        dsm_path: Path to DSM GeoTIFF.
        lat, lon: Location (WGS84).
        dt: Datetime (UTC or timezone-aware).
        radius: Radius in meters to extract around the point (unused if DSM
                is already clipped; reserved for future tile selection).

    Returns:
        ShadowResult with shadow mask and metadata.
    """
    dsm, meta = load_dsm(dsm_path)  # upscales to 1m by default
    sun = get_sun_position(lat, lon, dt)

    if sun.altitude <= 0:
        shadow = np.ones(dsm.shape, dtype=bool)
    else:
        shadow = compute_shadows(dsm, sun.altitude, sun.azimuth, meta["pixel_size"])
        # Scale cleaning thresholds to pixel area (25m² ≈ 4px at 2.5m)
        min_area_m2 = 25.0
        px_area = meta["pixel_size"] ** 2
        min_px = max(4, int(min_area_m2 / px_area))
        shadow = clean_shadow_mask(shadow, min_shadow_pixels=min_px, min_hole_pixels=min_px)

    # Compute bounds in WGS84
    bounds = meta["bounds"]
    transformer = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds.left, bounds.bottom)
    east, north = transformer.transform(bounds.right, bounds.top)

    return ShadowResult(
        shadow_mask=shadow,
        sun=sun,
        dsm_bounds=(west, south, east, north),
        pixel_size=meta["pixel_size"],
        crs=meta["crs"],
    )


class BarSunStatus(NamedTuple):
    is_sunny: bool
    sun_altitude: float
    sun_azimuth: float
    next_change: str | None       # ISO time of next transition, or None
    next_change_type: str | None  # "sun" or "shade"
    sunset: str | None            # ISO time of sunset


def query_point_shadow(
    dsm: np.ndarray,
    meta: dict,
    lat: float,
    lon: float,
    shadow_mask: np.ndarray,
) -> bool:
    """Check if a WGS84 point is in shadow on the given mask.

    Returns True if in shadow, False if sunny.
    Returns True (shadow) if point is outside the DSM bounds.
    """
    to_utm = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    px, py = to_utm.transform(lon, lat)

    bounds = meta["bounds"]
    if not (bounds.left <= px <= bounds.right and bounds.bottom <= py <= bounds.top):
        return True  # outside coverage = assume shadow (unknown)

    pixel_size = meta["pixel_size"]
    col = int((px - bounds.left) / pixel_size)
    row = int((bounds.top - py) / pixel_size)

    row = max(0, min(row, shadow_mask.shape[0] - 1))
    col = max(0, min(col, shadow_mask.shape[1] - 1))

    return bool(shadow_mask[row, col])


def predict_bar_status(
    dsm_path: Path,
    lat: float,
    lon: float,
    dt: datetime,
    step_minutes: int = 15,
    max_hours: float = 8.0,
) -> BarSunStatus:
    """Full per-bar query: current status + when does it change?

    Loads DSM once, computes shadow at current time, then scans forward
    in `step_minutes` increments to find the next sun/shade transition
    and sunset time.
    """
    dsm, meta = load_dsm(dsm_path)
    pixel_size = meta["pixel_size"]

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    sun = get_sun_position(lat, lon, dt)

    # Current shadow status
    if sun.altitude <= 0:
        return BarSunStatus(
            is_sunny=False,
            sun_altitude=sun.altitude,
            sun_azimuth=sun.azimuth,
            next_change=None,
            next_change_type=None,
            sunset=None,
        )

    shadow_now = compute_shadows(dsm, sun.altitude, sun.azimuth, pixel_size)
    is_shadowed_now = query_point_shadow(dsm, meta, lat, lon, shadow_now)

    # Scan forward to find next transition and sunset
    from datetime import timedelta
    step = timedelta(minutes=step_minutes)
    max_steps = int(max_hours * 60 / step_minutes)

    next_change_time = None
    next_change_type = None
    sunset_time = None

    for i in range(1, max_steps + 1):
        future_dt = dt + step * i
        future_sun = get_sun_position(lat, lon, future_dt)

        # Found sunset
        if future_sun.altitude <= 0:
            sunset_time = future_dt.isoformat()
            if next_change_time is None and not is_shadowed_now:
                # Currently sunny, will become shade at sunset
                next_change_time = future_dt.isoformat()
                next_change_type = "shade"
            break

        future_shadow = compute_shadows(dsm, future_sun.altitude, future_sun.azimuth, pixel_size)
        is_shadowed_future = query_point_shadow(dsm, meta, lat, lon, future_shadow)

        if next_change_time is None and is_shadowed_future != is_shadowed_now:
            next_change_time = future_dt.isoformat()
            next_change_type = "shade" if is_shadowed_future else "sun"

        # If we found both transition and sunset, stop early
        if next_change_time is not None and sunset_time is not None:
            break

    return BarSunStatus(
        is_sunny=not is_shadowed_now,
        sun_altitude=sun.altitude,
        sun_azimuth=sun.azimuth,
        next_change=next_change_time,
        next_change_type=next_change_type,
        sunset=sunset_time,
    )


# --- CLI for quick testing ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shadow_engine.py <dsm.tif> [lat lon yyyy-mm-ddTHH:MM]")
        print("  Defaults: lat=40.4401, lon=-3.7110, time=now UTC")
        sys.exit(1)

    dsm_path = Path(sys.argv[1])
    lat = float(sys.argv[2]) if len(sys.argv) > 2 else 40.4401
    lon = float(sys.argv[3]) if len(sys.argv) > 3 else -3.7110

    if len(sys.argv) > 4:
        dt = datetime.fromisoformat(sys.argv[4]).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    print(f"DSM: {dsm_path}")
    print(f"Location: {lat}, {lon}")
    print(f"Time: {dt.isoformat()}")

    result = compute_shadows_for_location(dsm_path, lat, lon, dt)

    total = result.shadow_mask.size
    shadowed = int(np.sum(result.shadow_mask))
    sunny = total - shadowed
    pct = sunny / total * 100

    print(f"Sun altitude: {result.sun.altitude:.1f} deg")
    print(f"Sun azimuth: {result.sun.azimuth:.1f} deg")
    print(f"Grid: {result.shadow_mask.shape[0]}x{result.shadow_mask.shape[1]}")
    print(f"Pixel size: {result.pixel_size}m")
    print(f"Sunny: {sunny}/{total} cells ({pct:.1f}%)")
    print(f"Shadow: {shadowed}/{total} cells ({100-pct:.1f}%)")
