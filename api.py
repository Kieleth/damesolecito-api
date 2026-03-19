"""
FastAPI shadow service.

Single endpoint: given a location and time, returns shadow data.
Loads pre-processed DSM tiles from data/dsm/, computes shadows on demand,
caches results for 15 minutes.
"""

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pyproj import Transformer
from shapely.geometry import mapping, shape
import rasterio.features

from pydantic import BaseModel
from shadow_engine import (
    compute_shadows_for_location, get_sun_position, load_dsm,
    compute_shadows, clean_shadow_mask, query_point_shadow, predict_bar_status,
)

app = FastAPI(title="DameSolecito Shadow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DSM_DIR = Path(__file__).parent / "data" / "dsm"

# Simple in-memory cache: key -> (timestamp, result)
_cache: dict[str, tuple[float, dict]] = {}
CACHE_TTL = 900  # 15 minutes


def _cache_key(lat: float, lon: float, dt: datetime) -> str:
    """Cache key: location + time rounded to 15 minutes."""
    rounded_minute = (dt.minute // 15) * 15
    time_key = dt.replace(minute=rounded_minute, second=0, microsecond=0).isoformat()
    raw = f"{lat:.4f},{lon:.4f},{time_key}"
    return hashlib.md5(raw.encode()).hexdigest()


def _find_dsm(lat: float, lon: float) -> Optional[Path]:
    """Find a DSM tile that covers the given location.

    Returns None if no tile covers the point — caller should fetch on demand.
    """
    dsm_files = list(DSM_DIR.glob("*.tif"))
    if not dsm_files:
        return None

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    px, py = transformer.transform(lon, lat)

    for dsm_path in dsm_files:
        with rasterio.open(dsm_path) as src:
            bounds = src.bounds
            if bounds.left <= px <= bounds.right and bounds.bottom <= py <= bounds.top:
                return dsm_path

    return None  # no tile covers this point — must fetch


def _shadow_mask_to_geojson(
    shadow_mask: np.ndarray,
    transform,
    src_crs: str,
) -> dict:
    """Convert a boolean shadow mask to GeoJSON polygons in WGS84.

    No polygon simplification — the 2.5m pixel staircase is the honest
    representation of our data resolution. Douglas-Peucker at any useful
    tolerance creates worse zigzag artifacts than the raw staircase.
    """
    shapes_list = list(rasterio.features.shapes(
        shadow_mask.astype(np.uint8),
        mask=shadow_mask,
        transform=transform,
    ))

    if not shapes_list:
        return {"type": "FeatureCollection", "features": []}

    proj_transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    features = []
    for geom, value in shapes_list:
        if value == 1:  # shadow
            s = shape(geom)

            coords = []
            for ring in [s.exterior] + list(s.interiors):
                transformed = [
                    proj_transformer.transform(x, y)
                    for x, y in ring.coords
                ]
                coords.append(transformed)

            from shapely.geometry import Polygon
            transformed_poly = Polygon(coords[0], coords[1:])

            features.append({
                "type": "Feature",
                "geometry": mapping(transformed_poly),
                "properties": {"type": "shadow"},
            })

    return {"type": "FeatureCollection", "features": features}


@app.get("/")
async def root():
    """Health check."""
    dsm_files = list(DSM_DIR.glob("*.tif"))
    return {
        "service": "DameSolecito Shadow API",
        "status": "ok",
        "dsm_tiles": len(dsm_files),
        "dsm_dir": str(DSM_DIR),
    }


@app.get("/shadows")
async def get_shadows(
    lat: float = Query(..., description="Latitude (WGS84)", ge=-90, le=90),
    lon: float = Query(..., description="Longitude (WGS84)", ge=-180, le=180),
    time: Optional[str] = Query(
        None,
        description="ISO 8601 datetime (default: now UTC). Example: 2026-03-08T14:00:00",
    ),
):
    """Compute shadow map for a location at a given time.

    Returns GeoJSON FeatureCollection of shadow polygons, plus sun position
    and summary stats.
    """
    # Parse time
    if time:
        try:
            dt = datetime.fromisoformat(time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, f"Invalid time format: {time}")
    else:
        dt = datetime.now(timezone.utc)

    # Check cache
    cache_key = _cache_key(lat, lon, dt)
    if cache_key in _cache:
        cached_time, cached_result = _cache[cache_key]
        if (time_module_time() - cached_time) < CACHE_TTL:
            return JSONResponse(cached_result)

    # Find or fetch DSM tile on demand
    dsm_path = _ensure_dsm(lat, lon)

    # Compute shadows (loads DSM at 1m upscaled resolution)
    result = compute_shadows_for_location(dsm_path, lat, lon, dt)

    # Convert shadow mask to GeoJSON using the upscaled grid's transform
    dsm_data, meta = load_dsm(dsm_path)
    geojson = _shadow_mask_to_geojson(result.shadow_mask, meta["transform"], meta["crs"])

    # Stats
    total = result.shadow_mask.size
    shadowed = int(np.sum(result.shadow_mask))
    sunny_pct = (total - shadowed) / total * 100

    response = {
        "sun": {
            "altitude": round(result.sun.altitude, 2),
            "azimuth": round(result.sun.azimuth, 2),
            "is_up": bool(result.sun.altitude > 0),
        },
        "shadows": geojson,
        "stats": {
            "total_cells": int(total),
            "shadowed_cells": shadowed,
            "sunny_percent": round(float(sunny_pct), 1),
            "pixel_size_m": float(result.pixel_size),
            "grid_shape": list(result.shadow_mask.shape),
        },
        "bounds": {
            "west": float(result.dsm_bounds[0]),
            "south": float(result.dsm_bounds[1]),
            "east": float(result.dsm_bounds[2]),
            "north": float(result.dsm_bounds[3]),
        },
        "dsm_file": dsm_path.name,
        "time": dt.isoformat(),
    }

    # Cache it
    _cache[cache_key] = (time_module_time(), response)

    return JSONResponse(response)


@app.get("/shadows/png")
async def get_shadow_png(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[str] = Query(None),
):
    """Return shadow mask as a transparent PNG for map overlay."""
    from io import BytesIO
    from PIL import Image

    # Parse time
    if time:
        try:
            dt = datetime.fromisoformat(time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, f"Invalid time format: {time}")
    else:
        dt = datetime.now(timezone.utc)

    dsm_path = _ensure_dsm(lat, lon)
    result = compute_shadows_for_location(dsm_path, lat, lon, dt)

    # Create RGBA image: shadow = semi-transparent dark, sun = fully transparent
    h, w = result.shadow_mask.shape
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[result.shadow_mask, 0] = 30   # R
    img[result.shadow_mask, 1] = 30   # G
    img[result.shadow_mask, 2] = 60   # B
    img[result.shadow_mask, 3] = 120  # A (semi-transparent)

    pil_img = Image.fromarray(img, "RGBA")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Bounds-West": str(result.dsm_bounds[0]),
            "X-Bounds-South": str(result.dsm_bounds[1]),
            "X-Bounds-East": str(result.dsm_bounds[2]),
            "X-Bounds-North": str(result.dsm_bounds[3]),
            "X-Sun-Altitude": str(round(result.sun.altitude, 2)),
            "X-Sun-Azimuth": str(round(result.sun.azimuth, 2)),
        },
    )


@app.get("/sun")
async def get_sun(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[str] = Query(None),
):
    """Get sun position only (no shadow computation)."""
    if time:
        try:
            dt = datetime.fromisoformat(time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, f"Invalid time format: {time}")
    else:
        dt = datetime.now(timezone.utc)

    sun = get_sun_position(lat, lon, dt)

    return {
        "altitude": round(sun.altitude, 2),
        "azimuth": round(sun.azimuth, 2),
        "is_up": bool(sun.altitude > 0),
        "time": dt.isoformat(),
    }


class BarLocation(BaseModel):
    id: str
    lat: float
    lon: float


class BarsStatusRequest(BaseModel):
    bars: list[BarLocation]
    time: str | None = None


def _parse_time(time_str: str | None) -> datetime:
    if time_str:
        dt = datetime.fromisoformat(time_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _ensure_dsm(lat: float, lon: float) -> Path:
    """Find cached DSM or fetch on demand from IGN WCS API."""
    dsm_path = _find_dsm(lat, lon)
    if dsm_path is not None:
        return dsm_path

    # On-demand fetch
    from fetch_dsm import fetch_dsm
    return fetch_dsm(lat, lon, radius=200.0)


@app.get("/bar-status")
async def get_bar_status(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    time: Optional[str] = Query(None),
):
    """Is this bar sunny? When does the status change?

    Returns current sun/shade status plus prediction of next transition.
    """
    dt = _parse_time(time)
    dsm_path = _ensure_dsm(lat, lon)

    status = predict_bar_status(dsm_path, lat, lon, dt)

    return {
        "is_sunny": status.is_sunny,
        "sun_altitude": round(status.sun_altitude, 1),
        "sun_azimuth": round(status.sun_azimuth, 1),
        "next_change": status.next_change,
        "next_change_type": status.next_change_type,
        "sunset": status.sunset,
        "time": dt.isoformat(),
    }


@app.post("/bars-status")
async def get_bars_status(req: BarsStatusRequest):
    """Batch query: sun/shade status for multiple bars.

    Optimized: loads DSM once, computes shadow per time step once,
    queries all bars against the same mask.
    """
    dt = _parse_time(req.time)

    if not req.bars:
        return {"bars": []}

    # Group bars by DSM tile (for now: one tile covers all)
    center_lat = req.bars[0].lat
    center_lon = req.bars[0].lon
    dsm_path = _ensure_dsm(center_lat, center_lon)

    dsm, meta = load_dsm(dsm_path)
    pixel_size = meta["pixel_size"]
    sun = get_sun_position(center_lat, center_lon, dt)

    results = []

    if sun.altitude <= 0:
        # Sun is down — all bars in shade
        for bar in req.bars:
            results.append({
                "id": bar.id,
                "is_sunny": False,
                "next_change": None,
                "next_change_type": None,
                "sunset": None,
            })
    else:
        # Compute current shadow mask once
        shadow_now = compute_shadows(dsm, sun.altitude, sun.azimuth, pixel_size)
        min_area_m2 = 25.0
        min_px = max(4, int(min_area_m2 / (pixel_size ** 2)))
        shadow_now = clean_shadow_mask(shadow_now, min_shadow_pixels=min_px, min_hole_pixels=min_px)

        # Query each bar's current status
        bar_status_now = {}
        for bar in req.bars:
            bar_status_now[bar.id] = query_point_shadow(dsm, meta, bar.lat, bar.lon, shadow_now)

        # Scan forward to find transitions and sunset
        from datetime import timedelta
        step = timedelta(minutes=15)
        max_steps = 32  # 8 hours

        next_change: dict[str, dict] = {}  # bar_id -> {time, type}
        sunset_time = None

        for i in range(1, max_steps + 1):
            future_dt = dt + step * i
            future_sun = get_sun_position(center_lat, center_lon, future_dt)

            if future_sun.altitude <= 0:
                sunset_time = future_dt.isoformat()
                break

            future_shadow = compute_shadows(dsm, future_sun.altitude, future_sun.azimuth, pixel_size)
            future_shadow = clean_shadow_mask(future_shadow, min_shadow_pixels=min_px, min_hole_pixels=min_px)

            for bar in req.bars:
                if bar.id in next_change:
                    continue  # already found this bar's transition
                is_shadowed = query_point_shadow(dsm, meta, bar.lat, bar.lon, future_shadow)
                if is_shadowed != bar_status_now[bar.id]:
                    next_change[bar.id] = {
                        "time": future_dt.isoformat(),
                        "type": "shade" if is_shadowed else "sun",
                    }

            # All bars resolved?
            if len(next_change) >= len(req.bars):
                # Still need sunset — keep scanning
                if sunset_time is not None:
                    break

        # Build response
        for bar in req.bars:
            change = next_change.get(bar.id)
            results.append({
                "id": bar.id,
                "is_sunny": not bar_status_now[bar.id],
                "next_change": change["time"] if change else None,
                "next_change_type": change["type"] if change else None,
                "sunset": sunset_time,
            })

    return {
        "bars": results,
        "sun": {
            "altitude": round(sun.altitude, 1),
            "azimuth": round(sun.azimuth, 1),
            "is_up": bool(sun.altitude > 0),
        },
        "time": dt.isoformat(),
    }


def time_module_time():
    """Wrapper for time.time() to avoid name collision with query param."""
    return time.time()
