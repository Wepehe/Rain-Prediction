"""Simple object linking for independent initiation-event catalogues."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening, label

from .detect import detect_event_masks
from .geometry import SOUTHERN_ONTARIO_POLYGON_WGS84, polygon_mask


@dataclass(frozen=True)
class LinkedInitiation:
    event_id: str
    representative_anchor_time: str
    start_time: str
    end_time: str
    centre_lat: float
    centre_lon: float
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    component_count: int
    pixel_count: int
    future_max_mm_hr: float
    apparent_initiation: bool
    advective_entry_like: bool
    touches_domain_boundary: bool


@dataclass
class _Component:
    anchor: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    y_centre: float
    x_centre: float
    centre_lat: float
    centre_lon: float
    pixel_count: int
    future_max: float
    advective_entry_like: bool
    touches_domain_boundary: bool


def _ensure_2d_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != lon.shape:
        raise ValueError("latitude and longitude must either be matching 2D arrays or 1D axes")
    return lat, lon


def _bbox_iou(a: _Component, b: _Component) -> float:
    height = max(0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min) + 1)
    width = max(0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min) + 1)
    intersection = height * width
    if intersection == 0:
        return 0.0
    area_a = (a.y_max - a.y_min + 1) * (a.x_max - a.x_min + 1)
    area_b = (b.y_max - b.y_min + 1) * (b.x_max - b.x_min + 1)
    return intersection / (area_a + area_b - intersection)


def _morphology(mask: np.ndarray, opening_pixels: int, closing_pixels: int) -> np.ndarray:
    result = mask
    if opening_pixels > 0:
        structure = np.ones((opening_pixels * 2 + 1, opening_pixels * 2 + 1), dtype=bool)
        result = binary_opening(result, structure=structure)
    if closing_pixels > 0:
        structure = np.ones((closing_pixels * 2 + 1, closing_pixels * 2 + 1), dtype=bool)
        result = binary_closing(result, structure=structure)
    return result


def _boundary(mask: np.ndarray, buffer_pixels: int) -> np.ndarray:
    if buffer_pixels < 1:
        return mask & ~binary_erosion(mask)
    structure = np.ones((buffer_pixels * 2 + 1, buffer_pixels * 2 + 1), dtype=bool)
    return mask & ~binary_erosion(mask, structure=structure)


def _components(
    rates: np.ndarray,
    times: pd.DatetimeIndex,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    history_steps: int,
    horizon_steps: int,
    rain_threshold: float,
    strong_threshold: float,
    max_missing_fraction: float,
    minimum_pixels: int,
    opening_pixels: int,
    closing_pixels: int,
    current_rain_distance_pixels: int,
    boundary_buffer_pixels: int,
    polygon: list[list[float]] | tuple[tuple[float, float], ...],
) -> list[_Component]:
    del times
    lat, lon = _ensure_2d_coordinates(latitude, longitude)
    domain = polygon_mask(lon, lat, polygon)
    domain_boundary = _boundary(domain, boundary_buffer_pixels)
    masks = detect_event_masks(
        rates,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        rain_threshold=rain_threshold,
        strong_threshold=strong_threshold,
        max_missing_fraction=max_missing_fraction,
    )
    structure = np.ones((3, 3), dtype=int)
    components: list[_Component] = []
    for anchor in range(history_steps - 1, len(rates) - horizon_steps):
        onset = _morphology(
            masks.clear_to_rain[anchor] & domain,
            opening_pixels=opening_pixels,
            closing_pixels=closing_pixels,
        )
        current_wet = np.isfinite(rates[anchor]) & (rates[anchor] >= rain_threshold)
        nearby_current_wet = binary_dilation(
            current_wet,
            structure=np.ones(
                (current_rain_distance_pixels * 2 + 1, current_rain_distance_pixels * 2 + 1),
                dtype=bool,
            ),
        )
        labels, count = label(onset, structure=structure)
        for component_id in range(1, count + 1):
            ys, xs = np.where(labels == component_id)
            if len(ys) < minimum_pixels:
                continue
            component_mask = labels == component_id
            future = rates[anchor + 1 : anchor + horizon_steps + 1, ys, xs]
            touches_boundary = bool(np.any(component_mask & domain_boundary))
            near_existing_rain = bool(np.any(component_mask & nearby_current_wet))
            components.append(
                _Component(
                    anchor=anchor,
                    y_min=int(ys.min()),
                    y_max=int(ys.max()),
                    x_min=int(xs.min()),
                    x_max=int(xs.max()),
                    y_centre=float(np.mean(ys)),
                    x_centre=float(np.mean(xs)),
                    centre_lat=float(np.mean(lat[ys, xs])),
                    centre_lon=float(np.mean(lon[ys, xs])),
                    pixel_count=len(ys),
                    future_max=float(np.nanmax(future)),
                    advective_entry_like=touches_boundary or near_existing_rain,
                    touches_domain_boundary=touches_boundary,
                )
            )
    return components


def _link_components(
    components: list[_Component],
    *,
    max_gap_steps: int,
    max_distance_pixels: float,
    min_iou: float,
) -> list[list[_Component]]:
    tracks: list[list[_Component]] = []
    for component in sorted(components, key=lambda item: item.anchor):
        best_track = None
        best_score = -np.inf
        for track in tracks:
            previous = track[-1]
            gap = component.anchor - previous.anchor
            if gap < 0 or gap > max_gap_steps:
                continue
            distance = np.hypot(component.y_centre - previous.y_centre, component.x_centre - previous.x_centre)
            iou = _bbox_iou(component, previous)
            if iou < min_iou and distance > max_distance_pixels:
                continue
            score = iou - distance / max(max_distance_pixels, 1.0)
            if score > best_score:
                best_track = track
                best_score = score
        if best_track is None:
            tracks.append([component])
        else:
            best_track.append(component)
    return tracks


def build_independent_initiation_catalog(
    rates: np.ndarray,
    times: pd.DatetimeIndex,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    interval_minutes: int = 6,
    resolution_km: float = 2.0,
    history_minutes: int = 30,
    horizon_minutes: int = 120,
    rain_threshold: float = 0.1,
    strong_threshold: float = 5.0,
    max_missing_fraction: float = 0.1,
    minimum_pixels: int = 12,
    opening_pixels: int = 1,
    closing_pixels: int = 1,
    link_max_gap_minutes: int = 24,
    link_max_centroid_distance_km: float = 35.0,
    link_min_bbox_iou: float = 0.05,
    polygon: list[list[float]] | tuple[tuple[float, float], ...] = SOUTHERN_ONTARIO_POLYGON_WGS84,
) -> pd.DataFrame:
    """Build one row per linked initiation object for event-level splitting."""
    history_steps = max(1, history_minutes // interval_minutes)
    horizon_steps = max(1, horizon_minutes // interval_minutes)
    components = _components(
        np.asarray(rates, dtype=np.float32),
        times,
        latitude,
        longitude,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        rain_threshold=rain_threshold,
        strong_threshold=strong_threshold,
        max_missing_fraction=max_missing_fraction,
        minimum_pixels=minimum_pixels,
        opening_pixels=opening_pixels,
        closing_pixels=closing_pixels,
        current_rain_distance_pixels=max(1, round(16.0 / resolution_km)),
        boundary_buffer_pixels=4,
        polygon=polygon,
    )
    tracks = _link_components(
        components,
        max_gap_steps=max(1, link_max_gap_minutes // interval_minutes),
        max_distance_pixels=link_max_centroid_distance_km / resolution_km,
        min_iou=link_min_bbox_iou,
    )
    linked = []
    for index, track in enumerate(tracks, start=1):
        representative = max(track, key=lambda item: (item.future_max, item.pixel_count))
        linked.append(
            LinkedInitiation(
                event_id=f"initiation_track_{index:04d}",
                representative_anchor_time=times[representative.anchor].isoformat(),
                start_time=times[min(item.anchor for item in track)].isoformat(),
                end_time=times[max(item.anchor for item in track)].isoformat(),
                centre_lat=representative.centre_lat,
                centre_lon=representative.centre_lon,
                y_min=min(item.y_min for item in track),
                y_max=max(item.y_max for item in track),
                x_min=min(item.x_min for item in track),
                x_max=max(item.x_max for item in track),
                component_count=len(track),
                pixel_count=sum(item.pixel_count for item in track),
                future_max_mm_hr=max(item.future_max for item in track),
                apparent_initiation=not any(item.advective_entry_like for item in track),
                advective_entry_like=any(item.advective_entry_like for item in track),
                touches_domain_boundary=any(item.touches_domain_boundary for item in track),
            )
        )
    return pd.DataFrame(asdict(item) for item in linked)
