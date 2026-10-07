"""Pure visualization helpers for the frozen operational forecast product."""
from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .inference import OCCURRENCE_THRESHOLD, OperationalForecast

LEADS_MINUTES = tuple(range(6, 121, 6))
RATE_BREAKS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0)
RATE_COLORS = (
    "#d9f0ff", "#78c7ff", "#32b56b", "#f1dc45",
    "#f39a32", "#e34a33", "#9d2389", "#53145f",
)


@dataclass(frozen=True)
class ForecastProduct:
    timestamps: np.ndarray
    pysteps_rate_mm_h: np.ndarray
    residual_probability: np.ndarray
    residual_wet_mask: np.ndarray
    residual_rate_mm_h: np.ndarray
    validity_mask: np.ndarray
    metadata: dict[str, Any]


def lead_to_index(lead_minutes: int) -> int:
    if lead_minutes not in LEADS_MINUTES:
        raise ValueError("lead must be one of +6, +12, ..., +120 minutes")
    return lead_minutes // 6 - 1


def load_forecast_npz(path_or_file: str | Path | BytesIO) -> ForecastProduct:
    with np.load(path_or_file, allow_pickle=False) as data:
        required = {
            "timestamps", "pysteps_rate_mm_h", "residual_probability",
            "residual_wet_mask", "residual_rate_mm_h", "validity_mask", "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"forecast product is missing arrays: {sorted(missing)}")
        product = ForecastProduct(
            timestamps=data["timestamps"].copy(),
            pysteps_rate_mm_h=data["pysteps_rate_mm_h"].astype(np.float32),
            residual_probability=data["residual_probability"].astype(np.float32),
            residual_wet_mask=data["residual_wet_mask"].astype(bool),
            residual_rate_mm_h=data["residual_rate_mm_h"].astype(np.float32),
            validity_mask=data["validity_mask"].astype(bool),
            metadata=json.loads(str(data["metadata_json"])),
        )
    for name in ("pysteps_rate_mm_h", "residual_probability", "residual_rate_mm_h"):
        if getattr(product, name).shape != (20, 128, 128):
            raise ValueError(f"{name} must have shape (20, 128, 128)")
    if not np.all((product.residual_probability >= 0) & (product.residual_probability <= 1)):
        raise ValueError("probability field is outside [0, 1]")
    return product


def point_series(product: ForecastProduct, y: int, x: int) -> dict[str, np.ndarray]:
    if not (0 <= y < 128 and 0 <= x < 128):
        raise ValueError("point indices must be between 0 and 127")
    return {
        "lead_minutes": np.asarray(LEADS_MINUTES),
        "pysteps_rate_mm_h": product.pysteps_rate_mm_h[:, y, x],
        "residual_rate_mm_h": product.residual_rate_mm_h[:, y, x],
        "residual_probability": product.residual_probability[:, y, x],
    }


def point_summary(product: ForecastProduct, y: int, x: int) -> dict[str, Any]:
    series = point_series(product, y, x)
    probability = series["residual_probability"]
    rate = series["residual_rate_mm_h"]
    wet = probability >= OCCURRENCE_THRESHOLD
    maximum_index = int(np.argmax(rate))
    return {
        "first_wet_lead_minutes": int(series["lead_minutes"][np.argmax(wet)]) if wet.any() else None,
        "maximum_probability": float(probability.max()),
        "maximum_corrected_rate_mm_h": float(rate[maximum_index]),
        "maximum_rate_lead_minutes": int(series["lead_minutes"][maximum_index]),
        "wet_forecast_frames": int(wet.sum()),
    }


def area_summary(product: ForecastProduct, lead_minutes: int) -> dict[str, float]:
    index = lead_to_index(lead_minutes)
    probability = product.residual_probability[index]
    rate = product.residual_rate_mm_h[index]
    return {
        "mean_probability": float(probability.mean()),
        "wet_area_fraction": float((probability >= OCCURRENCE_THRESHOLD).mean()),
        "mean_corrected_rate_mm_h": float(rate.mean()),
        "maximum_corrected_rate_mm_h": float(rate.max()),
    }


def _extent(product: ForecastProduct) -> tuple[float, float, float, float] | None:
    grid = product.metadata.get("grid", {})
    if "x" in grid and "y" in grid and len(grid["x"]) == 128 and len(grid["y"]) == 128:
        x, y = np.asarray(grid["x"]), np.asarray(grid["y"])
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return None


def dashboard_figure(
    history_rate: np.ndarray,
    product: ForecastProduct,
    lead_minutes: int,
    *,
    show_mask: bool = False,
    selected_point: tuple[int, int] | None = None,
):
    index = lead_to_index(lead_minutes)
    history = np.asarray(history_rate)
    if history.shape != (10, 128, 128):
        raise ValueError("history_rate must have shape (10, 128, 128)")
    cmap = ListedColormap(RATE_COLORS)
    norm = BoundaryNorm(RATE_BREAKS, cmap.N, extend="max")
    extent = _extent(product)
    panels = (
        (history[-1], "Observed radar — t0", "rate"),
        (product.pysteps_rate_mm_h[index], "PySTEPS precipitation rate", "rate"),
        (product.residual_probability[index], "P(precipitation > 0.1 mm/h)", "probability"),
        (product.residual_rate_mm_h[index], "Residual V1 corrected rate", "rate"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    rate_image = probability_image = None
    for axis, (field, title, kind) in zip(axes.flat, panels, strict=True):
        kwargs = {"origin": "lower", "extent": extent, "interpolation": "nearest"}
        if kind == "rate":
            image = axis.imshow(field, cmap=cmap, norm=norm, **kwargs)
            rate_image = image
        else:
            image = axis.imshow(field, cmap="viridis", vmin=0, vmax=1, **kwargs)
            probability_image = image
        axis.set_title(title)
        axis.set_xlabel("Projected x" if extent else "Grid x")
        axis.set_ylabel("Projected y" if extent else "Grid y")
        if show_mask and kind != "probability":
            axis.contour(product.residual_probability[index] >= OCCURRENCE_THRESHOLD,
                         levels=[0.5], colors=["white"], linewidths=0.7,
                         origin="lower", extent=extent)
        if selected_point is not None:
            y, x = selected_point
            px, py = (x, y)
            if extent:
                grid = product.metadata["grid"]
                px, py = grid["x"][x], grid["y"][y]
            axis.plot(px, py, marker="+", color="magenta", markersize=10, markeredgewidth=2)
    fig.colorbar(rate_image, ax=[axes[0, 0], axes[0, 1], axes[1, 1]], label="mm/h", shrink=0.8)
    fig.colorbar(probability_image, ax=axes[1, 0], label="Probability (35% operational threshold)")
    fig.suptitle(f"Southern Ontario nowcast — +{lead_minutes} min", fontsize=15)
    return fig


def figure_png_bytes(figure) -> bytes:
    target = BytesIO()
    figure.savefig(target, format="png", dpi=150, bbox_inches="tight")
    return target.getvalue()


def product_from_operational(value: OperationalForecast) -> ForecastProduct:
    """Convert an in-memory operational result without changing its values."""
    return ForecastProduct(
        timestamps=np.asarray([x.isoformat() for x in value.timestamps]),
        pysteps_rate_mm_h=value.pysteps_rate_mm_h,
        residual_probability=value.residual_probability,
        residual_wet_mask=value.residual_wet_mask,
        residual_rate_mm_h=value.residual_rate_mm_h,
        validity_mask=value.validity_mask,
        metadata=value.metadata,
    )
