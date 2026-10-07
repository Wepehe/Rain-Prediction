"""Public-facing live Southern Ontario precipitation nowcast."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from ontario_nowcast.operational import forecast, load_operational_model
from ontario_nowcast.operational.live_mrms import (
    DEFAULT_CACHE,
    DOMAIN_BBOX_WGS84,
    LiveDataUnavailable,
    forecast_cache_identity,
    geocode_location,
    location_in_domain,
    prepare_live_history,
)
from ontario_nowcast.operational.live_product import freshness_minutes, location_summary

LOGGER = logging.getLogger("live_nowcast")
BUNDLE = Path(os.environ.get("NOWCAST_BUNDLE_DIR", "artifacts/operational/residual_v1"))
STALE_MINUTES = int(os.environ.get("NOWCAST_STALE_WARNING_MINUTES", "20"))


@st.cache_resource(show_spinner=False)
def model(device: str):
    return load_operational_model(BUNDLE, device=device)


@st.cache_data(ttl=300, show_spinner=False)
def resolve_place(query: str):
    return geocode_location(query)["candidates"]


@st.cache_data(ttl=300, max_entries=32, show_spinner=False)
def live_history(latitude: float, longitude: float, refresh_token: int):
    del refresh_token
    return prepare_live_history(latitude, longitude, cache_dir=DEFAULT_CACHE)


@st.cache_data(ttl=600, max_entries=16, show_spinner=False)
def cached_forecast(history, timestamps, validity, grid_metadata, device: str,
                    cache_identity: str, bundle_identity: str):
    del cache_identity, bundle_identity
    return forecast(model(device), history, timestamps, validity, grid_metadata, diagnostics=False)


def run_live(latitude: float, longitude: float, device: str, refresh_token: int,
             manifest: dict):
    live = live_history(latitude, longitude, refresh_token)
    identity = forecast_cache_identity(live.tile, live.issue_time, manifest["model_version"],
                                       manifest["config_sha256"])
    grid_metadata = {
        "grid": {"x": live.tile.x.tolist(), "y": live.tile.y.tolist(),
                 "longitude": live.tile.longitude.tolist(), "latitude": live.tile.latitude.tolist()},
        "radar_issue_time": live.issue_time.isoformat(),
        "source_keys": list(live.source_keys),
        "missing_timestamps": [value.isoformat() for value in live.missing_timestamps],
    }
    result = cached_forecast(live.history_rate, live.timestamps, live.validity_mask,
                             grid_metadata, device, identity,
                             f'{manifest["bundle_version"]}:{manifest["config_sha256"]}')
    user_metadata = {
        **result.metadata,
        "selected_location": {"latitude": latitude, "longitude": longitude,
                              "x_index": live.tile.point_x_index, "y_index": live.tile.point_y_index},
        "latest_rate_at_location_mm_h": float(live.history_rate[-1, live.tile.point_y_index,
                                                                  live.tile.point_x_index]),
    }
    return live, replace(result, metadata=user_metadata), identity


def rate_map(live, result, lead: int, layer: str):
    if lead == 0:
        field, label, vmin, vmax = live.history_rate[-1], "Observed radar (mm/h)", 0, 20
    else:
        index = lead // 6 - 1
        if layer == "Residual V1 precipitation rate":
            field, label, vmin, vmax = result.residual_rate_mm_h[index], "Forecast rate (mm/h)", 0, 20
        elif layer == "Rain probability":
            field, label, vmin, vmax = result.residual_probability[index] * 100, "Rain probability (%)", 0, 100
        elif layer == "Deterministic PySTEPS":
            field, label, vmin, vmax = result.pysteps_rate_mm_h[index], "PySTEPS rate (mm/h)", 0, 20
        else:
            field, label, vmin, vmax = (result.residual_probability[index] >= 0.35).astype(float), "Operational rain mask", 0, 1
    fig, axis = plt.subplots(figsize=(8, 6))
    extent = [float(live.tile.longitude.min()), float(live.tile.longitude.max()),
              float(live.tile.latitude.min()), float(live.tile.latitude.max())]
    image = axis.imshow(field, origin="lower", extent=extent, cmap="turbo" if vmax > 1 else "Blues",
                        vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    axis.plot(live.tile.requested_longitude, live.tile.requested_latitude, "m+", ms=14, mew=3)
    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"{label} — {'Now' if lead == 0 else f'+{lead} min'}")
    fig.colorbar(image, ax=axis, label=label)
    return fig


def main() -> None:
    st.set_page_config(page_title="Southern Ontario Live Nowcast", page_icon="🌧️", layout="wide")
    st.title("Southern Ontario 0–2 h Precipitation Nowcast")
    st.caption("Experimental short-range precipitation guidance — not an official warning service")
    st.subheader("Where do you want a forecast?")
    query = st.text_input("City, address, or place", placeholder="Toronto, Waterloo, Hamilton…")
    selected = None
    if query:
        try:
            candidates = resolve_place(query)
            names = [item["name"] for item in candidates]
            selected = candidates[names.index(st.selectbox("Resolved place", names))]
        except (requests.RequestException, LookupError, ValueError, RuntimeError) as exc:
            st.warning(f"Location search is temporarily unavailable: {exc}. Use manual coordinates below.")
    with st.expander("Advanced: manual location or operational input"):
        manual = st.checkbox("Use manual latitude/longitude")
        cols = st.columns(2)
        manual_lat = cols[0].number_input("Latitude", 40.0, 48.0, 43.6532, format="%.5f")
        manual_lon = cols[1].number_input("Longitude", -87.0, -74.0, -79.3832, format="%.5f")
        st.caption("Historical NPZ upload remains available in `app/nowcast_demo.py`.")
    if manual:
        selected = {"name": "Manually selected location", "latitude": manual_lat, "longitude": manual_lon}
    if selected:
        latitude, longitude = float(selected["latitude"]), float(selected["longitude"])
        st.success(f'Resolved: {selected["name"]} ({latitude:.4f}, {longitude:.4f})')
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=7)
        if not location_in_domain(latitude, longitude):
            st.error("This location is outside the current Residual V1 forecast domain.")
            st.caption(f"Supported bounding region: {DOMAIN_BBOX_WGS84}")
            return
    else:
        st.info("Search for a Southern Ontario location, or use manual coordinates if search is unavailable.")
        return
    device = "cuda" if os.environ.get("NOWCAST_DEVICE") == "cuda" else "cpu"
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = 0
    actions = st.columns([2, 1])
    generate = actions[0].button("Generate 2-hour nowcast", type="primary", use_container_width=True)
    refresh = actions[1].button("Refresh latest radar", use_container_width=True)
    if refresh:
        st.session_state.refresh_token += 1
        generate = True
    request_key = f"{latitude:.5f}:{longitude:.5f}:{st.session_state.refresh_token}"
    if generate:
        try:
            with st.status("Finding latest radar…", expanded=True) as status:
                st.write("Preparing the last 60 minutes…")
                st.write("Estimating storm motion and generating the 2-hour forecast…")
                live, result, identity = run_live(latitude, longitude, device,
                                                  st.session_state.refresh_token, manifest)
                st.write("Preparing maps…")
                status.update(label="Nowcast ready", state="complete", expanded=False)
            previous = st.session_state.get("issue_time")
            st.session_state.update(live=live, result=result, identity=identity,
                                    request_key=request_key, issue_time=live.issue_time)
            if refresh and previous == live.issue_time:
                st.info("No newer radar scan is available yet.")
        except (ValueError, LiveDataUnavailable) as exc:
            st.error(str(exc)); return
        except Exception:
            LOGGER.exception("Live forecast failed")
            st.error("Live radar or forecast service is temporarily unavailable. Please try again shortly.")
            return
    if st.session_state.get("request_key") != request_key:
        st.info("Click **Generate 2-hour nowcast** to retrieve the latest radar and build the forecast.")
        return
    live, result = st.session_state.live, st.session_state.result
    age = freshness_minutes(live.issue_time, datetime.now(UTC))
    info = st.columns(3)
    info[0].metric("Radar observations through", live.issue_time.strftime("%Y-%m-%d %H:%M UTC"))
    info[1].metric("Latest radar age", f"{age:.0f} min")
    info[2].metric("Forecast generated", datetime.now(UTC).strftime("%H:%M UTC"))
    if age > STALE_MINUTES:
        st.warning(f"Radar data delayed — latest observation is {age:.0f} minutes old.")
    y, x = live.tile.point_y_index, live.tile.point_x_index
    summary = location_summary(result, y, x)
    st.subheader(summary["status"])
    columns = st.columns(3)
    columns[0].metric("Next 30 min — max probability", f'{summary["next_30_probability_max"]:.0%}',
                      help=f'Maximum forecast rate {summary["next_30_rate_max_mm_h"]:.1f} mm/h')
    columns[1].metric("Next 60 min — max probability", f'{summary["next_60_probability_max"]:.0%}',
                      help=f'Maximum forecast rate {summary["next_60_rate_max_mm_h"]:.1f} mm/h')
    columns[2].metric("Next 2 h — rain likely", "Yes" if summary["rain_likely"] else "No",
                      help=f'First operational wet lead: {summary["first_wet_lead_minutes"]}')
    leads = np.arange(6, 121, 6)
    chart = pd.DataFrame({"Residual V1 probability (%)": result.residual_probability[:, y, x] * 100,
                          "Residual V1 rate (mm/h)": result.residual_rate_mm_h[:, y, x]}, index=leads)
    st.subheader("Forecast at your location")
    st.line_chart(chart, x_label="Minutes from now", y_label="Forecast value")
    with st.expander("Compare with deterministic baseline"):
        baseline = pd.DataFrame({"PySTEPS rate (mm/h)": result.pysteps_rate_mm_h[:, y, x],
                                 "Residual V1 rate (mm/h)": result.residual_rate_mm_h[:, y, x]}, index=leads)
        st.line_chart(baseline)
    st.subheader("Surrounding precipitation")
    lead = st.select_slider("Forecast time", options=[0, *range(6, 121, 6)], value=30,
                            format_func=lambda value: "Now" if value == 0 else f"+{value} min")
    layer = st.selectbox("Map layer", ("Residual V1 precipitation rate", "Rain probability",
                                       "Operational P≥0.35 mask", "Deterministic PySTEPS"))
    figure = rate_map(live, result, lead, layer)
    st.pyplot(figure, use_container_width=True); plt.close(figure)
    with st.expander("Advanced / About this forecast"):
        st.write(f'Model `{result.metadata["model_version"]}`; frozen threshold `0.35`; cache `{st.session_state.identity[:12]}`.')
        st.write("Residual V1 corrects a deterministic radar-extrapolation forecast. Raw probabilities are not perfectly calibrated. Reliable radar-blind initiation has not been demonstrated.")
    st.warning("This is an experimental short-range precipitation nowcast validated on selected Southern Ontario weather events. It is not an official weather warning service. Use Environment and Climate Change Canada warnings for safety-critical decisions.")


if __name__ == "__main__":
    main()
