"""Local Streamlit dashboard for the validated Residual V1 operational product."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ontario_nowcast.operational import forecast, load_operational_model
from ontario_nowcast.operational.visualization import (
    area_summary,
    dashboard_figure,
    figure_png_bytes,
    point_series,
    point_summary,
    product_from_operational,
)

LOGGER = logging.getLogger("nowcast_demo")
EXAMPLE = Path("tests/fixtures/operational_residual_v1/dev_input.npz")
BUNDLE = Path("artifacts/operational/residual_v1")
THRESHOLD = 0.35


def _read_input(payload: bytes):
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        required = {"history_rate", "timestamps", "validity_mask"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Input is missing arrays: {sorted(missing)}")
        history = data["history_rate"].astype(np.float32)
        timestamps = data["timestamps"].astype(str).tolist()
        validity = data["validity_mask"].astype(bool)
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
    return history, timestamps, validity, metadata


@st.cache_resource(show_spinner=False)
def _model(device: str):
    return load_operational_model(BUNDLE, device=device)


@st.cache_data(show_spinner=False, max_entries=4)
def _forecast(payload: bytes, input_hash: str, device: str, bundle_identity: str):
    del input_hash, bundle_identity  # included solely in the cache key
    history, timestamps, validity, metadata = _read_input(payload)
    result = forecast(_model(device), history, timestamps, validity, metadata, diagnostics=True)
    return history, result


def _point_charts(product, y: int, x: int) -> None:
    series = point_series(product, y, x)
    rate_fig, rate_ax = plt.subplots(figsize=(8, 3))
    rate_ax.plot(series["lead_minutes"], series["pysteps_rate_mm_h"], label="PySTEPS")
    rate_ax.plot(series["lead_minutes"], series["residual_rate_mm_h"], label="Residual V1")
    rate_ax.set(xlabel="Lead (minutes)", ylabel="Rate (mm/h)")
    rate_ax.legend(); rate_ax.grid(alpha=0.25)
    st.pyplot(rate_fig); plt.close(rate_fig)
    probability_fig, probability_ax = plt.subplots(figsize=(8, 2.7))
    probability_ax.plot(series["lead_minutes"], series["residual_probability"] * 100)
    probability_ax.axhline(35, color="tab:red", linestyle="--", label="Frozen threshold")
    probability_ax.set(xlabel="Lead (minutes)", ylabel="Probability (%)", ylim=(0, 100))
    probability_ax.legend(); probability_ax.grid(alpha=0.25)
    st.pyplot(probability_fig); plt.close(probability_fig)
    st.caption("Probability is raw model output and is not perfectly calibrated.")


def main() -> None:
    st.set_page_config(page_title="Southern Ontario Nowcast", layout="wide")
    st.title("Southern Ontario 0–2 h Precipitation Nowcast")
    st.caption("Validated PySTEPS + Residual U-Net V1")
    with st.sidebar:
        st.header("Forecast input")
        mode = st.radio("Source", ("Example historical case — demonstration only", "Upload operational NPZ input"))
        if mode.startswith("Example"):
            payload = EXAMPLE.read_bytes() if EXAMPLE.is_file() else None
            st.info("Non-FINAL DEV software fixture. Not new validation evidence.")
        else:
            uploaded = st.file_uploader("Operational radar input", type=("npz",))
            payload = uploaded.getvalue() if uploaded else None
        device_options = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                device_options.append("cuda")
        except ImportError:
            pass
        device = st.selectbox("Device", device_options)
        run = st.button("Generate forecast", type="primary", disabled=payload is None)
    if payload is None:
        st.info("Choose the bundled example or upload a valid ten-frame operational NPZ.")
        return
    input_hash = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    bundle_identity = f'{manifest["bundle_version"]}:{manifest["config_sha256"]}'
    key = f"{input_hash}:{device}:{bundle_identity}"
    if run or st.session_state.get("forecast_key") == key:
        try:
            with st.status("Preparing forecast", expanded=True) as status:
                st.write("Validating radar input…")
                st.write("Estimating motion, running PySTEPS, and applying Residual V1…")
                history, result = _forecast(payload, input_hash, device, bundle_identity)
                st.write("Preparing visualization…")
                status.update(label="Forecast ready", state="complete", expanded=False)
            st.session_state.forecast_key = key
            st.session_state.history = history
            st.session_state.result = result
        except Exception as exc:
            LOGGER.exception("Forecast generation failed")
            st.error(f"Forecast could not be generated: {exc}")
            return
    if st.session_state.get("forecast_key") != key:
        st.info("Select **Generate forecast**. A new radar history is the only action that reruns inference.")
        return
    history, result = st.session_state.history, st.session_state.result
    product = product_from_operational(result)
    timing = result.metadata["timing_seconds"]
    header = st.columns(4)
    header[0].metric("Initialization", result.metadata["initialization_timestamp"])
    header[1].metric("Model", result.metadata["model_version"])
    header[2].metric("Operational threshold", "35%")
    header[3].metric(f"Generation ({device})", f'{timing["total"]:.2f} s')

    if "lead" not in st.session_state:
        st.session_state.lead = 60
    quick = st.columns([2, 1, 1, 1, 1, 1])
    quick[0].slider("Forecast lead", 6, 120, key="lead", step=6)
    for column, value in zip(quick[1:], (30, 60, 90, 120), strict=False):
        if column.button(f"+{value} min"):
            st.session_state.lead = value
            st.rerun()
    play_column, pause_column, mask_column = st.columns([1, 1, 3])
    if play_column.button("▶ Play forecast"):
        st.session_state.playing = True
    if pause_column.button("⏸ Pause"):
        st.session_state.playing = False
    show_mask = mask_column.toggle("Show operational rain mask (P ≥ 0.35)")

    point_columns = st.columns(2)
    y = point_columns[0].number_input("Point grid y", 0, 127, 64)
    x = point_columns[1].number_input("Point grid x", 0, 127, 64)
    figure = dashboard_figure(history, product, st.session_state.lead,
                              show_mask=show_mask, selected_point=(int(y), int(x)))
    st.pyplot(figure, use_container_width=True)
    png = figure_png_bytes(figure); plt.close(figure)
    st.download_button("Download current panels (PNG)", png,
                       file_name=f"nowcast_plus_{st.session_state.lead:03d}min.png", mime="image/png")

    summary = area_summary(product, st.session_state.lead)
    area = st.columns(4)
    area[0].metric("Mean rain probability", f'{summary["mean_probability"]:.1%}')
    area[1].metric("Wet-area fraction", f'{summary["wet_area_fraction"]:.1%}')
    area[2].metric("Mean corrected rate", f'{summary["mean_corrected_rate_mm_h"]:.2f} mm/h')
    area[3].metric("Maximum corrected rate", f'{summary["maximum_corrected_rate_mm_h"]:.2f} mm/h')

    st.subheader(f"Point forecast — grid ({int(x)}, {int(y)})")
    _point_charts(product, int(y), int(x))
    point = point_summary(product, int(y), int(x))
    st.json(point)

    export = st.columns(2)
    forecast_buffer = BytesIO()
    np.savez_compressed(forecast_buffer,
        timestamps=np.asarray([t.isoformat() for t in result.timestamps]),
        pysteps_rate_mm_h=result.pysteps_rate_mm_h,
        residual_probability=result.residual_probability,
        residual_wet_mask=result.residual_wet_mask,
        residual_rate_mm_h=result.residual_rate_mm_h,
        validity_mask=result.validity_mask,
        motion_u=result.motion_u, motion_v=result.motion_v,
        metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True)))
    export[0].download_button("Download forecast.npz", forecast_buffer.getvalue(), "forecast.npz")
    export[1].download_button("Download forecast_summary.json",
        json.dumps(result.summary(), indent=2), "forecast_summary.json", mime="application/json")

    with st.expander("About this forecast"):
        st.write("Deterministic PySTEPS supplies motion/extrapolation. Residual V1 learns corrections to precipitation evolution and was independently validated on weather systems. It was strongest at correcting growth underprediction. Reliable radar-blind initiation has not been demonstrated.")
    with st.expander("Independent validation"):
        st.table({"Metric": ["F1", "FSS18", "Brier", "Rate MAE"],
                  "PySTEPS": [0.5644, 0.7007, 0.2245, 0.5468],
                  "Residual V1": [0.6810, 0.7842, 0.1117, 0.5018]})
        st.caption("12-system independent FINAL event corpus; F1 improved in 12/12 systems. This was an event-focused evaluation, not continuous climatological validation.")
    st.warning("Limitations: radar-blind initiation remains unreliable; displacement and decay correction remain weaker than growth correction; raw probabilities are not perfectly calibrated; validation used selected event windows; PySTEPS is essential; threshold 0.35 is frozen.")

    if st.session_state.get("playing"):
        lead_to_next = st.session_state.lead + 6
        if lead_to_next > 120:
            lead_to_next = 6
        time.sleep(0.45)
        st.session_state.lead = lead_to_next
        st.rerun()


if __name__ == "__main__":
    main()
