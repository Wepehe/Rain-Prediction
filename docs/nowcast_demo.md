# Local Interactive Nowcast Demo

The Streamlit dashboard is a visualization layer for the frozen, validated Residual V1 operational API. It does not contain or duplicate forecast logic, train a model, tune a threshold, or calculate new scientific performance.

![Illustrative +60-minute dashboard from the bundled non-FINAL DEV fixture](assets/operational_nowcast_demo_60min.png)

The image is illustrative software-demo output from a non-FINAL DEV fixture, not additional validation evidence.

## Installation

Install the project with the demo and baseline dependencies:

```powershell
uv sync --extra baseline --extra demo
```

Launch from the repository root:

```powershell
uv run streamlit run app/nowcast_demo.py
```

## Accepted input

Choose either the bundled **Example historical case — demonstration only** or upload an operational NPZ containing:

- `history_rate`: `[10,128,128]` precipitation rates in mm/h;
- `timestamps`: ten chronological ISO-8601 timestamps at six-minute cadence;
- `validity_mask`: `[10,128,128]` validity values;
- optional scalar `metadata_json`, including grid coordinates when available.

The bundled case is a DEV-derived software regression fixture and is not from the consumed FINAL systems. Input validation and forecast generation go through `ontario_nowcast.operational.forecast`.

## Controls

- Generate a forecast once after choosing an input and CPU/CUDA device.
- Move the lead slider from +6 through +120 minutes, or use +30/+60/+90/+120 shortcuts.
- Play or pause the 20-frame forecast animation.
- Toggle the frozen operational `P ≥ 0.35` rain mask.
- Select grid x/y indices for point charts and a point summary.

Changing the lead, point, mask, or animation frame reads the cached result and does not rerun inference. Cache identity includes the radar-input hash, bundle version, and frozen configuration hash.

## Displayed products

The coordinated panels show latest observed radar, deterministic PySTEPS precipitation, raw Residual V1 occurrence probability, and Residual V1 corrected precipitation rate. Rate panels share fixed meteorological breakpoints; probabilities always use 0–100% without frame-dependent rescaling.

Point charts show PySTEPS and corrected rate plus raw probability. The first threshold-crossing lead is only an operational decision summary, not a calibrated onset interval.

## Export

The dashboard can download:

- the unchanged operational `forecast.npz` format;
- `forecast_summary.json`;
- a static four-panel PNG for the selected lead.

## Example workflow

1. Start the dashboard.
2. Keep the bundled demonstration case selected.
3. Select CPU and generate the forecast.
4. Choose +60 minutes and optionally enable the operational mask.
5. Move the x/y point and inspect its two time-series charts.
6. Download the operational product, JSON summary, or current PNG.

## Limitations

- Radar-blind initiation remains unreliable.
- Displacement and decay correction remain weaker than growth correction.
- Raw probabilities are useful but not perfectly calibrated.
- Validation used selected weather-event windows rather than continuous climatology.
- PySTEPS remains an essential input.
- The 0.35 threshold is frozen and cannot be changed in the dashboard.
