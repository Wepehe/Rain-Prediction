---
title: Southern Ontario Precipitation Nowcast
emoji: 🌧️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Southern Ontario precipitation nowcast

An experimental live 0–2 hour precipitation-nowcasting application using deterministic PySTEPS and the independently validated Residual U-Net V1.

## Run the public application locally

With Docker installed:

```powershell
git clone https://github.com/Wepehe/Rain-Prediction.git
cd Rain-Prediction
docker build -t ontario-nowcast-v1 .
docker run --rm -p 7860:7860 --memory=4g ontario-nowcast-v1
```

Open `http://localhost:7860`. Search for a Southern Ontario location and select **Generate 2-hour nowcast**. The application downloads current NOAA MRMS radar observations and runs the frozen operational pipeline automatically.

For a direct Python launch:

```powershell
uv sync --python 3.12 --extra data --extra baseline --extra demo --extra operational --extra dev
uv run streamlit run app/live_nowcast.py
```

## Hugging Face Spaces

This repository is configured as a Docker Space on port `7860`. Create a public Docker Space, then push or import this repository. The Space requires no secrets. It needs outbound HTTPS access to NOAA MRMS and OpenStreetMap Nominatim; its radar cache is temporary and safe to recreate.

See [the live-app deployment guide](docs/live_nowcast_app.md) for complete instructions and limitations.

## Validated model

The frozen `PySTEPSResidualUNetV1` has 3,060,440 parameters and uses ten six-minute radar frames to produce twenty forecast frames from +6 to +120 minutes. Its operating occurrence threshold is fixed at `0.35`.

On the independent 12-system FINAL event corpus:

| Metric | PySTEPS | Residual V1 |
|---|---:|---:|
| F1 | 0.5644 | 0.6810 |
| FSS18 | 0.7007 | 0.7842 |
| Brier | 0.2245 | 0.1117 |
| Rate MAE | 0.5468 | 0.5018 |

F1 improved in 12/12 systems. This was an event-focused evaluation, not continuous climatological validation. Reliable radar-blind initiation was not demonstrated.

## Research repository

The repository also preserves the data-source audits, event-level experiments, verification code, and scientific reports that led to the operational model. Large raw datasets and generated research artifacts are intentionally excluded from Git. The byte-frozen operational checkpoint, normalization, configuration, and model-source identity are included and verified at load time.

Useful references:

- [Cycle-2 Residual V1 report](docs/cycle2_residual_v1.md)
- [Operational inference](docs/operational_residual_v1.md)
- [Live application](docs/live_nowcast_app.md)
- [Project results summary](docs/project_results_summary.md)

## Safety and limitations

This is an experimental research nowcast, not an official weather-warning service. Radar-blind initiation remains difficult, probabilities are not perfectly calibrated, and PySTEPS is an essential input. Use official Environment and Climate Change Canada warnings for safety-critical decisions.
