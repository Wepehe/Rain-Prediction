# Validated Residual V1 — Operational Inference

## What the system does

Residual V1 produces a 0–2 hour precipitation nowcast for one 128 × 128 tile on the existing 2 km EPSG:3978 grid. It uses ten chronologically ordered radar frames at six-minute cadence, computes the frozen deterministic Lucas–Kanade PySTEPS extrapolation, and applies the validated residual U-Net. The product includes the baseline, learned occurrence probability, frozen-threshold decision, and corrected rain rate for 20 future six-minute leads.

This is software packaging of the frozen `PySTEPSResidualUNetV1`; it is not a new model or scientific evaluation.

## Validated evidence

On the independent, subsequently consumed 12-system FINAL weather-event corpus:

| Metric | PySTEPS | Residual V1 |
|---|---:|---:|
| Event-macro F1 | 0.5644 | 0.6810 |
| Event-macro FSS18 | 0.7007 | 0.7842 |
| Brier score | 0.2245 | 0.1117 |
| Rate MAE (mm/h) | 0.5468 | 0.5018 |

F1 improved in 12/12 FINAL systems. This evidence comes from a selected weather-event corpus, not continuous climatology.

## Inputs

The Python API accepts:

- `history_rate`: numeric array `[10, 128, 128]` in mm/h, oldest to newest;
- `timestamps`: ten ISO-8601 strings or datetimes, strictly increasing at exactly six-minute cadence;
- `validity_mask`: boolean-compatible array `[10, 128, 128]`;
- optional metadata, including `grid: {x: [...], y: [...]}` for coordinates.

Invalid or masked values are replaced with zero only after the explicit validity mask is combined with finite-value checks. Negative finite rates are clipped to zero. The API rejects wrong dimensions, cadence, ordering, or wholly invalid histories. It never recomputes TRAIN-derived normalization.

The CLI input is a compressed NPZ containing `history_rate`, `timestamps`, and `validity_mask`, plus optional scalar `metadata_json`. Research TRAIN/DEV/FINAL manifests are neither required nor accessed.

## Outputs

- `timestamps`: +6, +12, …, +120 minute valid times.
- `pysteps_rate_mm_h`: deterministic PySTEPS extrapolation `[20,H,W]`.
- `residual_probability`: raw learned probability that rate exceeds 0.1 mm/h `[20,H,W]`.
- `residual_wet_mask`: operational decision `residual_probability >= 0.35`.
- `residual_rate_mm_h`: learned intensity correction around PySTEPS `[20,H,W]`.
- `validity_mask`: input validity used by the model.
- optional `motion_u` and `motion_v`, plus immutable hashes, version, fallback, and timing metadata.

The CLI writes `forecast.npz` and `forecast_summary.json`.

## Example

```powershell
uv run python scripts/run_operational_nowcast.py `
  --input latest_radar.npz `
  --output forecast `
  --device cpu `
  --diagnostics
```

```python
import numpy as np
from ontario_nowcast.operational import forecast, load_operational_model

model = load_operational_model(device="cpu")
with np.load("latest_radar.npz") as data:
    product = forecast(
        model,
        data["history_rate"],
        data["timestamps"].astype(str).tolist(),
        data["validity_mask"],
        metadata={"grid": {"x": x_coordinates, "y": y_coordinates}},
    )
product.save_npz("forecast/forecast.npz")
```

## Reproducibility and benchmark

The frozen DEV software fixture `cycle2_20201019__active__20201019T1254__11` reproduced the fresh research evaluation pathway exactly:

- PySTEPS maximum absolute difference: `0.0`;
- occurrence-probability maximum absolute difference: `0.0`;
- corrected-rate maximum absolute difference: `0.0`;
- differing threshold-mask pixels: `0`.

The frozen fixture input SHA-256 is `febe87339851126758c22216e388a8358e1accc58839c0a304ead1bf7aa8a6b4`; its canonical output-array SHA-256 is `2f7269e47d8dbcdd6675cc6705ebb78a8ce0a4344710865ec01668ca88d11188`.

On the available CPU, one 128 × 128 tile took 15.906 s for PySTEPS, 0.240 s for neural inference, and 16.146 s total. Python-tracked peak memory was 66.8 MiB; this does not include all native-library allocations. CUDA was unavailable.

## Limitations

- Validation covers an event corpus, not continuous climatology.
- Reliable radar-blind initiation was not demonstrated.
- Displacement and decay correction remain weaker than growth correction.
- Raw probabilities are useful but imperfectly calibrated; no post-hoc calibration is applied.
- The model depends on a functioning PySTEPS forecast. Dry-scene LK failures use the frozen zero-velocity/latest-field fallback.
- The 0.35 occurrence threshold is frozen.

## Versioning

Bundle version: `1.0.0`; model version: `cycle2-residual-v1`; architecture: `PySTEPSResidualUNetV1`; trainable parameters: 3,060,440.

| Frozen item | SHA-256 |
|---|---|
| Epoch-14 checkpoint | `b536ca16daaaf8904b9c326ed00c01169105d764c2072249092a9259ba9a22c5` |
| TRAIN normalization | `23dc3b31edad1247fc77abd4ec7d44688cdd79e21b3d0f832260130df36c192c` |
| Configuration | `13456f0e6a940f3a8a41f68e7c9e7f7efdf4eaf5e24273bcc96f5b967250495b` |
| Model source | `2a84095e36009174ca662566b2158e8e9fb38e3e17fba280ceb8017e2f739ea2` |

The bundle is at `artifacts/operational/residual_v1/`. Loading fails on a missing artifact or hash mismatch.
