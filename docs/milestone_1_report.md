# Milestone 1 report

Status: **small vertical slice validated; scale-up is not yet authorized scientifically**. This report
records completed evidence and the remaining gates as of 2026-08-31.

## Executive result

The repository can retrieve authoritative historical radar, satellite, NWP, and station files for
southern-Ontario events, preserve their provenance, construct precipitation cubes, align
availability by UTC, mine clear-to-rain candidates, and evaluate persistence and dense optical-flow
baselines. On the single Toronto 2024-07-16 pilot, optical flow improves 0.1 mm h⁻¹ CSI through 30
minutes, is effectively tied at 60 minutes, and is worse by 90–120 minutes. This is the expected
failure mode of constant-motion extrapolation and is a useful lower bar for learned evolution.

A second 2024-08-17 storm window behaves differently: optical flow beats persistence at every tested
lead, though its advantage narrows to 0.013 CSI by 120 minutes. The contrast is strong evidence that
one case cannot support model-selection claims and that event-stratified verification is required.

These are diagnostic single-event results, not a held-out claim. The most consequential Stage 0
finding is a data constraint: the ideal ECCC 1 km quantitative radar composite has only a three-hour
public GeoMet window, while the verified official historical interface is image-oriented. NOAA MRMS
enables the southern-Ontario pilot from 2020 onward but ends near 55°N.

## Domain and grid decision

- Research context box: **96°W–72°W, 40.5°N–57.5°N**, approximately 1,600 × 1,900 km. It includes
  Ontario and upstream Great Lakes/US/Quebec/Manitoba context.
- Pilot box: **86.5°W–72.5°W, 40.5°N–49.5°N**, approximately 1,050 × 1,000 km. MRMS/HRRR make this
  a practical southern-Ontario vertical slice.
- Analysis CRS: **EPSG:3978** (Canada Atlas Lambert), initially 2 km. Radar ingestion preserves native
  quantitative values before reprojection. Training uses overlapping 256 × 256 tiles (512 km square).
- Baselines retain the MRMS 0.01° event crop and use 2× max pooling for runtime. Separately, a
  projection-aware fused sample has been built on the stated 2 km EPSG:3978 grid; this keeps baseline
  preprocessing changes from being confused with fusion changes.

## Storage estimate before scale-up

The estimator assumes 33 channels (1 radar, 4 satellite, 24 NWP, 4 station-derived), a 10-minute
processed cadence, float16 values plus a one-byte validity mask, 2.5:1 compression, and raw storage
equal to 1.35× processed. These assumptions are intentionally visible in
`configs/data/milestone_1.yaml`.

| Domain | Years | Processed | Raw + processed |
|---|---:|---:|---:|
| Full research, 800 × 950 at 2 km | 1 | 1.44 TiB | 3.38 TiB |
| Full research | 3 | 4.32 TiB | 10.15 TiB |
| Full research | 5 | 7.20 TiB | 16.92 TiB |
| Southern pilot, 525 × 500 at 2 km | 1 | 0.50 TiB | 1.17 TiB |
| Southern pilot | 3 | 1.49 TiB | 3.51 TiB |
| Southern pilot | 5 | 2.49 TiB | 5.84 TiB |

This rules out a blind province-wide download. The next build should store cropped/tiled arrays and
sample event windows plus matched hard negatives before considering continuous multi-year tensors.

## Real sample built

Event: Toronto / western Lake Ontario, 2024-07-16 12:00–18:00 UTC.

- 61 MRMS precipitation-rate frames at exactly 6-minute spacing; native crop shape 300 × 350.
- Seven GOES-16 ABI channel-13 CMI NetCDF files (25.6 MB total), decoded range 193.8–300.0 K.
- Seven HRRR analysis subsets (59.5 MB total): MSLP, 2 m T/Td, 10 m U/V, and surface CAPE. Each
  subset decodes to four GRIB groups on the 1059 × 1799 HRRR grid.
- Two ECCC monthly hourly station files (Toronto Pearson and Toronto City Centre).
- All 61 radar timestamps have a sampled GOES, HRRR, and station timestamp within ±31 minutes. The
  table records actual offsets rather than pretending the modalities are simultaneous.
- Seven hourly slices are spatially aligned into a 7 × 203 × 184 tensor in EPSG:3978. Radar is
  bilinearly interpolated from native lat/lon, GOES is bilinearly interpolated in its fixed-grid
  geostationary projection, HRRR uses the nearest native grid point, and the two station records use
  inverse-distance-squared features masked beyond 150 km. Every modality retains a finite-value mask.
- 17 first-pass hard-negative components were mined where fully observed radar remained below
  0.1 mm h⁻¹ through a two-hour future window despite surface CAPE ≥500 J kg⁻¹ and GOES C13
  brightness temperature ≤260 K. These are candidates for later environmental matching, not proof
  that convection was forecast operationally.
- 114 de-duplicated clear-to-rain component candidates passed the first event miner. This is a
  candidate catalogue: large connected areas and repeat lifecycle fragments still require tracking
  and Ontario masking before they become independent cases.

Second event: southern Ontario, 2024-08-17 14:00–22:00 UTC.

- 81 expected 6-minute slots over a 400 × 600 native crop; the absent 20:24 UTC scan is an explicit
  all-NaN frame. It is not interpolated and later scans do not shift cadence.
- 875 preliminary initiation components reflect a much more fragmented convective scene and reinforce
  the need for object tracking/de-duplication before treating components as independent events.
- The second event currently has radar baseline results and a montage; multimodal payload retrieval
  was validated on the Toronto event only.

The radar montage and a synchronized radar/GOES-C13/HRRR-CAPE/HRRR-temperature panel are saved under
`artifacts/milestone_1/toronto_flood_2024_07_16/`; metrics, synchronization offsets, and candidates
are beside them. Artifacts are reproducible and intentionally excluded from Git.

## Baseline methodology and initial results

Persistence repeats the latest rate. Optical flow estimates dense Farnebäck motion from the two most
recent log-scaled rate fields and semi-Lagrangian-advects the latest quantitative field. Both are
evaluated on identical anchors after 2× maximum pooling. The table shows event-aggregated metrics at
the 0.1 mm h⁻¹ occurrence threshold; FSS uses a 9-pixel neighbourhood.

| Lead | Persistence CSI | Flow CSI | Persistence FAR | Flow FAR | Persistence FSS | Flow FSS |
|---:|---:|---:|---:|---:|---:|---:|
| 6 min | 0.831 | **0.883** | 0.087 | **0.062** | **0.996** | 0.993 |
| 18 min | 0.680 | **0.766** | 0.176 | **0.095** | **0.969** | 0.966 |
| 30 min | 0.593 | **0.665** | 0.233 | **0.120** | 0.926 | **0.927** |
| 60 min | **0.484** | 0.481 | 0.313 | **0.179** | **0.810** | 0.806 |
| 90 min | **0.421** | 0.388 | 0.373 | **0.218** | **0.724** | 0.703 |
| 120 min | **0.352** | 0.309 | 0.461 | **0.267** | **0.640** | 0.604 |

Interpretation: motion sharply reduces false alarms and raises short-lead CSI, but its POD falls as
growth, decay, and direction changes accumulate. Persistence retains more broad wet area later in this
particular event, producing higher CSI despite worse FAR. This is exactly why the learned model must
be judged on initiation/dissipation and calibration, not just animation quality. No promotion decision
is valid from one event.

For the independent August storm window, 0.1 mm h⁻¹ CSI is 0.682/0.775 at 6 minutes,
0.364/0.551 at 30 minutes, 0.261/0.384 at 60 minutes, and 0.194/0.207 at 120 minutes for
persistence/optical flow respectively. These are still case-study scores, not a held-out leaderboard.

## Event detection

For every valid anchor and pixel, the detector requires 30 minutes below 0.1 mm h⁻¹, then searches the
next 120 minutes for rain. Missing values are excluded using a missing-fraction gate. Eight-connected
components of at least nine pooled pixels are catalogued with timestamp, bounding box, centroid,
pixel count, and maximum future rate; overlapping candidates within 30 minutes are suppressed.

Next refinements are object tracking, Ontario polygon masking, morphological noise control, explicit
minimum dry-area duration, initiation-versus-advection separation, convective/stratiform labels, and
hard-negative matching to positive cases on season, location, CAPE, moisture, and synoptic regime.

## Engineering verification

- Python 3.12 environment locked with `uv`; Python 3.14 is intentionally excluded until GRIB/HDF
  dependencies consistently support it.
- Automated tests cover timezone-aware nearest alignment, non-filled scan gaps, event extraction,
  missing-versus-dry handling, onset timing, categorical/Brier/FSS calculations, source timestamp
  parsing, and linear storage scaling.
- Current result: **21 tests pass** and Ruff reports no violations.
- Raw downloads are no-overwrite and recorded with SHA-256. Processed builds record source files,
  timestamps, domain, units, cadence, preprocessing description, and their own hash.

## Issues and limitations

1. Two radar events and one multimodal event have been processed; this is insufficient for a
   leaderboard or train/validation/test split.
2. Quantitative historical ECCC composite access is unresolved, so northern Ontario is not covered by
   the historical pilot.
3. The fused tensor uses only GOES C13, six surface HRRR fields, nearest-neighbour NWP mapping, and two
   Toronto stations. Add satellite tendencies/quality flags, pressure levels, conservative masks,
   better NWP resampling, and broader station coverage before training a multimodal model.
4. The optical-flow baseline is competent but not yet PySTEPS STEPS/extrapolation; add that before
   judging a learned model.
5. FSS physical neighbourhood size changes with current pooling and must be reported in kilometres.
6. ECCC station LST handling is explicit but needs a metadata test across jurisdictions/time zones.
7. HRRR radar assimilation can confound the core atmospheric-information ablation.
8. No train/validation/test split exists yet because a multi-event catalogue is not complete.

## Logical review and recommended next step

Do **not** scale to years yet. First complete a 10–20-event southern-Ontario benchmark with multiple
seasons and paired hard negatives. Generalize the validated EPSG:3978 fusion build, strengthen coverage
masks, add a PySTEPS baseline, and validate candidate independence. In parallel, contact ECCC or verify
an authorized archive route for machine-readable historical composite/volume data. Only after those
gates pass should event-window acquisition expand, followed by continuous data.

The first learned experiment should remain radar-only. Radar + NWP and radar + satellite must use the
same held-out weather events, issue-time-safe inputs, tiles, targets, and metrics so the central
pre-radar initiation hypothesis can be tested rather than assumed.
