# Milestone 1.5 benchmark

This milestone constructs the first southern-Ontario benchmark before learned-model training.
The benchmark manifest currently contains 18 candidate weather-event windows:
18 positive precipitation windows, 10 explicit initiation-focused
windows, and 4 event-level test windows, plus 18 matched
candidate hard-negative windows. Events are split only at the whole-event level; linked storm
fragments from a test event must not enter training.

## Status

- Evaluated MRMS samples available locally: 18
- Manifest events not yet materialized locally: 0
- Mean linked initiation objects per evaluated event: 146.1
- Verified matched hard-negative windows: 18/18
- Sparse-radar hard-negative windows: 18/18
- Strictly all-pixel dry hard-negative windows: 8/18
- Conservative hard-negative objects currently mined from fused multimodal data: 17
- PySTEPS status from this machine: available
- Stage 3 radar-only learned-model gate: authorized

PySTEPS is installed in the local environment and the deterministic extrapolation baseline was included in the current aggregate metrics.

The wrapper follows the public PySTEPS extrapolation example and API documentation:
https://pysteps.readthedocs.io/en/latest/auto_examples/plot_extrapolation_nowcast.html and
https://pysteps.readthedocs.io/en/stable/generated/pysteps.nowcasts.steps.forecast.html.

## Selected Events

| id | split | class | type_labels | paired_negative_id |
| --- | --- | --- | --- | --- |
| toronto_flood_2024_07_16 | dev | positive_initiation | summer, urban_flooding, organized_convection, initiation | toronto_favourable_dry_2024_07_15 |
| southern_ontario_storms_2024_08_17 | dev | positive_initiation | summer, thunderstorms, initiation, organized_convection | southern_ontario_favourable_dry_2024_08_18 |
| spring_frontal_rain_2024_04_03 | train | positive_precipitation | spring, frontal_rain, stratiform | spring_cloudy_dry_2024_04_04 |
| early_summer_convection_2024_05_21 | train | positive_initiation | summer, isolated_convection, initiation | early_summer_favourable_dry_2024_05_20 |
| june_frontal_convection_2024_06_05 | train | positive_initiation | summer, frontal_convection, initiation | june_favourable_dry_2024_06_04 |
| late_june_organized_storms_2024_06_22 | test | positive_initiation | summer, organized_convection, thunderstorms, initiation | late_june_favourable_dry_2024_06_21 |
| july_isolated_convection_2024_07_24 | train | positive_initiation | summer, isolated_convection, initiation | july_favourable_dry_2024_07_25 |
| august_severe_convection_2024_08_27 | test | positive_initiation | summer, rapid_thunderstorm_development, initiation | august_favourable_dry_2024_08_26 |
| september_stratiform_2024_09_24 | train | positive_precipitation | autumn, stratiform, frontal_rain | september_cloudy_dry_2024_09_23 |
| october_frontal_rain_2024_10_31 | train | positive_precipitation | autumn, frontal_rain, weakening | october_cloudy_dry_2024_10_30 |
| lake_effect_precip_2024_11_29 | test | positive_precipitation | cool_season, lake_effect, bands | lake_effect_setup_dry_2024_11_28 |
| winter_synoptic_precip_2025_02_12 | train | positive_precipitation | winter, stratiform, synoptic | winter_cloudy_dry_2025_02_11 |
| spring_showers_2025_04_29 | train | positive_initiation | spring, showers, weak_initiation | spring_favourable_dry_2025_04_30 |
| early_summer_cells_2025_06_17 | train | positive_initiation | summer, isolated_convection, initiation | early_summer_favourable_dry_2025_06_16 |
| july_organized_convection_2025_07_13 | test | positive_initiation | summer, organized_convection, initiation | july_favourable_dry_2025_07_12 |
| august_dissipating_system_2025_08_19 | train | positive_precipitation | summer, organized_convection, dissipation | august_favourable_dry_2025_08_18 |
| september_frontal_rain_2025_09_05 | train | positive_precipitation | autumn, frontal_rain, stratiform | september_cloudy_dry_2025_09_04 |
| lake_effect_precip_2025_12_02 | train | positive_precipitation | cool_season, lake_effect, bands | lake_effect_setup_dry_2025_12_01 |

## Matched Candidate Hard Negatives

| id | match_to | split | type_labels |
| --- | --- | --- | --- |
| toronto_favourable_dry_2024_07_15 | toronto_flood_2024_07_16 | dev | summer, favourable_environment, no_initiation |
| southern_ontario_favourable_dry_2024_08_18 | southern_ontario_storms_2024_08_17 | dev | summer, favourable_environment, no_initiation |
| spring_cloudy_dry_2024_04_04 | spring_frontal_rain_2024_04_03 | train | spring, cloudy, no_precipitation_initiation |
| early_summer_favourable_dry_2024_05_20 | early_summer_convection_2024_05_21 | train | summer, favourable_environment, no_initiation |
| june_favourable_dry_2024_06_04 | june_frontal_convection_2024_06_05 | train | summer, favourable_environment, no_initiation |
| late_june_favourable_dry_2024_06_21 | late_june_organized_storms_2024_06_22 | test | summer, favourable_environment, no_initiation |
| july_favourable_dry_2024_07_25 | july_isolated_convection_2024_07_24 | train | summer, favourable_environment, no_initiation |
| august_favourable_dry_2024_08_26 | august_severe_convection_2024_08_27 | test | summer, favourable_environment, no_initiation |
| september_cloudy_dry_2024_09_23 | september_stratiform_2024_09_24 | train | autumn, cloudy, no_precipitation_initiation |
| october_cloudy_dry_2024_10_30 | october_frontal_rain_2024_10_31 | train | autumn, cloudy, no_precipitation_initiation |
| lake_effect_setup_dry_2024_11_28 | lake_effect_precip_2024_11_29 | test | cool_season, lake_effect_setup, no_bands |
| winter_cloudy_dry_2025_02_11 | winter_synoptic_precip_2025_02_12 | train | winter, cloudy, no_precipitation_initiation |
| spring_favourable_dry_2025_04_30 | spring_showers_2025_04_29 | train | spring, favourable_environment, no_initiation |
| early_summer_favourable_dry_2025_06_16 | early_summer_cells_2025_06_17 | train | summer, favourable_environment, no_initiation |
| july_favourable_dry_2025_07_12 | july_organized_convection_2025_07_13 | test | summer, favourable_environment, no_initiation |
| august_favourable_dry_2025_08_18 | august_dissipating_system_2025_08_19 | train | summer, favourable_environment, no_initiation |
| september_cloudy_dry_2025_09_04 | september_frontal_rain_2025_09_05 | train | autumn, cloudy, no_precipitation_initiation |
| lake_effect_setup_dry_2025_12_01 | lake_effect_precip_2025_12_02 | train | cool_season, lake_effect_setup, no_bands |

## Evaluated Event Summary

| event_id | split | class | frames | anchors_evaluated | independent_initiation_events | apparent_initiation_events | pysteps_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| toronto_flood_2024_07_16 | dev | positive_initiation | 61 | 7 | 21 | 0 | available |
| southern_ontario_storms_2024_08_17 | dev | positive_initiation | 81 | 10 | 151 | 4 | available |
| spring_frontal_rain_2024_04_03 | train | positive_precipitation | 81 | 10 | 184 | 5 | available |
| early_summer_convection_2024_05_21 | train | positive_initiation | 81 | 10 | 52 | 2 | available |
| june_frontal_convection_2024_06_05 | train | positive_initiation | 81 | 10 | 153 | 16 | available |
| late_june_organized_storms_2024_06_22 | test | positive_initiation | 81 | 10 | 119 | 4 | available |
| july_isolated_convection_2024_07_24 | train | positive_initiation | 81 | 10 | 176 | 3 | available |
| august_severe_convection_2024_08_27 | test | positive_initiation | 81 | 10 | 134 | 21 | available |
| september_stratiform_2024_09_24 | train | positive_precipitation | 101 | 14 | 290 | 24 | available |
| october_frontal_rain_2024_10_31 | train | positive_precipitation | 101 | 14 | 162 | 26 | available |
| lake_effect_precip_2024_11_29 | test | positive_precipitation | 121 | 17 | 340 | 7 | available |
| winter_synoptic_precip_2025_02_12 | train | positive_precipitation | 121 | 17 | 106 | 21 | available |
| spring_showers_2025_04_29 | train | positive_initiation | 91 | 12 | 158 | 23 | available |
| early_summer_cells_2025_06_17 | train | positive_initiation | 81 | 10 | 62 | 14 | available |
| july_organized_convection_2025_07_13 | test | positive_initiation | 81 | 10 | 72 | 8 | available |
| august_dissipating_system_2025_08_19 | train | positive_precipitation | 101 | 14 | 180 | 2 | available |
| september_frontal_rain_2025_09_05 | train | positive_precipitation | 101 | 14 | 154 | 13 | available |
| lake_effect_precip_2025_12_02 | train | positive_precipitation | 121 | 17 | 115 | 2 | available |

## Aggregate Baselines

The table below aggregates currently materialized events only. All baselines use the same anchors,
valid masks, lead times, thresholds, and observations. FSS is reported by physical neighbourhood
radius, not pixel count. The current benchmark pass evaluated 216 forecast anchors across
the positive windows; per-event anchor counts are shown above so strided PySTEPS runs remain auditable.

| lead_minutes | model | threshold_mm_hr | csi | far | fss_radius_18km |
| --- | --- | --- | --- | --- | --- |
| 6 | optical_flow | 1.000 | 0.720 | 0.153 | 0.994 |
| 6 | persistence | 1.000 | 0.592 | 0.260 | 0.996 |
| 6 | pysteps_extrapolation | 1.000 | 0.725 | 0.127 | 0.995 |
| 12 | optical_flow | 1.000 | 0.619 | 0.215 | 0.983 |
| 12 | persistence | 1.000 | 0.464 | 0.370 | 0.989 |
| 12 | pysteps_extrapolation | 1.000 | 0.630 | 0.192 | 0.988 |
| 18 | optical_flow | 1.000 | 0.546 | 0.267 | 0.969 |
| 18 | persistence | 1.000 | 0.397 | 0.437 | 0.984 |
| 18 | pysteps_extrapolation | 1.000 | 0.562 | 0.245 | 0.979 |
| 30 | optical_flow | 1.000 | 0.439 | 0.347 | 0.931 |
| 30 | persistence | 1.000 | 0.320 | 0.518 | 0.978 |
| 30 | pysteps_extrapolation | 1.000 | 0.467 | 0.324 | 0.956 |
| 48 | optical_flow | 1.000 | 0.335 | 0.434 | 0.870 |
| 48 | persistence | 1.000 | 0.259 | 0.588 | 0.956 |
| 48 | pysteps_extrapolation | 1.000 | 0.377 | 0.409 | 0.915 |
| 60 | optical_flow | 1.000 | 0.285 | 0.482 | 0.830 |
| 60 | persistence | 1.000 | 0.232 | 0.621 | 0.941 |
| 60 | pysteps_extrapolation | 1.000 | 0.336 | 0.451 | 0.886 |
| 90 | optical_flow | 1.000 | 0.206 | 0.565 | 0.736 |
| 90 | persistence | 1.000 | 0.192 | 0.674 | 0.901 |
| 90 | pysteps_extrapolation | 1.000 | 0.263 | 0.532 | 0.811 |
| 120 | optical_flow | 1.000 | 0.157 | 0.634 | 0.658 |
| 120 | persistence | 1.000 | 0.162 | 0.719 | 0.862 |
| 120 | pysteps_extrapolation | 1.000 | 0.214 | 0.591 | 0.744 |

## Event Independence Logic

The detector first requires a dry radar history, future precipitation within the configured horizon,
and sufficient observed pixels. It applies a southern-Ontario polygon mask, simple morphological
opening/closing, and minimum object size filtering. Components are linked into one object track when
they are close in time and satisfy either bounding-box overlap or centroid-distance criteria. One
representative initiation is kept per linked object, which suppresses lifecycle fragments and nearby
duplicate anchor times.

Apparent initiation is separated from likely advective entry by two conservative flags: objects near
the southern-Ontario polygon boundary and objects adjacent to rain already present at issue time are
marked `advective_entry_like`. These rows are retained for audit but should not be treated as clean
initiation positives.

Known failure modes: the polygon is deliberately approximate, lake-effect bands crossing the mask
edge may be over-flagged as advective entry, split/merge storm behaviour is represented by simple
component links, and no environmental wind vector is yet used to distinguish growth from advection.

## Hard Negatives

Hard negatives are intended to match favourable environments that do not initiate precipitation.
The verifier records radar coverage first, then promotes only sparse-radar windows to GOES/HRRR
pair matching. `strict_dry_window` means no observed pixel in the event crop exceeds 0.1 mm h⁻¹.
`sparse_dry_window` is a pragmatic large-domain criterion that permits tiny radar speckle/edge
areas while rejecting windows with spatially extensive precipitation. Windows rejected by radar are
kept in the table so the negative set is auditable instead of silently curated.

| positive_event_id | negative_event_id | verified_hard_negative | rejection_reason | wet_pixel_fraction | max_frame_wet_pixel_fraction | heavy_pixel_fraction | p99_rate_mm_hr | mean_abs_standardized_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toronto_flood_2024_07_16 | toronto_favourable_dry_2024_07_15 | True | nan | 0.002 | 0.017 | 0.000 | 0.000 | 1.240 |
| southern_ontario_storms_2024_08_17 | southern_ontario_favourable_dry_2024_08_18 | True | nan | 0.010 | 0.020 | 0.002 | 0.000 | 1.847 |
| spring_frontal_rain_2024_04_03 | spring_cloudy_dry_2024_04_04 | True | nan | 0.000 | 0.005 | 0.000 | 0.000 | 1.459 |
| early_summer_convection_2024_05_21 | early_summer_favourable_dry_2024_05_20 | True | nan | 0.000 | 0.002 | 0.000 | 0.000 | 1.741 |
| june_frontal_convection_2024_06_05 | june_favourable_dry_2024_06_04 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.885 |
| late_june_organized_storms_2024_06_22 | late_june_favourable_dry_2024_06_21 | True | nan | 0.007 | 0.024 | 0.004 | 0.000 | 1.774 |
| july_isolated_convection_2024_07_24 | july_favourable_dry_2024_07_25 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.817 |
| august_severe_convection_2024_08_27 | august_favourable_dry_2024_08_26 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.884 |
| september_stratiform_2024_09_24 | september_cloudy_dry_2024_09_23 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.562 |
| october_frontal_rain_2024_10_31 | october_cloudy_dry_2024_10_30 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 0.978 |
| lake_effect_precip_2024_11_29 | lake_effect_setup_dry_2024_11_28 | True | nan | 0.000 | 0.001 | 0.000 | 0.000 | 1.445 |
| winter_synoptic_precip_2025_02_12 | winter_cloudy_dry_2025_02_11 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.419 |
| spring_showers_2025_04_29 | spring_favourable_dry_2025_04_30 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.641 |
| early_summer_cells_2025_06_17 | early_summer_favourable_dry_2025_06_16 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.895 |
| july_organized_convection_2025_07_13 | july_favourable_dry_2025_07_12 | True | nan | 0.000 | 0.001 | 0.000 | 0.000 | 1.281 |
| august_dissipating_system_2025_08_19 | august_favourable_dry_2025_08_18 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.407 |
| september_frontal_rain_2025_09_05 | september_cloudy_dry_2025_09_04 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.232 |
| lake_effect_precip_2025_12_02 | lake_effect_setup_dry_2025_12_01 | True | nan | 0.000 | 0.000 | 0.000 | 0.000 | 1.462 |

## Stage 3 Gate Decision

Stage 3 is authorized for the first radar-only learned
experiment. The gate requires all
18 positive events to be materialized/evaluated, PySTEPS to be included, event-level split
independence to remain intact, and all 18 paired hard negatives to pass radar plus GOES/HRRR
verification. If any of those checks fail, the next action is benchmark repair rather than learned
model training. The Stage 3 split is frozen in
`configs/experiments/stage_3_split_manifest.yaml`; held-out test events are reserved for one final
evaluation after architecture, loss weights, normalization, and stopping rules are selected from
train/dev results.

## Multimodal Features

Retained verified features are GOES C13 brightness temperature, HRRR 2 m temperature, 2 m dew point,
10 m U/V wind, mean sea-level pressure, surface CAPE, and nearby ECCC hourly station observations.
HRRR reprojection now uses projection-aware linear interpolation in EPSG:3978 with nearest fill only
outside the local convex hull. Candidate additions such as CIN, precipitable water, humidity and wind
aloft, vertical velocity, convergence, shear, and lapse-rate variables remain gated on unit,
valid-time, and archive-continuity checks.

## Missing Data

| event_id | frames | missing_frames | mean_missing_fraction | max_missing_fraction |
| --- | --- | --- | --- | --- |
| toronto_flood_2024_07_16 | 61 | 0 | 0.000 | 0.000 |
| southern_ontario_storms_2024_08_17 | 81 | 1 | 0.012 | 1.000 |
| spring_frontal_rain_2024_04_03 | 81 | 0 | 0.000 | 0.000 |
| early_summer_convection_2024_05_21 | 81 | 0 | 0.000 | 0.000 |
| june_frontal_convection_2024_06_05 | 81 | 0 | 0.000 | 0.000 |
| late_june_organized_storms_2024_06_22 | 81 | 0 | 0.000 | 0.000 |
| july_isolated_convection_2024_07_24 | 81 | 2 | 0.025 | 1.000 |
| august_severe_convection_2024_08_27 | 81 | 1 | 0.012 | 1.000 |
| september_stratiform_2024_09_24 | 101 | 1 | 0.010 | 1.000 |
| october_frontal_rain_2024_10_31 | 101 | 0 | 0.000 | 0.000 |
| lake_effect_precip_2024_11_29 | 121 | 0 | 0.000 | 0.000 |
| winter_synoptic_precip_2025_02_12 | 121 | 0 | 0.000 | 0.000 |
| spring_showers_2025_04_29 | 91 | 1 | 0.011 | 1.000 |
| early_summer_cells_2025_06_17 | 81 | 1 | 0.012 | 1.000 |
| july_organized_convection_2025_07_13 | 81 | 1 | 0.012 | 1.000 |
| august_dissipating_system_2025_08_19 | 101 | 1 | 0.010 | 1.000 |
| september_frontal_rain_2025_09_05 | 101 | 0 | 0.000 | 0.000 |
| lake_effect_precip_2025_12_02 | 121 | 0 | 0.000 | 0.000 |

Missing or unmaterialized events: none.

## Visual Checks

Per-event overview plots are written beside each evaluated event under
`artifacts/milestone_1_5/<event_id>/`. Dedicated audit panels are under
`artifacts/milestone_1_5/visual_audit/`:

- `optical_flow_success_late_june_2024.png`
- `initiation_growth_failure_spring_showers_2025.png`
- `dissipation_case_august_2025.png`
- `verified_hard_negative_august_2024.png`

These panels compare issue-time radar, +60 minute observations, optical-flow forecasts, PySTEPS
forecasts, and forecast errors for representative positive cases, plus a sparse-radar verified
negative-window overview.

## Storage Estimate For First Learned Experiment

The first learned experiment should stay radar-only until it beats PySTEPS on this benchmark. A
reasonable starting payload is 18 windows x roughly 8 hours x 6-minute cadence x one 2 km southern
Ontario crop. That is small enough to keep as event tensors plus masks and metadata, well below the
multi-year storage estimates from Milestone 1. Full province-wide continuous storage remains
explicitly out of scope.

## Smallest Learned Experiment After This Milestone

Use a radar-only sequence-to-sequence baseline with 60 minutes of MRMS input and 0-120 minutes of
output on fixed 256 x 256 km tiles sampled from the benchmark objects. Train only on event-level
training windows, tune on dev windows, and report once on held-out test events. It must beat
persistence, Farneback, and PySTEPS deterministic extrapolation at initiation-heavy leads before any
multimodal model is worth training.
