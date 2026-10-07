# Stage 6 transparent A+/HRRR hybrid validation

Status: **COMPLETE — NEGATIVE.** The protected holdout was scored once under
the frozen procedure. H1 was not scored and no post-holdout tuning occurred.

## Scientific question

Does a low-degree-of-freedom combination of a precise radar-based nowcast and
coarse operational HRRR precipitation provide robust incremental initiation
skill on independent weather events?

## Frozen procedure

The hybrid is exactly:

`P_hybrid = 0.50 * P_A+ + 0.50 * I(HRRR APCP >= 0.1 mm/h)`

Its operating probability threshold is 0.30. A+ is immutable. HRRR f02 hourly
APCP is used as the 0–60-minute context and f03 as the 60–120-minute context;
neither is described as a six-minute forecast. There is no GOES input,
calibration, nonlinear gate, lead-dependent weight, or protected-data tuning.

This experiment is legitimate external validation because the hybrid structure
and candidate weights 0, 0.25 and 0.50 were declared before H1 evaluation. The
0.50 weight and 0.30 threshold were selected on DEV only. Neither H1 nor this
hybrid has been scored on the protected 2023 set, and Stage 6 makes no new
tuning choice.

The machine-readable manifest was frozen before predictions with SHA-256
`992a00d7926a404c927539b7e9b8be98776950a438e0f5973da2dd9edeb4bc03`.
The 68-row holdout hash remained
`bbf001f7512dc5aae36158941affb878ec7f3e1d011dc0fadec39fed755b3f5c`.
All 68 caches and 70 issue-safe APCP products passed integrity checks, including
bit-exact repeated A+ output, no future radar, and no later HRRR cycle.

## Primary all-initiation result

The endpoint contains 56 rows from three independent positive events. Event
macro is the primary summary; row macro is also retained.

| Model | Precision | Recall | F1 | False initiation | Onset MAE | Bias | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| A+ | 0.483 | 0.145 | 0.181 | 0.517 | 39.49 min | -16.83 min | 0.228 |
| Raw HRRR | 0.477 | 0.084 | 0.121 | 0.244 | 39.82 min | +8.03 min | 0.278 |
| Hybrid | 0.604 | 0.124 | 0.164 | 0.291 | 33.46 min | -2.48 min | 0.238 |
| PySTEPS | 0.600 | 0.270 | 0.350 | 0.237 | 15.91 min | -1.72 min | 0.231 |

The hybrid improved precision, false-initiation fraction, onset MAE and bias
relative to A+, but lost recall. Its event-macro F1 was lower by **0.0162**.
Its descriptive three-event bootstrap interval for F1 is 0.074–0.256, versus
0.162–0.208 for A+; these intervals describe event sensitivity and are not
high-precision inferential estimates. Row-macro F1 was 0.195 for the hybrid and
0.176 for A+, showing why row pooling alone would give a misleading conclusion.

Individual-event F1 changes (hybrid minus A+) were:

| Independent event | A+ F1 | Hybrid F1 | Change |
|---|---:|---:|---:|
| 2023-08-03 severe storms | 0.208 | 0.163 | -0.045 |
| 2023-07-12 storms | 0.171 | 0.074 | -0.097 |
| 2023-07-20 supercells | 0.162 | 0.256 | +0.094 |

Only one of three events improved.

## Radar-limited and strict V2 endpoints

For the immutable 28-row `RADAR_LIMITED_INITIATION` subset, event-macro F1 was
0.126 for A+, 0.119 for the hybrid, 0.093 for HRRR and 0.225 for PySTEPS. The
hybrid reduced A+ F1 by 0.007 and therefore did not support the intended
mechanism on the broader radar-limited definition.

The immutable 19-row `RADAR_POOR_INITIATION_V2` subset gives a valuable but
non-general result: hybrid F1 was 0.102 versus 0.048 for A+, an improvement of
0.055. Hybrid precision/recall were 0.620/0.059, onset MAE 34.33 minutes, bias
+7.2 minutes and Brier 0.214. A+ precision/recall were 0.328/0.031, onset MAE
47.25 minutes, bias +1.25 minutes and Brier 0.207. Thus coarse HRRR helped some
strictly radar-poor cases, but detection remained sparse and the gain did not
extend to the full cohort or a majority of events.

## Hard negatives

On eight frozen dry rows, event-macro results were:

| Model | Mean probability | Maximum | Wet area | False initiation | Brier |
|---|---:|---:|---:|---:|---:|
| A+ | 0.0903 | 0.5585 | 0.0289 | 1.000 | 0.0226 |
| Raw HRRR | 0.0096 | 0.6250 | 0.0096 | 0.625 | 0.0220 |
| Hybrid | 0.0500 | 0.5254 | 0.0102 | 0.625 | 0.0167 |

The hybrid reduced A+ false coverage and Brier, but narrowly exceeded the
pre-existing 0.01 wet-area criterion and exceeded the 0.40 false-initiation
criterion. Hard-negative behavior is therefore not acceptable under the frozen
research policy.

## Heavy rain and broad precipitation

The four heavy-rain rows remain a separate secondary endpoint. At 0.1 mm/h,
hybrid F1 ranged from 0.430 to 0.453 at 30–90 minutes and was 0.433 at 120
minutes. This improved on A+ at every reported lead, while PySTEPS remained
stronger at 30–120 minutes. The hybrid underpredicted mean rate: forecasts were
0.356, 0.361, 0.412 and 0.406 mm/h at 30, 60, 90 and 120 minutes, versus
observed means of 0.772, 0.767, 0.843 and 0.764 mm/h. Its rate MAE nevertheless
improved on A+ at each lead. The complete 0.1/1/2.5/5 mm/h CSI, POD, FAR, F1,
6/18/36-km FSS and Brier tables are preserved in
`artifacts/stage_6/evaluation/heavy_rain_metrics.csv` and
`broad_metrics.csv`; expected-rate summaries are stored separately. HRRR values retain their hourly-interval interpretation;
30- and 90-minute labels are evaluation sampling points, not claims of exact
HRRR temporal resolution.

## Where the hybrid helped and hurt

Post-score descriptive grouping assigned 23 rows to “HRRR correctly adds rain
missed by A+”, two to correct suppression/confidence adjustment, 11 to false
addition, 19 to “A+ already correct and HRRR hurts”, and one to both systems
failing. Representative panels were generated after scoring and had no role in
selection. This mixed row-level behavior is consistent with strong regime and
event dependence.

## Final interpretation

Stage 6 is **NEGATIVE** under the predeclared interpretation: the hybrid
materially worsened the primary event-macro endpoint, improved only one of
three independent events, failed the broader radar-limited endpoint and did not
meet hard-negative constraints. The strict V2 gain and heavy-rain improvements
are retained as secondary evidence rather than used to revise the decision.

The blend must not be retuned on 2023. Any revised temporal treatment, regime
gate or hybrid requires a new future evaluation set. The appropriate next
investigations are event diversity, HRRR hourly timing, displacement errors and
why DEV overfavoured HRRR—not another blend search on this holdout.
