# Stage 12 — MRMS reflectivity and vertical-echo precursor analysis

Status: **DIAGNOSTIC COMPLETE — RICHER MRMS RADAR PRECURSORS NOT USEFUL.** Stage 12 did not justify a learned PySTEPS correction and produced no promotable forecast model. The protected six-system final set remained sealed: it was not accessed, materialized, summarized, or scored.

## Scope and frozen question

Stage 12 tested whether richer, issue-time-safe radar structure identifies precipitation initiation or growth failures that the existing PySTEPS representation misses. It used only the seven consumed development weather systems: 260 positive rows and 12 consumed hard negatives. The target was frozen from Stage 9 as `no_initiation_signal OR growth_underprediction`; future truth was used only to define the diagnostic outcome.

Exactly five MRMS products were admitted:

1. merged/composite reflectivity (`MergedReflectivityQCComposite_00.50`);
2. reflectivity at lowest altitude (`ReflectivityAtLowestAltitude_00.50`);
3. 30-dBZ echo top (`EchoTop_30_00.50`);
4. 50-dBZ echo top (`EchoTop_50_00.50`);
5. vertically integrated liquid (`VIL_00.50`).

No additional radar product, forecast model, or sealed-final information entered the experiment.

## Historical availability gate

The NOAA MRMS public operational archive begins in October 2020 and is ongoing; the required 2021–2022 development dates were audited directly. All five products have nominal two-minute cadence and 0.01-degree (approximately 1-km) grids. Reflectivity is in dBZ, echo tops are in km MSL, and VIL is in kg m-2. Product-specific missing/no-coverage codes were respected rather than treated as meteorological zero.

The frozen retention gate required at least 90% of expected current and 30-minute-baseline records in every independent development system. All five products achieved **100%** in every system and were retained. The audit covers 272 rows and 1,975 unique source files. The source manifest preserves archive key, requested and selected times, source age, ETag, local path, byte count, and SHA-256.

Every selected observation time is at or before its requested issue/history time, with maximum permitted source age of four minutes. No future radar frame was used. Current and historical source selections are explicit in `source_availability_audit.csv`; `availability_gate.json` records that every selection passed the causality check.

## Materialization and fixed descriptors

All 272 rows were materialized successfully. Each product had valid fraction 1.0 in every row after distinguishing missing echo from absent spatial coverage. Processing stayed on the existing 128 x 128, 2-km analysis tiles.

The predeclared physical thresholds were:

- reflectivity echo: at least 20 dBZ;
- reflectivity growth: at least +5 dBZ over 30 minutes;
- echo-top growth: at least +1 km over 30 minutes;
- VIL growth: at least +2 kg m-2 over 30 minutes;
- surface-rain exclusion: existing precipitation rate at most 0.1 mm h-1.

Descriptors include coverage fractions, upper quantiles, maxima, 30-minute growth coverage and magnitude, reflectivity-only coverage, and `elevated_echo_without_surface_rain`. The latter requires composite reflectivity of at least 20 dBZ, no coincident surface rain, and positive EchoTop30, EchoTop50, or VIL support. Thresholds and feature definitions were fixed before the failure labels were joined.

## Pixel-level precursor diagnostic

Among future-initiation pixels missed by PySTEPS, event-macro precursor prevalence was very small:

| Precursor at/before issue | Event-macro fraction |
|---|---:|
| Composite reflectivity >=20 dBZ | 0.007405 |
| EchoTop30 present | 0.000912 |
| EchoTop50 present | 0.000005 |
| VIL present | 0.053758 |
| Vertical-growth precursor | 0.000852 |
| Elevated echo without surface rain | 0.007405 |

Accordingly, 99.2579% of missed initiation pixels were category A, **no detectable radar precursor**. Reflectivity-only precursor coverage was 0.6569% and vertical-growth precursor coverage was 0.0852%. The apparent lead distribution is conditional on the rare pixels with an elevated precursor: 49.80% appeared within 15 minutes, 19.45% at 15–30 minutes, 12.85% at 30–60 minutes, and 17.91% more than 60 minutes beforehand. These percentages do not imply broad precursor coverage.

Correctly forecast initiation pixels also showed little elevated-echo precursor coverage (event-macro 0.03218). Hard negatives were essentially clean: elevated-echo coverage was zero event-macro, with only isolated tiny reflectivity or VIL traces. The descriptors therefore did not achieve usefulness by broadly firing on dry cases, but their scarcity left little usable initiation signal.

## Failure-category and growth analysis

The richer products did not preferentially mark the priority failures. Mean descriptor values for `no_initiation_signal` were composite echo coverage 0.01055, EchoTop30 coverage 0.00327, EchoTop50 coverage 0.000053, VIL coverage 0.02401, and elevated-echo coverage 0.00326. For `growth_underprediction`, the corresponding values were 0.03666, 0.01269, 0.000487, 0.08887, and 0.00670. The adequate/other group was stronger still at 0.05275, 0.01463, 0.000221, 0.12004, and 0.00890.

System-level growth-minus-adequate comparisons were inconsistent. Composite growth coverage was higher for growth failures in only 2 of 7 systems; the differences ranged from -0.03597 to +0.03468. Echo-top and VIL trends likewise changed sign across systems. This is not a stable development-wide precursor signature.

## Leave-one-event-out predictability test

The compact context model used only lead fraction, current radar wet fraction, and PySTEPS wet fraction. The richer-radar model added the frozen Stage 12 descriptors. Both were L2 logistic regressions with lambda 1.0. For every held-out system, normalization and coefficients were fit on the other six systems only.

| LOEO event-macro metric | Context only | Context + richer radar |
|---|---:|---:|
| ROC AUC | **0.77662** | 0.73747 |
| Precision | 0.71025 | 0.71249 |
| Recall | 0.99622 | 0.98471 |
| F1 | **0.82622** | 0.82405 |
| Brier score | 0.19875 | **0.19783** |

The richer descriptors changed AUC by **-0.03915**, reduced F1 slightly, and improved Brier by only 0.00092. Only **3 of 7** independent systems had a positive event-AUC change. Event changes were -0.22856, -0.07827, -0.06000, +0.04776, +0.05805, +0.04608, and -0.05912.

For the two primary categories, the richer model was effectively unchanged for `no_initiation_signal` (AUC 0.77830 versus 0.77627) and substantially worse for `growth_underprediction` (0.54148 versus 0.62447). Calibration-bin results and all held-out probabilities are preserved in the artifacts.

## Destruction and product-group tests

The same trained richer-radar checkpoints were evaluated after removing or shuffling their radar descriptors. Removing all richer radar raised event-macro AUC to 0.77685, approximately the context-only result. Global shuffling produced 0.67794 and within-event shuffling 0.69011, showing that the fitted model reacted to the descriptors, but that reaction did not translate into useful normal-condition generalization.

With the same checkpoints and no retraining, removal-group event-macro AUCs were:

- reflectivity removed: 0.75971;
- echo tops removed: 0.73798;
- VIL removed: 0.73974;
- elevated-echo mask removed: 0.73971.

Removing reflectivity improved the result toward the context baseline; removing the other groups made little difference. No product group supplied a reproducible positive increment.

## Decision

Stage 12 is classified **NOT USEFUL**. The source audit and materialization succeeded, so this is not an availability or causality failure. Instead:

- detectable pre-initiation echo was absent from nearly all missed pixels;
- growth-failure descriptors were not consistently stronger than adequate cases;
- the richer model lost 0.039 event-macro AUC and improved only 3 of 7 systems;
- destroying all richer radar information recovered the context baseline;
- no product group showed stable incremental value.

Therefore a learned PySTEPS correction experiment is **not justified**, and no Stage 12 model is promotable. This conclusion is limited to these five products, frozen representations, seven development systems, and the specified failure target; it is not a claim that all three-dimensional radar information is intrinsically useless.

## Reproducibility artifacts

All machine-readable outputs are under `artifacts/stage_12/`, including product definitions, availability and causal-source audits, downloaded-source hashes, feature predeclaration, row/lead descriptors, pixel precursor diagnostics, event consistency, growth-versus-adequate comparisons, LOEO predictions/fits/metrics/calibration, product-group removal results, the decision record, and `stage12_manifest.json` containing output hashes.

Stage 10 remains the frozen HRRR diagnostic, Stage 11 remains the frozen GOES diagnostic, and PySTEPS remains the principal reference forecast. No conclusion from those stages was rewritten.

At the project level, reliable radar-blind initiation information has not been demonstrated from deterministic hourly HRRR; 15-minute HRRR APCP, PRATE, or REFC; causal HRRR displacement correction; GOES C13/cooling; or the richer MRMS reflectivity, echo-top, and VIL descriptors evaluated here. Source-by-source precursor searching therefore stops after Stage 12. PySTEPS remains the principal validated 0–2-hour operational backbone.
