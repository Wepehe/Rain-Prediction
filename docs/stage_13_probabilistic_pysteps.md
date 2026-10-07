# Stage 13 — Probabilistic PySTEPS and operational uncertainty

Status: **FINAL COMPLETE — MIXED; DEVELOPMENT CATEGORICAL BENEFIT DID NOT CLEANLY GENERALIZE. PROBABILISTIC OUTPUT RETAINED AS OPERATIONAL SUPPLEMENT.**

Timing uncertainty remains **UNDER-DISPERSED**. The six-system final holdout is permanently consumed.

## Scientific scope

Stage 13 asks whether PySTEPS can express uncertainty around radar-observed precipitation more usefully than the deterministic reference. It does not claim to predict radar-blind initiation. The experiment uses only the seven consumed Stage 8 systems, 260 positive rows, and 12 consumed hard negatives.

The project-level evidence remains unchanged: reliable radar-blind initiation information has not been demonstrated from deterministic hourly HRRR; 15-minute HRRR APCP, PRATE, or REFC; causal HRRR displacement correction; GOES C13/cooling; or richer MRMS reflectivity, echo-top, and VIL descriptors. PySTEPS remains the principal validated 0–2-hour operational backbone.

## Existing deterministic reference

The reference was already deterministic, not ensemble-based. Its exact implementation is:

- PySTEPS 1.21.5;
- `extrapolation` nowcast method;
- Lucas–Kanade (`LK`) motion from the final three radar frames;
- dB rain-rate transform, 0.1 mm h-1 threshold, -15 dB zero value;
- 20 six-minute forecast steps;
- deterministic semi-Lagrangian extrapolation;
- no cascade/AR/noise or velocity perturbations;
- output inverse-transformed to non-negative mm h-1.

The existing Stage 8 deterministic fields were reused unchanged. The implementation remains `pysteps_deterministic_extrapolation` in `models/baselines.py`.

## Frozen ensemble configuration

The ensemble procedure was frozen before skill evaluation in `stage13_predeclaration.json` (initial SHA-256 `d78f57bb212f5ad2cead122920b2e0cd2f9f9397f79169cd3dbfc6796f5bcb2c`):

- standard PySTEPS `STEPS` method;
- eight members;
- ten six-minute input frames;
- Lucas–Kanade motion;
- 2-km pixels and six-minute timestep;
- dB transform and 0.1 mm h-1 precipitation threshold;
- PySTEPS defaults: six cascade levels, AR(2), nonparametric precipitation noise, and default velocity perturbations;
- NumPy FFT, one worker;
- fixed per-row seed: first 32 bits of SHA-256(`row_id + ':stage13:v1'`).

An exact repeat of the first row, treating identical missing-value locations as equal, produced identical arrays and maximum finite difference 0.0. All 272 rows passed probability monotonicity.

## Probability and rate products

Member exceedance fractions provide `P(rate > 0.1)`, `P(rate > 1.0)`, `P(rate > 2.5)`, and `P(rate > 5.0 mm h-1)` at every lead. Every field satisfies the required nesting. Expected rate is the direct ensemble-member mean; it is not probability multiplied by conditional intensity. Ensemble-member standard deviation supplies spread.

For storage-conscious continuous verification, the reported CRPS is a Gaussian approximation based on cached member mean and spread (mean 0.00101 in the evaluated scale), not exact empirical-member CRPS. The exact limitation is recorded rather than presenting the approximation as empirical CRPS.

## Probabilistic verification

Across event/threshold combinations, the raw ensemble's event-macro AUC is **0.5451** and Brier score is **0.05371**. Reliability tables by six-minute lead and threshold, probability sharpness, and individual-event AUC/Brier are preserved in the artifacts.

At the 0.1 mm h-1 threshold:

| Lead | Ensemble Brier | Deterministic Brier | Ensemble F1 | Deterministic F1 | Ensemble FSS18 | Deterministic FSS18 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 min | 0.0432 | 0.0522 | 0.2371 | 0.2465 | 0.4268 | 0.4713 |
| 60 min | 0.0753 | 0.0884 | 0.1026 | 0.1207 | 0.1965 | 0.2382 |
| 90 min | 0.1072 | 0.1214 | 0.0436 | 0.0743 | 0.1243 | 0.1794 |
| 120 min | 0.1402 | 0.1537 | 0.0205 | 0.0480 | 0.0866 | 0.1312 |

The same pattern holds at 1, 2.5, and 5 mm h-1: Brier improves, while the frozen probability >=0.5 categorical policy usually reduces CSI/F1. Full CSI, POD, FAR, F1, and FSS at 6/18/36 km are in `metrics_event_macro.csv`.

Across all thresholds and selected leads, event-macro F1 falls from **0.06307 to 0.04795**, and FSS18 falls from **0.24767 to 0.22166**. Brier improves in all seven independent systems, but F1 worsens in all seven. FSS18 improves in only two systems. The largest Brier improvements occur for September 22 (+0.01653) and the shared September 7/8 system (+0.00949); neither retains categorical skill.

## Onset and cessation uncertainty

The project persistence rule is two consecutive six-minute wet frames. Operational onset probabilities are derived by 30, 60, 90, and 120 minutes. The 10th/50th/90th-percentile timing approximation uses two-frame persistent marginal probabilities; because full member trajectories were not retained after probability construction, its joint-time dependence is approximate and is labeled accordingly.

Event-macro onset results are:

- ensemble median onset MAE: **45.12 min**;
- deterministic onset MAE: **43.37 min**;
- 10–90% interval coverage: **0.263**;
- mean interval width: **20.77 min**.

The raw ensemble is therefore under-dispersive for onset and does not improve median timing.

For initially wet valid pixels, cessation results are:

- ensemble median cessation MAE: **23.97 min**;
- deterministic cessation MAE: **24.19 min**;
- 10–90% interval coverage: **0.370**;
- mean interval width: **20.12 min**.

Cessation timing improves only marginally, and interval coverage remains inadequate.

## Failure-aware spread

Spread is not a radar-blind initiation signal. It is exactly zero on the `no_initiation_signal` cases while mean absolute expected-rate error is 0.6951: all members remain dry when the radar state contains no advectable precipitation. Growth-underprediction spread-error correlation is -0.022, also uninformative. Correlation is positive for displacement (0.289) and decay persistence (0.552), indicating some useful uncertainty response only where an existing echo can be perturbed.

This supports the intended interpretation: stochastic extrapolation can describe uncertainty in existing precipitation motion/evolution, but cannot create information absent from radar.

## Hard negatives

All 12 consumed hard negatives have:

- mean and maximum `P(>0.1)` equal to 0;
- wet-area coverage above the 0.5 policy equal to 0;
- probabilistic false alarm equal to 0;
- deterministic false initiation equal to 0;
- Brier score equal to 0.

Dry-case behavior is fully controlled on this small consumed set.

## Calibration decision

Reliability diagnostics show conservative/under-dispersive behavior, but a calibrator is **not justified in this pass**. The ensemble's discrimination is weak, its onset interval coverage is poor, and its 0.5 policy loses F1 in all seven systems. Platt scaling could adjust marginal probabilities but cannot restore absent initiation information or spatial structure. No calibrator was fit, avoiding an unnecessary grouped-CV optimization after the raw system failed to establish broad operational value.

## Classification and final-set gate

Stage 13 is classified **MIXED**:

- positive: lower Brier in all seven systems, reproducible probabilities, strict monotonicity, direct expected rate, and perfect control of the 12 hard negatives;
- negative: lower F1 in all seven systems, lower aggregate FSS18, worse onset MAE, poor onset/cessation interval coverage, and zero spread for radar-blind initiation failures.

The raw configuration is therefore **not frozen for final evaluation**, and the sealed six-system final set remains untouched. A future Stage 13 amendment would require a separately predeclared operating-policy or dispersion correction with weather-system-level validation; it must not reopen source-by-source precursor searching or imply that stochastic spread solves unseen initiation.

## Reproducibility artifacts

`artifacts/stage_13/` contains the immutable predeclaration, ensemble source/cache manifest with hashes and seeds, exact-repeat check, monotonic exceedance probabilities, deterministic/ensemble metrics by row/event/lead, reliability data, onset and cessation diagnostics, failure-category spread, hard-negative results, decision record, and a hash manifest. No artifact refers to or derives from the sealed final set.

## Stage 13A — Nested operating policy and exact trajectory uncertainty

Status: **PROMISING OPERATIONAL BASELINE — DEVELOPMENT PROCEDURE FROZEN; FINAL NOT SCORED.** Timing uncertainty is separately classified **UNDER-DISPERSED**.

### Input and regeneration audit

The Stage 13 predeclaration SHA-256 remains `d78f57bb212f5ad2cead122920b2e0cd2f9f9397f79169cd3dbfc6796f5bcb2c`. The existing Stage 13 evaluation was rerun before this amendment and reproduced deterministic F1 0.06307, raw ensemble Brier 0.05371, fixed-0.5 ensemble F1 0.04795, deterministic/ensemble FSS18 of 0.24767/0.22166, and zero hard-negative wet area under its original across-threshold summary.

Full members were regenerated with the unchanged row seeds. On five sampled rows, all four exceedance-probability arrays matched exactly. Ensemble mean and spread matched within their cached float16 quantization bounds (maximum differences no greater than 0.03125 and 0.015625), with identical missing-value masks. No ensemble parameter or probability field was refit.

### Nested operating threshold

For each outer weather system, the complete system and its hard negatives were excluded. The seven natural eight-member cut points were scored on the remaining systems. Selection maximized weather-event-macro F1 subject to hard-negative wet area no greater than 0.01 and false initiation no greater than 0.40, followed by the frozen FSS18/FAR/onset-MAE/higher-threshold tie-breaks.

Every outer fold independently selected **0.125**:

| Untouched outer system | Selected threshold |
|---|---:|
| Shared September 7/8 2021 | 0.125 |
| August 11 2021 | 0.125 |
| August 29 2022 | 0.125 |
| June 21 2021 | 0.125 |
| May 21 2022 | 0.125 |
| November 21 2021 | 0.125 |
| September 22 2021 | 0.125 |

Thus the outer result is genuinely system-held-out even though the selected policy is uniform.

### Categorical comparison

Metrics below use only untouched outer-system predictions at the 0.1 mm h-1 occurrence threshold:

| Event-macro metric | Deterministic | Ensemble at 0.5 | Stage 13A nested |
|---|---:|---:|---:|
| CSI | 0.1054 | 0.0997 | **0.1238** |
| POD | 0.1358 | 0.1194 | **0.2303** |
| FAR | 0.5287 | **0.3266** | 0.4567 |
| F1 | 0.1716 | 0.1600 | **0.1932** |
| FSS 6 km | 0.2029 | 0.1857 | **0.2145** |
| FSS 18 km | 0.24584 | 0.21858 | **0.24587** |
| FSS 36 km | **0.2797** | 0.2414 | 0.2729 |
| Onset MAE | 42.92 min | 44.70 min | **39.11 min** |
| Onset bias | +39.55 min | +41.89 min | **+30.42 min** |

The original categorical loss was therefore substantially an operating-policy problem: requiring four of eight members was too conservative. Requiring at least one member restores and modestly improves categorical skill, although FAR rises relative to the 0.5 policy.

### Event consistency

| System | Deterministic F1 | Fixed-0.5 F1 | Stage 13A F1 | Change vs deterministic | Deterministic FSS18 | Stage 13A FSS18 |
|---|---:|---:|---:|---:|---:|---:|
| Shared Sep 7/8 | 0.2657 | 0.2514 | 0.2831 | +0.0173 | 0.3801 | 0.3510 |
| Aug 11 | 0.1175 | 0.0970 | 0.1106 | -0.0069 | 0.1633 | 0.1493 |
| Aug 29 | 0.1944 | 0.1817 | 0.2348 | +0.0404 | 0.2920 | 0.3186 |
| Jun 21 | 0.1474 | 0.1562 | 0.1985 | +0.0510 | 0.2462 | 0.2713 |
| May 21 | 0.0899 | 0.0921 | 0.0853 | -0.0045 | 0.1257 | 0.1117 |
| Nov 21 | 0.0358 | 0.0232 | 0.0291 | -0.0067 | 0.0520 | 0.0433 |
| Sep 22 | 0.3502 | 0.3181 | 0.4109 | +0.0608 | 0.4616 | 0.4759 |

Four systems improve, one is within 0.005 below deterministic, and two miss that tolerance. The predefined consistency gate therefore passes for five of seven systems.

### Exact member timing

Each of the eight actual member trajectories was evaluated using the two-consecutive-frame persistence rule. The 12.5th and 87.5th percentiles are the finest requested central endpoints supportable by eight members and should not be interpreted as smoothly estimated quantiles.

For observed onset locations, event-macro member summaries are:

- onset by 30/60/90/120 minutes: 0.156/0.190/0.210/0.219 of member-location trajectories;
- mean earliest and latest member onset: 94.19 and 112.75 minutes;
- mean 12.5th/median/87.5th percentiles: 97.92/104.47/110.01 minutes;
- median-onset MAE: **44.49 minutes**;
- median bias: **+41.88 minutes**;
- 12.5–87.5% interval coverage: **0.198**;
- interval width: **12.09 minutes**;
- no member onset despite observed onset: **0.669**.

Exact trajectory coverage is worse than the previous marginal approximation (0.198 versus 0.263), with a narrower interval. Poor coverage was not an artifact of marginal timing; the ensemble is genuinely under-dispersed and frequently has no wet member.

For initially wet cases, exact cessation median MAE is **23.98 minutes**, bias is **-16.34 minutes**, interval coverage is **0.266**, and width is **10.32 minutes**. This is essentially unchanged in error from deterministic cessation (24.19 minutes) and also under-dispersed.

### Radar-blind initiation and spread-error behavior

Across `no_initiation_signal` row/leads, 99.936% of pixels have zero wet members, 0.064% have at least one wet member, and approximately 0.0001% have at least four. Noise does not supply meaningful radar-blind initiation information.

Full-field spread-error correlation is -0.027 overall. By failure category it is +0.289 for displacement, +0.552 for decay persistence, -0.022 for growth underprediction, and undefined for no-initiation because spread is zero. Spread has some relationship to advective/decay uncertainty but does not diagnose the central growth/initiation failures.

### Hard negatives and promotion gates

Under each hard-negative system's outer-selected threshold, mean wet area and false-initiation fraction are both **0.000**; mean and maximum wet probability remain zero.

| Frozen gate | Result |
|---|---|
| A. F1 no worse than deterministic by 0.005 | Pass: +0.02162 |
| B. FSS18 no worse by 0.01 | Pass: +0.00003 |
| C. At least 4/7 systems within F1 tolerance | Pass: 5/7 |
| D. Ensemble Brier better in at least 6/7 | Pass: 7/7 |
| E. Hard-negative wet area <=0.01 | Pass: 0.000 |
| F. Hard-negative false initiation <=0.40 | Pass: 0.000 |

All six gates pass. Stage 13A is therefore **PROMISING OPERATIONAL BASELINE**, while exact timing uncertainty remains **UNDER-DISPERSED**. This classification preserves the raw Brier/AUC/reliability/sharpness results; threshold selection does not improve or recalibrate probabilities.

### Frozen procedure and stopping point

Following the predefined pass branch, all-development grouped selection chose deployment threshold **0.125**. The unchanged ensemble, seed policy, probability products, persistence/timing definitions, categorical policy, evaluation code, source hashes, and decision hash are frozen in `artifacts/stage_13a/final_procedure_manifest.json`.

The probabilistic system now merits a separately authorized one-time final evaluation. That evaluation was **not** performed here. No sealed-final feature, radar input, forecast, member, summary, or score was generated.

## Stage 13A — One-time final evaluation

Status: **FINAL MIXED — DEVELOPMENT BENEFIT DID NOT CLEANLY GENERALIZE.** Timing is separately classified **TIMING UNCERTAINTY REMAINS UNDER-DISPERSED.** The six-system holdout is now permanently consumed; no post-final retuning is permitted.

### Integrity and execution

Before access, the frozen procedure, evaluation-code, ensemble-manifest, and Stage 13A decision hashes matched. The final positive-object manifest matched SHA-256 `52e0f3df309862584e3391da9ae0b40caba58583f28df68e45f4cd940445359f`; the procedure manifest matched `6b891216dd79cbceb266864e2d50850a4699567fb58859431961ffe29b352313`. Counts matched the freeze: six independent systems, 142 positive rows, 124 unique issue times, and eight hard negatives in four systems.

Only radar histories, deterministic PySTEPS, and the frozen seeded eight-member STEPS ensemble were materialized. No HRRR, GOES, richer-MRMS precursor, Stage 8 gate, or source-advantage input was used. All 150 rows were retained.

The first attempted execution stopped on its first row before writing results because of a diagnostic-sampler indexing error. The identical seeded procedure was restarted after that mechanical correction. After all forecasts completed, a reporting-only decimal-column-name error was corrected by summarizing the already-written outputs; forecasts were not regenerated.

### Six-system operational comparison

| Event-macro metric | Deterministic PySTEPS | Ensemble at 0.5 | Frozen Stage 13A at 0.125 |
|---|---:|---:|---:|
| CSI | **0.1819** | 0.1721 | 0.1803 |
| POD | 0.2353 | 0.2048 | **0.3757** |
| FAR | 0.4922 | **0.3479** | 0.5856 |
| F1 | **0.2818** | 0.2669 | 0.2787 |
| FSS 6 km | **0.3316** | 0.3107 | 0.3123 |
| FSS 18 km | **0.3936** | 0.3604 | 0.3590 |
| FSS 36 km | **0.4380** | 0.3910 | 0.3948 |
| Onset MAE | 37.56 min | 39.73 min | **32.54 min** |
| Onset bias | +34.15 min | +37.05 min | **+21.22 min** |

The 0.125 policy raises detection and improves categorical onset timing, but its false alarms erase the development F1 gain. Final F1 is 0.0031 below deterministic rather than 0.0216 above it, and FSS18 is 0.0346 lower rather than equal. The categorical development benefit therefore did not cleanly generalize.

### Final event consistency

| Final system | Deterministic F1 | Stage 13A F1 | Difference | Deterministic FSS18 | Stage 13A FSS18 | Deterministic onset MAE | Stage 13A onset MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aug 20 2022 | 0.4163 | 0.4143 | -0.0020 | 0.5643 | 0.5226 | 31.68 | 26.19 |
| Jul 19 2022 | 0.1398 | 0.1365 | -0.0033 | 0.1997 | 0.1777 | 37.76 | 35.77 |
| Jul 24 2022 | 0.3652 | 0.3710 | +0.0058 | 0.5058 | 0.4718 | 37.76 | 31.46 |
| Jul 28 2022 | 0.3245 | 0.3204 | -0.0041 | 0.4594 | 0.4090 | 36.15 | 28.66 |
| Jun 16 2022 | 0.1631 | 0.1760 | +0.0129 | 0.2496 | 0.2472 | 43.47 | 38.46 |
| May 15 2022 | 0.2818 | 0.2538 | -0.0281 | 0.3828 | 0.3257 | 38.55 | 34.67 |

Using a ±0.005 descriptive tie band, two systems improve, three are approximately tied, and one worsens. Stage 13A improves onset MAE in all six, but FSS18 is lower in five of six.

### Final probabilities

Raw probabilities improve Brier over deterministic occurrence in all six systems and every reported threshold/lead group. For `>0.1 mm h-1`, event-macro Brier is 0.0496/0.0632/0.0751/0.0926 over 0–30/30–60/60–90/90–120 minutes, versus deterministic 0.0546/0.0723/0.0857/0.1006. Corresponding AUC falls from 0.679 to 0.654, 0.615, and 0.546 as lead increases.

The same Brier direction holds for 1.0, 2.5, and 5.0 mm h-1. Early AUCs are 0.691, 0.677, and 0.646, declining toward 0.55 by 90–120 minutes. Sharpness also falls with lead. Full lead-group reliability bins are retained in `reliability.csv`. These probabilities have reproducible discrimination and Brier value, but they were not recalibrated and should not be described as fully calibrated.

### Exact trajectory timing

Final exact-member onset results are:

- onset by 30/60/90/120 minutes: 0.204/0.256/0.284/0.296;
- mean member median onset: 96.84 minutes;
- median-onset MAE: 38.13 minutes;
- median bias: +35.68 minutes;
- 12.5–87.5% coverage: 0.299;
- interval width: 17.96 minutes;
- no member onset despite observed onset: 0.551.

Coverage improves from development's 0.198 but remains far below a useful central interval, and more than half of observed-onset locations have no onset member. Timing remains under-dispersed.

For initially wet cases, deterministic cessation MAE is 18.31 minutes versus ensemble-median 16.68 minutes. Ensemble cessation bias is -9.95 minutes, coverage is 0.444, and width is 15.91 minutes. The median improves modestly, but the interval remains under-dispersed.

### Failure-aware uncertainty and radar-blind initiation

Spread-error correlation is 0.855 overall, 0.767 for displacement, 0.898 for decay persistence, 0.864 for growth underprediction, and -0.089 for no-initiation signal. Spread tracks errors well where radar echoes exist, but not where initiation signal is absent.

For final `no_initiation_signal` cases, 99.841% of evaluated pixels have zero wet members, 0.159% have at least one, and none have four. Rare stochastic wet members are not evidence of atmospheric initiation skill.

### Final hard negatives

All eight frozen rows and all four hard-negative systems have mean/max `P(>0.1)`, Stage 13A wet area, false initiation, and Brier equal to zero. The excellent dry-case result generalizes, but does not offset categorical spatial degradation on positive systems.

### Final interpretation

The final scientific classification is **FINAL MIXED — DEVELOPMENT BENEFIT DID NOT CLEANLY GENERALIZE**:

- Brier improvement generalizes strongly: 6/6 systems and every lead/threshold group;
- hard-negative behavior remains perfect;
- the 0.125 policy improves POD and onset timing;
- overall F1 is approximately tied but slightly worse;
- neighborhood skill degrades, and only two systems show F1 gains beyond 0.005;
- timing intervals remain under-dispersed.

Project-level conclusions after consuming the final set:

A. **Best deterministic operational forecast:** deterministic Lucas–Kanade PySTEPS remains the best categorical/spatial reference.

B. **Best probabilistic operational forecast:** frozen eight-member STEPS supplies useful raw exceedance probabilities and expected rate, but its 0.125 categorical policy does not replace deterministic PySTEPS cleanly.

C. **Probability reliability:** Brier improves consistently and early-lead discrimination is useful; reliability is not sufficient to claim complete calibration, especially at long lead.

D. **Timing reliability:** exact onset and cessation intervals remain under-dispersed and should not be presented as calibrated confidence intervals.

E. **Radar-blind initiation:** no reliable capability has been demonstrated. Nearly all no-initiation pixels remain dry in every member.

No Stage 13A threshold, seed, member count, perturbation, cascade, AR setting, calibration, spread, or event membership may now be changed using this consumed final set.

## Final operational configuration

### Primary categorical/spatial forecast

Deterministic Lucas–Kanade PySTEPS.

### Supplemental probabilistic forecast

Frozen eight-member PySTEPS STEPS ensemble.

### Probability outputs

- `P(rate > 0.1 mm h-1)`
- `P(rate > 1.0 mm h-1)`
- `P(rate > 2.5 mm h-1)`
- `P(rate > 5.0 mm h-1)`

### Expected precipitation rate

Direct eight-member ensemble mean.

### Stage 13A 0.125 categorical mask

The `P(rate > 0.1 mm h-1) >= 0.125` mask is retained as an evaluated experimental operating policy, but it is not the primary categorical forecast. Final F1 was 0.2787 versus 0.2818 for deterministic PySTEPS, and final FSS18 was 0.3590 versus 0.3936. Although POD and categorical onset timing improved, increased false alarms and degraded neighborhood skill prevent clean replacement of deterministic PySTEPS.

### Probability result

The frozen STEPS ensemble provides useful raw probabilistic information and expected precipitation rate. Its Brier-score advantage generalized across independent development and final systems, but reliability is insufficient to describe the probabilities as fully calibrated, particularly at longer lead times.

Brier improved in all seven development systems and all six final systems. The same direction held across the reported lead groups and precipitation thresholds, while final hard-negative behavior remained perfect.

### Timing result

**TIMING UNCERTAINTY REMAINS UNDER-DISPERSED.** Final onset interval coverage was 0.299, mean interval width was 17.96 minutes, the no-member onset fraction was 0.551, and cessation interval coverage was 0.444. The empirical 12.5–87.5% member interval is not a calibrated confidence interval.

### Radar-blind initiation result

Reliable radar-blind precipitation initiation has not been demonstrated in this project. In the final Stage 13A `no_initiation_signal` cases, 99.841% of evaluated pixels had zero wet ensemble members, 0.159% had at least one, and none had four. This conclusion is limited to the tested data, representations, models, and weather systems; it is not a claim that radar-blind initiation prediction is impossible in general.
