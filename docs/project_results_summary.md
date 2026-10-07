# Southern Ontario 0–2 h Precipitation Nowcasting — Project Results

## Objective

Develop and evaluate a Southern Ontario 0–2 h precipitation nowcasting system for:

- precipitation occurrence;
- onset and cessation timing;
- intensity;
- spatial evolution;
- uncertainty;
- precipitation initiation.

## Final operational system

### Deterministic backbone

Deterministic Lucas–Kanade PySTEPS is the primary categorical and spatial forecast.

### Probabilistic supplement

The frozen eight-member PySTEPS STEPS ensemble supplements the deterministic forecast with exceedance probabilities for 0.1, 1.0, 2.5, and 5.0 mm h-1 and a direct ensemble-mean expected precipitation rate. Its probability layer is operationally useful, but the Stage 13A 0.125 mask is not designated as the primary categorical forecast.

## Major experimental findings

1. Optical-flow and PySTEPS baselines established that radar extrapolation is strong.
2. Learned radar-only models did not consistently exceed PySTEPS.
3. GOES augmentation produced interesting earlier isolated evidence but did not demonstrate transferable incremental precursor value under later controlled PySTEPS-centered evaluation.
4. HRRR contains some complementary information, but its benefit is highly regime-dependent.
5. Fixed HRRR blending and learned conditional HRRR gating did not generalize.
6. Continuous APCP, causal displacement correction, 15-minute HRRR APCP/PRATE/REFC, and radar-like HRRR representations did not justify fusion.
7. GOES C13/cooling did not improve held-out PySTEPS failure discrimination.
8. Richer MRMS reflectivity, echo-top, and VIL features did not provide useful reproducible pre-initiation signal.
9. Probabilistic STEPS improved Brier consistently.
10. The low-probability Stage 13A categorical policy improved development skill but did not cleanly generalize on the final holdout.
11. Ensemble timing distributions remain under-dispersed.
12. Reliable radar-blind initiation has not been demonstrated.

## Final final-holdout comparison

| Metric | Deterministic PySTEPS | Stage 13A 0.125 |
|---|---:|---:|
| CSI | 0.1819 | 0.1803 |
| POD | 0.2353 | 0.3757 |
| FAR | 0.4922 | 0.5856 |
| F1 | 0.2818 | 0.2787 |
| FSS 6 km | 0.3316 | 0.3123 |
| FSS 18 km | 0.3936 | 0.3590 |
| FSS 36 km | 0.4380 | 0.3948 |
| Onset MAE | 37.56 min | 32.54 min |
| Onset bias | +34.15 min | +21.22 min |

The 0.125 policy detects substantially more precipitation and improves categorical onset timing, but raises false alarms and reduces neighborhood skill. It therefore does not cleanly replace deterministic PySTEPS.

## Probabilistic result

Brier improvement generalized in all six final systems after appearing in all seven development systems. Probabilistic discrimination is strongest at shorter leads and declines toward 120 minutes. The raw probabilities are useful but are not claimed to be fully calibrated.

The frozen STEPS ensemble provides useful raw probabilistic information and expected precipitation rate. Its Brier-score advantage generalized across independent development and final systems, but reliability is insufficient to describe the probabilities as fully calibrated, particularly at longer lead times.

## Timing uncertainty

Final exact-member onset coverage was 0.299, interval width was 17.96 minutes, and 55.1% of observed onset locations had no member onset. Cessation interval coverage was 0.444. **TIMING UNCERTAINTY REMAINS UNDER-DISPERSED.** The empirical member interval must not be presented as a calibrated confidence interval.

## Radar-blind initiation

Negative or non-transferable initiation evidence was obtained from deterministic and subhourly HRRR representations, GOES C13/cooling, richer MRMS radar descriptors, and stochastic PySTEPS. In final Stage 13A no-initiation cases, 99.841% of evaluated pixels had zero wet members, 0.159% had at least one, and none had four.

The experiments did not identify a reproducible, issue-time-safe source of precipitation-initiation information that generalized across independent weather systems. This conclusion applies to the tested data, models, representations, and systems; it does not establish that radar-blind initiation prediction is impossible in general.

## Main methodological lesson

Many apparent improvements were event- or regime-specific. Future work should emphasize:

- substantially more independent weather systems;
- event-level validation;
- new development and final splits;
- stronger model/data regimes rather than repeated one-channel feature additions;
- PySTEPS as the reference baseline.

## Holdout status

**The six-system Stage 13A final holdout has now been permanently consumed.**

It must never be used for:

- threshold selection;
- model tuning;
- calibration selection;
- feature selection;
- architecture selection.

Any future research experiment requires new independent development and evaluation systems.

## Reproducibility references

- [Stage 13 predeclaration](../artifacts/stage_13/stage13_predeclaration.json) — SHA-256 `d78f57bb212f5ad2cead122920b2e0cd2f9f9397f79169cd3dbfc6796f5bcb2c`
- [Stage 13A final procedure](../artifacts/stage_13a/final_procedure_manifest.json) — SHA-256 `6b891216dd79cbceb266864e2d50850a4699567fb58859431961ffe29b352313`
- [Stage 13A final evaluation](../artifacts/stage_13a/final_evaluation/stage13a_final_evaluation.json) — recorded final-evaluation SHA-256 `f582242f5126133546fd31ce007e6e2264bcd820f6fcfac5e600d4e1fdd9215e`
- [Stage 9 source-advantage diagnosis](stage_9_source_advantage_analysis.md)
- [Stage 10/10B HRRR representation diagnosis](stage_10_hrrr_representation_analysis.md)
- [Stage 11 GOES diagnosis](stage_11_goes_pysteps_precursor_analysis.md)
- [Stage 12 MRMS precursor diagnosis](stage_12_mrms_radar_precursor_analysis.md)

The recorded Stage 13A final-evaluation hash refers to the completed scientific artifact before this documentation-only closure. No forecasts or metrics were recomputed during closure.
