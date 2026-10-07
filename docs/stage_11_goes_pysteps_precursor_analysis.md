# Stage 11 — GOES precursor value for PySTEPS failure

Status: **DIAGNOSTIC COMPLETE — GOES C13/COOLING NOT USEFUL.** This was a diagnostic-only experiment. No promotable forecast model or PySTEPS correction was trained. The analysis used the seven consumed Stage 8 development systems, 260 clean positive-initiation rows, and 12 consumed hard negatives. The sealed six-system final set was not accessed, materialized, summarized, or scored.

## Scientific question and frozen design

Stage 11 asked whether issue-time-safe GOES cloud-top information contains reproducible precursor information for precipitation initiation or growth that PySTEPS cannot observe from radar history. The descriptors and physical thresholds were frozen before joining future-truth failure labels:

- current GOES-East ABI C13 mean, minimum, 10th percentile, and fraction colder than **235 K**;
- mean C13 cooling over nominal 10, 20, 30, and 60-minute windows;
- 30-minute cooling 90th percentile and fraction cooling faster than **8 K h⁻¹**;
- cold/rapid-cooling fractions not covered by current radar or the PySTEPS forecast;
- context limited to forecast lead, current radar wet fraction, and PySTEPS wet fraction.

Cooling is baseline C13 minus current C13, so positive values mean colder cloud tops. C13 is treated only as infrared cloud-top brightness temperature; neither C13 nor cooling is converted to rainfall. The fixed Stage 4B linear remapping was reused on the existing 128×128, 2-km radar tiles with an observation-validity fraction.

The predefined diagnostic target is a PySTEPS initiation/growth failure: `no_initiation_signal` or `growth_underprediction` under the existing Stage 9 taxonomy. Labels use future truth only as outcomes and never enter a GOES feature. One L2-regularized logistic regression (`lambda=1`) was evaluated by leave-one-independent-system-out validation. Every held-out prediction used normalization and coefficients fit on the other six systems only.

## Historical availability and causal provenance

GOES-16 was verified—not assumed—to be the applicable GOES-East satellite for all required 2021–2022 issue times. The audited product was NOAA `ABI-L2-CMIPC`, channel 13. For the 223 unique development issue times, the current and four cooling-baseline requirements produced 1,115 source records and 761 unique archive files. **All 1,115 records were available and causal.**

The latest nominal source time was issue minus 10 minutes. Candidate files were accepted only when both scan end and archive creation were at or before the simulated issue time. The audit preserves satellite, product, scan start/end, archive creation time, source key, local path, ETag, byte count, and SHA-256. Actual scan separations were 0–10, 0–20, 0–30, and 0–85 minutes for the nominal 10/20/30/60-minute baselines; medians were exactly 10/20/30/60 minutes. The actual times—not nominal labels—remain in the provenance table.

Of the 260 positive rows, 255 had complete current-plus-cooling tile coverage. Five Aug 29 rows at the CONUS product edge lacked complete coverage and were excluded rather than imputed. All seven systems remained represented. All 12 hard negatives were usable.

## GOES conditions by PySTEPS outcome

Event-row-lead descriptive means were:

| Existing PySTEPS category | C13 mean (K) | Cold fraction | Mean 30-min cooling (K) | Rapid-cooling fraction | Interpretation |
|---|---:|---:|---:|---:|---|
| No initiation signal | 272.42 | 0.067 | -1.087 | 0.236 | Misses were warmer and warming on average, not marked by a strong cold/cooling precursor. |
| Growth underprediction | 266.86 | 0.097 | +0.246 | 0.273 | Slight mean cooling exists, but separation is small. |
| Other/adequate | 266.44 | 0.128 | -0.337 | 0.251 | Similar to or colder than growth failures. |
| Decay persistence | 256.55 | 0.257 | +1.313 | 0.324 | The strongest cloud signal occurs with existing/decaying systems, not cleanly with new initiation. |
| Displacement | 268.50 | 0.113 | -0.103 | 0.234 | No distinctive precursor. |
| Intensity error | 258.71 | 0.198 | +3.245 | 0.386 | Strong cooling is concentrated in an already-active/intensity regime. |

Thus raw cold cloud and cooling are meteorologically meaningful, but they do not uniquely identify the failure mode where a non-radar precursor is most valuable. In particular, `no_initiation_signal` cases do not show stronger average cooling than successful/adequate cases. The `growth_underprediction` separation is weak relative to event-to-event variation.

## Lead dependence

Event-macro held-out AUC by future lead window was:

| Model | 0–30 | 30–60 | 60–90 | 90–120 |
|---|---:|---:|---:|---:|
| Radar/PySTEPS context only | 0.774 | 0.711 | 0.717 | 0.739 |
| Context + current C13 | 0.726 | 0.623 | 0.609 | 0.649 |
| Context + cooling | 0.695 | 0.660 | 0.619 | 0.594 |
| Context + all GOES | 0.666 | 0.638 | 0.597 | 0.586 |

GOES does not reveal a useful precursor window before radar/PySTEPS. The all-GOES diagnostic is worse than context alone in every lead group, including 0–30 minutes.

## Held-out-event diagnostic performance

Primary event-macro results across the seven untouched systems were:

| Inputs | AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| Context only | **0.773** | 0.706 | 0.996 | 0.823 | 0.2005 |
| Context + current C13 | 0.731 | 0.707 | 0.996 | 0.823 | 0.2025 |
| Context + cooling | 0.740 | 0.706 | 0.994 | 0.822 | **0.1996** |
| Context + current C13 + cooling | 0.712 | 0.707 | 0.992 | 0.823 | 0.2012 |

The event-balanced target prevalence is high (event-macro 0.701), so threshold-0.5 precision/recall/F1 are close to the always-failure behavior and are not evidence of incremental GOES value. The discriminative endpoint is AUC, supported by Brier/calibration. The training-event base-rate null has AUC 0.5 in every held-out event; context adds substantial discrimination. GOES does not add to that context.

All-GOES-minus-context AUC changes by system were -0.109, -0.121, -0.089, +0.006, -0.200, +0.062, and +0.022. Only three of seven systems improve at all, and only two improve by more than 0.01. This is regime-dependent sign reversal, not reproducible precursor value.

## Destruction tests and raw C13 versus cooling

For the same fitted all-GOES checkpoints:

| Evaluation condition | Event-macro AUC | Brier |
|---|---:|---:|
| Normal GOES | 0.712 | 0.2012 |
| GOES removed at training normalization means | 0.773 | 0.2006 |
| GOES globally shuffled | 0.709 | 0.2009 |
| GOES shuffled within held-out event | 0.709 | 0.2004 |

Removal improves AUC back to the context-only level, while both shuffles are nearly indistinguishable from normal GOES. This is the opposite of the dependence pattern required for a useful precursor.

Cooling alone is less harmful than current C13 or their combination: AUC is 0.740 versus 0.731 for current C13 and 0.712 for both. Its small Brier improvement (0.0009) does not survive as discrimination and is accompanied by lower AUC. Raw C13 and cooling therefore do not demonstrate complementary incremental value.

## Failure-category discrimination

One-versus-rest event-macro AUC for the full GOES diagnostic versus context was:

| Category | Context AUC | Context + GOES AUC | Change |
|---|---:|---:|---:|
| No initiation signal | 0.775 | 0.698 | -0.077 |
| Growth underprediction | 0.623 | 0.623 | -0.001 |
| Displacement | 0.467 | 0.481 | +0.014 |
| Decay persistence | 0.395 | 0.467 | +0.072 |
| Intensity error | 0.019 | 0.234 | +0.215 |

The apparent gains are concentrated in decay/intensity categories, where cold active cloud is physically expected but does not solve radar-blind initiation. GOES degrades `no_initiation_signal` discrimination and is neutral for growth underprediction—the two priority categories.

## Hard negatives and event consistency

Across the five hard-negative systems, event-macro cold-cloud fraction was 0.043, rapid-cooling fraction 0.145, 30-minute strongest-cooling percentile 7.89 K, and radar-uncovered cold fraction 0.043. Some dry cases therefore contain substantial rapid cooling or cold cloud; for example, event means reach rapid-cooling fractions of 0.339 and 30-minute 90th-percentile cooling of 20.0 K. A simple cold/cooling trigger would not be reliably dry-safe.

Descriptor directions also change by system. Cold fraction and radar/PySTEPS-uncovered cold fraction are positively associated with the failure target in only two systems and negatively associated in five. Mean cooling descriptors have somewhat more consistent positive signs (four to six systems depending on window), but the held-out model shows that these correlations do not translate into incremental discrimination. Pooled pixel or row-lead correlations are therefore not treated as independent evidence.

## Decision

GOES precursor value is classified **NOT USEFUL** for this frozen diagnostic and development set:

- held-out-event AUC falls by 0.061 versus radar/PySTEPS context;
- gains do not replicate across a majority of independent systems;
- destruction does not cause meaningful degradation;
- no-initiation and growth-underprediction do not improve;
- physically strong satellite signatures also occur in active/decay regimes and hard negatives.

A separately predeclared PySTEPS+GOES initiation correction is **not justified**. PySTEPS remains the strongest validated 0–2-hour operational backbone in this project, while reliable exogenous initiation information has not been demonstrated. Further improvement requires genuinely new sources or a substantially different modeling/data regime, rather than another deterministic HRRR/GOES gate over the same information.

Machine-readable provenance, descriptors, held-out predictions, fold fits, calibration, lead/category summaries, destruction results, and hashes are under `artifacts/stage_11/`.
