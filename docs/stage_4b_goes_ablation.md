# Stage 4B GOES C13 ablation report

Status: **complete**. The procedure remains frozen as **A+ / NULL GOES RESULT**
from DEV. The exact protected holdout was subsequently materialized and scored
once, producing a **STRONG GOES EFFECT on the fresh holdout**. This discrepancy
did not reopen model selection.

Stage 4B has advanced beyond the first B1/B2 DEV pass. B1 and B2 were trained
from the frozen Stage 3.1 radar-only control, DEV lifecycle diagnostics were
run, and one modest radar-only A+ capacity control was trained because B2 showed
a meaningful DEV advantage over B1. The added A+ result makes the DEV evidence
mixed rather than cleanly freeze-ready.

Machine-readable decision records:

- `artifacts/stage_4b/stage_4b_dev_decision_manifest.json`
- `artifacts/stage_4b/final_dev_decision/dev_final_decision.json`
- `artifacts/stage_4b/stage_4b_frozen_procedure_manifest.json`

The final holdout used 68 exact issue-time/object tiles. The earlier 20 coarse
positive event tensors were not substituted for them. No threshold, checkpoint,
calibration, lifecycle definition, or selection rule was changed.

## Gate status

The Stage 4 gate remains modality-specific:

| Gate family | Status |
|---|---|
| B: radar + GOES C13 | pass |
| C-F: NWP/static | not complete |

B is not blocked by HRRR/NWP or full static-feature materialization. Those
remain prerequisites for Ablations C-F only.

## Models evaluated on DEV

All learned models use the same Stage 4 DEV-selection tensors. The final
initiation holdout remains sealed.

| Model | Inputs | Best DEV loss | Params | Notes |
|---|---|---:|---:|---|
| Stage 3.1 radar control | radar only | frozen earlier | ~528k | immutable baseline |
| B1 | radar + raw C13 history | 1.1998 | 584,962 | weak GOES-dependence diagnostic |
| B2 | radar + raw C13 + 10/20/30/60 min C13 cooling | 1.1973 | 585,538 | best broad expected-rate CSI/FAR/FSS |
| A+ | radar-only widened capacity control | 1.1981 | 579,125 | one modest capacity control, no sweep |

Artifacts:

- B1: `artifacts/stage_4b/b1_raw_c13/`
- B2: `artifacts/stage_4b/b2_c13_cooling/`
- A+: `artifacts/stage_4b/a_plus_radar/`
- lifecycle diagnostics: `artifacts/stage_4b/lifecycle/`

## DEV expected-rate metrics at 0.1 mm/h

These categorical metrics use the expected precipitation-rate field. Brier score
uses the corresponding probability-head exceedance probability. The evaluated
set is DEV + Stage 4 DEV repair only: 24 tensors.

| Model | Lead | CSI | POD | FAR | F1 | FSS 18 km | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| PySTEPS | 30 | 0.292 | 0.453 | 0.549 | 0.452 | 0.536 | 0.188 |
| B1 | 30 | 0.291 | 0.990 | 0.708 | 0.451 | 0.328 | 0.188 |
| B2 | 30 | 0.311 | 0.983 | 0.687 | 0.475 | 0.344 | 0.169 |
| A+ | 30 | 0.284 | 0.992 | 0.715 | 0.443 | 0.321 | 0.157 |
| PySTEPS | 60 | 0.221 | 0.360 | 0.637 | 0.362 | 0.521 | 0.204 |
| B1 | 60 | 0.264 | 0.992 | 0.736 | 0.417 | 0.295 | 0.217 |
| B2 | 60 | 0.289 | 0.971 | 0.709 | 0.448 | 0.322 | 0.190 |
| A+ | 60 | 0.277 | 0.977 | 0.721 | 0.434 | 0.309 | 0.181 |
| PySTEPS | 90 | 0.155 | 0.273 | 0.737 | 0.268 | 0.403 | 0.214 |
| B1 | 90 | 0.253 | 0.998 | 0.747 | 0.403 | 0.286 | 0.232 |
| B2 | 90 | 0.265 | 0.974 | 0.733 | 0.419 | 0.303 | 0.197 |
| A+ | 90 | 0.260 | 0.956 | 0.736 | 0.413 | 0.292 | 0.206 |
| PySTEPS | 120 | 0.105 | 0.201 | 0.820 | 0.190 | 0.336 | 0.207 |
| B1 | 120 | 0.235 | 0.999 | 0.765 | 0.381 | 0.265 | 0.251 |
| B2 | 120 | 0.240 | 0.980 | 0.759 | 0.387 | 0.274 | 0.225 |
| A+ | 120 | 0.160 | 0.970 | 0.839 | 0.275 | 0.266 | 0.195 |

B2 is still the best learned model on broad expected-rate CSI/FAR/FSS at these
lead times. A+ is not a categorical-skill win, but it nearly matches B2 DEV loss
and improves Brier at several leads, so the B2 advantage cannot yet be attributed
cleanly to satellite information rather than training/capacity/probability
calibration.

## GOES-dependence diagnostics on DEV

B1 does not show healthy dependence on raw C13 alone: zeroing GOES is often
slightly better than normal GOES. B2 behaves more plausibly: zeroing GOES
degrades CSI and Brier at 30-120 minutes. Shuffling GOES weakens B2 less than
zeroing, so the satellite-use signal is positive but not decisive.

| Model | Perturbation | 60-min CSI | 60-min Brier | 120-min CSI | 120-min Brier |
|---|---|---:|---:|---:|---:|
| B1 | normal | 0.264 | 0.217 | 0.235 | 0.251 |
| B1 | zeroed GOES | 0.279 | 0.208 | 0.236 | 0.242 |
| B1 | shuffled GOES | 0.265 | 0.218 | 0.235 | 0.250 |
| B2 | normal | 0.289 | 0.190 | 0.240 | 0.225 |
| B2 | zeroed GOES | 0.277 | 0.210 | 0.236 | 0.245 |
| B2 | shuffled GOES | 0.285 | 0.197 | 0.240 | 0.230 |

## DEV probability-policy diagnostics

Lifecycle metrics separate the probability head from expected-rate categorical
rain. The threshold sweep is DEV-only and should not be retuned on test or final
holdout data.

Candidate initiation probability thresholds:

| Model | P threshold | Precision | Recall | F1 | Onset MAE min | Median bias min | False-initiation frac | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.30 | 0.639 | 0.960 | 0.767 | 26.75 | -12 | 0.361 | 0.206 |
| B2 | 0.30 | 0.654 | 0.872 | 0.747 | 25.85 | -12 | 0.346 | 0.216 |
| A+ | 0.30 | 0.674 | 0.850 | 0.752 | 26.62 | -12 | 0.326 | 0.217 |
| B2 | 0.50 | 0.703 | 0.767 | 0.734 | 22.50 | -12 | 0.297 | 0.216 |

The simple DEV selection score currently ranks B1 at threshold 0.30 highest, but
that is not a sufficient reason to select B1 for Stage 4B because B1 fails the
GOES-dependence diagnostic and has weaker broad expected-rate skill. B2 at 0.30
or 0.50 and A+ at 0.30 remain plausible DEV policies depending on whether the
priority is recall, false-initiation control, or attribution to GOES.

The final common rule selected A+ at 0.35 and B2 at 0.20. A+ is the operational
Stage 4B model; calibration is the identity mapping.

## DEV dissipation diagnostics

Dissipation probability-head metrics show B2 has a real DEV advantage over B1
and the Stage 3.1 radar control at low probability thresholds, but optical flow
remains a strong radar-only motion baseline.

| Model / field | Threshold | Precision | Recall | F1 | Cessation MAE min | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Stage 3.1 probability head | 0.05 | 0.996 | 0.672 | 0.802 | 37.26 | 0.772 |
| B1 probability head | 0.05 | 0.996 | 0.666 | 0.798 | 27.80 | 0.748 |
| B2 probability head | 0.05 | 0.997 | 0.944 | 0.969 | 26.33 | 0.659 |
| PySTEPS expected field | n/a | 0.996 | 0.743 | 0.851 | 29.19 | 0.259 |
| optical-flow expected field | n/a | 0.997 | 0.941 | 0.968 | 28.64 | 0.062 |

Expected-rate lifecycle fields remain weak for some cessation questions; the
probability head is the more informative learned output for lifecycle timing.

## DEV hard-negative behavior

The two active DEV hard-negative events were evaluated for B1/B2. These are not
the final-holdout hard negatives.

| Model | Event | Mean P(rain > 0.1) | Max P(rain > 0.1) | Wet-area fraction P >= 0.5 | Brier |
|---|---|---:|---:|---:|---:|
| B1 | stage4_dev_dry_favourable_2026_07_10 | 0.043 | 0.775 | 0.0029 | 0.0070 |
| B1 | stage4_dev_post_storm_dry_2026_05_21 | 0.020 | 0.322 | 0.0000 | 0.0007 |
| B2 | stage4_dev_dry_favourable_2026_07_10 | 0.055 | 0.726 | 0.0002 | 0.0056 |
| B2 | stage4_dev_post_storm_dry_2026_05_21 | 0.047 | 0.363 | 0.0000 | 0.0028 |

B2 raises mean rain probability on both DEV hard negatives but keeps the
P >= 0.5 wet-area fraction near zero.

## Current decision answers

1. Does raw C13 add predictive skill? Not convincingly; B1 is not robust to
   GOES removal.
2. Do C13 cooling tendencies add skill beyond raw C13? Yes on broad DEV
   expected-rate CSI/FAR/Brier versus B1.
3. Does B2 beat the radar-only capacity control? Mixed. B2 has better broad
   categorical expected-rate skill, but A+ nearly matches DEV loss and has
   competitive Brier/lifecycle behavior.
4. Is useful initiation proven? Not yet. Probability-head thresholds are
   informative, but false-initiation control and model selection remain mixed.
5. Does GOES help dissipation? B2 improves learned probability-head dissipation
   metrics on DEV, though optical flow remains very strong.
6. Does B2 justify selection? No. It failed the predeclared initiation and
   satellite-dependence gates despite small broad-field gains.
7. Does this justify Ablation C? Ablation C remains out of scope for this pass;
   it may proceed later if its NWP gates pass, including after this null result.

## Final DEV-only decision pass

The rule was declared before opening the holdout: require no more than 1% of DEV
hard-negative pixels above the operating threshold and no more than 40% false
initiations; maximize weather-event-macro initiation F1; then minimize onset MAE
and Brier. B2 promotion additionally required wins on at least two of three
initiation criteria, two of three broad criteria, and consistent degradation
under global shuffle, within-event shuffle, and zeroing.

| Model | Threshold | Precision | Recall | F1 | False initiation | Onset MAE | Hard-negative fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A+ | 0.35 | 0.709 | 0.804 | 0.749 | 0.291 | 25.97 min | 0.0059 |
| B2 | 0.20 | 0.643 | 0.912 | 0.750 | 0.357 | 27.77 min | 0.0069 |

The F1 delta was only +0.0009 for B2 (event bootstrap 95% interval -0.0168 to
+0.0141). B2 worsened false initiation by +0.0662 (interval +0.0404 to +0.1152)
and onset MAE by +1.80 minutes (interval -0.56 to +5.55). At 60 and 90 minutes,
mean CSI deltas were only +0.0036 and +0.0021 and both intervals crossed zero.

B2 normal initiation F1 was 0.7501. Global shuffle was better at 0.7526,
within-event shuffle was 0.7481, and all-GOES-zero was 0.7505. Cooling-zero
was 0.7496; raw-C13-zero was better at 0.7547. This fails the required consistent
satellite-information ordering. Reliability tables were retained, but no
post-hoc calibration was fitted because only six DEV weather events are present
and calibration would add another unstable selection degree of freedom.

Therefore Stage 4B is frozen as **A+ / NULL GOES RESULT**. This is a DEV selection
conclusion, not yet the protected-holdout interpretation.

## Exact holdout materialization

The original coarse tensor mismatch was repaired without changing the frozen
manifest. Exact tensors were created for all 68 clean rows at their 63 unique
issue times and object-specific tiles. Multiple objects at one issue time reuse
source scans but retain distinct identities and tiles.

Integrity results:

| Check | Result |
|---|---|
| Frozen manifest SHA-256 | `0d347e2bff4ab174af248b2345e709f0c0b12505a0542dcca573e10d2af08783` (unchanged) |
| Exact clean rows | 68/68 |
| Failures/unavailable rows | 0 |
| Radar input / GOES / target shapes | 10×128×128 / 5×5×128×128 / 20×128×128 |
| Exact issue timestamps and tiles | pass |
| Future radar in inputs | none |
| GOES latency | pass; causal scans at least 10 minutes behind issue time |
| Validity masks and dimensions | pass |

## Frozen final clean-initiation evaluation

Metrics below are row-macro results over the 68 exact rows. Event-level results
are retained separately for all five independent weather events.

| Model | Policy threshold | Precision | Recall | F1 | False initiation | Onset MAE | Median bias | Brier | Detect ≤30/60/90/120 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PySTEPS | deterministic | 0.707 | 0.371 | 0.449 | 0.249 | 16.60 | -0.55 | 0.265 | 0.499 / 0.446 / 0.407 / 0.371 |
| Stage 3.1 radar | 0.05 | 0.352 | 0.998 | 0.503 | 0.648 | 45.19 | -41.21 | 0.192 | 0.961 / 0.977 / 0.988 / 0.998 |
| A+ radar | **0.35 frozen** | 0.466 | 0.430 | 0.393 | 0.534 | 41.71 | -33.55 | 0.263 | 0.558 / 0.496 / 0.455 / 0.430 |
| B1 raw C13 | 0.30 | 0.463 | 0.722 | 0.524 | 0.537 | 38.77 | -27.82 | 0.249 | 0.774 / 0.767 / 0.743 / 0.722 |
| B2 C13 + cooling | **0.20 frozen** | 0.485 | 0.675 | 0.522 | 0.515 | 39.97 | -27.04 | 0.233 | 0.804 / 0.751 / 0.702 / 0.675 |

B2 improved over the capacity-matched A+ control by +0.128 F1, +0.246 recall,
+0.019 precision, -0.019 false-initiation fraction, -1.74 minutes onset MAE,
and -0.030 Brier. B2 had higher row-macro F1 in four of five independent events.
B1 and B2 were similar overall: B1 had slightly higher F1/recall, while B2 had
better precision, false-initiation control, Brier, and early-window detection.

| Independent event | A+ F1 | B2 F1 | B2 - A+ |
|---|---:|---:|---:|
| Aug 30 eastern Ontario | 0.575 | 0.516 | -0.059 |
| Jul 27 southwest Ontario | 0.341 | 0.489 | +0.148 |
| Jun 21 derecho | 0.465 | 0.596 | +0.131 |
| May 20 southwest Ontario | 0.333 | 0.472 | +0.139 |
| Sep 2 southern Ontario | 0.437 | 0.559 | +0.121 |

The unweighted five-event macro F1 is 0.430 for A+ and 0.526 for B2.

## Strict pre-radar subset

The subset definition was fixed before neural scoring: full radar history dry at
eventual-initiation pixels, valid radar coverage, no advective-entry designation,
and PySTEPS rain at no more than 5% of eventual-initiation pixels. Zero of 68 rows
met all conditions (zero independent events). The definition was not weakened
after seeing results, so no strict-subset performance estimate is reported.

## Protected final hard negatives

Sixteen frozen tensors across four independent hard-negative events were scored.

| Model | Mean P(rain) | Max P(rain) | Wet-area fraction above policy | False-initiation fraction | Brier |
|---|---:|---:|---:|---:|---:|
| PySTEPS | 0.00004 | 0.250 | 0.00004 | 0.250 | 0.00026 |
| Stage 3.1 | 0.0529 | 0.438 | 0.3605 | 0.997 | 0.00661 |
| A+ | 0.0713 | 0.456 | 0.000245 | 0.977 | 0.00674 |
| B2 | 0.0497 | 0.380 | 0.00309 | 0.989 | 0.00347 |

B2 triggered a larger wet area than A+ at their different frozen operational
thresholds, but the absolute affected area remained about 0.31%. B2 had lower
mean probability, maximum probability, and Brier than A+. The very high learned
false-initiation fractions reflect that almost every rare predicted initiation
on these dry events is false; wet-area fraction conveys their spatial extent.

## Final Stage 4B interpretation

The final scientific classification is **STRONG INCREMENTAL GOES EFFECT OVER
CAPACITY-MATCHED RADAR-ONLY ON THE FRESH INITIATION-FOCUSED HOLDOUT**. It does
not establish strict pre-radar initiation forecasting: the predeclared
`STRICT_PRE_RADAR_INITIATION` subset contained zero qualifying rows. It also
supports a GOES-information effect, not a cooling-specific effect. B1 and B2
were similar on the final holdout, so attribution to explicit cooling tendencies
remains uncertain. This contradicts the DEV **A+ / NULL GOES RESULT**, where B2
failed the satellite-dependence gate. The discrepancy is retained as
generalization uncertainty and does not reopen selection. Stage 4B is closed.
