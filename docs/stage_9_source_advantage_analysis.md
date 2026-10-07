# Stage 9 — Source-advantage and PySTEPS-centered diagnosis

Status: **DIAGNOSTIC COMPLETE — SOURCE ADVANTAGE WEAK / REGIME-DEPENDENT.** No promotable forecast model was trained. PySTEPS is the principal operational radar reference. The sealed Stage 8 six-system final set was not read, featurized, predicted, or scored.

## Scope and data

This analysis uses only previously consumed data: the three positive 2023 Stage 6 evaluation systems (plus their already-consumed heavy-rain and dry controls) and the seven Stage 8 nested-development systems. Weather-system identity remains the replication unit. The diagnostic table preserves collection, independent system, row/object identity, issue time, and forecast lead.

The oracle analysis contains 6,769 valid row-leads from 340 rows. Truth-derived source-advantage values and labels are analysis targets only; they are not causal inputs. Positive `delta_brier_hrrr_vs_pysteps` means HRRR has lower pixel Brier score than PySTEPS. The analogous hybrid quantity uses a diagnostic 50/50 PySTEPS/HRRR occurrence mixture. Stage 8 was never fitted as one all-development deployable gate, so it is reported only on its legitimate seven outer-held-out systems and is not retrospectively applied to Stage 6.

## Cross-event baseline hierarchy

| Collection | Model | Events | Mean F1 | SD | Range | Precision | Recall | False initiation | Onset MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 6 consumed 2023 | PySTEPS | 3 | 0.350 | 0.102 | 0.203 | 0.600 | 0.270 | 0.237 | 15.91 min |
| Stage 6 consumed 2023 | A+ | 3 | 0.181 | 0.024 | 0.046 | 0.483 | 0.145 | 0.517 | 39.49 min |
| Stage 6 consumed 2023 | Raw HRRR | 3 | 0.121 | 0.098 | 0.191 | 0.477 | 0.084 | 0.244 | 39.82 min |
| Stage 6 consumed 2023 | Stage 6 hybrid | 3 | 0.164 | 0.091 | 0.182 | 0.604 | 0.124 | 0.291 | 33.46 min |
| Stage 8 nested DEV | PySTEPS | 7 | 0.350 | 0.213 | 0.588 | 0.645 | 0.290 | 0.248 | 14.59 min |
| Stage 8 nested DEV | A+ | 7 | 0.277 | 0.211 | 0.631 | 0.483 | 0.266 | 0.517 | 34.91 min |
| Stage 8 nested DEV | Raw HRRR | 7 | 0.318 | 0.204 | 0.563 | 0.525 | 0.320 | 0.341 | 30.73 min |
| Stage 8 nested DEV | Stage 6 hybrid | 7 | 0.346 | 0.190 | 0.519 | 0.530 | 0.358 | 0.390 | 28.96 min |
| Stage 8 nested DEV | Stage 8 gate | 7 | 0.263 | 0.211 | 0.634 | 0.541 | 0.239 | 0.391 | 34.70 min |

PySTEPS beats A+ in all 10 independent positive systems. It beats raw HRRR in all three Stage 6 systems and four of seven Stage 8 systems. Its raw event variance is not uniformly smaller than A+'s—the Stage 6 F1 standard deviation is larger—but its ranking, false-initiation behavior, and onset error are substantially more consistent. The evidence supports stability in comparative performance, not a claim that PySTEPS always has the smallest numerical variance.

## Oracle source advantage

On Stage 8 development, the row-lead oracle labels comprise 996 cases where both HRRR and the combination improve PySTEPS, 43 where HRRR alone improves it, 821 where only the combination improves it, 1,152 where PySTEPS is clearly better, and 2,172 approximate ties. On consumed Stage 6 positives/heavy rain, the respective counts are 252, 0, 200, 176, and 561.

Raw HRRR beats PySTEPS on mean Brier only for May 21 2022 (+0.0017) and July 12 2023 (+0.0016); it is worse for the other nine positive/heavy-rain events, including June 21 2021 (-0.218), September 22 2021 (-0.144), and November 21 2021 (-0.103). The diagnostic PySTEPS/HRRR mixture improves mean Brier in eight of eleven positive/heavy-rain events, but loses badly on June 21 and November 21. This is complementarity, not reliable raw-HRRR superiority.

Raw HRRR has negative mean advantage on Stage 8 radar-limited cases (-0.0246) and Stage 6 radar-limited cases (-0.0103). It is slightly positive on the already-consumed Stage 6 strict-V2 cohort (+0.0013), while the combination is +0.0068. Stage 8 has zero strict-V2 cases, so no Stage 8 strict radar-poor conclusion is made.

## PySTEPS failures and HRRR response

The failure categories are descriptive post-forecast heuristics, not training labels. On Stage 8 development:

- For no-initiation-signal cases, HRRR clearly corrects 7.3%, partially corrects 4.8%, also misses 65.0%, and worsens 22.9%.
- For growth underprediction, it clearly corrects 17.4%, partially corrects 26.0%, also misses 15.2%, and worsens 41.4%.
- For displacement, it clearly corrects 23.1%, partially corrects 24.7%, also misses 10.3%, and worsens 41.9%.
- For decay persistence, it clearly corrects 36.3% and partially corrects 29.7%, but worsens 23.6%.
- For intensity error, it worsens 94.4%.

Consumed Stage 6 cases show the central limitation again: HRRR clearly corrects only 2.2% of PySTEPS no-initiation failures and also misses 76.7%. Coarse APCP supplies little reliable direct initiation signal where extrapolation has none.

## HRRR failures

Spatial displacement dominates: 3,016 Stage 8 row-leads and 672 Stage 6 row-leads. HRRR beats PySTEPS in only 19.4% and 21.9% of these cases. Broad-area/poor-localization cases are strongly harmful (mean advantage -0.126 on Stage 8). The first-hour “hourly signal too early” diagnostic contains 27 Stage 8 cases and HRRR wins only 3.7%. These patterns identify displacement and hourly temporal coarseness as the main limitations of the current APCP representation.

## Exploratory predictability

A small λ=1 logistic diagnostic was evaluated with grouped leave-one-event-out prediction on the seven Stage 8 systems. Inputs were issue-time-safe lead, radar coverage, PySTEPS coverage, A+ entropy, HRRR coverage, PySTEPS/HRRR disagreement, and radar-support overlap. Recent radar growth existed in consumed Stage 6 source tensors but not in the detached Stage 8 feature cache; it was therefore non-informative in this common-feature diagnostic. No event identity was supplied.

| Held-out event | AUC | Accuracy | Selected Brier | Always PySTEPS | Fixed mixture |
|---|---:|---:|---:|---:|---:|
| Sep 7/8 shared | 0.636 | 0.539 | 0.1211 | 0.1211 | 0.0978 |
| Aug 11 | 0.601 | 0.723 | 0.0708 | 0.0708 | 0.0662 |
| Aug 29 | 0.599 | 0.627 | 0.0817 | 0.0817 | 0.0743 |
| Jun 21 | undefined | 1.000 | 0.0623 | 0.0623 | 0.1022 |
| May 21 | 0.521 | 0.477 | 0.0636 | 0.0636 | 0.0540 |
| Nov 21 | 0.654 | 0.814 | 0.0533 | 0.0533 | 0.0726 |
| Sep 22 | 0.374 | 0.773 | 0.2207 | 0.2207 | 0.1734 |

Event-macro AUC is 0.564. The classifier's choices are identical to always choosing PySTEPS: mean selected Brier is 0.09621 for both, while the fixed mixture is better at 0.09150. Accuracy is inflated by event-specific base rates. Held-out AUC ranges from 0.374 to 0.654, and one event has no positive advantage targets. Source advantage is not reliably captured by the current causal features.

## Why Stage 8 g became nearly constant

The exact eight Stage 8 features contain substantial event structure. Between-event variance accounts for 63.0% of HRRR wet-fraction variance, 45.6% of current-radar wet-fraction variance, and 36.2% of PySTEPS wet-fraction variance. Row-level associations therefore partly encode regimes that do not transfer cleanly across held-out systems.

All exact features except lead have negative aggregate association with raw-HRRR advantage. HRRR wet fraction is strongest (`ρ=-0.531`) and negative in all seven systems; A+/HRRR disagreement is `ρ=-0.419`, and HRRR/radar support is `ρ=-0.372`. Several weaker effects change sign between systems. This favors general HRRR attenuation rather than selective positive trust.

The λ=1 penalty term is only 0.010–0.032% of final objective value, so it does not dominate total loss numerically. Under the event-balanced pixel-Brier gradient, however, standardized slopes remain below 0.0042 in absolute value and fold intercepts explain most gate variation. Constant-like `g` is therefore a joint consequence of weak/inconsistent conditional gradients, event-balanced Brier, and regularization—not evidence that λ alone caused failure. Stage 8 was not refitted.

## Brier versus initiation utility

Stage 8 improves Brier over A+ in six of seven systems but improves initiation F1 in only three. On Aug 11, Aug 29, and May 21, Brier improves while both F1 and recall decline. Brier and F1 directions agree in only four of seven systems; onset MAE disagrees with Brier improvement in three.

Pixel calibration rewards conservative changes over many pixels, while promotion depends on detecting small initiation footprints at useful times without excess false initiation. This is a material objective mismatch, documented diagnostically without revising Stage 8.

## Operational reference and decision

PySTEPS should become the principal operational radar reference. It beats A+ on all 10 positive systems, roughly halves onset MAE in both collections, and maintains much lower false initiation. A+ remains a learned-radar research comparator, but beating A+ alone is no longer a sufficient multimodal result.

**Source-advantage classification: WEAK / REGIME-DEPENDENT.** There is oracle complementarity, especially from conservative mixing, but current issue-time-safe features do not turn it into reliable held-out-event source selection. Another gate over the same coarse APCP representation is not justified. Any future PySTEPS-centered experiment requires a separate predeclaration and genuinely improved information or representation, such as higher-temporal-resolution or ensemble NWP and displacement-aware processing. The sealed Stage 8 final set remains untouched.

Machine-readable tables and hashes are under `artifacts/stage_9/source_advantage/`.
