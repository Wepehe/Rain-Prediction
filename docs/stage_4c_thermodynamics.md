# Stage 4C thermodynamic ablation

Status: source/causality, spatial-tensor, normalization, deterministic-baseline,
fresh-holdout, exact-tensor, and C1 pipeline/gradient gates passed. The 68-row
fresh holdout is frozen and fully materialized. C1 convergence training is
complete; its selected epoch-2 checkpoint has DEV loss 1.18364 and SHA-256
`d35dd3b975bd6495937985e11987f3b01dc28810da9fc519b42274079e677a4f`.
The final 2023 holdout remains unscored and the Stage C procedure is not yet
frozen.

## C1 DEV decision

The completed epoch-2 checkpoint was evaluated once on all 24 common DEV tensors
(12 positive-initiation anchors, 8 hard-negative anchors, 4 other positive-
precipitation anchors; six independent events). Before inference, the threshold
grid, constraints, policy order, comparators, and thermodynamic-attribution rule
were written to `artifacts/stage_4c/c1/dev_selection_rule.json`. The protected
2023 holdout was not scored.

The predeclared rule selects probability threshold **0.35**. It passes the
hard-negative wet-area constraint narrowly (0.00982 <= 0.01) and the initiation
false-fraction constraint (0.2864 <= 0.40). Its event-macro lifecycle results are:

| Precision | Recall | F1 | False initiation | Onset MAE | Median bias | Brier |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7136 | 0.7968 | 0.7444 | 0.2864 | 25.59 min | -12 min | 0.1992 |

Detection recall within 30/60/90/120 minutes is 0.9333/0.8768/0.8292/0.7968.
Against frozen A+ at 0.35, C1 changes event-macro initiation F1 by **-0.00485**.
The weather-event bootstrap 95% interval is [-0.0153, 0.0132] (three independent
events, so this interval is descriptive and low-powered). Frozen B2 at 0.20 has
event-macro F1 0.7501; A+ has 0.7492. C1 therefore improves neither comparator.

At the 0.1 mm h-1 expected-rate threshold, C1 CSI at 30/60/90/120 minutes is
0.2873/0.2769/0.2591/0.1633. It is nearly tied with A+
(0.2844/0.2772/0.2604/0.1597), behind B2 at every lead, and well behind raw HRRR
APCP. Raw HRRR CSI is 0.3938/0.3386/0.3805/0.3275, but its 30/60 and 90/120
values respectively reuse hourly accumulation fields and are not precise
30-minute timing forecasts. C1 exceeds PySTEPS CSI after 30 minutes, while
PySTEPS retains substantially higher 18-km FSS. Complete results for all four
rain thresholds and leads are in `dev_broad_model_comparison.csv`; C1-only FSS
is additionally reported at 6, 18, and 36 km.

Thermodynamic destruction produces no meaningful degradation. At threshold
0.35, event-macro F1 is 0.744389 normally, 0.744380 with all thermodynamics
zeroed, 0.744219 after global shuffle, and 0.744333 after within-event shuffle.
Normal-minus-perturbed deltas are at most 0.00017, far below the 0.005 floor used
to distinguish a meaningful effect from numerical/sign-only changes. Removing
moisture gives F1 0.744358; removing CAPE/CIN gives 0.744328. Thus the tiny-gate
moisture sensitivity does not translate to DEV skill, and neither result is
feature importance.

The frozen historical `RADAR_LIMITED_INITIATION` membership remains unchanged
(14 exact object/time rows, 13 issue times, four events; membership SHA-256
`1b4749e6cd1c165bef26726f86eb27850d0088f12764d4ebad3c1f46ec5ed883`). It is
not scoreable on the common 24-anchor DEV tensors: those exact object/time rows
belong to a different historical diagnostic set, and compatible exact C1
thermodynamic tensors were never materialized. Substituting coarse hourly
anchors or redefining membership would be scientifically invalid, so no
RADAR_LIMITED score is fabricated.

**DEV classification: NULL.** C1 is constraint-compliant but does not improve
event-macro initiation over A+, does not show meaningful dependence on the
thermodynamic inputs, and does not improve broad precipitation consistently.
Accordingly C-MOISTURE/C-INSTABILITY training is not authorized, the Stage C
procedure is not frozen for promotion, and the protected 2023 holdout remains
sealed. This documents the thermodynamic null before any later Stage D dynamics
experiment.

## Scientific question

Does issue-time-safe atmospheric thermodynamic information add precipitation-
initiation skill beyond the competent, capacity-matched A+ radar-only model?

The primary C1 input set is radar plus HRRR 2 m temperature, 2 m dew point,
surface CAPE, surface CIN, and column precipitable water. Wind and dynamics are
reserved for Stage D. The radar pathway, loss family, targets, lifecycle
definitions, and verification philosophy remain aligned with A+.

## Stage 4B closure carried forward

Stage 4B remains frozen as A+ / NULL GOES RESULT on DEV. Its fresh holdout showed
a strong incremental GOES-information effect over capacity-matched radar-only,
but did not prove strict pre-radar initiation because the frozen strict subset
had zero rows. B1 and B2 were similar on that holdout, so explicit cooling-
tendency attribution remains uncertain. Stage 4C does not reopen those choices.

## NWP-specific gate

The gate is limited to the five C1 thermodynamic variables. Every record must
retain model cycle issue time, forecast valid time, forecast hour, units, product
identity, and simulated availability. A future-valid field is allowed only when
its source cycle was available by nowcast issue time. No later cycle may be used.

Required checks before training:

- complete source materialization and checksums;
- decoded units and physical ranges;
- model-issue/valid-time causality and automated leakage rejection;
- spatial remapping, interpolation, missingness, and explicit masks;
- train-only normalization;
- exact object/time tensor support separated from shared source caching;
- operational raw HRRR precipitation baseline availability.

Completed source-gate evidence:

| Check | Result |
|---|---|
| Required records | 2,100 |
| Shared HRRR products | 420/420 |
| Variables decoded in every product | 2t, 2d, CAPE, CIN, PWAT: 420/420 each |
| Missing/corrupt products after repair | 0 |
| Model cycles available by nowcast issue | pass |
| Future-valid records from already available cycles | 1,400 |
| Negative forecast hours | 0 |
| Dynamics fields required for this gate | none |

The full header audit detected five partial cached files left by an interrupted
download, quarantined them, rematerialized those exact products, and then passed
all 420 products. Source caching remains separate from future object-specific
tensor generation.

## RADAR_POOR_INITIATION_V2 predeclaration

This is a new Stage 4C definition and does not modify the Stage 4B strict subset.
Membership is determined before any Stage C prediction using radar, geometry,
and PySTEPS only:

1. valid radar coverage is at least 95% through the ten-frame history and target;
2. at least 95% of eventual-initiation pixels are below 0.1 mm h-1 in the final
   input frame;
3. mean history wet-area fraction above 0.1 mm h-1 within the object tile is at
   most 2%;
4. PySTEPS predicts rain at no more than 20% of eventual-initiation pixels during
   the two-hour horizon;
5. no tracked upstream object is classified as an obvious advective entry;
6. the eventual-initiation footprint is at least 18 km from the tensor boundary.

These thresholds are frozen before Stage C scoring. Counts by object row, unique
issue time, and independent weather event must be computed from radar evidence
and repeated unchanged on the new protected Stage C holdout.

The definition was applied once to the 68 historical Stage 4B clean-initiation
rows solely to quantify its likely support. It selected zero object rows, zero
unique issue times, and zero independent events across all five historical
events. The criteria were not relaxed after observing that result. This means
`RADAR_POOR_INITIATION_V2` remains a valid predeclared diagnostic, but it may be
empty on the new Stage C holdout and cannot be assumed to provide an adequately
powered primary result. Its frozen membership file has SHA-256
`2f7dcc6ad762c7da755253868704f9c004ec810debb292885b79285dfa46c838`.

## Spatial tensor and normalization gate

All 140 existing train/dev/diagnostic anchors were remapped to the radar grid.
Each thermodynamic tensor has shape `3 x 5 x 128 x 128`: three forecast-valid
contexts (issue, +60, and +120 minutes) from an already-issued HRRR cycle and
the five frozen thermodynamic variables. Explicit validity masks and the model
issue, forecast valid time, forecast hour, product identity, and tile identity
are retained with every tensor.

| Check | Result |
|---|---|
| Requested anchors | 140 |
| Materialized anchors | 140 |
| Failed anchors | 0 |
| Finite, nonconstant channels | pass |
| Stage C training started | no |

Projection-aware interpolation uses cached Delaunay/barycentric weights in the
radar projection, with nearest-neighbour values only outside the source-grid
triangulation. Visual overlays were inspected for convective initiation,
organized convection, stratiform precipitation, dissipation, and favourable
dry weather. Across all five cases the fields were geographically coherent and
showed no visible flip, transpose, tile-offset, constant-channel, or invalid-mask
artifact.

Normalization statistics were calculated from the 48 training tensors only.
Temperature, dew point, and precipitable water use direct standardization; CAPE
uses `log1p(max(x, 0))`; CIN uses signed `log1p(abs(x))`. No clipping is applied.
The stored normalization record includes counts, means, standard deviations,
raw ranges, and the source tensor-manifest hash so that DEV and future holdout
processing cannot refit it.

## Deterministic precipitation baselines

Operational raw HRRR precipitation was materialized as the one-hour APCP interval
from the same already-issued cycle used for C1. Forecast hour 2 supplies the
0--60 minute interval and forecast hour 3 supplies 60--120 minutes. Because each
field is a one-hour accumulation in kg m-2, its numeric value is also the
interval-average rate in mm h-1. The 30/60-minute checks use forecast hour 2 and
the 90/120-minute checks use forecast hour 3. All 280 exact APCP messages passed
finite-value, unit, interval, checksum, remapping, and provenance checks.

DEV-only results at 0.1 mm h-1 are:

| Model | Lead | CSI | POD | FAR | F1 |
|---|---:|---:|---:|---:|---:|
| PySTEPS | 30 | 0.292 | 0.453 | 0.549 | 0.452 |
| raw HRRR APCP | 30 | 0.394 | 0.545 | 0.414 | 0.565 |
| PySTEPS | 60 | 0.221 | 0.360 | 0.637 | 0.362 |
| raw HRRR APCP | 60 | 0.339 | 0.489 | 0.476 | 0.506 |
| PySTEPS | 90 | 0.155 | 0.273 | 0.737 | 0.268 |
| raw HRRR APCP | 90 | 0.380 | 0.598 | 0.489 | 0.551 |
| PySTEPS | 120 | 0.105 | 0.201 | 0.820 | 0.190 |
| raw HRRR APCP | 120 | 0.327 | 0.553 | 0.555 | 0.493 |

CSI, POD, FAR, F1, and 18 km FSS were also stored for 0.1, 1.0, 2.5,
and 5.0 mm h-1, together with event-level results. HRRR and PySTEPS are
deterministic here, so Brier score is explicitly not applicable and no
probability was manufactured.

The transparent PySTEPS--HRRR baseline searched only HRRR weights
`0, 0.25, 0.5, 0.75, 1` independently at 30/60/90/120 minutes, selecting pooled
DEV F1 at 0.1 mm h-1. DEV selected weight 1.0 at every lead. Thus the frozen
simple-combination baseline is honestly identical to raw HRRR; there is no DEV
evidence that linear blending with PySTEPS improves this control.

## Fresh Stage C holdout gate

The protected candidate specification is
`configs/data/stage_4c_fresh_holdout.yaml`. It uses four independent 2023
weather periods absent from every Stage 3, 3.1, 4B-selection, and 4B-holdout
split: three convective periods and one prolonged heavy-rain period, plus four
paired dry candidates. This was the pre-freeze candidate definition; exact
objects, issue times, tiles, and accepted negatives were subsequently frozen as
described below. No C-model prediction has been generated for those frozen rows.

Radar acquisition is complete for all eight candidate periods. Radar-only
object mining found 11, 30, and 15 clean pre-radar initiation objects in the
July 12, July 20, and August 3 convective events respectively. The August 23
period is retained as non-convective/heavy-rain generalization evidence rather
than an initiation event.

Of the four predeclared negative candidates, July 19 and August 2 passed the
sparse-dry radar gate (mean wet fractions 0.00580 and 0.00841). July 11 and
August 22 were rejected before neural scoring because their mean wet fractions
were 0.03608 and 0.01659, above the 0.01 sparse-dry limit. They will not be
silently replaced after C1 results. Environmental matching for the retained
negatives remained incomplete because its source retrieval stalled. The
predeclared sparse-dry fallback was therefore used, after which the exact
holdout was frozen and C1 training proceeded.

## Frozen radar-limited hierarchy

`RADAR_LIMITED_INITIATION` is the primary radar-poor diagnostic. Its thresholds
are 95% valid coverage, 90% final-frame dryness at eventual-initiation pixels,
5% maximum historical tile wet fraction, 40% maximum PySTEPS coverage, no
advective entry, and 18 km object-footprint boundary clearance. The proposed
thresholds were not adjusted. A geometry implementation check was corrected to
measure the tracked initiation footprint rather than unrelated future rain
elsewhere in the tile. Historical support is 14 rows, 13 issue times, and four
independent events; the frozen membership hash is
`1b4749e6cd1c165bef26726f86eb27850d0088f12764d4ebad3c1f46ec5ed883`.

`RADAR_POOR_INITIATION_V2` remains the strict secondary diagnostic, unchanged,
with zero historical rows.

## Frozen 2023 holdout and exact tensors

The fresh Stage C manifest was frozen before any C1 prediction with SHA-256
`bbf001f7512dc5aae36158941affb878ec7f3e1d011dc0fadec39fed755b3f5c`.
It contains 68 rows: 56 clean initiation rows (52 unique issue times, three
independent positive events), eight hard-negative rows across the two retained
sparse-dry events, and four heavy-rain generalization rows. On this protected
set, the precomputed radar-only classifications contain 28
`RADAR_LIMITED_INITIATION` rows (26 issue times, all three positive events) and
19 strict V2 rows. These memberships are immutable.

Environmental matching remains explicitly incomplete: the legacy full-field
source retrieval produced no output for more than 12 minutes. Under the
predeclared fallback, July 19 and August 2 were retained based on their passed
sparse-dry radar gate. No substitute negatives were introduced.

All 129 unique issue-safe HRRR source products and all 68 exact holdout tensors
materialized successfully. Non-hourly object times use linear interpolation
between bracketing hourly valid fields from one already-issued cycle. Integrity
checks passed for tensor dimensions, masks, finite/nonconstant channels, radar/
target timing, causal model cycles, and exact reuse of the train-only
normalization hash. No future radar or later HRRR cycle enters an input.

## C1 pipeline gate

C1 has 636,173 trainable parameters: 484,587 in the A+-style radar pathway,
55,248 in the thermodynamic branch, 1,800 in fusion, and 94,538 in the decoder
and heads. This is about 9.8% above the 579,125-parameter A+ control and is not a
material capacity expansion requiring another radar control.

A two-sample, 64 x 64 pipeline overfit reduced loss from 1.1458 to 0.3661.
Radar, thermodynamic, and fusion gradient sums were all nonzero after the
zero-initialized fusion projection received its first update. Mean absolute
logit changes were 0.966 for all thermodynamics zeroed, 0.560 when shuffled,
0.728 with moisture zeroed, and 0.184 with CAPE/CIN zeroed. This establishes
working modality paths only; it is not a scientific result.

## C1 convergence training

C1 was trained on all 48 TRAIN tensors per epoch and evaluated on all 24 DEV
tensors, with the frozen holdout excluded from both loaders. Training used the
A+/Stage 4B loss family, train-derived occurrence weights, AdamW at 0.0003,
best-DEV checkpointing, learning-rate reduction, and six-epoch early stopping.
The CPU run stopped after eight epochs (479 seconds). Best DEV loss was 1.18364
at epoch 2; subsequent epochs did not improve it. The selected checkpoint hash
is `d35dd3b975bd6495937985e11987f3b01dc28810da9fc519b42274079e677a4f`.

Training loss declined from 0.6630 to 0.5783 while later DEV loss worsened to
1.2464, so the best checkpoint is retained and the divergence is treated as
overfitting rather than evidence for additional training. The protected 2023
holdout remained unopened during training. DEV meteorological and lifecycle
evaluation is now complete and classified NULL as documented above; no Stage C
promotion procedure is frozen.

## Closure state

The independent-holdout gate passed and C1 convergence plus DEV evaluation are
complete. Because C1 is NULL on DEV, the protected holdout is intentionally not
opened and no C1 sub-ablation is trained. Stage C is closed as a documented DEV
null; a separately predeclared Stage D dynamics hypothesis may proceed without
reinterpreting this result.
