# Stage 4D atmospheric dynamics ablation

Status: Stage 4D is closed as **DEV NULL — DYNAMICS INFORMATION NOT USED**. D1
passed its engineering gates and completed convergence training, but failed
promotion gates A–D. It passed only the hard-negative constraint. Neither the
historical exact radar-limited diagnostic nor the protected 2023 holdout was
scored. Stage 4C remains closed and all protected memberships remain sealed.

## Scientific hypothesis

Stage C tested environmental potential. Stage D instead asks whether issue-time-
safe dynamical forcing and triggering information improves precipitation
initiation beyond A+ radar-only, especially where radar extrapolation contains
weak precursor information. A null D1 will not trigger a large feature search.

## Frozen D1 inputs

D1 uses native HRRR 10 m and 850 hPa U/V wind, 700 hPa vertical velocity
(omega), and 500 hPa U/V wind. On the radar grid it derives:

- 850 hPa convergence: `-(du/dx + dv/dy)`, positive for convergence, in s-1;
- 850 hPa relative vorticity: `dv/dx - du/dy`, in s-1;
- approximate deep-layer shear: 500 hPa minus 850 hPa U/V and magnitude, in
  m s-1.

Derivatives use EPSG:3978 projected coordinates and the 2 km radar-grid spacing.
The five Stage C thermodynamic variables are prohibited. Three issue-safe HRRR
valid times (issue, +60, +120 minutes) use the established conservative 60-minute
cycle-availability lag. Every tensor will preserve cycle, valid time, forecast
hour, source product, checksums, and masks.

## Controlled architecture

D1 retains the A+ radar pathway, decoder, four exceedance heads, conditional-
intensity head, loss family, and targets. A modest recurrent dynamics encoder
adds a zero-initialized latent residual, analogous to C1. Parameter count and
memory will be recorded before training; a capacity control is required only if
the increase materially exceeds roughly 15% over A+.

The implemented D1 has 636,893 trainable parameters: 484,587 in the unchanged
radar pathway, 55,968 in the dynamics branch, 1,800 in fusion, and 94,538 in the
unchanged decoder/heads. This is 9.97% above the 579,125-parameter A+ control,
so the predeclared capacity criterion does not require another radar control.

## Frozen DEV promotion rule

The candidate probability grid is 0.05–0.95 by 0.05. Thresholds must keep
hard-negative wet area at or below 0.01 and initiation false fraction at or
below 0.40. Promotion additionally requires:

1. at least +0.01 weather-event-macro initiation F1 over A+;
2. at least 0.005 event-macro F1 degradation under each principal destroyed-
   dynamics condition (zeroed, global shuffle, within-event shuffle), with
   event-level consistency;
3. no unacceptable hard-negative degradation.

Onset MAE, Brier, exact historical `RADAR_LIMITED_INITIATION`, and 60–120 minute
broad skill are secondary endpoints. The 2023 holdout cannot be opened merely
because D1 trains; a complete DEV pass is required first.

## Data-gate plan

Initial development uses exactly the 48 TRAIN and 24 DEV/repair anchors used by
A+, B2, and C1. It excludes all protected 2023 rows. Before training, the gate
must establish complete products and checksums, issue-time causality, decoded
units and ranges, projection/alignment, masks, derived fields, TRAIN-only
normalization, and visual panels spanning initiation, organized convection,
stratiform rain, dissipation, and dynamically active dry weather.

## Completed data gate

The development audit contains exactly 72 established anchors: 48 TRAIN and 24
DEV/repair. It expands to 1,512 native-variable records and 432 unique HRRR
surface/pressure products. All 432 products materialized with checksums and zero
failures. No source cycle is later than its forecast issue time, the minimum
availability lag is 60 minutes, forecast hours are nonnegative, and the 2023
holdout is absent.

All 72 dynamics tensors materialized with shape `3 x 10 x 128 x 128`, explicit
masks, source-cycle provenance, and zero failures. TRAIN-only normalization uses
48 tensors (2,359,296 valid values per channel), with positive standard
deviation in every channel. Across TRAIN, convergence spans approximately
-3.21e-4 to 6.04e-4 s-1, vorticity -5.07e-4 to 5.73e-4 s-1, 700 hPa omega
-20.65 to 5.37 Pa s-1, and shear magnitude 0.03 to 42.55 m s-1. These ranges are
physically plausible for the sampled systems.

Five visual panels cover convective initiation, organized convection,
stratiform precipitation, dissipation, and dynamically active dry weather.
Manual inspection confirms coherent wind, omega, convergence/vorticity, and
shear structures with consistent orientation and no missing bands. The derived
gradient fields retain some expected fine-scale interpolation texture; this is
documented rather than smoothed post hoc.

## Tiny D1 pipeline gate

On two 64 x 64 TRAIN crops, loss fell from 1.14584 to 0.36726 over 160 updates
(ratio 0.321). Absolute gradient sums were nonzero for radar (264.57), dynamics
(0.430), and fusion (0.426). Mean absolute logit changes were 0.745 for all
dynamics zeroed, 0.611 for sample shuffle, 0.415 for convergence removal, 0.277
for vertical-motion removal, and 0.285 for shear removal. Masks and all ten
channels were finite/nonconstant. These are path checks only, not evidence of
forecast skill. They authorize one controlled convergence-training run using
the already frozen configuration.

## D1 convergence training

D1 trained on all 48 TRAIN tensors per epoch and evaluated on all 24 DEV/repair
tensors. AdamW used learning rate 0.0003, the frozen A+/C1 loss family and
TRAIN-derived occurrence weights, best-DEV checkpointing, LR reduction, and
six-epoch early stopping. The CPU run completed nine epochs in 373 seconds.

Epoch 3 produced the selected DEV loss 1.18908. Later TRAIN loss declined to
0.56397 while DEV failed to improve and ended at 1.22232, consistent with
overfitting. The retained checkpoint SHA-256 is
`5b5c9d22b78482f4b34f481160c76cfe845f4961a4ab13c66703c73e9e2d93b1`.
This loss was not used as a promotion result. The subsequent frozen DEV sweep,
meteorological verification, hard-negative analysis, and dynamics-destruction
tests below determined the classification. The protected 2023 holdout remains
unopened.

## Frozen DEV evaluation

The immutable epoch-3 checkpoint was evaluated on all 24 common DEV/repair
tensors under normal inputs, all-zero dynamics, global shuffle, within-event
shuffle, convergence removal, vertical-motion removal, and shear removal. The
threshold sweep covered 0.05–0.95 by 0.05. Threshold **0.45** maximized the
predeclared event-macro policy among eligible thresholds: hard-negative wet area
was 0.00892 and initiation false fraction 0.24083.

At 0.45, D1 event-macro precision/recall/F1 were
0.75917/0.73650/**0.73888**. False initiation was 0.24083, onset MAE 23.14
minutes, median onset bias -9.5 minutes, and Brier 0.20129. Detection within
30/60/90/120 minutes was 0.9102/0.8455/0.7797/0.7365.

### Primary comparisons

D1 trailed frozen A+ initiation F1 by **-0.01035**, rather than clearing the
required +0.01. It gained precision (+0.0501), reduced false initiation
(-0.0501), improved onset MAE (-2.83 minutes), and improved Brier (-0.00889),
but lost recall (-0.0676). The weather-event bootstrap 95% interval for the F1
delta was [-0.0363, 0.0200] across only three events. Event deltas were +0.0200
for the August 17 storms, -0.0363 for the June 18 initiation event, and -0.0148
for the Toronto flood.

D1 also trailed B2 F1 by -0.01127 and C1 by -0.00551. Relative to B2 it traded
substantially higher precision (+0.1163) for lower recall (-0.1760). Relative to
C1 it improved precision and onset MAE but again lost recall. D1 therefore does
not beat either prior learned multimodal reference on the primary endpoint.

### Dynamics-dependence and component tests

Normal-minus-destroyed event-macro F1 was only +0.000414 for all-zero dynamics,
+0.000504 for global shuffle, and +0.000685 for within-event shuffle. All fail
the frozen +0.005 attribution requirement. This mirrors C1: the current late
residual pathway can fit while making negligible operational use of its
non-radar branch. Dynamics are therefore not the first demonstrated NWP
information effect.

Removing convergence changed F1 by +0.000009, removing vertical motion by
+0.000400, and removing shear by +0.000044 in favour of normal D1. The
corresponding 60/90/120-minute CSI values were essentially invariant near
0.280/0.258/0.165. None of convergence, omega, or shear has detectable DEV skill
in this trained representation; these are diagnostic removals, not alternative
models.

### Broad precipitation and hard negatives

At 0.1 mm h-1, D1 CSI at 30/60/90/120 minutes was
0.2906/0.2801/0.2582/0.1645. It was close to A+ and C1, behind B2 at all four
leads, and far behind raw HRRR APCP. D1 exceeded PySTEPS CSI after 30 minutes,
but PySTEPS retained much higher 18-km FSS. Complete six-model results for all
four rain thresholds are in `dev_broad_model_comparison.csv`; learned-model
Brier scores remain separate from expected-rate categorical metrics.

On the two DEV hard-negative events, D1 mean probability was 0.0841, maximum
probability 0.5859, wet area 0.00892, and Brier 0.01185. Its wet-area constraint
passed, but Brier was worse than A+ (0.00894), B2 (0.00415), and C1 (0.01104).
There is no evidence that useful dynamics gains were purchased through broad
dry-event rain prediction because no useful dynamics gain was present.

The exact historical `RADAR_LIMITED_INITIATION` tensors were not materialized.
That endpoint was secondary and cannot alter failure of mandatory gates A–D;
spending additional retrieval effort after the primary null would violate the
predeclared stopping logic. No coarse-anchor substitution was made.

## Final promotion outcome

Gate A (D1 >= A+ +0.01): fail. Gates B–D (normal exceeds zero/global/within-
event destruction by >=0.005): all fail. Gate E (hard negatives): pass.
**DEV classification: NULL.** Stage D is closed without a larger model,
pressure-level search, added variables, thermo/GOES fusion, or protected
inference. The next justified experiment is a separately predeclared
architectural or hybrid test of whether the training/fusion design forces use
of non-radar information.

## Questions to close at completion

The final report will answer whether D1 beats A+, B2, C1, PySTEPS, and raw HRRR;
whether its skill depends on dynamics; which of convergence, vertical motion,
and shear matter; whether radar-limited initiation and onset timing improve;
whether dry-event false alarms remain controlled; which leads benefit; whether
effects repeat across events; and whether combined GOES+dynamics fusion is
justified.
