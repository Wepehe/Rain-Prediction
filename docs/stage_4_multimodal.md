# Stage 4 controlled multimodal nowcasting

Status: Stage 4B is complete. Stage 4C is closed as **DEV NULL — THERMODYNAMIC
INFORMATION NOT USED**: the tested C1 representation learned no detectable
incremental thermodynamic skill. Its protected 2023 holdout remains sealed.
Stage 4D is also closed as **DEV NULL — DYNAMICS INFORMATION NOT USED**. Its D1
model did not improve the frozen radar control and was insensitive to destroyed
dynamics inputs. Frozen Stage 3.1 and Stage 4B/C/D artifacts remain unchanged;
the protected 2023 holdout remains sealed. Stage 5 is a separate frozen-baseline
residual-fusion experiment and does not reopen any Stage 4 decision.

The Stage C conclusion is deliberately narrow: C1 did not improve event-macro
initiation over A+, broad expected-rate skill was approximately tied with A+ and
behind B2, and zero/shuffle/moisture/instability perturbations caused negligible
DEV degradation. Raw HRRR APCP remained strong. This does not establish that
atmospheric thermodynamics are intrinsically useless.

The Stage 4 scientific question is:

> Does atmospheric and satellite information provide incremental predictive
> skill beyond radar history alone, particularly for precipitation initiation,
> growth, decay, and 60-120 minute forecasts?

## Frozen radar-only control

The Stage 3.1 radar-only control is frozen as:

- `multiscale_convlstm_radar_control_528k`

The Stage 4 audit records hashes for the frozen radar-only control in:

- `artifacts/stage_4/audit/radar_control_freeze_manifest.json`

The freeze manifest covers:

- Stage 3.1 experiment config
- best-dev checkpoint
- train-only normalization
- Stage 3.1 split manifest
- Stage 3.1 sample manifest
- DEV, existing-test, and fresh-holdout metric outputs
- DEV probability-threshold sweep
- lifecycle metrics

The Stage 3.1 architecture, radar preprocessing, target definitions, tile size,
input/output history, metrics, and threshold-selection methodology should not be
changed after Stage 4 results are inspected.

## Stage 4 split repair

Stage 4 split policy is defined in:

- `configs/experiments/stage_4_split_manifest.yaml`

New candidate repair/holdout windows are defined in:

- `configs/data/stage_4_evaluation_repair.yaml`

The new windows are intentionally marked `candidate_unvalidated`. The eight
main repair/holdout event windows have now been radar-materialized, but they
must still pass event-object mining, hard-negative verification, and multimodal
issue-time checks before they can support final Stage 4 claims.

### DEV repair candidates

| Event | Purpose |
|---|---|
| `stage4_dev_initiation_2026_06_18` | add independent initiation DEV coverage |
| `stage4_dev_dissipation_2026_05_20` | add independent dissipation DEV coverage |
| `stage4_dev_dry_favourable_2026_07_10` | add favourable-environment dry DEV coverage |
| `stage4_dev_favourable_dry_2026_06_17` | matched hard-negative candidate |
| `stage4_dev_post_storm_dry_2026_05_21` | matched/post-storm hard-negative candidate |

### Stage 4 initiation holdout candidates

| Event | Purpose |
|---|---|
| `stage4_init_holdout_june21_2025_derecho` | fresh initiation-heavy final holdout candidate |
| `stage4_init_holdout_may20_2026_southwest_ontario` | fresh severe-convection initiation candidate |
| `stage4_init_holdout_july27_2026_southwest_ontario` | fresh unstable-airmass initiation candidate |
| `stage4_init_holdout_aug30_2026_eastern_ontario` | fresh eastern-Ontario/National-Capital initiation candidate |
| `stage4_init_holdout_sep02_2026_southern_ontario` | fresh late-summer severe-convection candidate |

Matched favourable-environment dry candidates are included in the same final
holdout split so hard negatives remain protected from model-development
feedback:

- `stage4_holdout_favourable_dry_2025_06_20`
- `stage4_holdout_favourable_dry_2026_05_18`
- `stage4_holdout_favourable_dry_2026_07_26`
- `stage4_holdout_favourable_dry_2026_08_29`
- `stage4_holdout_favourable_dry_2026_09_01`

The existing Stage 3.1 fresh holdout had no initiation-centered samples, so it
is not sufficient for the central Stage 4 initiation hypothesis.

## Audit artifacts

The Stage 4 audit command is:

```powershell
nowcast-stage4-audit `
  --config configs/experiments/stage_4_multimodal.yaml `
  --output-dir artifacts/stage_4/audit
```

Current audit outputs:

- `artifacts/stage_4/audit/event_inventory.csv`
- `artifacts/stage_4/audit/multimodal_availability_audit.csv`
- `artifacts/stage_4/audit/radar_control_freeze_manifest.json`
- `artifacts/stage_4/audit/stage4_audit_summary.json`

Current audit summary, refreshed 2026-09-07 after excluding rejected candidates
from the active split manifest:

| Quantity | Value |
|---|---:|
| frozen radar-control files hashed | 11 |
| total documented inventory events, including rejected candidates | 55 |
| active split-manifest events | 35 |
| radar-materialized documented events | 55 |
| radar-materialized active events | 35 |
| active issue-time availability records | 14,140 |
| training started | false |

Materialization status by split:

| Split | Radar materialized? | Active events |
|---|---|---:|
| train | yes | 12 |
| dev | yes | 2 |
| existing/test-labelled Stage 3 events | yes | 4 |
| Stage 3.1 fresh holdout | yes | 4 |
| Stage 4 DEV repair | yes | 4 |
| Stage 4 initiation holdout | yes | 9 |

The eight main Stage 4 repair/holdout event windows were materialized with:

- 0 missing events
- 0 failed events
- no model evaluation
- no training

Radar materialization summary for those main windows:

| Event | Frames | Missing frames | Mean missing fraction |
|---|---:|---:|---:|
| `stage4_dev_initiation_2026_06_18` | 101 | 3 | 0.030 |
| `stage4_dev_dissipation_2026_05_20` | 101 | 0 | 0.000 |
| `stage4_dev_dry_favourable_2026_07_10` | 101 | 0 | 0.000 |
| `stage4_init_holdout_june21_2025_derecho` | 101 | 1 | 0.010 |
| `stage4_init_holdout_may20_2026_southwest_ontario` | 121 | 0 | 0.000 |
| `stage4_init_holdout_july27_2026_southwest_ontario` | 121 | 0 | 0.000 |
| `stage4_init_holdout_aug30_2026_eastern_ontario` | 111 | 3 | 0.027 |
| `stage4_init_holdout_sep02_2026_southern_ontario` | 131 | 0 | 0.000 |

The seven explicitly matched Stage 4 hard-negative windows plus the DEV
favourable-dry event-table candidate have been radar-materialized and verified.

### Hard-negative verification

Hard-negative verification is recorded in:

- `artifacts/stage_4/evaluation_repair/hard_negative_pair_matching.csv`
- `artifacts/stage_4/gate/hard_negative_decisions.csv`
- `artifacts/stage_4/gate/hard_negative_decisions_summary.json`

Result after excluding the two rejected candidates from the active split:

| Quantity | Value |
|---|---:|
| active pairs attempted | 6 |
| verified active hard negatives | 6 |
| rejected active hard negatives | 0 |
| verifier failures | 0 |

Two previously rejected windows were rescued by selecting clean 256 km
verification tiles inside the already materialized radar crops:

| Candidate | Verification tile | Mean wet fraction | Max-frame wet fraction |
|---|---|---:|---:|
| `stage4_dev_dry_favourable_2026_07_10` | `[-83.3750, 43.6250, -80.8250, 46.1750]` | 0.00082 | 0.00352 |
| `stage4_holdout_favourable_dry_2026_08_29` | `[-77.2550, 43.0050, -74.7050, 45.5550]` | 0.00387 | 0.02838 |

Two candidates remain rejected because precipitation is too extensive even after
checking for clean subtiles:

| Rejected candidate | Mean wet fraction | Max-frame wet fraction | Mean heavy fraction | Max-frame heavy fraction |
|---|---:|---:|---:|---:|
| `stage4_dev_favourable_dry_2026_06_17` | 0.22085 | 0.47135 | 0.12494 | 0.28072 |
| `stage4_holdout_favourable_dry_2026_05_18` | 0.09618 | 0.22611 | 0.06415 | 0.17617 |

These two rejected cases are retained in the repair config with rejection
metadata, but they have been removed from the active Stage 4 split manifest and
are ignored by the hard-negative verifier. They should not be used as dry hard
negatives unless replaced by new verified candidates.

## GOES C13 source-file materialization

The GOES C13 materialization command is:

```powershell
nowcast-stage4-materialize-goes `
  --audit-csv artifacts/stage_4/audit/multimodal_availability_audit.csv `
  --output-dir artifacts/stage_4/materialization `
  --data-root data `
  --config configs/experiments/stage_4_multimodal.yaml
```

It writes:

- `artifacts/stage_4/materialization/stage4_goes_c13_required_sources.csv`
- `artifacts/stage_4/materialization/stage4_goes_c13_materialized_sources.csv`
- `artifacts/stage_4/materialization/stage4_goes_c13_materialization_summary.json`
- `artifacts/stage_4/materialization/stage4_goes_c13_decode_spotcheck.csv`

Current GOES source-file status:

| Quantity | Value |
|---|---:|
| required active anchor source records | 3,500 |
| unique required C13 source times | 1,010 |
| materialized unique C13 source times | 1,010 |
| truncated by development limit | false |
| complete | true |

A representative decode spot check successfully opened sampled GOES-16 and
GOES-19 ABI C13 files from 2024, 2025, and 2026. Each sampled file decoded as a
1500 x 2500 `float32` CMI brightness-temperature field with about 98.7% finite
pixels. This confirms actual file contents and values exist; it is not just an
availability-table count.

## Radar+GOES anchor tensor materialization

The radar+GOES tensor materialization command is:

```powershell
python -m ontario_nowcast.training.stage4_materialize `
  --mode radar-goes-tensors `
  --audit-csv artifacts/stage_4/audit/multimodal_availability_audit.csv `
  --output-dir artifacts/stage_4/materialization `
  --data-root data `
  --config configs/experiments/stage_4_multimodal.yaml
```

Current radar+GOES anchor-tensor status:

| Quantity | Value |
|---|---:|
| active audit anchors requested | 140 |
| anchors materialized | 140 |
| failures | 0 |
| truncated by development limit | false |
| complete | true |

Each `.npz` sample contains 10 radar input frames, 20 radar target frames,
validity masks, five GOES-derived channels over five satellite lags, and static
latitude/longitude grids. The GOES tensor layout is `(channel, lag, y, x)`.

A tensor-layout bug was found during sanity checking and corrected before the
current artifacts were frozen: the first materialization used `(lag, channel,
y, x)` while the metadata expected `(channel, lag, y, x)`. The full 140-anchor
radar+GOES tensor set was regenerated after the fix.

Corrected tensor spot checks show physically plausible C13 brightness
temperatures and nonconstant cooling channels; sampled raw C13 values span
approximately 218-299 K with no decoded values below or equal to 100 K.

## Radar+GOES visual and tiny-overfit sanity checks

Radar+GOES visual sanity panels were generated from the corrected tensors for:

- clean convective initiation;
- organized storm;
- stratiform precipitation;
- dissipation;
- matched favourable dry negative.

The visual sanity summary reports 5/5 requested panels created. These panels
verify radar/GOES synchronization and qualitative field alignment for Ablation B
inputs, but they intentionally do not include NWP fields.

The tiny multimodal overfit gate also now passes for the corrected radar+GOES
tensors:

| Quantity | Value |
|---|---:|
| samples | 2 |
| optimization steps | 80 |
| device | CPU |
| parameters | 25,320 |
| initial loss | 0.7849 |
| final loss | 0.5085 |
| loss ratio | 0.6479 |
| radar branch receives gradients | true |
| satellite branch receives gradients | true |
| fusion layers receive gradients | true |
| removing satellite changes output | true |
| shuffling satellite changes output | true |
| raw C13 nonconstant | true |
| cooling channels nonconstant | true |
| masks nonempty | true |

This is a synchronization/gradient-path gate only. It is not an Ablation B model
result and does not inspect final-holdout neural performance.

## Stage 4 gate artifacts

The metadata-only gate command is:

```powershell
nowcast-stage4-gate `
  --repair-config configs/data/stage_4_evaluation_repair.yaml `
  --output-dir artifacts/stage_4/gate `
  --data-root data
```

It writes:

- `artifacts/stage_4/gate/event_object_validation.csv`
- `artifacts/stage_4/gate/event_object_summary.csv`
- `artifacts/stage_4/gate/stage4_final_initiation_holdout_manifest.csv`
- `artifacts/stage_4/gate/stage4_final_initiation_holdout_manifest.sha256.json`
- `artifacts/stage_4/gate/stage4_training_gate_summary.json`

The final initiation-holdout manifest was frozen as metadata only; no neural or
multimodal model performance was inspected on the final holdout.

Current object-mining summary:

| Quantity | Value |
|---|---:|
| Stage 4 events validated | 15 |
| initiation-like objects mined | 1,406 |
| clean pre-radar initiation objects across Stage 4 repair/holdout windows | 72 |
| final initiation-holdout object-manifest rows | 650 |
| clean final initiation-holdout rows | 68 |
| final holdout manifest SHA-256 | `0d347e2bff4ab174af248b2345e709f0c0b12505a0542dcca573e10d2af08783` |

## Multimodal inputs in the audit

First-pass Stage 4 modalities are defined in:

- `configs/experiments/stage_4_multimodal.yaml`

The audit currently records these issue-time-safe input families:

| Modality | Records | Initial variables |
|---|---:|---|
| radar | 2,200 | MRMS precipitation-rate history |
| satellite | 5,500 | GOES-East ABI C13 brightness temperature and C13 cooling tendencies |
| NWP | 13,200 | HRRR surface, thermodynamic, wind, moisture, and upper-air candidates |
| static | 1,320 | lat/lon, elevation, land/water, Great Lakes mask, shoreline distance |

The audit explicitly distinguishes:

- forecast issue time
- source timestamp
- input availability timestamp
- NWP model issue time
- NWP model valid time

Forecast valid times from an already-available HRRR cycle are permitted, but a
model cycle that was not available by the simulated nowcast issue time is
rejected.

The audit initially caught a GOES leakage bug: a nominal satellite lag of zero
plus 10 minutes of product latency made the current scan unavailable at issue
time. The audit now treats "current GOES" as the latest scan available by issue
time, i.e. issue time minus product latency.

## HRRR/NWP source materialization status

The HRRR source-materialization mode is now implemented with exact GRIB
byte-range selectors for the 20 Stage 4 NWP variables in
`configs/experiments/stage_4_multimodal.yaml`.

The command is:

```powershell
nowcast-stage4-materialize hrrr-nwp-sources `
  --audit-csv artifacts/stage_4/audit/multimodal_availability_audit.csv `
  --output-dir artifacts/stage_4/materialization `
  --data-root data `
  --config configs/experiments/stage_4_multimodal.yaml
```

For safe smoke testing, the command supports `--max-products`. A capped real
NOAA HRRR run was completed with `--max-products 2`:

| Quantity | Value |
|---|---:|
| required active NWP audit records | 8,400 |
| unique required HRRR products | 840 |
| attempted HRRR products | 2 |
| materialized HRRR products | 2 |
| failures | 0 |
| truncated by development limit | true |
| complete | false |

The smoke run materialized one `wrfprs` subset with 12 pressure-level fields and
one `wrfsfc` subset with 8 surface/column fields for the same HRRR cycle and
forecast hour. The decode spot check verified finite values on the native HRRR
Lambert grid (`1799 x 1059`) for sampled messages.

This is a source-file materialization proof, not a full NWP tensor gate. Full
HRRR materialization remains deliberately incomplete because it requires 840
unique HRRR products for the active Stage 4 audit anchors.

The HRRR caveat for later interpretation remains: HRRR cycles may assimilate
radar and other recent observations. Stage 4 NWP ablations must treat HRRR as an
operational atmospheric-state control with possible radar-informed analysis
content, not as an independent non-radar information source.

## Leakage tests added

Automated Stage 4 leakage/split tests are in:

- `tests/test_stage4_audit.py`

They cover:

- Stage 4 DEV repair and final initiation holdout disjointness
- HRRR future valid times from already-available cycles
- rejection of unavailable/future HRRR cycles
- train-only normalization metadata
- protection against scoring `stage4_initiation_holdout` before Stage 4 freeze
- audit generation without training artifacts

Additional existing tests already cover basic timestamp alignment, issue-time
availability, train-only Stage 3 normalization, hard-negative mining, and
Stage 3/3.1 model/data behavior.

## Controlled ablation sequence

The planned sequence is encoded in `configs/experiments/stage_4_multimodal.yaml`:

| ID | Model | Inputs |
|---|---|---|
| A | frozen Stage 3.1 radar control | radar |
| B | radar + GOES C13 cooling | radar, satellite |
| C | radar + basic thermodynamics | radar, temperature, dew point, CAPE, CIN, precipitable water |
| D | radar + wind/dynamics | radar, winds, convergence/shear/vertical velocity candidates |
| E | radar + satellite + thermodynamics | radar, satellite, basic NWP thermodynamics |
| F | radar + satellite + full verified NWP | radar, satellite, full validated NWP |

Do not jump directly to the optional full model. Each ablation should keep the
radar branch, target heads, samples, optimizer, DEV procedure, and metrics as
close as practical to the frozen radar control.

## Stage 4 modality-specific gates before training

Stage 4 now uses modality-specific gates. The radar+GOES gate authorizes
Ablation B only. HRRR/NWP and full static-feature materialization are separate
prerequisites for Ablations C-F and must not block B.

The common pre-training checks remain:

1. rejected hard-negative candidates are replaced or excluded from the active
   split;
2. initiation/dissipation/hard-negative category verification passes for all
   new Stage 4 windows;
3. multimodal files are materialized for all train/dev/test/holdout events;
4. every modality record has source timestamp, availability timestamp, valid
   timestamp, resolution, units, missing fraction, interpolation method, and
   train-only normalization method;
5. issue-time leakage tests pass on the materialized records;
6. a tiny multimodal overfit verifies synchronization, masks, gradients, and
   modality-specific signal paths;
7. the Stage 4 procedure is frozen before any final initiation-holdout scoring.

Current gate status:

| Gate | Status | Applies to |
|---|---|---|
| Stage 3.1 radar control preserved | pass | B-F |
| all radar windows materialized | pass | B-F |
| hard negatives verified | pass: 6/6 active verified | B-F |
| event objects validated | pass | B-F |
| final initiation-holdout manifest frozen | pass | B-F |
| final holdout has clean pre-radar initiation objects | pass | B-F |
| GOES C13 source files materialized | pass | B |
| radar+GOES anchor tensors materialized | pass: 140/140 | B |
| radar+GOES visual sanity panels completed | pass: 5/5 | B |
| issue-time-safe GOES latency handling | pass | B |
| tiny radar+GOES overfit completed | pass | B |
| HRRR/NWP source files materialized | not complete: 2/840 smoke only | C-F |
| full multimodal NWP/static tensors materialized | not complete | C-F |
| full multimodal NWP-inclusive visual panels completed | not complete | C-F |

Ablation B is complete. DEV froze A+ / NULL GOES RESULT; the subsequent exact
68-row protected holdout produced a strong B2-over-A+ initiation result without
reopening selection. All exact tensors passed timing, tile, mask, dimensional,
latency, and manifest-hash integrity checks. Ablations C-F remain blocked only
until their NWP/static gates pass.

## Current decision answers

The current status is:

1. GOES initiation improvement: strong on the protected holdout despite a DEV
   null; B2 row-macro F1 0.522 versus A+ 0.393 and B2 won 4/5 events.
2. Thermodynamic-state improvement: not evaluated.
3. Wind/dynamics improvement: not evaluated.
4. Largest data-source gain: not evaluated.
5. Satellite/NWP complementarity: not evaluated.
6. Beat Stage 3.1 radar-only: B2 improves holdout precision, F1, false-initiation
   control, timing, and Brier; Stage 3.1 has near-unity recall through broad rain.
7. Beat PySTEPS: B2 beats PySTEPS on DEV pixel CSI/Brier at 30-120 min for
   the 0.1 mm/h threshold, but PySTEPS has better FSS.
8. Lead-time dependence: preliminary B2-B1 gains are strongest at 30-90 min.
9. Fresh initiation-holdout gain: B2 F1 +0.128 and recall +0.246 versus A+.
10. False-alarm-controlled improvement: B2 false-initiation fraction 0.515
    versus A+ 0.534 at their frozen 0.20/0.35 thresholds.
11. Hard-negative calibration: B2 affected 0.00309 of area versus A+ 0.000245,
    but had lower mean probability and Brier across four protected events.
12. Beat raw NWP precipitation: not evaluated.
13. Beat PySTEPS/NWP blend: not evaluated.
14. Useful atmospheric variables: not evaluated.
15. Remaining multimodal failure modes: not evaluated.

This is the intended Stage 4 starting line: the experiment is now structured so
future improvements can be attributed to additional information rather than to a
quietly changed radar-only baseline. The current Stage 4B record is maintained
in `docs/stage_4b_goes_ablation.md`.
