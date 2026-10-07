# Cycle 2 Progress

## Current status

- Current phase: validated Residual V1 operational inference package complete.
- Systems completed: 96/96.
- Process state: final scientific conclusion frozen.
- Replacement split: 72 TRAIN / 12 DEV / 12 FINAL.
- V1 training: complete; 22 epochs, best epoch 14, early stopping triggered.
- DEV promotion: complete; all eight frozen gates passed.
- FINAL evaluation: complete; all 353 rows scored exactly once.
- FINAL status: `CONSUMED_FINAL`.
- Final classification: `FINAL VALIDATED — RESIDUAL V1 GENERALIZES`.
- Operationalization: production API, CLI, hash-verified bundle, DEV golden fixture, and structured output complete.
- Interactive V1 forecast demo implemented. This is an engineering milestone only.
- Live user-facing nowcast interface implemented. Live operation does not constitute additional scientific validation of V1.
- Next action: preserve V1 unchanged. Any V2 requires a new predeclaration and new independent holdout.

Residual V1 scientific validation is complete. Operational packaging is a software-engineering phase and does not modify the frozen model.

## Completed milestones

1. Audited 1,704 archive days with zero metadata failures.
2. Recorded 57 legacy events and excluded them from NEW DEV/FINAL eligibility.
3. Sparse-screened 473 systems; 435 qualified, with zero decode failures.
4. Selected a 96-event slate: 18 winter, 24 spring, 30 summer, and 24 autumn.
5. Materialized all 96 systems; one genuinely absent MRMS scan remains explicitly missing.
6. Corrected the timestamp-resolution validator before an accepted freeze.
7. Created and tested the first 74-system initiation-gated freeze.
8. Pre-training review identified clean-initiation-gated system qualification as unsuitable for residual learning.
9. Archived the first split and manifest as `SUPERSEDED_PRETRAINING`; old final-manifest SHA-256: `32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e`.
10. Re-mined all 96 systems using residual-relevant qualification: integrity plus at least 12 causal active-precipitation candidates before capping.
11. Qualified all 96 systems and recovered all 22 formerly rejected systems.
12. Froze the replacement 72/12/12 split with explicit TRAIN season/regime reserves.
13. Sealed replacement NEW FINAL unscored; SHA-256: `d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3`.
14. Passed the expanded Cycle-2 integrity suite: 11 passed, 0 failed.

## Current running phase

`All-96 residual-relevant row mining and replacement freeze` is complete.

- Qualified systems: 96.
- Rejected systems: 0.
- Formerly rejected systems recovered: 22/22.
- Clean-initiation rows: 653.
- Active-precipitation rows: 2,304.
- Hard negatives: 60.
- Minimum valid active candidates in any qualified system: 38.
- Qualified systems with fewer than three clean initiations: 22.
- Exceptions affecting accepted outputs: none.

## Issues and corrections

### Missing MRMS scan

One source scan was genuinely absent and remains explicitly missing. No interpolation, substitution, or fabrication was introduced.

### Timestamp-unit validation bug

The installed pandas build represents timestamps as `datetime64[us, UTC]`; integer comparison against nanosecond `Timedelta.value` falsely rejected valid cadence. Cadence now uses direct timedelta equality. This was validation-only and did not alter materialized data.

### Hard-negative issue-frame validity

Hard negatives now require both aggregate-window and issue-frame validity of at least 0.90. Negatives were generated with finite issue-time maxima before either accepted freeze.

### Initiation-gated first freeze

The first freeze required three clean-initiation objects per system, yielding 74 qualified systems and only one winter TRAIN system. This was inappropriate for a model whose primary target is correction of existing precipitation evolution. Because no training, Cycle-2 PySTEPS forecast, prediction, or metric existed, the first freeze was archived as `SUPERSEDED_PRETRAINING` rather than scientifically consumed.

Replacement qualification uses only source integrity and at least 12 valid issue-time-selected active rows before the 24-row cap. Clean initiation remains a dedicated subset and hard-negative availability does not gate inclusion. All 22 previously rejected systems were recovered.

## Scientific safeguards

- [x] >=60 qualified independent systems
- [x] replacement NEW DEV contains 12 systems
- [x] replacement NEW FINAL contains 12 systems
- [x] no system crosses splits
- [x] legacy events excluded from DEV/FINAL
- [x] season/year/regime split audit complete
- [x] active-row eligibility uses issue-time radar only
- [x] residual qualification does not use future truth or forecast skill
- [x] clean initiation does not gate inclusion
- [x] clean-initiation subset retained
- [x] hard-negative manifests retained
- [x] TRAIN seasonal support constraints pass
- [x] rare descriptive regimes remain in TRAIN when avoidable
- [x] TRAIN/DEV normalization isolation tested
- [x] FINAL loaders blocked from TRAIN/DEV
- [x] replacement FINAL manifest sealed unscored
- [x] superseded manifest/hash retained
- [x] no FINAL forecasts or metrics generated
- [x] no training occurred before replacement freeze
- [x] expanded leakage/integrity suite passes

## Dataset counts

| Quantity | First freeze | Replacement freeze |
| --- | ---: | ---: |
| Materialized systems | 96 | 96 |
| Qualified systems | 74 | 96 |
| Rejected systems | 22 | 0 |
| NEW TRAIN systems | 50 | 72 |
| NEW DEV systems | 12 | 12 |
| NEW FINAL systems | 12 | 12 |
| Clean initiation rows | 631 | 653 |
| Active precipitation rows | 1,776 | 2,304 |
| Hard negatives | 48 | 60 |

### Replacement systems by season

| Split | Autumn | Spring | Summer | Winter |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 18 | 18 | 24 | 12 |
| DEV | 3 | 3 | 3 | 3 |
| FINAL | 3 | 3 | 3 | 3 |

The first freeze had only one winter TRAIN system; the replacement has 12.

### Replacement systems by year

| Split | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 4 | 13 | 10 | 18 | 9 | 18 |
| DEV | 1 | 2 | 2 | 2 | 2 | 3 |
| FINAL | 1 | 2 | 3 | 2 | 2 | 2 |

### Replacement systems by descriptive regime

| Split | Convective | Mixed | Stratiform |
| --- | ---: | ---: | ---: |
| TRAIN | 1 | 71 | 0 |
| DEV | 1 | 11 | 0 |
| FINAL | 1 | 11 | 0 |

The simple classifier labels 3 systems convective and 93 mixed; none are labeled stratiform. This describes the coarse classifier/materialization slate and is not a claim that stratiform meteorology is absent. Unlike the first freeze, TRAIN retains a convective-labeled system.

## Qualification failures

No systems failed replacement qualification. Every system had valid source integrity and at least 38 active candidates, exceeding the frozen threshold of 12.

## Tests

- Command: `uv run pytest tests/test_cycle2_residual.py tests/test_cycle2_dataset_freeze.py -q`
- Result: `11 passed, 0 failed`.
- Coverage includes active-row qualification, qualification with fewer than three clean initiations, no model/PySTEPS qualification fields, split disjointness, exact DEV/FINAL counts, TRAIN season/regime support, legacy exclusion, sealed-final exclusion, row geometry/timing, hard-negative validity, loader guards, superseded-manifest provenance, model identity/parameter checks, and replacement-final sealing.

## Residual V1 training

**Complete — promoted on DEV, evaluated once on FINAL, and permanently frozen.**

The guarded cache contains 2,283 TRAIN and 381 DEV rows, with zero FINAL rows. TRAIN-only normalization completed successfully. The replacement FINAL manifest retained SHA-256 `d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3` and was not used for forecasts, predictions, targets, errors, or metrics.

Training completed 22 epochs on CPU and stopped at the frozen patience of eight. It remained numerically healthy: no NaN/Inf loss, invalid gradient, or split-boundary failure occurred.

| Training item | Completed value |
| --- | --- |
| Completed epochs | 22 |
| Early stopping | Yes, patience 8/8 |
| Best epoch | 14 |
| Best checkpoint score | 0.934005 |
| Best F1 / FSS18 / onset MAE at checkpoint threshold 0.50 | 0.585108 / 0.724875 / 6.7699 min |
| Best checkpoint SHA-256 | `b536ca16daaaf8904b9c326ed00c01169105d764c2072249092a9259ba9a22c5` |
| Final recorded learning rate | 0.000187 |
| Normalization SHA-256 | `23dc3b31edad1247fc77abd4ec7d44688cdd79e21b3d0f832260130df36c192c` |
| Configuration SHA-256 | `13456f0e6a940f3a8a41f68e7c9e7f7efdf4eaf5e24273bcc96f5b967250495b` |

## Residual V1 DEV evaluation

The complete 0.10–0.90 threshold sweep selected **0.35** using the frozen weather-system-macro rule and hard-negative constraints. At that threshold, Residual V1 improved event-macro F1 from 0.5371 to 0.6416 and FSS18 from 0.7111 to 0.7722. Occurrence Brier improved from 0.1966 to 0.1042. All 12 DEV systems had strictly higher F1.

Diagnostics are complete:

- active-precipitation F1: 0.6130 → 0.7141;
- strongest failure-category gain: growth underprediction, ΔF1 +0.140;
- zeroing PySTEPS: ΔF1 −0.224 and ΔFSS18 −0.233;
- reducing history to the latest frame: ΔF1 −0.035;
- zeroing explicit LK motion: negligible categorical effect;
- radar-blind diagnostic detection: 16.3% with 9.9% false-area cost, insufficient for an initiation-capability claim;
- hard-negative operational wet area and false initiation: both 0.0.

All eight frozen promotion gates passed. The scientific classification is:

**PROMOTE — FREEZE FOR ONE-TIME FINAL EVALUATION**

The procedure was frozen after DEV and has now completed its authorized one-time FINAL evaluation. Its current status is `CONSUMED_FINAL`. Full results are in `docs/cycle2_residual_v1.md`.

## Next action

V1 is closed. Do not retrain, retune, recalibrate, alter the checkpoint, or reuse the consumed systems as an independent holdout. Any further model must be a separately predeclared V2 with new FINAL data.

## One-time independent FINAL evaluation

The frozen epoch-14 checkpoint was evaluated once on October 6, 2026 after all required hashes matched. All 353 predictions across 12 systems were generated and hash-sealed before scoring.

| FINAL event-macro metric | PySTEPS | Residual V1 | Delta |
| --- | ---: | ---: | ---: |
| F1 | 0.5644 | **0.6810** | **+0.1166** |
| FSS18 | 0.7007 | **0.7842** | **+0.0835** |
| Brier | 0.2245 | **0.1117** | **−0.1128** |
| AUC | 0.6800 | **0.8527** | **+0.1727** |
| Rate MAE | 0.5468 | **0.5018** | **−0.0450** |

F1 improved strictly in 12/12 systems. The strong growth-underprediction correction replicated, categorical/spatial gains widened with lead, intensity skill improved, and operational hard-negative wet area remained zero. Radar-blind initiation remained weak and costly, so it is not part of the validated claim.

**Final classification: `FINAL VALIDATED — RESIDUAL V1 GENERALIZES`.**

NEW FINAL is permanently `CONSUMED_FINAL`; the consumption timestamp and frozen procedure hashes are recorded under `artifacts/cycle2/residual_v1/final_evaluation/`.
