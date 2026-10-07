# Cycle 2 residual-learning dataset freeze

**Status:** `FROZEN_PRETRAINING_REPLACEMENT`. Training is not authorized by this document.

## Why the first freeze was superseded

The first freeze qualified systems only when they contained at least three clean-initiation objects. That produced 74 qualified systems, rejected 22 solely for initiation count, and left TRAIN with one winter system. This qualification was misaligned with `PySTEPSResidualUNetV1`, whose primary purpose is to correct growth, decay, displacement, and intensity of radar-observed precipitation.

The correction occurred before model training, Cycle-2 PySTEPS generation, learned predictions, model metrics, or FINAL scoring. The first split and manifest are preserved under `artifacts/cycle2/data/superseded_initiation_gated_freeze/` with status `SUPERSEDED_PRETRAINING` and manifest SHA-256 `32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e`.

## Replacement qualification

A system qualifies when:

1. source, cadence, and integrity checks pass;
2. complete issue-time geometry supports the 60-minute history and 120-minute horizon; and
3. at least 12 valid active-precipitation rows are available before the 24-row cap.

Active rows are selected at fixed 30-minute spacing from current radar only, using components above 0.1 mm/h containing at least 12 analysis-grid pixels. Future evolution, PySTEPS errors, model outputs, and model skill are not selection inputs. Clean initiation remains a capped evaluation subset but does not gate systems. Hard-negative availability is not required.

All 96 systems qualified; the minimum active-candidate count was 38. All 22 systems rejected by the first freeze were recovered, and 22 qualified systems have fewer than three clean initiations.

## Replacement split

| Split | Systems | Autumn | Spring | Summer | Winter |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 72 | 18 | 18 | 24 | 12 |
| DEV | 12 | 3 | 3 | 3 | 3 |
| FINAL | 12 | 3 | 3 | 3 | 3 |

Year counts:

| Split | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 4 | 13 | 10 | 18 | 9 | 18 |
| DEV | 1 | 2 | 2 | 2 | 2 | 3 |
| FINAL | 1 | 2 | 3 | 2 | 2 | 2 |

Coarse descriptive regime counts:

| Split | Convective | Mixed | Stratiform |
| --- | ---: | ---: | ---: |
| TRAIN | 1 | 71 | 0 |
| DEV | 1 | 11 | 0 |
| FINAL | 1 | 11 | 0 |

These labels are not meteorological ground truth. The absence of a `stratiform` label reflects the simple classifier and/or slate; it does not establish the absence of stratiform meteorology.

## Row manifests

| Split | Clean initiation | Active precipitation | Hard negatives |
| --- | ---: | ---: | ---: |
| TRAIN | 512 | 1,728 | 43 |
| DEV | 85 | 288 | 8 |
| FINAL | 56 | 288 | 9 |
| **Total** | **653** | **2,304** | **60** |

Rows are capped at 12 clean-initiation, 24 active-precipitation, and one hard-negative row per system. Hard negatives require aggregate-window and issue-frame validity of at least 0.90.

## FINAL seal and provenance

- Replacement status: `SEALED_UNSCORED`.
- Replacement final-manifest SHA-256: `d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3`.
- Superseded manifest SHA-256: `32d83c5c69e07cee1b5f072892500c05dbaf20679bb027b1d248b2555c64196e`.
- No replacement-FINAL PySTEPS forecasts, learned inputs, predictions, or metrics exist.
- TRAIN/DEV manifests contain no replacement-FINAL rows.
- All source hashes are recorded in `artifacts/cycle2/data/dataset_freeze_report.json`.

## Integrity verification

Command:

`uv run pytest tests/test_cycle2_residual.py tests/test_cycle2_dataset_freeze.py -q`

Result: **11 passed, 0 failed**.

Verified properties include the ≥12 active-row qualification, qualification with fewer than three clean initiations, lack of model-derived qualification fields, system-disjoint splits, exact 12-system DEV/FINAL partitions, TRAIN seasonal/regime support, legacy exclusion, FINAL loader guards, superseded-manifest provenance, hard-negative validity, and sealed replacement FINAL.

## Review boundary

The dataset is frozen but model training remains stopped pending scientific review of this replacement freeze.
