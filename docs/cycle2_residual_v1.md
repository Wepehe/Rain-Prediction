# Cycle-2 PySTEPS Residual U-Net V1

## Executive summary

- **Question:** can a compact learned residual model improve radar-observed 0–2 h precipitation evolution beyond deterministic Lucas–Kanade PySTEPS?
- **Independent dataset:** 96 new weather systems split into 72 TRAIN, 12 DEV, and 12 one-time FINAL systems.
- **Model:** a 3,060,440-parameter U-Net consuming radar history, deterministic PySTEPS, LK motion, and explicit validity masks.
- **Training:** completed 22 CPU epochs; early stopping triggered after eight consecutive non-improvements.
- **Best checkpoint:** epoch 14, selected at the fixed 0.50 checkpoint threshold with score **0.9340**.
- **Frozen DEV operating threshold:** **0.35**, selected from the predeclared grid under both hard-negative constraints.
- **Headline DEV result:** event-macro F1 improved from **0.5371 to 0.6416** and FSS18 from **0.7111 to 0.7722**; all 12 systems improved in F1.
- **Final result:** **FINAL VALIDATED — RESIDUAL V1 GENERALIZES**. FINAL F1 improved from 0.5644 to 0.6810 and all 12 systems improved.

## 1. Scientific question

V1 tests whether learning a correction around a competent deterministic motion forecast improves the evolution of precipitation already represented in radar. It predicts occurrence and a residual correction to PySTEPS in log-rate space.

This is not primarily a radar-blind initiation model. Any apparent initiation behavior is reported separately and is not the basis of promotion.

## 2. Independent dataset

The Cycle-2 corpus contains 96 independent systems spanning autumn, spring, summer, and winter.

| Split | Systems | Active precipitation | Clean initiation | Hard negatives |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 72 | 1,728 | 512 | 43 |
| DEV | 12 | 288 | 85 | 8 |
| FINAL, now consumed | 12 | 288 | 56 | 9 |

Each row contains ten 6-minute radar-history frames and twenty 6-minute targets on a 128×128 tile at 2 km resolution. The first initiation-gated freeze was superseded before training because it was not representative for residual evolution learning; no result from it was reused.

## 3. Frozen model

| Item | Frozen value |
| --- | --- |
| Architecture | `PySTEPSResidualUNetV1` |
| Trainable parameters | 3,060,440 |
| Inputs | 10 MRMS frames; 20 deterministic PySTEPS frames; LK u/v; 10 validity masks |
| Outputs | 20 occurrence logits; 20 residual log-rate corrections |
| Encoder widths | 40 / 80 / 160 / 320 |
| Optimizer | AdamW, LR 3×10⁻⁴, weight decay 10⁻⁴ |
| Scheduler | Cosine |
| Loss | focal occurrence + 0.5 Huber log-rate + 0.25 18-km spatial + 0.05 residual L1 |
| Batch policy | Batch size 4; 571 event-balanced batches/epoch |
| Hard negatives | One negative in approximately 25% of batches; never more than one |
| Training limit | 50 epochs; patience 8; gradient clip 1.0 |
| Checkpoint threshold | 0.50 |
| Sampler seed | 2718 |

The residual intensity projection was exactly zero-initialized. Before training, its maximum absolute correction was 0.0 and reconstructed intensity differed from raw PySTEPS by only 9.54×10⁻⁷ from floating-point roundoff.

## 4. Training

Training completed cleanly after 22 epochs and stopped at patience 8/8. There were no NaN/Inf losses, invalid gradients, or split-boundary violations. All 43 TRAIN hard negatives appeared in every audited epoch; the maximum repeat count was four.

The immutable best checkpoint is epoch 14. Its selection used only the fixed 0.50 probability threshold and the frozen rule `F1 + 0.5 × FSS18 − 0.002 × onset MAE`.

| Epoch | Train loss | DEV F1 @ 0.50 | DEV FSS18 @ 0.50 | Onset MAE | Score | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.1914 | 0.4998 | 0.6299 | 5.49 min | 0.8038 | Initial best |
| **14** | **0.1406** | **0.5851** | **0.7249** | **6.77 min** | **0.9340** | **Frozen best** |
| 22 | 0.1330 | 0.5559 | 0.6892 | 5.21 min | 0.8901 | Early stop |

The decreasing training loss with variable later DEV scores supports best-checkpointing: the final epoch was not the best generalizing model. The complete curve is retained in `artifacts/cycle2/residual_v1/training_history.csv`.

## 5. DEV operating threshold

After checkpoint freeze, the probability threshold was selected once on DEV from 0.10–0.90 in 0.05 increments. Threshold **0.35** maximized positive-row weather-system-macro F1 among candidates satisfying hard-negative wet-area increase ≤0.01 and false-initiation fraction ≤0.40.

| Threshold | F1 | FSS18 | FAR | Onset MAE | Negative wet-area Δ | False initiation | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **0.35** | **0.6482** | **0.7773** | 0.3681 | 9.75 | 0.0000 | 0.00 | **Yes** |
| 0.40 | 0.6395 | 0.7750 | 0.3246 | 8.61 | 0.0000 | 0.00 | Yes |
| 0.45 | 0.6202 | 0.7591 | 0.2799 | 7.70 | 0.0000 | 0.00 | Yes |
| 0.30 | 0.6460 | 0.7647 | 0.4184 | 10.73 | 0.000006 | 1.00 | **No** |

The full sweep is in `artifacts/cycle2/residual_v1/dev_threshold_sweep.csv`. The main comparison uses all DEV row types and therefore differs slightly from the positive-only selection F1.

## 6. Main DEV result

Primary aggregation is the unweighted mean over 12 independent weather systems.

| Metric | Deterministic PySTEPS | Residual V1 | Delta |
| --- | ---: | ---: | ---: |
| CSI | 0.3967 | **0.5133** | **+0.1166** |
| POD | 0.4962 | **0.7145** | **+0.2184** |
| FAR | **0.3550** | 0.3681 | +0.0131 |
| F1 | 0.5371 | **0.6416** | **+0.1046** |
| FSS6 | 0.6519 | **0.7261** | **+0.0742** |
| FSS18 | 0.7111 | **0.7722** | **+0.0611** |
| FSS36 | 0.7495 | **0.8030** | **+0.0535** |
| Onset MAE | **9.37 min** | 9.75 min | +0.39 min |
| Onset bias | −0.40 min | −1.08 min | −0.67 min |
| Cessation MAE | 8.64 min | **5.09 min** | **−3.55 min** |
| Occurrence Brier | 0.1966 | **0.1042** | **−0.0924** |
| Occurrence AUC | 0.6830 | **0.8679** | **+0.1849** |
| Rate MAE | 0.4651 | **0.4513** | **−0.0138** |
| Log-rate MAE | **0.1933** | 0.2061 | +0.0129 |
| Heavy-rain conditional error | 6.7161 | **6.5629** | **−0.1532** |

The main gain comes from much higher detection and spatial coverage with only a modest FAR increase. Probability quality, cessation timing, linear-rate MAE, and heavy-rain error improve; log-rate MAE and onset timing remain slightly better for deterministic PySTEPS.

## 7. Lead-time behavior

| Lead | System | F1 | CSI | POD | FAR | FSS18 | Rate MAE | Brier |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min | PySTEPS | 0.6415 | 0.5066 | 0.6165 | **0.3012** | 0.8428 | 0.4130 | 0.1505 |
|  | Residual V1 | **0.7116** | **0.5923** | **0.7862** | 0.3144 | **0.8580** | **0.3900** | **0.0827** |
| 60 min | PySTEPS | 0.5036 | 0.3675 | 0.4703 | 0.4022 | 0.7085 | 0.4924 | 0.2085 |
|  | Residual V1 | **0.6189** | **0.4961** | **0.6935** | **0.3814** | **0.7655** | **0.4692** | **0.1075** |
| 90 min | PySTEPS | 0.3947 | 0.2744 | 0.3589 | 0.4846 | 0.5868 | 0.5284 | 0.2426 |
|  | Residual V1 | **0.5528** | **0.4335** | **0.6422** | **0.4409** | **0.6942** | **0.5107** | **0.1238** |
| 120 min | PySTEPS | 0.3133 | 0.2104 | 0.2783 | 0.5249 | 0.4741 | **0.5442** | 0.2635 |
|  | Residual V1 | **0.4941** | **0.3825** | **0.5987** | **0.4948** | **0.6260** | 0.5659 | **0.1397** |

The categorical, neighborhood, and probabilistic gains persist and generally widen with lead. The exception is 120-minute rate MAE, where PySTEPS remains better by 0.0217 mm/h.

## 8. Independent-system consistency

| DEV system | PySTEPS F1 | V1 F1 | ΔF1 | PySTEPS FSS18 | V1 FSS18 | ΔFSS18 | PySTEPS onset | V1 onset | Δ onset | Δ Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-10-19 | 0.414 | 0.560 | +0.146 | 0.539 | 0.680 | +0.141 | 13.26 | 13.84 | +0.58 | −0.096 |
| 2021-05-28 | 0.666 | 0.772 | +0.106 | 0.762 | 0.848 | +0.086 | 4.19 | 3.41 | −0.78 | −0.119 |
| 2021-06-03 | 0.389 | 0.450 | +0.060 | 0.589 | 0.592 | +0.003 | 14.04 | 14.68 | +0.64 | −0.033 |
| 2022-02-22 | 0.468 | 0.590 | +0.121 | 0.627 | 0.699 | +0.072 | 10.85 | 13.29 | +2.44 | −0.106 |
| 2022-11-30 | 0.485 | 0.698 | **+0.212** | 0.628 | 0.824 | +0.195 | 8.47 | 10.64 | +2.17 | −0.167 |
| 2023-04-30 | 0.585 | 0.710 | +0.125 | 0.751 | 0.836 | +0.085 | 8.14 | 9.23 | +1.09 | −0.090 |
| 2023-05-03 | 0.577 | 0.651 | +0.074 | 0.759 | 0.776 | +0.016 | 10.13 | 9.81 | −0.32 | −0.096 |
| 2024-08-09 | 0.570 | 0.609 | **+0.039** | 0.818 | 0.789 | −0.030 | 8.96 | 7.79 | −1.17 | −0.064 |
| 2024-09-07 | 0.552 | 0.661 | +0.109 | 0.760 | 0.826 | +0.066 | 9.16 | 9.81 | +0.64 | −0.095 |
| 2025-01-01 | 0.589 | 0.646 | +0.056 | 0.813 | 0.800 | −0.012 | 10.15 | 10.50 | +0.35 | −0.060 |
| 2025-07-31 | 0.582 | 0.661 | +0.079 | 0.748 | 0.773 | +0.025 | 6.83 | 5.79 | −1.04 | −0.064 |
| 2025-12-19 | 0.566 | 0.692 | +0.126 | 0.739 | 0.824 | +0.085 | 8.24 | 8.26 | +0.02 | −0.118 |

F1 improves strictly in **12/12** systems; none is tied within 0.005 or worse by more than 0.005. Median ΔF1 is **+0.1074**. The largest gain is 2022-11-30 (+0.2123); the smallest is 2024-08-09 (+0.0389). Two systems lose FSS18 and five have worse onset MAE, so consistency is strong but not universal across every metric.

## 9. What did the model improve?

| Row type | System | F1 | FSS18 | Onset MAE | Cessation MAE | Rate MAE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Active precipitation | PySTEPS | 0.6130 | 0.7864 | 8.20 | 10.49 | 0.5543 | 0.2255 |
|  | Residual V1 | **0.7141** | **0.8413** | 8.23 | **6.17** | **0.5306** | **0.1180** |
| Clean initiation | PySTEPS | 0.2807 | 0.4563 | **14.85** | 2.24 | **0.1405** | 0.1121 |
|  | Residual V1 | **0.3966** | **0.5358** | 16.89 | **1.55** | 0.1767 | **0.0708** |

On the primary active-precipitation target, V1 gains 0.1011 F1, 0.0549 FSS18, 4.32 minutes in cessation MAE, and roughly halves Brier. Clean-initiation F1 and probability quality improve, but onset MAE and rate error worsen.

| PySTEPS failure category | Cases | ΔF1 | ΔFSS18 | Δ rate MAE | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Growth underprediction | 241 | **+0.140** | **+0.110** | **−0.038** | Clear principal gain |
| Decay persistence | 38 | +0.029 | −0.035 | +0.029 | Categorical gain, poorer structure/rate |
| Displacement | 84 | +0.046 | −0.037 | +0.053 | More detection, no spatial correction |
| Intensity error | 14 | +0.027 | +0.041 | −0.009 | Small broad improvement |
| No initiation signal | 4 | 0.000 | 0.000 | +0.011 | No recovery |

These categories are post-selection diagnostics, not training labels. V1’s strongest and most coherent contribution is correcting growth underprediction.

## 10. What did it not improve?

- FAR increases slightly overall: 0.3550 to 0.3681.
- Overall onset MAE worsens by 0.39 minutes, and clean-initiation onset MAE worsens by 2.03 minutes.
- Log-rate MAE is worse despite a modest improvement in linear-rate MAE.
- At 120 minutes, rate MAE is worse than PySTEPS.
- Two systems lose FSS18 even though all 12 gain F1.
- Displacement and decay-persistence categories gain detection but lose neighborhood or rate skill.
- The explicit LK motion channels appear largely redundant once radar history and PySTEPS are present.

## 11. Radar-blind initiation

Within clean-initiation rows, 123,730 pixel locations across 85 rows and all 12 systems met the strict diagnostic definition: dry radar history, dry deterministic PySTEPS, and a persistent future MRMS onset. V1 detected **16.3%**, with at least one detection in 10/12 systems, but incurred a **9.9%** false-area fraction over otherwise dry locations.

The four row-level cases classified as having no PySTEPS initiation signal had no F1/FSS recovery. This does not demonstrate reliable radar-blind initiation. It shows limited extrapolation beyond the deterministic occurrence footprint at a substantial false-area cost.

| Hard-negative metric | PySTEPS | Residual V1 |
| --- | ---: | ---: |
| Mean thresholded wet area | 0.0000 | 0.0000 |
| Maximum thresholded wet area | 0.0000 | 0.0000 |
| False-initiation fraction | 0.00 | 0.00 |
| Brier | **0.00017** | 0.00219 |
| Mean rain probability | 0.0000 | 0.0386 |
| Maximum rain probability | 0.0000 | 0.3093 |

The model retains sub-threshold uncertainty on dry cases without triggering operational rain at 0.35.

## 12. Destruction tests

| DEV variant | F1 | ΔF1 | FSS18 | ΔFSS18 | Onset MAE | Δ onset MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intact V1 | 0.6416 | — | 0.7722 | — | 9.75 | — |
| Zero PySTEPS channels | 0.4180 | **−0.2236** | 0.5389 | **−0.2332** | 7.76 | −1.99 |
| Latest radar frame only | 0.6061 | −0.0355 | 0.7184 | −0.0538 | 11.14 | +1.39 |
| Zero LK motion | 0.6410 | −0.0006 | 0.7721 | −0.0001 | 9.70 | −0.06 |
| Zero intensity residual | 0.6416 | 0.0000 | 0.7722 | 0.0000 | 9.75 | 0.00 |

PySTEPS is the dominant backbone, while multi-frame radar history supplies additional useful information. Explicit motion channels add almost nothing beyond those inputs. Zeroing the intensity residual leaves probability-head categorical metrics unchanged by construction; its contribution appears in the main rate MAE (−0.0138 mm/h) and heavy-rain error (−0.1532 mm/h).

## 13. Promotion decision

| Gate | Requirement | Observed | Pass/Fail |
| --- | --- | --- | :---: |
| A | F1 ≥ PySTEPS + 0.01 | +0.1046 | **Pass** |
| B | FSS18 ≥ PySTEPS | +0.0611 | **Pass** |
| C | Onset MAE no worse by >3 min | +0.39 min | **Pass** |
| D | ≥7/12 systems noninferior within 0.005 | 12/12 | **Pass** |
| E | ≥5/12 systems strictly better | 12/12 | **Pass** |
| F | Hard-negative wet-area increase ≤0.01 | 0.0000 | **Pass** |
| G | Hard-negative false initiation ≤0.40 | 0.00 | **Pass** |
| H | Brier no worse by >0.005 | −0.0924 | **Pass** |

**PROMOTE — FREEZE FOR ONE-TIME FINAL EVALUATION**

At the DEV decision point, the procedure was frozen as `FROZEN_AWAITING_FINAL_AUTHORIZATION`. Promotion was based on categorical, neighborhood, probabilistic, and cross-system gains—not radar-blind initiation. Section 16 records the subsequently authorized one-time FINAL result.

## 14. Limitations

- DEV contains only 12 independent systems.
- The corpus consists of selected weather-event windows, not continuous climatology.
- Regime labels are coarse; the simple classifier calls most systems “mixed.”
- Training used a CPU-only environment. This affects runtime rather than the frozen statistical comparison, but limits operational training practicality.
- NEW FINAL has now been evaluated exactly once and is permanently consumed; it cannot support later tuning or an independent V2 claim.
- No claim is made about climatological or production deployment validity.

## 15. Next step

Preserve V1 unchanged. Any subsequent model must be named V2, separately predeclared, and evaluated on a new independent holdout. The consumed Cycle-2 FINAL systems may be used only as historical or diagnostic data.

## 16. One-time independent FINAL evaluation

### Integrity and execution

The frozen procedure was evaluated exactly once on October 6, 2026 at 16:03 UTC (12:03 EDT). Before FINAL access, the manifest, checkpoint, normalization, configuration, model-source, threshold, and parameter-count checks all matched exactly. The run contained all 12 independent systems and all 353 rows: 288 active precipitation, 56 clean initiation, and 9 hard negatives. No system or row was replaced or dropped.

All predictions were generated before scoring began. Each row preserves hashes for its source, PySTEPS fields, normalized model inputs, raw model outputs, reconstructed rate, and serialized tensor file. The generation-complete marker recorded `metrics_computed=false` before the scoring phase was permitted.

### Headline FINAL comparison

Primary aggregation remains the unweighted mean over the 12 independent FINAL weather systems.

| Metric | Deterministic PySTEPS | Residual V1 | Delta |
| --- | ---: | ---: | ---: |
| CSI | 0.4270 | **0.5657** | **+0.1387** |
| POD | 0.5111 | **0.7479** | **+0.2368** |
| FAR | **0.2947** | 0.3236 | +0.0289 |
| F1 | 0.5644 | **0.6810** | **+0.1166** |
| FSS6 | 0.6490 | **0.7448** | **+0.0958** |
| FSS18 | 0.7007 | **0.7842** | **+0.0835** |
| FSS36 | 0.7361 | **0.8111** | **+0.0750** |
| Onset MAE | **8.18 min** | 8.45 min | +0.27 min |
| Onset bias | −0.07 min | −0.10 min | −0.03 min |
| Cessation MAE | 9.91 min | **5.35 min** | **−4.56 min** |
| Occurrence Brier | 0.2245 | **0.1117** | **−0.1128** |
| Occurrence AUC | 0.6800 | **0.8527** | **+0.1727** |
| Rate MAE | 0.5468 | **0.5018** | **−0.0450** |
| Log-rate MAE | 0.2289 | **0.2246** | **−0.0043** |
| Heavy-rain conditional error | 7.2596 | **6.8776** | **−0.3820** |

The principal DEV finding generalized: V1 produces substantially better categorical, neighborhood, probability, cessation, and intensity skill. Its cost is a modest FAR increase and 0.27-minute onset-MAE degradation.

### Lead behavior

| Lead | System | F1 | CSI | POD | FAR | FSS18 | Rate MAE | Brier |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 min | PySTEPS | 0.6826 | 0.5530 | 0.6481 | **0.2432** | 0.8487 | 0.4926 | 0.1616 |
|  | Residual V1 | **0.7556** | **0.6492** | **0.8165** | 0.2657 | **0.8780** | **0.4443** | **0.0842** |
| 60 min | PySTEPS | 0.5316 | 0.4010 | 0.4834 | 0.3396 | 0.6845 | 0.5864 | 0.2349 |
|  | Residual V1 | **0.6615** | **0.5519** | **0.7369** | **0.3387** | **0.7702** | **0.5216** | **0.1144** |
| 90 min | PySTEPS | 0.4313 | 0.3094 | 0.3839 | 0.3973 | 0.5663 | 0.6175 | 0.2833 |
|  | Residual V1 | **0.6060** | **0.4969** | **0.6985** | **0.3915** | **0.7049** | **0.5622** | **0.1362** |
| 120 min | PySTEPS | 0.3493 | 0.2418 | 0.3012 | **0.4248** | 0.4696 | 0.6274 | 0.3163 |
|  | Residual V1 | **0.5589** | **0.4534** | **0.6618** | 0.4384 | **0.6494** | **0.6133** | **0.1550** |

The DEV pattern of widening categorical and spatial benefit with lead generalizes. ΔF1 grows from +0.073 at 30 minutes to +0.210 at 120 minutes; ΔFSS18 grows from +0.029 to +0.180. Rate MAE and Brier improve at every reported lead.

### Independent-system consistency

| FINAL system | PySTEPS F1 | V1 F1 | ΔF1 | PySTEPS FSS18 | V1 FSS18 | ΔFSS18 | PySTEPS onset | V1 onset | Δ onset | Δ Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-11-15 | 0.506 | 0.653 | +0.147 | 0.636 | 0.772 | +0.136 | 11.03 | 12.04 | +1.01 | −0.125 |
| 2021-07-11 | 0.534 | 0.622 | +0.087 | 0.680 | 0.716 | +0.036 | 7.79 | 8.39 | +0.60 | −0.086 |
| 2021-10-25 | 0.591 | 0.653 | +0.062 | 0.744 | 0.761 | +0.018 | 7.38 | 8.37 | +0.98 | −0.093 |
| 2022-02-17 | 0.619 | 0.802 | +0.183 | 0.707 | 0.873 | +0.167 | 6.05 | 5.37 | −0.68 | −0.181 |
| 2022-04-18 | 0.537 | 0.691 | +0.154 | 0.646 | 0.783 | +0.137 | 12.60 | 12.71 | +0.11 | −0.110 |
| 2022-06-12 | 0.489 | 0.517 | **+0.028** | 0.725 | 0.676 | −0.049 | 8.77 | 8.66 | −0.12 | −0.036 |
| 2023-02-22 | 0.613 | 0.823 | +0.210 | 0.677 | 0.865 | +0.188 | 5.21 | 5.11 | −0.10 | −0.252 |
| 2023-08-10 | 0.520 | 0.559 | +0.040 | 0.717 | 0.710 | −0.007 | 9.28 | 8.39 | −0.89 | −0.036 |
| 2024-01-24 | 0.536 | 0.750 | **+0.214** | 0.635 | 0.834 | +0.200 | 8.54 | 10.30 | +1.76 | −0.164 |
| 2024-03-09 | 0.581 | 0.673 | +0.092 | 0.731 | 0.784 | +0.053 | 7.61 | 7.84 | +0.23 | −0.092 |
| 2025-03-20 | 0.621 | 0.734 | +0.113 | 0.745 | 0.837 | +0.092 | 7.37 | 7.64 | +0.26 | −0.104 |
| 2025-10-31 | 0.625 | 0.694 | +0.069 | 0.765 | 0.798 | +0.032 | 6.46 | 6.54 | +0.09 | −0.075 |

F1 improves strictly in **12/12** systems, with no ties within 0.005 and no adverse system. Median ΔF1 is **+0.1027** and median ΔFSS18 is **+0.0727**. The largest F1 gain is 2024-01-24 (+0.2140); the smallest is 2022-06-12 (+0.0276). Two systems lose FSS18, and onset timing changes remain mixed.

### Row types and hard negatives

| Row type | System | F1 | POD | FAR | FSS18 | Onset MAE | Cessation MAE | Rate MAE | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Active precipitation | PySTEPS | 0.6274 | 0.5719 | **0.2578** | 0.7623 | **7.51** | 11.43 | 0.6371 | 0.2494 |
|  | Residual V1 | **0.7447** | **0.8224** | 0.2961 | **0.8419** | 7.61 | **6.17** | **0.5790** | **0.1230** |
| Clean initiation | PySTEPS | 0.2703 | 0.2340 | 0.5355 | 0.4263 | **12.66** | **1.76** | **0.1897** | 0.0913 |
|  | Residual V1 | **0.3678** | **0.3829** | **0.5127** | **0.5101** | 13.82 | 1.83 | 0.2254 | **0.0606** |

Active-precipitation evolution—the primary target—generalizes strongly across categorical, spatial, cessation, rate, and probability metrics. Clean initiation improves in F1, POD, FAR, FSS18, and Brier, but worsens in onset and rate error.

| Hard-negative metric | PySTEPS | Residual V1 |
| --- | ---: | ---: |
| Mean probability | 0.0000 | 0.0386 |
| Maximum probability | 0.0000 | 0.3074 |
| Mean thresholded wet area | 0.0000 | 0.0000 |
| Maximum thresholded wet area | 0.0000 | 0.0000 |
| False-initiation fraction | 0.00 | 0.00 |
| Brier | **0.00023** | 0.00222 |

Dry-case probabilities are worse than deterministic zero, but remain below the frozen operational threshold and do not trigger false rain.

### Failure categories

| Frozen category | Cases | ΔF1 | ΔFSS18 | Δ rate MAE |
| --- | ---: | ---: | ---: | ---: |
| Growth underprediction | 241 | **+0.153** | **+0.132** | **−0.073** |
| Decay persistence | 36 | +0.020 | −0.046 | +0.038 |
| Displacement | 53 | +0.028 | −0.047 | +0.029 |
| Intensity error | 19 | +0.014 | −0.006 | −0.002 |
| No initiation signal | 4 | 0.000 | 0.000 | +0.009 |

The central diagnostic finding also replicates: V1 is especially effective at correcting PySTEPS growth underprediction. It does not resolve displacement, decay persistence, or absent initiation signal in a spatially coherent way.

### Radar-blind initiation

The strict FINAL diagnostic identified 80,290 eligible pixel locations across all 56 clean-initiation rows and all 12 systems. V1 detected **18.0%**, with at least one detection in 11/12 systems, while producing an **11.2%** false-area fraction over otherwise dry locations. The four row-level `no initiation signal` cases again show no F1 or FSS recovery.

This is not reliable radar-blind initiation. The validated claim remains learned correction of radar-observed precipitation evolution.

### Raw probability quality

Raw, uncalibrated Brier improves from 0.2245 to **0.1117**, and AUC improves from 0.6800 to **0.8527**. Probabilities show useful sharpness: 27.0% of valid pixels fall in the 0.0–0.1 bin, while 21.3% fall at or above 0.7.

Reliability is imperfect. The lowest bins overforecast occurrence (mean 0.049 versus observed 0.016; 0.146 versus 0.086), the 0.3–0.4 bin is nearly exact (0.349 versus 0.349), and bins above 0.4 are increasingly underconfident—for example, mean 0.650 corresponds to observed frequency 0.791. No FINAL calibration was fitted.

### Intensity result

The modest intensity benefit generalizes more clearly than on DEV:

- overall rate MAE improves by **0.0450 mm/h**;
- rate MAE improves at 30, 60, 90, and 120 minutes;
- log-rate MAE improves by **0.0043**;
- heavy-rain conditional error improves by **0.3820 mm/h**.

The residual is still chiefly an occurrence/evolution improvement, but the intensity correction is independently positive on FINAL.

### Final classification

**FINAL VALIDATED — RESIDUAL V1 GENERALIZES**

The defining DEV result survives on fully independent systems: substantial categorical and neighborhood gains, better probability quality, improved intensity error, controlled operational hard negatives, and strict F1 improvement in every FINAL system. Remaining limitations—slightly higher FAR, mixed onset timing, weak displacement/decay correction, and no reliable radar-blind initiation—do not reverse the central result.

NEW FINAL is now permanently `CONSUMED_FINAL`. No V1 retuning is permitted, and these systems cannot serve as an independent FINAL set for V2.

## Reproducibility

| Artifact | SHA-256 / value |
| --- | --- |
| Model source | `2a84095e36009174ca662566b2158e8e9fb38e3e17fba280ceb8017e2f739ea2` |
| Best checkpoint | `b536ca16daaaf8904b9c326ed00c01169105d764c2072249092a9259ba9a22c5` |
| Normalization file | `23dc3b31edad1247fc77abd4ec7d44688cdd79e21b3d0f832260130df36c192c` |
| Configuration | `13456f0e6a940f3a8a41f68e7c9e7f7efdf4eaf5e24273bcc96f5b967250495b` |
| TRAIN/DEV cache manifest | `b825ce113f1bcb3cc205df62203971b0b58164120e397b728d8d006ffb1fa506` |
| Sealed FINAL manifest | `d71dd81da4663c98c84858a6890ee824922033b68140fb38cedaea7928c9b0c3` |
| FINAL evaluation JSON | `31c7a118c6b79fd8101ed613343ee11d5d5df9f4217e6e3e7440578f5c3ac7f4` |
| FINAL directory hash manifest | `b2f380639c63240745f43d2c32512c5d4b46a41ddd8f97d86f00e8872be9ca34` |
| Environment | Python 3.12.13; PyTorch 2.14.0+cpu; PySTEPS 1.21.5; NumPy 2.5.2; pandas 3.0.5 |

Key artifacts are under `artifacts/cycle2/residual_v1/`: `training_history.csv`, `dev_threshold_sweep.csv`, `dev_metrics_by_system.csv`, `dev_metrics_by_row_type.csv`, `dev_lead_metrics.csv`, `failure_category_summary.csv`, `destruction_summary.csv`, `evaluation_summary.json`, and `final_procedure_manifest.json`.
