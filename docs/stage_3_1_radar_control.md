# Stage 3.1 radar-only neural control

Status: complete as a limited Stage 3.1 radar-only control.

Stage 4 multimodal training has not been started.

## Frozen Stage 3 baseline

The original Stage 3 run is preserved as `minimal_conv_lstm_stage_3`.

Immutable baseline hashes are recorded in:

- `artifacts/stage_3_1/minimal_conv_lstm_stage_3_baseline_manifest.json`

A post-run hash check confirmed that the recorded Stage 3 files still match
their baseline hashes. The following Stage 3 artifacts were not overwritten:

- `artifacts/stage_3/model_best.pt`
- `artifacts/stage_3/training_metadata.json`
- `artifacts/stage_3/dev_metrics.csv`
- `artifacts/stage_3/test_metrics.csv`
- `artifacts/stage_3/test_event_metrics.csv`
- `artifacts/stage_3/sample_manifest.csv`
- `artifacts/stage_3/sample_manifest_summary.json`
- `docs/stage_3_radar_only.md`

This baseline remains the 24,120-parameter lightly trained ConvLSTM. It is a
minimal-model failure baseline, not evidence that radar-only learned nowcasting
has saturated.

## Split hygiene

Stage 3.1 split policy is defined in:

- `configs/experiments/stage_3_1_split_manifest.yaml`

The Stage 3 train/dev events are retained. The previously inspected Stage 3
test events are relabeled as `existing_test` and were used only for diagnostic
comparison after the Stage 3.1 procedure was frozen.

Four new independent fresh-holdout events were identified and materialized
before final model scoring:

- `beryl_remnants_rain_2024_07_10`
- `midsummer_storms_2024_07_31`
- `debby_remnants_rain_2024_08_09`
- `spring_rain_2025_05_16`

Fresh-holdout materialization artifacts:

- `artifacts/stage_3_1/fresh_holdout_materialization/`

Stage 3.1 sample manifest:

- `artifacts/stage_3_1/sample_manifest.csv`

Manifest counts:

| Split | Samples |
|---|---:|
| train | 1,037 |
| dev | 84 |
| existing_test | 172 |
| fresh_holdout | 101 |

Sample-category counts:

| Split | Active | Dissipation | Dry/low | Initiation | Random |
|---|---:|---:|---:|---:|---:|
| train | 240 | 120 | 245 | 288 | 144 |
| dev | 30 | 0 | 0 | 36 | 18 |
| existing_test | 40 | 14 | 46 | 48 | 24 |
| fresh_holdout | 40 | 19 | 18 | 0 | 24 |

Audit note: the dev split has initiation, active, and random samples, but no
`dissipation` or `dry_low_activity` samples. This was inherited from Stage 3.
The two candidate dev dry-negative events did not yield strict dry/low-activity
128 x 128 windows under the existing tile criterion. DEV thresholding still
contains many dry pixels inside wet-event tiles, but sample-category coverage is
incomplete.

## Minimal ConvLSTM drizzle-collapse diagnosis

Frozen minimal-model head diagnostics were written to:

- `artifacts/stage_3_1/minimal_conv_lstm_stage_3_head_analysis/`

The analysis keeps these quantities separate:

- occurrence probability: `P(rate >= 0.1 mm/h)`
- conditional intensity: intensity head decoded to mm/h
- expected rate: occurrence probability multiplied by conditional intensity
- observed rate: MRMS target rate

DEV pixel-class summaries:

| Observed pixel class | P(rate >= 0.1) mean | Conditional rate mean | Expected rate mean | Observed rate mean |
|---|---:|---:|---:|---:|
| dry, <0.1 mm/h | 0.204 | 1.320 | 0.259 | 0.000 |
| light, 0.1-1 mm/h | 0.294 | 1.230 | 0.335 | 0.549 |
| moderate/heavy, >=1 mm/h | 0.345 | 1.171 | 0.375 | 9.614 |

Existing-test diagnostic pixel-class summaries:

| Observed pixel class | P(rate >= 0.1) mean | Conditional rate mean | Expected rate mean | Observed rate mean |
|---|---:|---:|---:|---:|
| dry, <0.1 mm/h | 0.159 | 1.383 | 0.224 | 0.000 |
| light, 0.1-1 mm/h | 0.288 | 1.230 | 0.330 | 0.517 |
| moderate/heavy, >=1 mm/h | 0.332 | 1.190 | 0.366 | 6.669 |

Interpretation:

1. The occurrence head was weakly informative but poorly separated.
2. The conditional intensity head was the clearest collapse point: it regressed
   toward about 1.2 mm/h across dry, light, and heavy observed classes.
3. The expected-rate product therefore stayed in the drizzle/light-rain range
   even where observed rain was heavy.
4. The deterministic rule `expected rate > 0.1 mm/h` is not equivalent to
   `P(rate >= 0.1 mm/h) > threshold`.

DEV-only probability thresholding for the minimal model selected 0.25 by
F1/CSI. DEV Brier score for `P(rate >= 0.1)` was 0.198.

## Stage 3.1 model

Primary Stage 3.1 config:

- `configs/experiments/stage_3_1_radar_control.yaml`

Architecture:

- multi-scale convolutional encoder
- ConvLSTM latent recurrence
- multi-scale decoder with skip connections
- four occurrence exceedance heads: 0.1, 1.0, 2.5, and 5.0 mm/h
- conditional log-intensity head

Trainable parameters: 528,490.

This is over 20 times larger than the 24,120-parameter pilot while still being
small enough to debug. The approximate receptive field is substantially larger
than the minimal model because recurrent processing happens at quarter
resolution after two downsampling stages and is decoded back to 128 x 128.

Loss:

- train-only weighted BCE for occurrence exceedance heads
- Smooth L1 on rainy-pixel log intensity
- occurrence thresholds: 0.1, 1.0, 2.5, 5.0 mm/h

Train-only positive weights were derived from the Stage 3.1 train split.

## Tiny-overfit preflight

Final tiny-overfit artifacts:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k_tiny_overfit/`

Result:

- device: CPU
- CUDA available: false
- steps: 120
- tiny batches: 2
- initial loss: 2.131
- final loss: 1.061
- final/initial loss ratio: 0.498
- mean gradient norm: 1.001
- finite target-mask fraction: 1.000
- final mean `P(rate >= 0.1)`, dry pixels: 0.250
- final mean `P(rate >= 0.1)`, rainy pixels: 0.743
- final expected-rate MAE: 0.765 mm/h
- final wet-pixel expected-rate MAE: 1.583 mm/h

This demonstrates that the Stage 3.1 architecture, masks, gradients, loss path,
and output heads can fit a very small sample. It is not a generalization
result.

## Main convergence training

Main training artifacts:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/`

CUDA was not available in this environment. The run therefore used CPU training
with event preloading and the complete 1,037-sample train manifest each epoch.

Training used best-dev checkpointing, ReduceLROnPlateau, and early stopping.
It stopped after epoch 20 with the best checkpoint at epoch 14.

| Epoch | Train loss | DEV loss | LR |
|---:|---:|---:|---:|
| 1 | 1.275 | 3.031 | 0.00050 |
| 5 | 0.736 | 1.863 | 0.00050 |
| 10 | 0.652 | 1.736 | 0.00050 |
| 14 | 0.582 | 1.674 | 0.00050 |
| 20 | 0.534 | 1.724 | 0.00025 |

Best checkpoint:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/model_best.pt`

The procedure was frozen before existing-test and fresh-holdout evaluation in:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/frozen_stage_3_1_manifest.json`

Frozen checkpoint SHA-256:

- `b84671ca06239acd99eaf0ec82e4a9a4df3e9674a061d9a65d5fc5ee549e17f2`

DEV selected operational occurrence probability threshold:

- `P(rate >= 0.1 mm/h) >= 0.85`

The threshold was selected on DEV only.

## Stage 3.1 head diagnostics

DEV diagnostics:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/head_analysis_dev/`

DEV pixel-class summaries:

| Observed pixel class | P(rate >= 0.1) mean | Conditional rate mean | Expected rate mean | Observed rate mean |
|---|---:|---:|---:|---:|
| dry, <0.1 mm/h | 0.565 | 1.651 | 1.126 | 0.000 |
| light, 0.1-1 mm/h | 0.813 | 2.321 | 2.005 | 0.550 |
| moderate/heavy, >=1 mm/h | 0.889 | 3.018 | 2.797 | 9.298 |

Fresh-holdout pixel-class summaries:

| Observed pixel class | P(rate >= 0.1) mean | Conditional rate mean | Expected rate mean | Observed rate mean |
|---|---:|---:|---:|---:|
| dry, <0.1 mm/h | 0.193 | 0.710 | 0.292 | 0.000 |
| light, 0.1-1 mm/h | 0.834 | 1.979 | 1.758 | 0.523 |
| moderate/heavy, >=1 mm/h | 0.909 | 3.021 | 2.864 | 5.079 |

The Stage 3.1 model corrected the worst minimal-model intensity collapse, but
it still underestimates heavier rain and its expected-rate field tends to
overpredict light rain. Probability and expected-rate products should therefore
continue to be reported separately.

DEV-only threshold sweep for `P(rate >= 0.1)`:

| Probability threshold | CSI | POD | FAR | F1 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.347 | 1.000 | 0.653 | 0.515 |
| 0.25 | 0.381 | 0.994 | 0.618 | 0.552 |
| 0.50 | 0.447 | 0.956 | 0.544 | 0.618 |
| 0.70 | 0.509 | 0.863 | 0.446 | 0.675 |
| 0.80 | 0.531 | 0.785 | 0.378 | 0.694 |
| 0.85 | 0.532 | 0.724 | 0.333 | 0.695 |
| 0.90 | 0.509 | 0.630 | 0.275 | 0.674 |
| 0.95 | 0.413 | 0.457 | 0.192 | 0.584 |

DEV Brier score for `P(rate >= 0.1)` was 0.272.

## Deterministic verification at 0.1 mm/h

These metrics use the expected-rate field thresholded at 0.1 mm/h, not the
probability-head threshold.

| Split | Model | 6 min CSI | 30 min CSI | 60 min CSI | 120 min CSI |
|---|---|---:|---:|---:|---:|
| DEV | Stage 3.1 learned | 0.469 | 0.366 | 0.357 | 0.329 |
| DEV | PySTEPS | 0.851 | 0.651 | 0.517 | 0.360 |
| Existing test | Stage 3.1 learned | 0.446 | 0.329 | 0.275 | 0.229 |
| Existing test | PySTEPS | 0.814 | 0.617 | 0.500 | 0.358 |
| Existing test | minimal Stage 3 | 0.135 | 0.140 | 0.147 | 0.155 |
| Fresh holdout | Stage 3.1 learned | 0.515 | 0.390 | 0.316 | 0.222 |
| Fresh holdout | PySTEPS | 0.825 | 0.630 | 0.520 | 0.423 |
| Fresh holdout | minimal Stage 3 | 0.143 | 0.131 | 0.119 | 0.104 |

Stage 3.1 beats the minimal ConvLSTM substantially at 6 minutes and from 30
minutes onward on the fresh holdout, and it greatly reduces the minimal model's
rain-everywhere false-alarm behavior. It does not beat PySTEPS.

## Multi-threshold verification example

Full metric CSVs report CSI, POD, FAR, F1, FSS, Brier, and reliability by lead
time and rain threshold:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/evaluation/dev_metrics.csv`
- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/evaluation/existing_test_metrics.csv`
- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/evaluation/fresh_holdout_metrics.csv`

At 60 minutes:

| Split | Model | Threshold | CSI | POD | FAR | F1 | FSS 18 km | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | Stage 3.1 | 0.1 | 0.357 | 0.999 | 0.643 | 0.526 | 0.544 | 0.301 |
| DEV | Stage 3.1 | 1.0 | 0.391 | 0.908 | 0.594 | 0.562 | 0.608 | 0.320 |
| DEV | Stage 3.1 | 2.5 | 0.326 | 0.540 | 0.548 | 0.492 | 0.625 | 0.376 |
| DEV | Stage 3.1 | 5.0 | 0.090 | 0.094 | 0.299 | 0.165 | 0.149 | 0.456 |
| DEV | PySTEPS | 0.1 | 0.517 | 0.615 | 0.235 | 0.682 | 0.845 | 0.232 |
| DEV | PySTEPS | 1.0 | 0.451 | 0.544 | 0.275 | 0.622 | 0.793 | 0.202 |
| DEV | PySTEPS | 2.5 | 0.360 | 0.472 | 0.397 | 0.529 | 0.729 | 0.171 |
| DEV | PySTEPS | 5.0 | 0.254 | 0.364 | 0.544 | 0.405 | 0.673 | 0.131 |
| Fresh holdout | Stage 3.1 | 0.1 | 0.316 | 0.991 | 0.683 | 0.481 | 0.265 | 0.114 |
| Fresh holdout | Stage 3.1 | 1.0 | 0.338 | 0.890 | 0.648 | 0.505 | 0.496 | 0.125 |
| Fresh holdout | Stage 3.1 | 2.5 | 0.274 | 0.599 | 0.664 | 0.430 | 0.470 | 0.134 |
| Fresh holdout | Stage 3.1 | 5.0 | 0.120 | 0.133 | 0.450 | 0.214 | 0.375 | 0.146 |
| Fresh holdout | PySTEPS | 0.1 | 0.520 | 0.698 | 0.329 | 0.684 | 0.684 | 0.086 |
| Fresh holdout | PySTEPS | 1.0 | 0.434 | 0.590 | 0.378 | 0.605 | 0.669 | 0.063 |
| Fresh holdout | PySTEPS | 2.5 | 0.337 | 0.487 | 0.478 | 0.504 | 0.670 | 0.041 |
| Fresh holdout | PySTEPS | 5.0 | 0.229 | 0.346 | 0.596 | 0.373 | 0.631 | 0.024 |

## Initiation and dissipation

Lifecycle metrics are written to:

- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/lifecycle/existing_test_lifecycle_metrics.csv`
- `artifacts/stage_3_1/multiscale_convlstm_radar_control_528k/lifecycle/fresh_holdout_lifecycle_metrics.csv`

Existing-test initiation:

| Model | Precision | Recall | MAE min | Median bias min | False fraction |
|---|---:|---:|---:|---:|---:|
| Stage 3.1 expected rate | 0.438 | 0.971 | 34.46 | -24 | 0.562 |
| Stage 3.1 probability head | 0.883 | 0.324 | 19.29 | 0 | 0.117 |
| Optical flow | 0.788 | 0.435 | 15.12 | 0 | 0.213 |
| PySTEPS | 0.794 | 0.526 | 15.43 | 0 | 0.206 |

The expected-rate field detects many initiations by raining too broadly, so it
does not demonstrate useful initiation skill. The DEV-thresholded probability
head controls false initiations better, but recall remains below PySTEPS.

Existing-test dissipation:

| Model | Precision | Recall | MAE min | Median bias min | False fraction |
|---|---:|---:|---:|---:|---:|
| Stage 3.1 expected rate | 1.000 | 0.137 | 55.77 | 72 | 0.000 |
| Stage 3.1 probability head | 1.000 | 0.980 | 11.74 | 0 | 0.000 |
| Optical flow | 1.000 | 0.996 | 7.54 | 0 | 0.000 |
| PySTEPS | 1.000 | 0.954 | 6.18 | 0 | 0.000 |

Fresh holdout did not include initiation-centered samples, so only dissipation
was scored there:

| Model | Precision | Recall | MAE min | Median bias min | False fraction |
|---|---:|---:|---:|---:|---:|
| Stage 3.1 expected rate | 1.000 | 0.161 | 52.38 | 72 | 0.000 |
| Stage 3.1 probability head | 1.000 | 0.815 | 14.95 | 0 | 0.000 |
| Optical flow | 0.998 | 0.903 | 12.81 | 6 | 0.002 |
| PySTEPS | 1.000 | 0.918 | 11.87 | 6 | 0.000 |

The probability head learns useful dissipation timing, though PySTEPS and
optical flow remain stronger. The expected-rate field dissipates too late and
misses most cessations.

## Optional PySTEPS residual control

The optional PySTEPS-plus-learned-residual control was not implemented in this
limited run. The main radar-only control, diagnostics, and final holdout
evaluation already consumed the available CPU budget, and adding PySTEPS
forecast generation inside a training loop would have materially expanded the
experiment. This remains a useful later control, but it should not block Stage
4.

## Stage 3.1 decision answers

1. Was the original Stage 3 failure primarily undertraining/capacity/output
   decoding?

   Largely yes for the minimal-model failure: the original 24k model was
   under-capacity/undertrained and its conditional intensity head collapsed to
   drizzle. Output decoding also contributed because expected-rate thresholding
   was conflated with occurrence-probability thresholding. Stage 3.1 removes
   those obvious explanations, but does not solve radar-only nowcasting.

2. Does a reasonably trained radar-only network beat the minimal ConvLSTM
   substantially?

   Yes, especially on the fresh holdout at 6 minutes and from 30 minutes onward
   by 0.1 mm/h expected-rate CSI, with much lower false-alarm behavior than the
   minimal model. The minimal model still shows some misleading short-lead CSI
   at 12-18 minutes because its broad rain field overlaps rain areas while
   producing excessive false alarms.

3. Does it approach or beat PySTEPS anywhere?

   It approaches PySTEPS only at long DEV lead times, but it does not beat
   PySTEPS on existing-test or fresh-holdout verification. PySTEPS remains the
   stronger radar-only operational baseline.

4. Does it show real initiation skill after controlling false alarms?

   Not convincingly. The expected-rate field gets high recall by raining too
   broadly. The probability head controls false initiations better, but recall
   is lower than PySTEPS.

5. Does it learn useful dissipation?

   Yes for the probability head: dissipation recall is high with near-zero false
   dissipation on existing-test and fresh holdout. The expected-rate field is
   poor for dissipation because it keeps light rain too long.

6. Is the probability head informative even when deterministic expected-rate
   output is poor?

   Yes. The probability head separates dry/light/heavy pixels much more clearly
   than the minimal model, supports a DEV-selected threshold of 0.85, and gives
   more useful lifecycle timing than the expected-rate field.

7. Does a learned correction to PySTEPS help?

   Not answered in this limited run. The optional residual control was deferred.

8. What remains impossible or weak using radar history alone?

   Heavy-rain intensity remains underestimated, expected-rate outputs still
   smear light rain, initiation remains weak after controlling false alarms, and
   PySTEPS remains stronger overall. This supports moving to Stage 4, but it
   should be framed as testing incremental atmospheric information over a
   competent radar-only control, not as proving radar-only learned nowcasting is
   inherently saturated.

## Stage 4 handoff

Stage 4 may proceed next. Freeze the Stage 3.1 split, target definition,
training/evaluation procedure, and primary radar-only architecture as much as
practical before adding multimodal inputs. The Stage 4 question is whether
atmospheric state information adds incremental skill over this competent
radar-only control.
