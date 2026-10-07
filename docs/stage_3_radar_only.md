# Stage 3 radar-only learned nowcasting experiment

Stage 3 tests one question: can a modest learned radar-evolution model outperform persistence,
Farnebäck optical flow, and PySTEPS deterministic extrapolation on held-out southern-Ontario events,
especially where precipitation evolves rather than simply translates?

Short answer: no. This first radar-only ConvLSTM does not beat PySTEPS overall, does not show useful
initiation skill, and does not reduce dissipation error. Its apparent detection of new precipitation
comes from predicting too much light rain, not from physically useful initiation modelling. The result
supports moving to Stage 4 radar plus atmospheric state, rather than increasing radar-only model
complexity just to force a win.

## Frozen benchmark and split

The benchmark gate from Milestone 1.5 is satisfied: 18 positive event windows are materialized and
evaluated, all 18 paired hard-negative windows pass radar plus GOES/HRRR verification, and PySTEPS is
available locally.

The Stage 3 split is frozen in `configs/experiments/stage_3_split_manifest.yaml`.

| split | positive events | use |
| --- | ---: | --- |
| train | 12 | model fitting and train-only normalization |
| dev | 2 | early stopping and procedure selection |
| test | 4 | one final held-out evaluation after the dev procedure was frozen |

Frozen test positives:

- `late_june_organized_storms_2024_06_22`
- `august_severe_convection_2024_08_27`
- `lake_effect_precip_2024_11_29`
- `july_organized_convection_2025_07_13`

Paired test hard negatives were included as dry/low-activity test samples, but neural performance was
not inspected until after the train/dev procedure was selected.

## Dataset

Source data are MRMS quantitative precipitation-rate event tensors only. No NWP, GOES, station,
CAPE, humidity, wind, or atmospheric fields enter the model.

Each sample uses:

- input: 10 historical 6-minute frames, approximately 60 minutes
- target: 20 future 6-minute frames, 120 minutes
- tile: 128 × 128 pixels at 2 km resolution, representing 256 × 256 km
- input channels: log-transformed precipitation rate and an explicit valid-radar mask
- target heads: occurrence above 0.1 mm h⁻¹ and log-intensity for observed rainy pixels

Missing radar is masked. It is not treated as zero precipitation.

Sampling manifest: `artifacts/stage_3/sample_manifest.csv`.

| split/category summary | samples |
| --- | ---: |
| train | 568 |
| dev | 60 |
| test | 176 |
| initiation-centered | 252 |
| active precipitation | 180 |
| dissipation | 78 |
| random weather-event | 108 |
| dry/low-activity | 186 |

Initiation-centred samples are intentionally oversampled. Dry/low-activity samples are included but
do not dominate training.

## Architecture

The model is a compact radar-only ConvLSTM encoder-decoder:

```text
(batch, 10, 2, 128, 128)
    -> 2-layer convolutional encoder
    -> ConvLSTM hidden state
    -> convolutional decoder
    -> occurrence logits: (batch, 20, 128, 128)
    -> conditional log-intensity: (batch, 20, 128, 128)
```

Parameter count: 24,120.

The run used CPU only; CUDA was not available. Peak GPU memory is therefore not applicable.

## Loss and training procedure

The loss is a simple composite objective:

- BCE-with-logits occurrence loss on valid target pixels
- Smooth L1 log-intensity loss on observed rainy target pixels
- weights: occurrence 1.0, intensity 0.5

Training used train-only normalization:

- mean log1p rain rate: 0.120598
- std log1p rain rate: 0.410283

Tiny overfit sanity check:

- 2 batches, 20 steps
- loss decreased from 0.7516 to 0.6140

Bounded train/dev run:

| epoch | train loss | dev loss |
| ---: | ---: | ---: |
| 1 | 0.7745 | 1.0878 |
| 2 | 0.6092 | 1.0878 |
| 3 | 0.5679 | 1.0337 |
| 4 | 0.4691 | 1.0586 |
| 5 | 0.4763 | 1.0661 |

Best checkpoint: epoch 3, `artifacts/stage_3/model_best.pt`.

## Final held-out test metrics

All systems were evaluated on identical test samples, anchors, lead times, masks, and thresholds.

At the 0.1 mm h⁻¹ occurrence threshold:

| lead | model | CSI | POD | FAR | F1 | FSS 18 km | Brier |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | learned ConvLSTM | 0.145 | 1.000 | 0.855 | 0.253 | 0.255 | 0.090 |
| 30 | optical flow | 0.534 | 0.650 | 0.251 | 0.696 | 0.761 | 0.082 |
| 30 | persistence | 0.472 | 0.622 | 0.339 | 0.641 | 0.735 | 0.101 |
| 30 | PySTEPS | 0.603 | 0.731 | 0.226 | 0.752 | 0.810 | 0.081 |
| 60 | learned ConvLSTM | 0.150 | 1.000 | 0.850 | 0.261 | 0.254 | 0.111 |
| 60 | optical flow | 0.354 | 0.429 | 0.331 | 0.523 | 0.566 | 0.117 |
| 60 | persistence | 0.389 | 0.534 | 0.411 | 0.560 | 0.562 | 0.126 |
| 60 | PySTEPS | 0.475 | 0.610 | 0.317 | 0.644 | 0.656 | 0.114 |
| 120 | learned ConvLSTM | 0.163 | 1.000 | 0.837 | 0.280 | 0.254 | 0.131 |
| 120 | optical flow | 0.154 | 0.175 | 0.438 | 0.267 | 0.291 | 0.156 |
| 120 | persistence | 0.296 | 0.419 | 0.497 | 0.457 | 0.438 | 0.162 |
| 120 | PySTEPS | 0.330 | 0.435 | 0.423 | 0.496 | 0.443 | 0.154 |

At the 1.0 mm h⁻¹ threshold, the learned model is essentially silent:

| lead | model | CSI | POD | FAR | F1 | FSS 18 km | Brier |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | learned ConvLSTM | 0.000 | 0.000 | n/a | 0.000 | 0.205 | 0.088 |
| 30 | PySTEPS | 0.512 | 0.633 | 0.273 | 0.677 | 0.788 | 0.060 |
| 60 | learned ConvLSTM | 0.000 | 0.000 | 0.800 | 0.000 | 0.193 | 0.093 |
| 60 | PySTEPS | 0.376 | 0.494 | 0.389 | 0.546 | 0.632 | 0.082 |
| 120 | learned ConvLSTM | 0.000 | 0.000 | n/a | 0.000 | 0.136 | 0.104 |
| 120 | PySTEPS | 0.237 | 0.314 | 0.510 | 0.383 | 0.410 | 0.105 |

## Event-level and category-level results

The learned model predicts rain somewhere in every test tile at the 0.1 mm h⁻¹ threshold, producing
high POD but severe false alarms.

| category | samples | learned CSI | PySTEPS CSI | learned FAR | PySTEPS FAR | learned wet fraction | observed wet fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| active precipitation | 40 | 0.218 | 0.346 | 0.782 | 0.505 | 1.000 | 0.213 |
| dissipation | 18 | 0.054 | 0.205 | 0.946 | 0.706 | 1.000 | 0.054 |
| dry/low-activity | 38 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| initiation-centered | 56 | 0.219 | 0.371 | 0.781 | 0.439 | 1.000 | 0.215 |
| random weather-event | 24 | 0.195 | 0.290 | 0.805 | 0.476 | 1.000 | 0.187 |

Event-level metrics are stored in `artifacts/stage_3/test_event_metrics.csv`.

## Initiation evaluation

On initiation-centred test tiles, the learned model reports perfect detection within all lead windows
only because it predicts precipitation everywhere. That is not useful initiation skill.

| model | observed onset pixels | predicted onset pixels | missed | false | median onset error | onset MAE | false-initiation fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned ConvLSTM | 435,447 | 917,504 | 0 | 482,057 | -12 min | 30.1 min | 0.525 |
| PySTEPS | 435,447 | 335,253 | 142,349 | 42,155 | 0 min | 8.7 min | 0.046 |

Within-window detection:

| model | 30 min | 60 min | 90 min | 120 min |
| --- | ---: | ---: | ---: | ---: |
| learned ConvLSTM | 1.000 | 1.000 | 1.000 | 1.000 |
| PySTEPS | 0.708 | 0.662 | 0.632 | 0.614 |

The learned model has fewer misses only by accepting an unacceptable false-alarm load.

## Dissipation evaluation

On dissipation samples, the learned model usually fails to cease precipitation:

| model | observed cessation pixels | predicted cessation pixels | missed cessation | false cessation | median timing error | timing MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned ConvLSTM | 39,968 | 3,878 | 36,090 | 0 | 6 min | 11.2 min |
| PySTEPS | 39,968 | 36,949 | 3,024 | 5 | 0 min | 8.4 min |

This does not support the hypothesis that the radar-only ConvLSTM learned useful decay physics.

## Calibration

Reliability artifacts:

- `artifacts/stage_3/diagnostics/test_reliability.csv`
- `artifacts/stage_3/diagnostics/test_reliability.png`

Test expected calibration error for the 0.1 mm h⁻¹ occurrence head: 0.064.

The reliability curve is imperfect but the more serious problem is categorical: the expected-rate
field crosses 0.1 mm h⁻¹ almost everywhere. So even if probability bins are not catastrophically
miscalibrated, the downstream expected-rain output is not operationally useful.

## Qualitative audit

Panels are stored in `artifacts/stage_3/diagnostics/`:

- `test_learned_relative_best.png`
- `test_pysteps_clear_win.png`
- `test_both_fail_initiation.png`
- `test_learned_hallucination.png`
- `test_dissipation_audit.png`
- `test_reliability.png`

There was no honest “learned model clearly beats PySTEPS” case in this run. The
`test_learned_relative_best.png` panel is therefore a relative-best learned example, not a claimed
victory. The hallucination and dissipation panels are more representative of the failure mode.

## Leakage controls

Automated tests cover:

- no target timestamp appears in input
- input end index precedes target start index
- train/dev/test event IDs are disjoint
- frozen test event list is unchanged
- normalization statistics come from training samples only
- missing radar remains masked rather than becoming dry precipitation
- model and loss output shapes/masks are valid

The test suite does not yet prove that every linked storm fragment stays within its event split at
the tile-manifest level; that logic is inherited from the Milestone 1.5 whole-event split and should
be strengthened before larger model comparisons.

## Compute cost

This run used CPU only:

- PyTorch: CPU wheel
- CUDA available: false
- train/dev bounded run: 5 epochs, 20 train batches per epoch, 10 dev batches
- final test evaluation: 176 samples with persistence, optical flow, PySTEPS, and learned model
- parameter count: 24,120

The local dependency used for this pilot was installed explicitly to avoid a surprise CUDA-heavy
default lock resolution:

```powershell
uv pip install torch --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cpu
```

## Answers to the Stage 3 questions

1. Does the learned model beat PySTEPS overall?
   No. PySTEPS wins clearly on CSI, FAR, F1, and FSS at the tested operational thresholds.

2. At what lead time does any advantage appear?
   No meaningful advantage appears. At 120 minutes and 0.1 mm h⁻¹, learned CSI barely exceeds optical
   flow but remains far below PySTEPS and has FAR 0.837.

3. Does it beat PySTEPS specifically for precipitation initiation?
   No. It detects initiation by raining everywhere, with false-initiation fraction 0.525 versus
   PySTEPS 0.046.

4. Does it reduce dissipation errors?
   No. It misses most observed cessation pixels.

5. Is any gain caused merely by predicting more precipitation and accepting additional false alarms?
   Yes. This is the dominant failure mode.

6. Does the result justify adding atmospheric information?
   Yes. This result supports the hypothesis that radar history alone is insufficient for useful
   initiation and decay prediction in this benchmark.

## Decision

Recommendation: B. Radar-only saturates for this first modest model; Stage 4 radar plus atmospheric
state should begin after preserving this Stage 3 result as the radar-only baseline.

Do not increase model complexity solely to force a Stage 3 victory. The useful evidence from Stage 3
is negative: radar-only learned evolution, at this scale and data volume, does not yet beat the
classical extrapolation baselines and fails for the reasons the multimodal plan was designed to test.
