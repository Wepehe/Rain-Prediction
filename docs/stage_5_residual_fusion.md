# Stage 5 frozen-baseline residual fusion

Status: **DEV FAIL — NOT PROMOTED; PROTECTED 2023 HOLDOUT REMAINS SEALED.**

H1 learned a real, mostly HRRR-dependent correction, but did not improve frozen
A+ and lost to a transparent HRRR/A+ hybrid. Stage 5 therefore stops without
scoring the protected holdout or building a larger multimodal network. Stages
4B/C/D remain frozen historical experiments.

## Frozen experiment and gates

H1 uses detached A+ logits, conditional intensity and expected rate; honest
HRRR APCP contexts for 0–60 and 60–120 minutes; and GOES C13 plus cooling
tendencies. Its bounded residual output was initialized to zero. A+ is absent
from the optimizer and cannot receive gradients.

All 72 development rows materialized (48 TRAIN, 24 DEV/repair), with no
protected split. A+ remained bit-identical at SHA-256
`63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70`.
TRAIN-only APCP `log1p` normalization has mean 0.109482 and standard deviation
0.308419.

The 90,981-parameter correction passed the tiny gate: loss fell from 1.23800 to
0.34091, every required gradient was nonzero, destruction changed predictions,
and residuals stayed finite and effectively unsaturated.

## Non-radar probe

The 74,953-parameter HRRR+GOES probe has no radar or A+ input. It stopped after
17 epochs with best DEV loss 0.94344. At threshold 0.20, event-macro initiation
F1 was 0.76895. HRRR shuffle reduced F1 to 0.72763 and all-input shuffle to
0.70761; GOES shuffle changed F1 only to 0.76698. The available non-radar fields
therefore contain learnable DEV signal, primarily HRRR, though DEV has only
three independent positive events.

## H1 convergence and corrections

H1 trained once and stopped after 10 epochs. The best checkpoint was epoch 4
(DEV loss 1.27411; SHA-256
`a11a8f31e901cada82f974dcbfbfe6bc905fd3f7a920f8a284921c6399612dcd`).
Later epochs increased correction magnitude while worsening DEV loss.

Mean absolute delta logits were about 0.29–0.31 for initiation, 0.29–0.31 for
active rain, and 0.23–0.25 for hard negatives over 30–120 minutes. Initiation
corrections became more positive with lead; only about 23–26% of hard-negative
corrections were positive.

## DEV initiation results

These are weather-event macro averages over three positive DEV events. H1 uses
its eligible DEV threshold 0.55; A+ retains its frozen 0.35 policy.

| Model | Threshold | Precision | Recall | F1 | False init. | Onset MAE | Bias | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A+ frozen | 0.35 | 0.708 | 0.810 | 0.746 | 0.292 | 25.82 | -11.5 | 0.209 |
| B2 reference | 0.20 | 0.640 | 0.916 | 0.748 | 0.360 | 27.53 | -12.5 | 0.208 |
| H1 | 0.55 | 0.768 | 0.705 | 0.715 | 0.232 | 21.93 | -7.0 | 0.197 |
| Raw HRRR | 0.50 | 0.858 | 0.561 | 0.672 | 0.142 | 25.11 | 0.0 | 0.308 |
| PySTEPS | 0.50 | 0.685 | 0.435 | 0.519 | 0.315 | 33.29 | +6.0 | 0.487 |
| A+/HRRR hybrid (0.50) | 0.30 | 0.778 | 0.775 | 0.763 | 0.222 | 23.43 | -7.0 | 0.194 |

H1 improves precision, onset error, bias and Brier relative to A+, but loses
recall and has F1 lower by 0.03136. The fixed hybrid exceeds H1 by 0.04756.
Imposing H1's threshold on A+ would be an invalid favorable comparison.

## Dependence, hard negatives and capacity control

At H1 threshold 0.55, F1 losses were 0.01138 for HRRR zeroing, 0.01066 for HRRR
shuffle, 0.00091 for GOES zeroing, 0.00053 for GOES shuffle, 0.01266 for all-
input zeroing and 0.01139 for all-input shuffle. H1 genuinely uses HRRR on DEV;
meaningful GOES dependence is not detected.

On hard negatives H1 has mean probability 0.0873, maximum 0.6313, wet-area
fraction 0.00137, false-initiation fraction 0.375 and Brier 0.01260. It passes
both constraints. The hybrid also passes (wet area 0.00234, false-initiation
0.375) with Brier 0.00336.

The identical-capacity A+-only correction control (90,981 parameters, all
non-radar inputs forced to zero) stopped after eight epochs, had worse best DEV
loss 1.30267, and best unconstrained event-macro F1 0.75881 at threshold 0.05.
Extra residual capacity alone does not explain H1's HRRR sensitivity, but this
control supplies no reason to promote H1.

## Frozen decision

| Promotion gate | Result |
|---|---|
| H1 F1 at least A+ + 0.01 | **FAIL** |
| All-non-radar zero degradation at least 0.005 | PASS |
| All-non-radar shuffle degradation at least 0.005 | PASS |
| Individual modality degradation at least 0.005 | PASS (HRRR) |
| H1 at least simple hybrid + 0.005 | **FAIL** |
| Hard-negative constraints | PASS |

H1 is **not promoted**. This is not an information null: the probe and H1 both
respond materially to HRRR. Rather, coarse HRRR is useful on DEV but the learned
residual does not exploit it better than a transparent blend and sacrifices too
much A+ recall. GOES remains unused. More independent events and investigation
of hourly-HRRR timing/resolution and label learnability are warranted; a larger
residual network is not.

Complete threshold/event tables, detection-window metrics, broad CSI/POD/FAR/
F1/FSS/Brier metrics, hard-negative rows, and residual diagnostics are in
`artifacts/stage_5/h1/evaluation/`. No 2023 prediction was generated.
