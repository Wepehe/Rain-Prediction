# Stage 7 HRRR timing, displacement, and regime diagnosis

Status: **DIAGNOSTIC COMPLETE — NO NEW MODEL OR GATE TRAINED.** Stage 6 remains
**COMPLETE — NEGATIVE**. The consumed 2023 set was used only for post-mortem
analysis; no weight, threshold, time offset, dilation, calibration, rule, or
architecture was selected from it.

## Scope and definitions

The analysis covers all 56 positive 2023 rows, the 28-row frozen
`RADAR_LIMITED_INITIATION` cohort, the 19-row strict V2 cohort, eight hard
negatives and four heavy-rain rows. Row-level outputs retain event, issue time,
object, frozen memberships, forecast states, onset timing and spatial
diagnostics.

For descriptive object summaries, observed onset is the median first 0.1 mm/h
crossing among the eventual wet footprint. Spatial distances use the 2 km grid.
The relevant HRRR field is f02 for onset through 60 minutes and f03 thereafter.
The 6/18/36 km neighborhoods and 30-minute early/late label are diagnostic
descriptions only, not candidate gate parameters.

## 1. Why DEV favored the hybrid

Stage 5 DEV had only three positive events and strongly favored cases with
early, colocated HRRR support. All 12 DEV initiation rows had some direct HRRR
overlap; HRRR covered 34.9% of the eventual footprint on average and median
object onset was 22 minutes. In contrast, only 76.8% of the 56 2023 rows had
any direct overlap, mean footprint coverage was 8.9%, and median object onset
averaged 66.5 minutes.

The DEV hybrid gain therefore arose in a small sample whose HRRR support was
both earlier and roughly four times denser. The DEV result was real for those
events but not representative of the independent 2023 regimes. This is a
sample/event-diversity explanation, not evidence that a different 2023-tuned
weight should be chosen.

## 2. Why 2023 rejected it

The fixed blend reduced recall enough to outweigh its improvements in
precision, false initiation, onset bias and Brier. Event F1 changed by -0.045
on August 3, -0.097 on July 12 and +0.094 on July 20. Only one event improved;
the broader radar-limited subset also declined, and frozen hard-negative limits
failed. These remain the governing Stage 6 results.

The row-level diagnostic labels contain 23 cases where HRRR added a previously
missed initiation, but also 19 where HRRR hurt A+, ten false additions, three
joint misses and only one other net-help case. A fixed global blend cannot
distinguish these situations.

## 3. Temporal mismatch

| True onset group | Rows | Correct HRRR interval/location | Hybrid >30 min early | Hybrid >30 min late |
|---|---:|---:|---:|---:|
| 0–30 min | 4 | 1.000 | 0.000 | 0.000 |
| 30–60 min | 17 | 0.941 | 0.412 | 0.000 |
| 60–90 min | 29 | 0.724 | 0.448 | 0.000 |
| 90–120 min | 6 | 0.333 | 0.167 | 0.000 |

Temporal coarseness is material. Correct interval/location support falls sharply
with lead, reaching one third for 90–120-minute onset. The hourly binary field
also makes the hybrid substantially early in roughly 41–45% of the two central
lead groups. No offset was shifted or optimized. Timing is not the sole failure:
many rows in the nominally correct interval still have weak spatial coverage.

## 4. Spatial displacement and coverage

HRRR directly overlapped the eventual footprint in 43 rows, was a near miss in
two, and completely missed 11. Direct-overlap rows gained 0.050 F1 on average;
complete misses lost 0.097. However, direct overlap was usually sparse: mean
overlap was only 11.6%, while mean HRRR/observed centroid separation among
overlap rows was 78.5 km. This combination indicates broad or displaced HRRR
areas clipping part of a footprint rather than precise localization.

The spatial diagnosis is therefore **mixed displacement and incomplete
coverage**, not a clean case for choosing an 18- or 36-km dilation. Those radii
remain descriptive hypotheses requiring new data.

## 5. Strict V2 post-mortem

Fifteen of 19 V2 rows had direct HRRR overlap; four had no useful HRRR signal.
Twelve rows with positive hybrid-minus-A+ F1 had, on average, earlier onset
(77.5 versus 91.7 minutes), lower A+ confidence (0.133 versus 0.149), much
higher HRRR footprint coverage (11.1% versus 0.7%), and larger observed
footprints (23,432 versus 12,049 km²).

The strict V2 gain came from a subset where radar confidence was extremely low
and coarse HRRR captured at least part of a relatively large future system.
Even there, event-macro recall remained only 0.059, so V2 supports the presence
of useful signal but not operational sufficiency.

## 6. RADAR_LIMITED versus V2

| Cohort | Rows | A+ probability | PySTEPS coverage | Final radar wet | HRRR coverage | Onset | Mean row F1 change |
|---|---:|---:|---:|---:|---:|---:|---:|
| RADAR_LIMITED, not V2 | 9 | 0.226 | 0.314 | 0.0296 | 0.0645 | 66.7 min | -0.125 |
| Strict V2 | 19 | 0.139 | 0.062 | 0.0095 | 0.0726 | 82.7 min | +0.081 |

The non-V2 radar-limited rows still contain appreciable radar/PySTEPS signal.
Giving HRRR a fixed 0.50 weight suppresses useful radar confidence there. V2
removes much more of that radar shortcut, so partial HRRR support can help.
This is a post-hoc hypothesis and must not become a V2-only rule without new
development and final sets.

## 7. Event dependence

| Event | HRRR direct-overlap rows | HRRR footprint coverage | Mean onset | Mean area | Mean row F1 change |
|---|---:|---:|---:|---:|---:|
| July 12 | 0.364 | 0.012 | 70.9 min | 21,801 km² | -0.097 |
| July 20 | 0.933 | 0.134 | 64.3 min | 26,054 km² | +0.094 |
| August 3 | 0.733 | 0.054 | 67.6 min | 13,142 km² | -0.045 |

July 20 differs mainly in HRRR spatial support: almost every row has overlap,
coverage is 2.5 times August 3 and 11 times July 12, and observed systems are
larger. Three events are insufficient to call this a causal regime; it is a
hypothesis about organization, scale and HRRR placement.

## 8. Hard-negative failures

HRRR triggered in five of eight dry rows. All five also had some A+ threshold
crossing, and in every case HRRR created additional hybrid crossings outside
the A+ wet area. Three rows were A+-only. Four hybrid false areas were
widespread (at least 1% of the tile) and four isolated. The worst rows had HRRR
wet fractions of 1.7–5.9% and simultaneous elevated A+ areas.

The dominant hard-negative mechanism is therefore **both sources active plus
fixed-blend expansion**, not an HRRR-only scalar trigger. The blend lowers mean
probability and Brier but cannot prevent spatially extensive HRRR false areas
from crossing 0.30.

## 9. Heavy-rain behavior

At 0.1 mm/h the hybrid preserved 97.8–98.6% POD while reducing A+ false alarms,
raising F1 at every 30–120-minute endpoint. Its mean-rate forecasts remained
too low, but rate MAE improved because the HRRR component suppressed A+'s broad
background rain. This is principally an occurrence/spatial-coverage correction,
not successful intensity reconstruction.

PySTEPS remained stronger because it preserves and advects the observed rain
structure: its FSS and precision were substantially higher, especially at
30–60 minutes. The A+/HRRR average smooths/suppresses false background but does
not retain motion-consistent spatial detail as well.

## 10. Dominant failure classification

The defensible classification is **MULTIPLE / UNRESOLVED**, with measured
support for:

- **REGIME DEPENDENCE / INSUFFICIENT EVENT DIVERSITY:** DEV and July 20 had much
  stronger HRRR support; only three events existed in each comparison.
- **OVER-TRUSTING HRRR WHEN RADAR IS ALREADY INFORMATIVE:** non-V2 radar-limited
  rows lost 0.125 mean row F1 while retaining much higher A+/PySTEPS signal.
- **HRRR FALSE POSITIVES:** five of eight dry rows had HRRR-triggered expansion.
- **TEMPORAL COARSENESS:** interval correctness deteriorated with lead and many
  hybrid onsets were more than 30 minutes early.
- **SPATIAL DISPLACEMENT/INCOMPLETE COVERAGE:** 13 rows were misses/near misses,
  and overlap was sparse even when nonzero.
- **A+ CALIBRATION INTERACTION:** a fixed average suppresses useful A+
  probabilities when HRRR is dry and promotes broad HRRR areas when it is wet.

No single factor explains all failures.

## 11. Next hypothesis and new event sets

A scientifically justified future hypothesis is:

> HRRR should modify A+ only when radar confidence is low and HRRR support is
> spatially and temporally plausible.

Potential inputs are A+ confidence, A+/HRRR disagreement, radar wet fraction,
PySTEPS coverage, HRRR occurrence, HRRR-to-target-proxy distance and lead. This
is a hypothesis only; no gate definition, radius, threshold or architecture is
frozen or trained in Stage 7.

Eight never-used candidate periods were declared for new gate TRAIN/DEV and six
separate candidate periods for a future final holdout. The date-window manifests
were frozen before any future gate evaluation. They span 2018–2022 severe,
frontal, organized, isolated and prolonged-rain candidates. Sources include
official ECCC annual weather summaries and statements. Radar availability and
radar-only object mining must still confirm valid events; no failed candidate
may be silently replaced after seeing gate results. If substantially more than
three independent positives do not survive, more periods must be predeclared
before development.

The future final set cannot be scored until object membership, paired hard
negatives, tensors and hashes are frozen. Its current period-manifest SHA-256 is
`d28d9f689be6ae70b44d41dbeeba55ca5c85373558ddfb02ba6cd4741db99d4b`.

## 12. Is a simple gate justified?

Yes as a **future low-degree-of-freedom research hypothesis**, because HRRR has
clear conditional value and the fixed blend fails in identifiable situations.
No gate is justified from the 2023 rows themselves. Development must use only
the new TRAIN/DEV events, compare A+, PySTEPS, raw HRRR and frozen Stage 6, and
be evaluated once on the separately frozen new final set.

## Artifacts

The diagnostic tables are under `artifacts/stage_7/`, including the complete
row error table, temporal and spatial summaries, V2 post-mortem,
RADAR_LIMITED/V2 contrast, event comparison, hard-negative analysis, DEV/2023
comparison, and frozen candidate-period manifests. All are descriptive; no
forecasting parameters were fitted.
