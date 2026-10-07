# Stage 10 — Causal HRRR representation and displacement diagnosis

Status: **DIAGNOSTIC COMPLETE — NO HRRR REPRESENTATION PROMOTED.** No fusion model was trained. Only the seven consumed Stage 8 development systems were used; the sealed six-system final set was not accessed or materialized.

## HRRR source audit

The NOAA HRRR archive index was checked for all 168 causal model cycles used by the consumed development rows. Each candidate was present in 168/168 cycles:

| Product | GRIB identifier | Units | Temporal meaning | f01 valid time and causal availability |
|---|---|---|---|---|
| Continuous APCP | `APCP:surface:0-1 hour acc fcst`, ecCodes `tp` | kg m⁻² (equivalent water depth in mm) | Accumulation over cycle +0 to +1 hour | Valid at cycle +1 h; treated available at cycle +1 h |
| Precipitation rate | `PRATE:surface:1 hour fcst`, `prate` | kg m⁻² s⁻¹ | Instantaneous forecast snapshot | Valid and available at cycle +1 h |
| Composite reflectivity | `REFC:entire atmosphere:1 hour fcst`, `refc` | dB | Instantaneous column-maximum reflectivity snapshot | Valid and available at cycle +1 h |
| 1000-m reflectivity | `REFD:1000 m above ground:1 hour fcst`, `refd` | dB | Instantaneous derived reflectivity snapshot | Valid and available at cycle +1 h |

The archive therefore contains richer continuous and radar-like fields, but not genuinely sub-hourly forecast guidance in the audited **standard surface-file sequence**. PRATE, REFC, and REFD in that sequence are hourly snapshots. They cannot resolve a 6-minute onset time without fabricating temporal information. Stage 10B separately audits the historical `wrfsubhf` product family below.

The existing future APCP fields retain their honest semantics: f02 is the cycle +1-to-+2-hour accumulation and f03 is +2-to-+3 hours. They are mapped to the 0–60 and 60–120-minute nowcast contexts; they are not described as six-minute forecasts.

## Fixed causal displacement procedure

For each issue time, f01 APCP was compared with radar accumulated over the exactly matching past interval `(cycle time, cycle +1 hour]`. Six-minute radar rates were multiplied by 0.1 hour and summed; no instantaneous frame was compared with an accumulation. Both fields were placed on the common 2-km tile grid and thresholded at 0.1 mm accumulated precipitation.

One deterministic FFT cross-correlation translation was estimated inside a fixed ±18-pixel (±36-km) window. The peak translation was scored by occurrence IoU; IoU ≥0.05 defined a meaningful match. The same issue-time vector was then applied unchanged to f02/f03. No future target influenced the vector, search window, threshold, or transformation.

## Current spatial mismatch

| System | Mean shift | Median shift | Mean match IoU | Meaningful matches |
|---|---:|---:|---:|---:|
| Sep 7/8 shared | 5.58 km | 0 km | 0.253 | 78.6% |
| Aug 11 | 5.54 km | 0 km | 0.153 | 47.1% |
| Aug 29 | 9.64 km | 4 km | 0.237 | 61.1% |
| Jun 21 | 8.09 km | 2 km | 0.356 | 94.4% |
| May 21 | 5.58 km | 0 km | 0.249 | 47.6% |
| Nov 21 | 12.31 km | 0 km | 0.047 | 28.6% |
| Sep 22 | 7.91 km | 0 km | 0.462 | 89.5% |

Most systems have a zero median shift even when their mean is nonzero, and match availability ranges from 28.6% to 94.4%. Current HRRR/radar correspondence is therefore strongly event-dependent and frequently insufficient to identify a reliable translation.

## Does current displacement persist?

Event-macro positive-row results are:

| Representation | Brier | AUC | CSI | POD | FAR | F1 | FSS 6 km | FSS 18 km | FSS 36 km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw APCP | 0.1921 | 0.6184 | 0.1520 | 0.2771 | 0.6184 | 0.2191 | 0.2569 | 0.2980 | 0.3384 |
| Causally translated APCP | 0.1909 | 0.6103 | 0.1444 | 0.2629 | 0.6148 | 0.2100 | 0.2477 | 0.2898 | 0.3320 |

Translation yields only a 0.0012 Brier improvement while degrading AUC, CSI, POD, F1, and all FSS radii. Brier improves in four of seven systems, but F1 improves only for the Sep 7/8 shared system. Nov 21 loses 0.0405 F1. The issue-time displacement does not persist consistently enough to support deterministic correction of future APCP.

## Lead-group and temporal diagnosis

| Lead group | Raw POD | Raw FAR | Raw F1 | Shifted POD | Shifted FAR | Shifted F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0–30 min | 0.192 | 0.664 | 0.165 | 0.183 | 0.659 | 0.161 |
| 30–60 min | 0.158 | 0.677 | 0.136 | 0.154 | 0.670 | 0.133 |
| 60–90 min | 0.209 | 0.575 | 0.205 | 0.197 | 0.572 | 0.197 |
| 90–120 min | 0.182 | 0.595 | 0.182 | 0.175 | 0.589 | 0.176 |

The first hourly field produces the largest false-alarm ratios, while the second hour has somewhat better discrimination but still weak recall. Translation does not repair the timing pattern. No product audited in the original standard surface-file pass provides genuinely higher-frequency future information, so a finer onset time cannot be inferred honestly from those fields. The later `wrfsubhf` amendment evaluates native 15-minute fields separately and does not relabel them as 6-minute forecasts.

## Continuous versus binary APCP

Raw continuous magnitude has modest ranking information: event-macro occurrence AUC is 0.618 and Spearman correlation is 0.201. Its AUC is nearly unchanged between rows where Stage 9 says HRRR is helpful (0.603) and harmful (0.599), although correlation is higher in helpful rows (0.228 versus 0.157). Continuous magnitude therefore retains some spatial ranking information lost by binarization, but does not cleanly identify when HRRR should be trusted.

Binarization was an information loss, but not the dominant failure. Continuous APCP remains hourly, displaced, broad, and regime-dependent.

## Spatial-tolerance upper bound

Raw FSS rises from 0.257 at 6 km to 0.298 at 18 km and 0.338 at 36 km. This improvement with neighborhood size shows that some forecasts contain the right broad precipitation region at the wrong location. However, even 36-km FSS remains modest, and causal translation reduces FSS at every radius. Spatial tolerance therefore provides a limited oracle-style upper bound, not evidence that one persistent translation can recover the missing skill.

A future-truth-optimized shift was deliberately not used as a forecast transformation. Neighborhood verification supplies the post-hoc spatial-tolerance diagnosis without contaminating the causal shift.

## PySTEPS failure categories

| Dominant PySTEPS category | Rows | Shift Brier improvement | Shift F1 change | Shift FSS18 change |
|---|---:|---:|---:|---:|
| No initiation signal | 47 | -0.00101 | -0.01131 | -0.01236 |
| Growth underprediction | 168 | +0.00017 | -0.00516 | -0.00465 |
| Displacement | 25 | +0.00590 | -0.00178 | -0.00057 |
| Decay persistence | 9 | +0.00001 | +0.00134 | +0.00231 |
| Other/adequate | 11 | +0.00085 | +0.00261 | +0.00332 |

Even for displacement-classified PySTEPS failures, the HRRR translation improves Brier but not categorical or neighborhood skill. It worsens the critical no-initiation category. Improved spatial representation does not materially increase the fraction of PySTEPS failures that HRRR corrects.

## Hard negatives

Only one of 12 dry rows has a meaningful current-time match; mean shift magnitude is 2.74 km. Translation reduces deterministic wet/dry error from 0.00166 to 0.00133 without changing F1 from zero, and does not expand 6/18/36-km false-area summaries. This transformation does not simply spread precipitation, but its positive-case benefit is absent.

## Representation decisions

| Candidate | Classification | Decision |
|---|---|---|
| Raw continuous hourly APCP | **WEAK / REGIME-DEPENDENT** | Retains modest ranking information but does not resolve timing or displacement. |
| Issue-time displacement-corrected APCP | **NOT USEFUL** | Tiny Brier gain with worse F1, AUC, and FSS; effects are inconsistent by event. |
| Historical `wrfsubhf` 15-minute APCP, PRATE, and REFC | **AVAILABLE; NOT USEFUL** | Complete for 168/168 causal development cycles, but every candidate lost event-macro F1 to the matched hourly APCP comparator in all seven systems. |
| Hourly REFC/REFD | **WEAK / REGIME-DEPENDENT as a source candidate** | Complete and causal, but not temporally richer; skill requires a separately frozen radar-like mapping and was not claimed here. |

## Stage 10B amendment — HRRR `wrfsubhf` source and representation audit

Stage 10B is a narrow amendment to the source claim, not a reopening of Stage 10. The completed conclusions for continuous hourly APCP, causal displacement correction, hourly PRATE/REFC/REFD, and PySTEPS as the principal operational reference remain frozen. No fusion model was trained, and only the seven already-consumed Stage 8 development systems were used. The sealed six-system final set was neither inspected nor materialized.

### Historical coverage, provenance, and causality

The historical NOAA HRRR `wrfsubhf` archive was queried for the same 168 causal HRRR cycles used by Stage 8. Both required forecast-hour files (`wrfsubhf01` and `wrfsubhf02`) and their `.idx` objects were present for every cycle: **168/168 complete cycles and 336/336 files**, spanning the required 2021 and 2022 development cases. The audit records source URLs, index records, file identifiers, and index hashes; the selected-message manifest additionally records byte ranges and SHA-256 hashes for every downloaded GRIB message.

The original Stage 8 cycle assignment and conservative simulated-availability rule were retained. Every field came from its row's already-approved causal cycle; no later cycle was substituted. Future-valid fields from that available cycle were permitted. Native offsets are +15, +30, +45, +60, +75, +90, +105, and +120 minutes. They remain 15-minute HRRR context and are never described or interpolated as six-minute NWP.

### Exact GRIB content and temporal semantics

Parsing the historical indexes and decoded messages established:

| Candidate | GRIB identity | Units | Native temporal meaning | Available offsets |
|---|---|---|---|---|
| Subhourly APCP | `APCP:surface`, ecCodes `tp` | kg m⁻² | Successive 15-minute interval accumulations: 0–15, 15–30, 30–45, 45–60 minutes in f01 and 60–75, 75–90, 90–105, 105–120 minutes in f02 | +15 through +120 every 15 min |
| Subhourly PRATE | `PRATE:surface`, ecCodes `prate` | kg m⁻² s⁻¹ | Instantaneous precipitation-rate snapshots | +15 through +120 every 15 min |
| Subhourly REFC | `REFC:entire atmosphere`, ecCodes `refc` | dB | Instantaneous column-maximum reflectivity snapshots | +15 through +120 every 15 min |
| Subhourly REFD | `REFD:1000 m above ground`, ecCodes `refd` | dB | Instantaneous 1000-m derived reflectivity snapshots | +15 through +120 every 15 min |

The compact scored set was frozen as APCP, PRATE, and REFC; REFD's existence and semantics were verified but it was not added as another representation search. A representative multi-system/multi-year decode gate preceded full materialization. All **4,032** selected APCP/PRATE/REFC messages were obtained with hashes, decoded fields were finite, and values changed between quarter-hour valid times. Orientation, projection, remapping, and plausible decoded ranges passed the same gate.

APCP was converted from each 15-minute accumulation to its equivalent interval-mean rate and compared with MRMS accumulated over the matching interval. PRATE was converted from kg m⁻² s⁻¹ to mm h⁻¹ and compared with instantaneous native-time MRMS rate. REFC used one physically motivated conversion frozen before scoring, Marshall–Palmer `Z = 200 R^1.6`, and was likewise compared at native times. No time offset was selected using truth.

### Native-time event-macro verification

The positive-system event-macro results at the four reporting windows were:

| Representation | Lead | CSI | POD | FAR | F1 | FSS 18 km | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15-min APCP | 0–30 | 0.075 | 0.178 | 0.838 | 0.115 | 0.192 | 0.547 | 0.140 |
| 15-min APCP | 30–60 | 0.067 | 0.140 | 0.833 | 0.101 | 0.164 | 0.534 | 0.143 |
| 15-min APCP | 60–90 | 0.077 | 0.137 | 0.809 | 0.113 | 0.191 | 0.531 | 0.167 |
| 15-min APCP | 90–120 | 0.085 | 0.142 | 0.782 | 0.122 | 0.204 | 0.529 | 0.206 |
| 15-min PRATE | 0–30 | 0.060 | 0.143 | 0.858 | 0.093 | 0.173 | 0.528 | 0.121 |
| 15-min PRATE | 30–60 | 0.061 | 0.115 | 0.819 | 0.092 | 0.163 | 0.524 | 0.121 |
| 15-min PRATE | 60–90 | 0.068 | 0.117 | 0.827 | 0.101 | 0.187 | 0.523 | 0.150 |
| 15-min PRATE | 90–120 | 0.075 | 0.122 | 0.792 | 0.109 | 0.197 | 0.523 | 0.182 |
| 15-min REFC | 0–30 | 0.073 | 0.210 | 0.871 | 0.113 | 0.208 | 0.560 | 0.155 |
| 15-min REFC | 30–60 | 0.079 | 0.185 | 0.840 | 0.119 | 0.208 | 0.554 | 0.157 |
| 15-min REFC | 60–90 | 0.087 | 0.185 | 0.827 | 0.130 | 0.228 | 0.558 | 0.184 |
| 15-min REFC | 90–120 | 0.094 | 0.189 | 0.807 | 0.140 | 0.250 | 0.557 | 0.221 |

FSS at 6/18/36 km, MAE, rank correlation, and all native-offset results are retained in the machine-readable outputs. REFC is the strongest of the three categorical candidates and is more radar-compatible in the limited sense of higher F1, FSS, AUC, and onset detection than PRATE. Its FAR remains 0.81–0.87, however, and its positive-case Brier score is worse than the alternatives.

Onset results tell the same story. APCP, PRATE, and REFC respectively detected 26.3%, 24.5%, and 35.0% of eligible onset pixels by 120 minutes. Their event-macro onset MAE values were 50.9, 51.0, and 46.9 minutes, with median biases of +11.6, +18.4, and +12.0 minutes. Detection by 30/60/90/120 minutes was 15.8/18.0/21.7/26.3% for APCP, 13.2/15.7/19.7/24.5% for PRATE, and 18.4/22.3/27.9/35.0% for REFC. The additional cadence therefore did not solve short-lead timing.

### Matched hourly comparison and event consistency

The comparison uses hourly APCP evaluated against the **same native-compatible truth definition** as each subhourly candidate. This avoids comparing interval APCP with instantaneous rain rate or reusing the original 30-minute aggregated Stage 10 target. Averaged over the seven independent positive systems:

| Candidate | Candidate F1 | Matched hourly APCP F1 | Change | Systems improved | Brier improvement |
|---|---:|---:|---:|---:|---:|
| 15-min APCP | 0.1125 | 0.1953 | -0.0827 | 0/7 | +0.0135 |
| 15-min PRATE | 0.0985 | 0.1787 | -0.0801 | 0/7 | +0.0312 |
| 15-min REFC | 0.1252 | 0.1787 | -0.0535 | 0/7 | -0.0045 |

No candidate improves F1 in even one independent system. APCP and PRATE produce lower Brier scores largely by forecasting less wet area, but that apparent calibration gain accompanies much lower detection and categorical skill. Subhourly QPF does not improve timing over hourly APCP; PRATE does not improve localization; and REFC's more radar-like representation is still materially weaker than the matched hourly comparator. None materially improves either 0–30 or 30–60-minute skill.

### PySTEPS failure categories and hard negatives

Against the matched hourly APCP comparator, F1 changes remain negative in every Stage 9 descriptive category. For the two primary categories:

| PySTEPS failure category | 15-min APCP | 15-min PRATE | 15-min REFC |
|---|---:|---:|---:|
| No initiation signal | -0.0610 | -0.0597 | -0.0476 |
| Growth underprediction | -0.0913 | -0.0922 | -0.0518 |

The same conclusion holds for displacement, decay persistence, and other/adequate rows. Richer cadence does not increase correction of the critical initiation or growth failures.

On the 12 consumed development hard negatives, mean false wet area was 0.00277 for APCP, 0.00123 for PRATE, and 0.00445 for REFC. Corresponding dry-pixel false-initiation fractions were 0.0124, 0.0079, and 0.0222; Brier scores were 0.00322, 0.00164, and 0.00485. The candidates do not cause unacceptable broad dry false alarms, but that restraint does not compensate for their weak positive-event detection.

### Stage 10B decision

Historical `wrfsubhf` is **AVAILABLE**, with complete causal coverage and genuine native 15-minute APCP, PRATE, REFC, and REFD fields. The three evaluated compact representations are nevertheless classified **NOT USEFUL** for this controlled development set: all lose materially to matched hourly APCP, all lose in all seven systems, and none repairs PySTEPS initiation/growth failures. A PySTEPS-centered fusion experiment is therefore **not justified** by Stage 10B. This is an empirical representation result, not a claim that the archive lacks subhourly guidance.

## Final interpretation

No Stage 10 representation currently merits a PySTEPS-centered fusion experiment. Continuous APCP is somewhat more informative than binary occurrence, but its advantage is too weak and event-dependent. The fixed causal displacement correction fails the event-consistency requirement and degrades the metrics most relevant to operational precipitation placement.

Future work should require genuinely new representation quality—such as demonstrably better ensemble or convection-aware NWP, or a separately predeclared displacement-aware radar-like procedure—before another fusion attempt. Merely increasing the HRRR cadence from hourly to 15 minutes did not supply that quality here. PySTEPS remains the primary operational reference. The sealed final set remains untouched.

Original machine-readable results and hashes are under `artifacts/stage_10/`. The Stage 10B archive audit, selected-message provenance, decode gate, native verification, matched comparator, event-consistency tables, and hashes are under `artifacts/stage_10b/`.
