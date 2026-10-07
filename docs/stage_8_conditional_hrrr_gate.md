# Stage 8 — Conditional HRRR gating

Status: **CLOSED — NEGATIVE — NOT PROMOTED.** The sealed six-system final set was not scored. The frozen features, λ, α, architecture, and threshold rules were not revised.

## Qualification result

**QUALIFICATION COMPLETE — ADEQUATE. GATE PREDECLARATION IS THE ONLY AUTHORIZED NEXT STEP.**

This pass performed source materialization and event qualification only. It did not define gate features, architecture, formula, thresholds, blend strength, spatial radii, cross-validation policy, comparators, or promotion criteria. It did not train or evaluate a gate. No A+, PySTEPS, HRRR-skill, blend, or gate predictions were generated on the final set, and the consumed 2023 evaluation set was not inspected.

The rules were frozen before object counts were inspected in `configs/data/stage_8_qualification.yaml` (SHA-256 `829a7ed666f937ac4ebe14400bdf4d2c697d4210d537796802902c5afb6970c3`): at most 10% missing expected radar scans; at least 90% finite support in retained 60-minute history plus 120-minute target tiles; and at least three distinct clean-initiation issue times for positive-period credit.

## Candidate outcomes

| Split | Candidate | Available / expected | Missing | Finite | Clean rows | Unique issues | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| DEV | `stage7_dev_jun21_2021_frontal_convection` | 160 / 161 | 0.62% | 99.38% | 18 | 15 | qualify |
| DEV | `stage7_dev_jul15_2021_barrie_outbreak` | 141 / 161 | 12.42% | 87.58% | 0 retained | 0 | **reject: radar missingness >10%** |
| DEV | `stage7_dev_sep22_2021_rain` | 418 / 421 | 0.71% | 99.29% | 19 | 16 | qualify |
| DEV | `stage7_dev_may21_2022_derecho` | 161 / 161 | 0% | 100% | 42 | 29 | qualify |
| DEV | `stage7_dev_aug29_2022_wet_windy` | 420 / 421 | 0.24% | 99.76% | 95 | 78 | qualify |
| DEV | `stage8_add_dev_aug11_2021` | 160 / 161 | 0.62% | 99.38% | 51 | 39 | qualify |
| DEV | `stage8_add_dev_sep07_2021` | 161 / 161 | 0% | 100% | 12 | 12 | qualify |
| DEV | `stage8_add_dev_sep08_2021` | 160 / 161 | 0.62% | 99.38% | 16 | 16 | qualify; shared-system credit |
| DEV | `stage8_add_dev_nov21_2021` | 161 / 161 | 0% | 100.00% | 7 | 7 | qualify |
| FINAL | `stage7_final_jun16_2022_cold_front` | 161 / 161 | 0% | 100% | 28 | 23 | qualify |
| FINAL | `stage8_add_final_may15_2022` | 161 / 161 | 0% | 100% | 13 | 12 | qualify |
| FINAL | `stage8_add_final_jul19_2022` | 161 / 161 | 0% | 100% | 27 | 24 | qualify |
| FINAL | `stage8_add_final_jul24_2022` | 160 / 161 | 0.62% | 99.38% | 36 | 33 | qualify |
| FINAL | `stage8_add_final_jul28_2022` | 161 / 161 | 0% | 100% | 17 | 16 | qualify |
| FINAL | `stage8_add_final_aug20_2022` | 161 / 161 | 0% | 100% | 21 | 16 | qualify |

All tensors span their exact predeclared timestamps on the native 500×930 cropped MRMS grid, preserve `mm h-1`, retain missing scans as NaNs, and record source hashes. Full audit details are in `artifacts/stage_8/qualification/stage8_candidate_qualification.csv`.

## Radar-only object mining

Objects were mined using radar, timing, and geometry only. Clean membership required apparent initiation, no advective-entry flag, no domain-boundary contact, future maximum at least 1 mm/h, complete history/target support, and the frozen validity threshold. Forecast performance was not available to the miner.

For each retained row, the frozen manifest records the exact object identity, system group, issue time, 128×128 tile, 10 history timestamps, 20 target timestamps, validity fraction, source tensor hash, PySTEPS eligibility, and A+ eligibility. A separate compressed bundle preserves the exact history and target validity masks.

| Split | Qualified periods | Independent systems | Clean object rows | Unique issue times |
|---|---:|---:|---:|---:|
| Development | 8 | **7** | 260 | 212 |
| Final | 6 | **6** | 142 | 124 |

The September 7 and September 8 development periods receive one combined independent-system credit. Adjacent dates were not inflated into two systems.

## HRRR causal qualification

The retained object and hard-negative issue times require 708 HRRR records across 354 distinct issue times and 336 unique APCP products. For every issue time, the materializer selected the latest cycle satisfying the frozen 60-minute simulated availability lag and materialized f02 and f03 hourly APCP.

- All 708 records were available by the simulated nowcast issue time.
- No later cycle was used to rescue a row.
- All products decoded as finite APCP fields.
- Every local product has a SHA-256 checksum.
- Cycle issue, simulated availability, forecast valid time, forecast hour, product identity, units, and accumulation step are recorded.

## Hard negatives

Hard-negative membership was selected before gate design using radar dryness only. The frozen criteria require at least 90% valid pixels, mean wet fraction ≤0.001, maximum-frame wet fraction ≤0.005, zero heavy-rain fraction, and 120-minute separation.

| Split | Rows | Independent systems represented |
|---|---:|---:|
| Development | 12 | 5 |
| Final | 8 | 4 |

Issue-safe HRRR support is retained for later evaluation but was not used to accept or reject these rows.

## Disjointness and sealing

The automated audit found no shared event ID, weather-system group, issue time, object ID, or overlapping configured Stage 3–6 storm window between development and final data. The audit covered the Milestone 1/1.5, Stage 3.1, Stage 4 repair, and consumed 2023 Stage 4C configurations. `stage8_prior_stage_overlap_audit.csv` contains zero rows.

The final manifest is sealed. Only materialization and integrity metadata have been inspected; no forecast-model output exists for it.

## Frozen hashes

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Development object manifest | 260 | `8744a4a925ca93bd8261c1c660f7a9dfa52724d22e591664dcf8a6f51dfb0280` |
| Final object manifest | 142 | `52e0f3df309862584e3391da9ae0b40caba58583f28df68e45f4cd940445359f` |
| Independent-system groups | 15 periods | `61d4f25a379b357f94d8aea35d46733f692e53b1358fd6cb58c7f229f7958205` |
| Hard-negative manifest | 20 | `220b14770d4be2eea5cebcefc8b4c3b33348837feceb49f4c600e82b4f111512` |
| Rejected-candidate table | 1 | `3c91fec0c91479093a9cca922dd2cfbc3410b2f01b0454f055d50dff21b735d2` |
| HRRR causal records | 708 | `70d5c181f4c5ede9c0d04edb9ece2710857a8d600908da616b57f69be0a43854` |
| Disjointness audit | — | `6e98c494da2262c6bcd44305e4276e451e50e94d4f18ec292ea4c3674c2b392f` |

`artifacts/stage_8/qualification/stage8_qualification_freeze.json` is the machine-readable freeze index.

## Decision

Stage 8 satisfies the required development gate with seven independent positive systems and has six independent final systems. It is authorized to proceed only to a separate gate-predeclaration pass. That next pass may define and freeze leakage-safe features, architecture, formula, event-level validation, comparators, and promotion criteria. Training and final-set prediction remain unauthorized.

## Gate predeclaration

**PREDECLARED AND FROZEN — NOT TRAINED.**

The hypothesis is: “HRRR should modify frozen A+ only when issue-time-safe radar-based information indicates that A+ is uncertain/weak and the HRRR precipitation signal is plausibly useful.” Stage 8 tests conditional trust, not another multimodal forecast generator.

### Immutable forecasts and gate equation

A+ remains the immutable checkpoint with SHA-256 `63e049c5ea75c3aa6af7b461fe6e9bca2db3c51198a667e71f02a7c6217efe70`. HRRR remains binary `I(APCP hourly-interval average rate >= 0.1 mm/h)` with the issue-safe f02 interval used through 60 minutes and f03 after 60 minutes. No fitted temporal interpolation or HRRR threshold search is permitted.

The gate produces one scalar for each 128×128 tile and each of the 20 six-minute leads:

`g(r,t) = sigmoid(beta_0 + sum_i beta_i z_i)`

`P_final(x,t) = (1 - 0.50 g(r,t)) P_A+(x,t) + 0.50 g(r,t) I_HRRR(x,t)`

Thus `g=0` is exactly A+, while `g=1` is exactly the frozen Stage 6 50/50 hybrid. The gate cannot exceed the already-tested HRRR influence `alpha=0.50`.

### Exact feature set

The ordered input is frozen to these eight tile/lead-level causal features:

1. `lead_fraction`: lead minutes divided by 120.
2. `current_radar_wet_fraction`: valid-pixel wet fraction in the final observed radar frame at >0.1 mm/h.
3. `a_plus_wet_fraction`: fraction with frozen A+ probability ≥0.35 at that lead.
4. `a_plus_mean_entropy`: mean Bernoulli entropy of A+ probability, in nats.
5. `pysteps_wet_fraction`: fraction with issue-time PySTEPS rate ≥0.1 mm/h.
6. `hrrr_wet_fraction`: fraction with applicable f02/f03 APCP hourly-interval average rate ≥0.1 mm/h.
7. `a_plus_hrrr_disagreement_fraction`: fraction where `(A+ >=0.35) XOR HRRR-wet`.
8. `hrrr_radar_forecast_support_fraction`: among HRRR-wet pixels, the fraction also supported by A+ ≥0.35 or PySTEPS ≥0.1 mm/h; defined as zero when HRRR has no wet pixels.

Fractions use the required sources’ jointly valid pixels. A row/lead with no valid denominator is rejected, not imputed. Entropy clips probability to `[1e-6, 1-1e-6]` only for evaluating logarithms.

No feature may use a future observed footprint or centroid, observed onset, future storm size, target distance/overlap, future radar, eventual-initiation label, or RADAR_LIMITED/V2 membership. Every materialized feature row must retain event/system ID, issue and lead time, source timestamps, applicable HRRR cycle/valid interval/availability time, units, transformation version, and source hashes. Any source timestamp or simulated availability later than issue time automatically invalidates the row.

### Architecture, normalization, and objective

The only learned model is a logistic gate with one intercept, eight coefficients, no hidden layer, interactions, or nonlinear expansion, and L2 penalty `lambda=1.0`; the intercept is unpenalized. Coefficients initialize at zero. The frozen solver is deterministic L-BFGS with at most 1,000 iterations and gradient tolerance `1e-9`. Regularization is not searched.

Each LOEO fold standardizes all eight features using the arithmetic mean and population standard deviation from its six training systems only. A zero-variance training feature uses divisor 1 and standardized value zero. No validation or final value contributes to normalization. If development promotes the gate, final-fit normalization uses all seven development systems only.

The loss is event-balanced Brier loss for observed occurrence >0.1 mm/h over all valid pixels and 20 leads. It first averages pixels within each row/lead, then row/leads within each independent system, and finally gives each of the six training systems equal weight before adding the L2 penalty. It never pools pixels or weights systems by object count.

### Seven-fold development procedure

Development uses exactly seven leave-one-independent-system-out folds. Both September 7 and September 8 periods stay in their single shared-system fold. Each fold fits normalization and coefficients on six systems and predicts only the seventh. All selection evidence comes from the combined out-of-fold predictions; in-sample skill is not evidence.

The Stage 8 operating threshold is selected from `0.05, 0.10, ..., 0.95`. A threshold is eligible only when development hard-negative wet area is ≤0.01 and initiation false fraction is ≤0.40. Among eligible thresholds, maximize unweighted weather-event-macro initiation F1, breaking ties by lower onset MAE, lower Brier, higher precision, then lower numeric threshold. It is frozen before the all-development fit.

### Comparators and rule diagnostic

The identical OOF rows compare frozen A+ at its 0.35 policy threshold, PySTEPS at 0.1 mm/h, raw binary HRRR, the frozen Stage 6 hybrid at its 0.30 threshold, and the Stage 8 logistic gate. H1 remains historical and is not a promotion comparator.

The untuned rule diagnostic is:

`g_rule = clip(1 - mean(abs(2 P_A+ - 1)), 0, 1)`

It uses the same α=0.50 equation and the Stage 8 OOF-selected threshold without separate tuning. No numeric superiority margin over this diagnostic is required, but its results must be reported.

### Frozen promotion rule

All conditions must pass:

- Stage 8 event-macro initiation F1 is at least A+ F1 +0.01.
- Stage 8 F1 is at least frozen Stage 6 hybrid F1 +0.005.
- At least four of seven systems have non-negative gate-minus-A+ F1, with at least three strictly positive.
- Development hard-negative wet area is ≤0.01.
- False-initiation fraction is ≤0.40.

Secondary reporting includes precision, recall, false initiation, onset MAE and bias, Brier score, detection within 30/60/90/120 minutes, CSI, FSS, and meaningful broad occurrence/rate metrics. Gate distributions and standardized coefficients are descriptive mechanism checks only.

### Radar-limited evaluation cohorts

The existing definitions are reused unchanged and are evaluation labels, never inputs:

- `RADAR_LIMITED_INITIATION`: radar validity ≥0.95, final dryness at eventual pixels ≥0.90, history wet area ≤0.05, PySTEPS eventual-pixel coverage ≤0.40, non-advective, and ≥18 km boundary clearance.
- `RADAR_POOR_INITIATION_V2`: the corresponding thresholds are ≥0.95, ≥0.95, ≤0.02, ≤0.20, non-advective, and ≥18 km.

Membership on new development data must be frozen before Stage 8 predictions are inspected. Performance is reported for all initiation, RADAR_LIMITED, and strict V2 when adequately populated, without changing the primary endpoint.

### Final-set and failure policy

The six-system final object manifest remains sealed at SHA-256 `52e0f3df309862584e3391da9ae0b40caba58583f28df68e45f4cd940445359f`. No final features or predictions may be generated until OOF development passes every promotion condition and the all-development normalization, coefficients, selected threshold, memberships, code, and dependency hashes are frozen.

If development fails, Stage 8 stops without final scoring or revisions to features, interactions, α, λ, architecture, or thresholds. Any revision is a new experiment requiring a new design.

The complete machine-readable specification was subsequently amended before training as documented below.

## Final pre-training validation amendment

**AMENDED BEFORE ANY STAGE 8 FIT OR PREDICTION.** At amendment time, no Stage 8 gate coefficient had been fitted, no Stage 8 prediction existed, no final-set forecast output had been generated, and no consumed 2023 result was consulted. The eight-feature set, model family, α=0.50, HRRR definition, A+ checkpoint, λ=1.0, L-BFGS solver, event-balanced Brier objective, comparators, radar-limited definitions, and sealed final object manifest remain unchanged.

### Hard-negative system audit

The 12 development hard-negative rows belong to five of the existing seven positive independent-system groups. None represents an additional negative-only system:

| Existing positive-system group | Negative rows | Candidate periods |
|---|---:|---|
| `dev_jun21_2021_frontal_convection` | 2 | `stage7_dev_jun21_2021_frontal_convection` |
| `dev_may21_2022_derecho` | 2 | `stage7_dev_may21_2022_derecho` |
| `dev_aug29_2022_wet_windy` | 2 | `stage7_dev_aug29_2022_wet_windy` |
| `dev_2021_09_07_08_shared_system` | 4 | `stage8_add_dev_sep07_2021`, `stage8_add_dev_sep08_2021` |
| `dev_nov21_2021` | 2 | `stage8_add_dev_nov21_2021` |

The explicit 272-row positive/negative membership table is `artifacts/stage_8/gate_predeclaration/stage8_development_row_system_membership.csv`, SHA-256 `63e6366d3ab995c7c7b03bcf563de47718cb4dd422672578610e479f1e34ed5d`. It records row ID, role, candidate/event ID, independent-system group, and relationship to positive systems. The same complete mapping is embedded in the machine-readable predeclaration.

### Universal grouped holdout policy

The development design retains seven outer system folds. Every positive and hard-negative row receives an evaluation prediction only from a fit that excluded its complete independent weather system. Hard negatives coincident with a positive outer system use that system's outer model and inner-selected threshold. There are no extra negative-only folds because no additional negative system exists.

For each outer positive-system fold:

1. Hold out the outer positive system and every negative row in that system.
2. On the six remaining systems, generate six-fold inner grouped-LOEO predictions with fold-local normalization.
3. Select that outer fold's threshold from `0.05, 0.10, ..., 0.95` using only inner predictions: require hard-negative wet area ≤0.01 and false initiation ≤0.40; maximize event-macro initiation F1; break ties by lower onset MAE, lower Brier, higher precision, then lower threshold.
4. Fit normalization and coefficients using all six permitted outer-training systems.
5. Apply the inner-selected threshold once to the untouched outer system and its hard negatives.

The promotion estimate combines only these seven untouched outer predictions. In-sample predictions, inner predictions, thresholds informed by the outer event, and a globally selected seven-system threshold are prohibited from the promotion calculation.

### Amended promotion consistency rule

All previously frozen gates remain, except the event-consistency wording is tightened. Promotion requires:

- nested outer event-macro F1 ≥ frozen A+ F1 +0.01;
- nested outer event-macro F1 ≥ frozen Stage 6 hybrid F1 +0.005;
- **at least four of seven independent positive systems have strictly positive Stage-8-minus-A+ event F1; ties do not count**;
- out-of-system hard-negative wet area ≤0.01;
- outer-evaluation false-initiation fraction ≤0.40.

No additional promotion requirement was introduced.

### Promotion versus final fitting

Fold-specific inner thresholds are used only for the nested development promotion estimate. If and only if every promotion gate passes, the threshold-free outer probabilities may serve as ordinary seven-system OOF probabilities for selecting one deployment threshold under the same frozen rule. Then normalization and one logistic gate are fitted using all seven development systems, and normalization, coefficients, deployment threshold, code/source hashes, comparators, and cohort memberships are frozen in a final procedure manifest. Only after that freeze may the six-system final holdout be scored once.

The amended machine-readable predeclaration SHA-256 is `18e2d0273f5e3a04028339e5cd534d9b2e86a3a8129d32234aaa252e7e94ec5d`.

## Nested development results

The frozen nested experiment was executed on 260 positive-initiation rows and 12 hard-negative rows from seven and five independent systems, respectively. `RADAR_LIMITED_INITIATION` contains 146 rows; the strict `RADAR_POOR_INITIATION_V2` cohort contains zero qualifying rows and is therefore reported as unpopulated, not estimated. Membership was frozen before gate fitting. The feature-manifest, cohort, and causality-audit SHA-256 values are, respectively, `bbbedc03603ae43d372dcc8c53fc6ff969a5120cf32b822765f1473959d425dc`, `629fc3878a31f89defdc6fe8553a55444f75992774757adbe14a3eaac8070c45`, and `b63a48f621ca8ed3aac56ac5cc5f3d34081aca8595af721f6418b89dd3607a61`.

All 272 rows passed the eight-feature, 20-lead causality audit. Every row received exactly one out-of-system prediction; all stored prediction hashes validate; all seven L-BFGS fits converged. No final-set feature or forecast was generated. The complete audit is in `artifacts/stage_8/nested_development/nested_execution_integrity.json`.

### Outer folds and individual-system results

| Untouched system | Threshold | Iterations | A+ F1 | Hybrid F1 | Stage 8 F1 | Stage 8 − A+ | Stage 8 − hybrid | Mean g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sep 7/8 shared | 0.45 | 7 | 0.4453 | 0.3696 | 0.3588 | -0.0865 | -0.0109 | 0.4111 |
| Aug 11 2021 | 0.40 | 7 | 0.1528 | 0.1681 | 0.1331 | -0.0198 | -0.0351 | 0.3885 |
| Aug 29 2022 | 0.40 | 7 | 0.2672 | 0.2813 | 0.2401 | -0.0271 | -0.0412 | 0.3612 |
| Jun 21 2021 | 0.40 | 10 | 0.2940 | 0.5005 | 0.3360 | +0.0420 | -0.1645 | 0.5245 |
| May 21 2022 | 0.40 | 7 | 0.1337 | 0.1744 | 0.1042 | -0.0295 | -0.0702 | 0.3761 |
| Nov 21 2021 | 0.45 | 9 | 0.0084 | 0.2403 | 0.0168 | +0.0084 | -0.2235 | 0.4769 |
| Sep 22 2021 | 0.45 | 8 | 0.6396 | 0.6868 | 0.6505 | +0.0109 | -0.0363 | 0.5347 |

Only three of seven systems improved strictly over A+. No system improved over the frozen hybrid.

### Comparator results

These are unweighted weather-system macro results from untouched outer predictions. The full tables include onset and 30/60/90/120-minute detection results.

| Model | Precision | Recall | F1 | False initiation | Onset MAE (min) | Median onset bias (min) | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| A+ | 0.4832 | 0.2659 | 0.2773 | 0.5168 | 34.91 | -21.90 | 0.2242 |
| PySTEPS | 0.6450 | 0.2896 | 0.3499 | 0.2480 | 14.59 | -3.18 | 0.2539 |
| Raw HRRR | 0.5249 | 0.3198 | 0.3177 | 0.3411 | 30.73 | -4.01 | 0.2886 |
| Stage 6 50/50 hybrid | 0.5299 | 0.3578 | 0.3459 | 0.3895 | 28.96 | -6.78 | 0.2187 |
| Stage 8 conditional gate | 0.5411 | 0.2394 | 0.2628 | 0.3905 | 34.70 | +0.09 | 0.2121 |
| Uncertainty rule | 0.4953 | 0.2371 | 0.2586 | 0.4155 | 33.11 | -13.82 | 0.2144 |

The corresponding row-macro Stage 8 F1 is 0.2406, versus 0.2645 for A+, 0.2951 for the hybrid, 0.3555 for PySTEPS, 0.2532 for raw HRRR, and 0.2420 for the uncertainty rule. Broad 30/60/90/120-minute CSI, F1, and 18-km FSS are stored in `broad_lead_metrics.csv`; Stage 8 F1 is 0.1261, 0.1070, 0.1032, and 0.0932 at those leads.

### Hard negatives and frozen mechanism cohorts

Across the 12 out-of-system hard-negative rows (five systems), Stage 8 wet area is 0.000078, false initiation is 0.1667, Brier score is 0.00423, mean probability is 0.05656, maximum probability is 0.45973, and mean gate value is 0.4270. Per-row results and their fold thresholds are in `hard_negative_outer_metrics.csv`.

For all initiation rows, Stage 8 event-macro F1 is 0.2628. For the 146-row `RADAR_LIMITED_INITIATION` cohort it is 0.1127, with precision 0.5443, recall 0.0736, false initiation 0.3927, onset MAE 39.41 minutes, median onset bias +6.48 minutes, and Brier 0.2162. Strict V2 has no qualifying development rows, so no V2 skill estimate is possible.

### Gate behavior and coefficients

Across all 5,440 row-leads, mean `g` is 0.4025 and median `g` is 0.3765; its 5th/25th/75th/95th percentiles are 0.3609/0.3613/0.4109/0.5340. Mean `g` is nearly flat with lead (0.4025 at 6 minutes and 0.4026 at 120 minutes), 0.3885 when HRRR is dry, 0.4077 when HRRR is wet, 0.4270 on hard negatives, and 0.3910 on `RADAR_LIMITED_INITIATION`. Mean `g` rises as A+ confidence falls (0.3971 in the highest-confidence bin versus 0.4445 in the lowest populated bin) and as A+/HRRR agreement falls (0.3918 above 0.9 agreement versus 0.4332 at or below 0.5).

The fitted standardized feature coefficients are very small under the frozen λ=1 penalty. Across folds, mean coefficients are: lead fraction +0.000079, current radar wet fraction +0.000417, A+ wet fraction +0.001172, A+ entropy +0.000761, PySTEPS wet fraction -0.000154, HRRR wet fraction -0.003347, A+/HRRR disagreement -0.000857, and HRRR/radar support +0.000921. Intercepts range from -0.5712 to +0.1476. Exact per-fold normalization, intercepts, coefficients, objectives, convergence records, probabilities, and categorical predictions are stored under `artifacts/stage_8/nested_development/`; `outer_coefficients_with_intercept.csv` and `coefficient_stability.csv` provide the coefficient tables.

### Promotion decision

| Frozen gate | Requirement | Result | Pass |
|---|---|---:|---|
| A | Stage 8 F1 ≥ A+ F1 + 0.01 | 0.2628 vs required 0.2873 | No |
| B | Stage 8 F1 ≥ hybrid F1 + 0.005 | 0.2628 vs required 0.3509 | No |
| C | At least 4/7 systems strictly improve over A+ | 3/7 | No |
| D | Hard-negative wet area ≤ 0.01 | 0.000078 | Yes |
| E | False initiation ≤ 0.40 | 0.3905 | Yes |

**Final development classification: NEGATIVE — NOT PROMOTED.** Stage 8 stops here. The protected 2023 final set remains sealed and unscored; no all-development deployment threshold or final gate was fitted.
