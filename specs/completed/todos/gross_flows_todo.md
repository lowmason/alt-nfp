# TODO: Concise gross-flows analysis for payroll provider vs official statistics

## Objective
Build an agent-executable pipeline for payroll-provider gross flows that replaces a brute-force review of thousands of cell-level measures with a concise diagnostics system.

The output should answer five questions:

1. **Are hires occurring in the right places relative to the benchmark?**
2. **Are separations occurring in the right places relative to the benchmark?**
3. **Are hire and separation rates too high or too low overall?**
4. **Is any mismatch mostly geography, mostly industry, or mostly geography-by-industry interaction?**
5. **Which small set of cells are the truly important outliers after shrinkage and reliability filtering?**

The implementation should support national, industry, geography, and geography-by-industry views while staying concise enough for routine monitoring.

---

## Scope and concept guardrail
This file covers **worker-flow measures** built from payroll rosters:

- hires
- separations
- continuing employment
- hire rate
- separation rate
- net growth rate
- churn rate

Important: the phrase "gross job flows" is overloaded. The implementation here is for **worker hires/separations at the worker-by-unit level**, not establishment-level job creation/job destruction in the BED/BDS sense.

Do **not** compare worker-flow measures directly to official job-flow statistics without an explicit concept bridge. If the benchmark is establishment job flows rather than worker flows, mark the comparison as concept-misaligned and either:

- stop, or
- route to a separate translation layer.

---

## Recommended organizing principle
Treat gross flows as **two different analysis objects**:

1. **Event composition**
   - where hires occur across cells
   - where separations occur across cells

2. **Rate intensity**
   - how high hire, separation, net, and churn rates are within cells

This is the key design choice that keeps the analysis concise.

---

## Non-goals
- Do **not** dump full cell tables into the executive summary.
- Do **not** run a separate significance test for every state x industry cell.
- Do **not** compare worker-flow measures to establishment job-flow measures without a concept-alignment layer.
- Do **not** count provider onboarding or offboarding as economic hires or separations.
- Do **not** treat continuing employment as a headline metric on equal footing with hires and separations; it is mainly an accounting and denominator object.
- Do **not** publish fine geographic flow diagnostics before handling degraded location information.
- Do **not** trust raw outlier rankings in sparse cells without shrinkage.

---

## Deliverables
- A reproducible pipeline that ingests provider microdata or pre-aggregated provider flow tables plus one or more official benchmarks.
- A machine-readable diagnostics object with two scorecards:

### Scorecard A: event composition
- `D_hires_overall`
- `D_hires_geography`
- `D_hires_industry`
- `decomposition_hires`
- `D_seps_overall`
- `D_seps_geography`
- `D_seps_industry`
- `decomposition_seps`
- `top_event_outlier_cells`

### Scorecard B: rate intensity
- `hire_rate_gap_national`
- `sep_rate_gap_national`
- `net_rate_gap_national`
- `churn_rate_gap_national`
- `hire_rate_surface_error`
- `sep_rate_surface_error`
- `decomposition_hire_rate`
- `decomposition_sep_rate`
- `top_rate_outlier_cells`

### Required metadata
- `employment_concept_used` (`active` or `qualified`)
- `flow_tracking_unit` (`client`, `pseudo_establishment`, or other configured unit)
- `flow_interval` (`t_minus_1_to_t`)
- `benchmark_source`
- `benchmark_concept_alignment`
- `grid_used`
- `location_regime`
- `share_allocated`
- `warnings`

- A human-readable markdown or HTML report with:
  - one executive summary
  - one event-composition summary
  - one rate-intensity summary
  - one short list of top outlier cells
  - one reliability/warnings section
- Unit tests for identities, aggregation, decomposition, and stabilization logic.

---

## Recommended project structure

```text
src/
  gross_flows/
    __init__.py
    config.py
    schemas.py
    provider.py
    benchmark.py
    stabilization.py
    unit_harmonization.py
    presence.py
    intervals.py
    location_regime.py
    allocation.py
    cell_assignment.py
    flow_builder.py
    metrics.py
    decomposition.py
    shrinkage.py
    reliability.py
    report.py
    cli.py

tests/
  test_presence.py
  test_flow_identities.py
  test_stabilization.py
  test_metrics.py
  test_decomposition.py
  test_shrinkage.py
  test_allocation.py
  test_cell_assignment.py

configs/
  default.yaml
  provider_<name>.yaml
  benchmark_<name>.yaml

outputs/
  tables/
  figures/
  reports/
```

---

## Core statistical design

## 1) Use interval-consistent flow definitions
Do **not** define hires using `(t vs t-1)` and separations using `(t vs t+1)` inside the same monthly scorecard. That mixes intervals.

Internally, define flows on the interval `(t-1, t)`.

Let `I[w, u, t] = 1` if worker `w` is present at unit `u` in period `t` under the selected employment concept.

For each unit `u`:

```text
H[u, t-1, t] = sum_w 1{I[w,u,t]=1 and I[w,u,t-1]=0}
S[u, t-1, t] = sum_w 1{I[w,u,t-1]=1 and I[w,u,t]=0}
C[u, t-1, t] = sum_w 1{I[w,u,t-1]=1 and I[w,u,t]=1}
```

Accounting identities:

```text
E[u, t-1] = C[u, t-1, t] + S[u, t-1, t]
E[u, t]   = C[u, t-1, t] + H[u, t-1, t]
Net[u]    = H[u] - S[u]
Churn[u]  = H[u] + S[u]
```

Default denominator for internal diagnostics:

```text
Ebar[u, t-1, t] = 0.5 * (E[u, t-1] + E[u, t])
```

Default rates:

```text
hire_rate[u]  = H[u] / Ebar[u]
sep_rate[u]   = S[u] / Ebar[u]
net_rate[u]   = (H[u] - S[u]) / Ebar[u]
churn_rate[u] = (H[u] + S[u]) / Ebar[u]
```

Configuration must allow alternative benchmark-aligned denominators:
- `start_employment`
- `end_employment`
- `average_employment`
- `benchmark_native`

Acceptance requirement:
- every reported rate must declare its denominator concept.

---

## 2) Build flows from a chosen employment concept
The flow builder must support at least two presence concepts:

- `active`: present on payroll during the pay period containing the reference date
- `qualified`: active and receiving qualifying pay types during that pay period

The agent must allow the user to compute both in parallel. Reports should state which concept drives the headline numbers.

---

## 3) Define the tracking unit explicitly
Flow measurement depends on the unit on which worker continuity is judged.

Supported options:
- `client`
- `pseudo_establishment`
- `worker_cell_direct` (advanced)

Default behavior:
- national analysis may use `client` if that is the only credible unit
- geography/industry cell analysis should prefer `pseudo_establishment` or another harmonized unit when possible

Required warning:
- if a client spans multiple states or industries and only `client` tracking is used, geographic and industry flow assignment may be biased.

---

## 4) Distinguish external hires/separations from internal transfers
For subnational or industry decomposition, worker movement across cells inside the same enterprise may be important.

Implement transfer handling modes:
- `ignore_internal_transfer` (simplest, acceptable for national totals)
- `same_unit_only` (default first pass)
- `count_cross_cell_transfer_as_sep_plus_hire` (preferred for cell-level flow tables when worker-cell assignment is credible)

The selected transfer rule must be written to metadata and report text.

---

## 5) Event-composition analysis
For a chosen grid `c in C` (for example `state x supersector`), compute event totals:

- provider hires: `H_P[c]`
- benchmark hires: `H_B[c]`
- provider separations: `S_P[c]`
- benchmark separations: `S_B[c]`

Then define event shares:

```text
pH[c] = H_P[c] / sum_c H_P[c]
qH[c] = H_B[c] / sum_c H_B[c]

pS[c] = S_P[c] / sum_c S_P[c]
qS[c] = S_B[c] / sum_c S_B[c]
```

Headline event-composition distances:

```text
D_hires_overall = 0.5 * sum_c |pH[c] - qH[c]|
D_seps_overall  = 0.5 * sum_c |pS[c] - qS[c]|
```

Margin distances:

```text
D_hires_geography = 0.5 * sum_g |pH[g,.] - qH[g,.]|
D_hires_industry  = 0.5 * sum_i |pH[.,i] - qH[.,i]|

D_seps_geography = 0.5 * sum_g |pS[g,.] - qS[g,.]|
D_seps_industry  = 0.5 * sum_i |pS[.,i] - qS[.,i]|
```

Interpretation:
- these are the fractions of hire-share or separation-share mass that would need to be reallocated for the provider event distribution to match the benchmark.

---

## 6) Event-composition decomposition
Decompose hire and separation composition mismatch into:
- geography
- industry
- interaction

Preferred first implementation:
- benchmark-relative Poisson log-linear model on counts

For hires:

```text
H_P[c] ~ Poisson(muH[c])
log(muH[c]) = log(sum_c H_P[c]) + log(qH[c]) + alphaH[g(c)] + betaH[i(c)] + gammaH[c]
```

For separations:

```text
S_P[c] ~ Poisson(muS[c])
log(muS[c]) = log(sum_c S_P[c]) + log(qS[c]) + alphaS[g(c)] + betaS[i(c)] + gammaS[c]
```

Required output:
- share of mismatch attributed to geography
- share of mismatch attributed to industry
- share of mismatch attributed to interaction

Implementation note:
- prefer a symmetric attribution rule such as averaged sequential deviance shares, not an order-dependent one-shot ANOVA decomposition.

Fallback if needed:
- weighted least squares on `log((p + eps) / (q + eps))`

---

## 7) Rate-intensity analysis
Rates do not sum to one across cells, so they should not be summarized only with compositional distances.

For each cell `c`, compute provider and benchmark rates:

```text
hP[c] = H_P[c] / Ebar_P[c]
sP[c] = S_P[c] / Ebar_P[c]
nP[c] = hP[c] - sP[c]
uP[c] = hP[c] + sP[c]

hB[c] = H_B[c] / Ebar_B[c]
sB[c] = S_B[c] / Ebar_B[c]
nB[c] = hB[c] - sB[c]
uB[c] = hB[c] + sB[c]
```

National headline gaps:

```text
hire_rate_gap_national  = hP[national] - hB[national]
sep_rate_gap_national   = sP[national] - sB[national]
net_rate_gap_national   = nP[national] - nB[national]
churn_rate_gap_national = uP[national] - uB[national]
```

Cell-level rate gap objects:

```text
dH[c] = hP[c] - hB[c]
dS[c] = sP[c] - sB[c]
```

Surface-error summaries should include at least one of:
- weighted MAE
- weighted RMSE
- weighted median absolute gap

Default weights:
- benchmark average employment `Ebar_B[c]`

---

## 8) Rate-surface decomposition
For hire and separation rates, decompose rate mismatch into geography, industry, and interaction.

Preferred first implementation:
- benchmark-relative Poisson model using benchmark rates as exposure terms

For hires:

```text
H_P[c] ~ Poisson(muHr[c])
log(muHr[c]) = log(Ebar_P[c]) + log(hB[c] + eps) + aH + alphaH[g(c)] + betaH[i(c)] + gammaH[c]
```

For separations:

```text
S_P[c] ~ Poisson(muSr[c])
log(muSr[c]) = log(Ebar_P[c]) + log(sB[c] + eps) + aS + alphaS[g(c)] + betaS[i(c)] + gammaS[c]
```

Interpretation:
- `aH`, `aS` capture national over/under-intensity
- `alpha` captures geography tilt in rates
- `beta` captures industry tilt in rates
- `gamma` captures state-by-industry residual rate mismatch

Fallback if benchmark counts are unavailable but benchmark rates are available:
- weighted regression on `log((rate_provider + eps) / (rate_benchmark + eps))`

Derived metrics:
- `net_rate_gap` and `churn_rate_gap` should be reported, but decomposition should still be built from primitive hire and separation models.

---

## 9) Shrinkage and outlier detection
Do **not** rank raw event-share gaps or raw rate gaps directly.

Implement shrinkage for:
- event-composition residuals
- hire-rate residuals
- separation-rate residuals

Preferred initial implementation:
- empirical Bayes shrinkage on residuals grouped by
  - geography parent
  - industry parent
  - size class if available
  - optional month-of-year or season bucket

Preferred final implementation:
- Bayesian partial pooling over geography, industry, size, and residual cell effects

Output:
- shrunken event residual per cell
- shrunken hire-rate residual per cell
- shrunken separation-rate residual per cell
- uncertainty / posterior SD
- top `K` materially abnormal cells by expected absolute deviation or z-like score

Required suppression rule:
- tiny cells that surface only due to noise should not dominate the outlier list.

---

## 10) Reliability and anomaly interpretation
High churn can reflect either:
- true economic behavior
- normal industry structure
- payroll-processing artifacts
- provider onboarding/offboarding artifacts
- seasonal spikes

Therefore create a reliability and interpretation layer.

Minimum required flags:
- `possible_onboarding_artifact`
- `possible_provider_churn_artifact`
- `possible_pay_frequency_artifact`
- `possible_seasonal_pattern`
- `concept_misalignment`
- `heavy_geographic_allocation`

Optional but recommended:
- a seasonal anomaly score comparing current cell rates with that cell's own month-of-year history.

---

## Methodological guardrails the agent must obey

### A. Benchmark concept alignment is mandatory
Before any comparison, write a benchmark-alignment object with:
- benchmark source name
- worker-flow vs job-flow classification
- employment concept used in benchmark
- denominator concept used in benchmark rates
- timing convention
- seasonal adjustment status
- geography/industry availability
- whether benchmark is national only, margin only, or full interaction grid

Possible alignment labels:
- `aligned`
- `aligned_with_minor_adjustment`
- `partially_aligned`
- `not_aligned`

If `not_aligned`, do not publish a numeric comparison as if concepts match.

### B. Stabilized panels are required before flow construction
The agent must not treat all raw client entries and exits as economic worker flows.

Implement at least one stabilization method:
- tenure-based stabilization
- change-point-based stabilization

Panel rule:
- only stabilized units are eligible to contribute to the measurement panel
- new units are not admitted mid-panel unless a configured rotating refresh occurs
- client/provider exits should not generate mass separations by default

### C. Worker presence must be built from pay periods carefully
Payroll data are fixed over pay periods, not calendar instants.

Implement rules for:
- weekly
- biweekly
- semi-monthly
- monthly

Required behavior:
- presence must be determined from the pay period containing the reference date
- reports must record pay-frequency composition
- diagnostic warnings should fire when mixed frequencies create obvious timing artifacts

### D. Choose the analysis grid adaptively
Default grid selection logic:

1. `geocode`
   - allow `state x supersector`
   - optionally finer internal cells

2. `zip_only`
   - allow `state x supersector`
   - if weak, fall back to `region x supersector`

3. `state_only`
   - do not trust fine geography as directly observed
   - default published grid should be conservative
   - often prefer `region x supersector`

4. `no_location`
   - do not publish cell-level geography as if observed
   - default to national and industry-only flow diagnostics

### E. Redistribution must happen before geographic diagnostics when location is degraded
If location is `state_only` or `no_location`, do not compute geographic flow diagnostics on the raw table.

Preprocessing options:
- classify likely single-establishment vs multi-establishment clients
- retain direct geography for small clients where credible
- redistribute large-client employment and flow exposure using benchmark geography shares
- optionally rake to benchmark geography margins

All post-redistribution cells must carry flags:
- `observed`
- `partially_observed`
- `allocated`

### F. Size class is a first-class dimension
At minimum:
- store employment and flow counts by size class if available
- allow decomposition and shrinkage models to condition on size class
- add a warning if size mix is unavailable

### G. Continuing employment is an internal accounting object, not a headline diagnostic
Continuing employment should be used to:
- verify identities
- build denominators
- support persistence diagnostics
- help detect implausible flow spikes

It should not receive equal billing to hires/separations in the executive summary unless the use case specifically demands it.

### H. Internal transfers must be explicit
If cell-level worker assignment is credible, the system should be able to flag within-firm cross-cell transfers separately from external hires/separations.

At minimum, report whether transfers are:
- ignored
- partially observed
- explicitly reclassified as sep-plus-hire across cells

---

## Implementation tasks

## Phase 0: Scaffolding
- [ ] Create the repo/module skeleton shown above.
- [ ] Add config system with:
  - provider name
  - benchmark name
  - analysis period
  - employment concept (`active` / `qualified`)
  - flow tracking unit
  - denominator type
  - transfer handling rule
  - geography and industry granularity
  - shrinkage method
  - top-K outlier count
- [ ] Add logging and deterministic seeds.
- [ ] Add one CLI entry point, for example:

```bash
python -m gross_flows.cli --config configs/provider_x.yaml
```

Acceptance criteria:
- pipeline runs end-to-end on a tiny toy dataset.

---

## Phase 1: Define data contracts
- [ ] Write schema definitions for provider input and benchmark input.
- [ ] Support both employee-level microdata and pre-aggregated flow tables.

Minimum provider microdata fields:
- `period`
- `worker_id`
- `client_id`
- `pseudo_establishment_id` if available
- `industry_code`
- `state` or other location fields if available
- `pay_frequency`
- `active_flag`
- `qualified_flag`
- `payment_type` if available
- `size_class` if available
- `location_quality_flag`

Minimum provider aggregated fields:
- `period`
- `cell_id`
- `employment_count_t_minus_1`
- `employment_count_t`
- `hire_count`
- `sep_count`
- `continuing_count`
- `industry_code`
- `geography_code`
- `size_class` if available
- metadata documenting how flows were constructed

Minimum benchmark fields:
- `period`
- `benchmark_source`
- `concept_type` (`worker_flow` or other)
- `geography_code`
- `industry_code`
- `employment_or_exposure`
- `hire_count` and/or `hire_rate`
- `sep_count` and/or `sep_rate`
- seasonal-adjustment status
- denominator concept

- [ ] Validate missing keys, duplicated rows, impossible counts, and inconsistent identities.

Acceptance criteria:
- invalid inputs fail loudly with useful error messages.

---

## Phase 2: Build stabilized measurement panels
- [ ] Implement tenure-based stabilization.
- [ ] Implement optional change-point-based stabilization.
- [ ] Implement rotating frozen panels with configurable refresh cadence.
- [ ] Exclude administrative provider entries/exits from economic flow measurement by default.
- [ ] Record stabilization status for each unit and period.

Acceptance criteria:
- units still onboarding are not allowed to contribute economic hires/separations.
- provider exits do not automatically create mass separations.

---

## Phase 3: Build worker presence and interval tables
- [ ] Construct worker presence under `active` and `qualified` concepts.
- [ ] Normalize pay-period timing to the chosen reference-date convention.
- [ ] Build interval tables for `(t-1, t)`.
- [ ] Derive `H`, `S`, `C`, `E_t_minus_1`, `E_t`, `Ebar`.
- [ ] Add recall flags when a worker reappears at the same unit after absence.
- [ ] Add optional transfer flags when worker-cell assignment changes across periods.

Acceptance criteria:
- accounting identities hold exactly on test data.
- interval labels are unambiguous.

---

## Phase 4: Benchmark alignment layer
- [ ] Implement benchmark concept classifier.
- [ ] Implement a benchmark-alignment report.
- [ ] Support benchmark availability patterns:
  - national only
  - industry only
  - geography only
  - geography x industry
- [ ] Support benchmark data that provide counts, rates, or both.
- [ ] Block invalid comparisons when concepts do not align.

Acceptance criteria:
- every run emits an explicit concept-alignment label.
- no report silently compares incomparable concepts.

---

## Phase 5: Unit harmonization and cell assignment
- [ ] Build unit harmonization layer for client vs pseudo-establishment analysis.
- [ ] Implement worker-to-cell assignment for geography and industry.
- [ ] Build NAICS to supersector mapping.
- [ ] Build state to Census region/division mapping.
- [ ] Support aggregation from fine codes to reporting cells.
- [ ] Flag cells as observed vs allocated geography.

Acceptance criteria:
- national totals are invariant to aggregation level.
- cell totals remain coherent after aggregation.

---

## Phase 6: Location-regime detection and geographic allocation
Skip allocation when geography is fully credible.

- [ ] Implement rule-based location-regime classifier.
- [ ] Emit one of:
  - `geocode`
  - `zip_only`
  - `state_only`
  - `no_location`
- [ ] For degraded geography, implement redistribution before geographic flow diagnostics.
- [ ] Support small-client retention of observed geography when credible.
- [ ] Support large-client redistribution using benchmark geography shares.
- [ ] Support optional raking / IPF.
- [ ] Track `share_allocated` and sensitivity to alternative priors.

Acceptance criteria:
- post-allocation employment and flow totals still sum correctly.
- geographic diagnostics are never run on raw degraded geography as if observed.

---

## Phase 7: Build comparable event and rate tables
- [ ] Construct provider and benchmark event tables for hires and separations.
- [ ] Construct provider and benchmark rate tables for hire, separation, net, and churn rates.
- [ ] Ensure denominators are benchmark-aligned when required.
- [ ] Produce national, geography, industry, and interaction views where benchmark support exists.
- [ ] Mark missing benchmark dimensions explicitly instead of backfilling silently.

Acceptance criteria:
- the system can produce a single comparable table per measure and period.

---

## Phase 8: Event-composition metrics
- [ ] Implement `D_hires_overall` and `D_seps_overall`.
- [ ] Implement geography and industry margin distances for hires and separations.
- [ ] Add optional weighted variants if needed.
- [ ] Add interval estimates later if easy; otherwise stage later.

Acceptance criteria:
- identical provider and benchmark event tables return zeros.
- obvious toy perturbations return expected values.

---

## Phase 9: Event-composition decomposition
- [ ] Fit benchmark-relative Poisson log-linear model for hires.
- [ ] Fit benchmark-relative Poisson log-linear model for separations.
- [ ] Return geography, industry, and interaction shares of mismatch.
- [ ] Add safeguards for zero cells.
- [ ] Produce concise narrative text, for example:
  - `Hire mismatch is mostly industry composition.`
  - `Separation mismatch is mainly geographic.`

Acceptance criteria:
- decomposition shares are stable on toy data.
- results are not sensitive to term ordering if symmetric attribution is used.

---

## Phase 10: Rate metrics and surface error
- [ ] Compute national hire, separation, net, and churn rate gaps.
- [ ] Compute weighted MAE and weighted RMSE for hire-rate and separation-rate surfaces.
- [ ] Add optional weighted median absolute gap.
- [ ] Ensure net and churn are derived, not independently estimated first.

Acceptance criteria:
- cells with equal provider and benchmark rates produce zero error.
- a high-churn / zero-net toy example is surfaced correctly as a rate-intensity mismatch.

---

## Phase 11: Rate-surface decomposition
- [ ] Fit benchmark-relative Poisson rate models for hires and separations.
- [ ] Return national intensity gap plus geography, industry, and interaction components.
- [ ] Produce concise narrative text, for example:
  - `Provider hire rates are nationally high, especially in leisure and hospitality.`
  - `Separation-rate mismatch is concentrated in a few state x industry cells.`

Acceptance criteria:
- national over/under-intensity is separated from compositional mismatch.
- interaction effects do not dominate unless the data support it.

---

## Phase 12: Shrinkage and outlier discovery
- [ ] Apply empirical Bayes shrinkage to event residuals.
- [ ] Apply empirical Bayes shrinkage to hire-rate and separation-rate residuals.
- [ ] Group by geography parent, industry parent, and size class when available.
- [ ] Suppress tiny unreliable cells.
- [ ] Rank top `K` cells separately for:
  - event composition
  - hire-rate intensity
  - separation-rate intensity

Acceptance criteria:
- sparse cells stop dominating the outlier lists.
- economically important large-cell outliers still surface.

Stretch goal:
- replace empirical Bayes with full hierarchical Bayes.

---

## Phase 13: Reliability, warnings, and anomaly checks
- [ ] Create reliability labels:
  - `reliable`
  - `marginal`
  - `insufficient`
- [ ] Base reliability on:
  - cell size
  - share allocated vs observed
  - stabilization status
  - concept alignment
  - shrinkage intensity
  - temporal stability if history is available
- [ ] Add warnings when:
  - geography is heavily model-based
  - many units are unstabilized
  - provider onboarding/offboarding may contaminate flows
  - pay-frequency mix can create timing artifacts
  - benchmark concepts are only partially aligned

Optional anomaly module:
- [ ] compare current rates to month-of-year history and flag unusual spikes.

Acceptance criteria:
- warnings appear in both machine-readable output and human report.

---

## Phase 14: Reporting layer
- [ ] Produce one concise summary table with:
  - benchmark source
  - concept-alignment status
  - employment concept used
  - tracking unit used
  - grid used
  - location regime
  - event-composition metrics
  - rate-intensity metrics
  - top outlier cells
- [ ] Produce markdown report containing:
  - executive summary
  - event-composition summary
  - rate-intensity summary
  - reliability and warnings
  - compact appendix
- [ ] Produce no more than 4 small charts in the executive section:
  - hire composition by geography or industry margin
  - separation composition by geography or industry margin
  - ranked hire-rate residuals
  - ranked separation-rate residuals or one residual heatmap

Acceptance criteria:
- the first section can be read in under 3 minutes.
- no giant cell dump appears in the executive summary.

---

## Phase 15: Temporal extension
This phase is optional for the first implementation but should be designed in from the start.

- [ ] Allow diagnostics to run month-by-month.
- [ ] Store histories of:
  - `D_hires_overall`
  - `D_seps_overall`
  - national rate gaps
  - surface-error metrics
  - top outlier persistence
- [ ] Add drift diagnostics:
  - does provider representativeness change over time?
  - do flow artifacts coincide with provider operational changes?
  - are spikes seasonal, cyclical, or administrative?

Acceptance criteria:
- the system can later support dashboards and backtests without redesign.

---

## Minimum viable implementation order
If the agent must ship quickly, do the following in order:

1. [ ] Build stabilized interval flow table.
2. [ ] Build benchmark alignment layer.
3. [ ] Construct comparable event tables.
4. [ ] Compute `D_hires_overall`, `D_seps_overall`, and rate gaps.
5. [ ] Fit simple geography/industry decomposition.
6. [ ] Add empirical Bayes shrinkage.
7. [ ] Generate concise markdown report.
8. [ ] Add geographic allocation for degraded location regimes.
9. [ ] Upgrade to hierarchical Bayes if needed.

---

## Suggested output schema

```json
{
  "provider": "string",
  "period": "YYYY-MM",
  "employment_concept_used": "active|qualified",
  "flow_interval": "t_minus_1_to_t",
  "flow_tracking_unit": "client|pseudo_establishment|...",
  "benchmark_source": "string",
  "benchmark_concept_alignment": "aligned|aligned_with_minor_adjustment|partially_aligned|not_aligned",
  "location_regime": "geocode|zip_only|state_only|no_location",
  "grid_used": "state_x_supersector|region_x_supersector|national_x_supersector|...",
  "share_allocated": 0.0,
  "totals": {
    "provider_employment_t_minus_1": 0,
    "provider_employment_t": 0,
    "provider_hires": 0,
    "provider_seps": 0,
    "provider_continuing": 0,
    "benchmark_hires": 0,
    "benchmark_seps": 0
  },
  "scorecard_event": {
    "D_hires_overall": 0.0,
    "D_hires_geography": 0.0,
    "D_hires_industry": 0.0,
    "D_seps_overall": 0.0,
    "D_seps_geography": 0.0,
    "D_seps_industry": 0.0,
    "decomposition_hires": {
      "geography_share": 0.0,
      "industry_share": 0.0,
      "interaction_share": 0.0,
      "method": "poisson_loglinear"
    },
    "decomposition_seps": {
      "geography_share": 0.0,
      "industry_share": 0.0,
      "interaction_share": 0.0,
      "method": "poisson_loglinear"
    }
  },
  "scorecard_rates": {
    "hire_rate_gap_national": 0.0,
    "sep_rate_gap_national": 0.0,
    "net_rate_gap_national": 0.0,
    "churn_rate_gap_national": 0.0,
    "hire_rate_surface_error": {
      "weighted_mae": 0.0,
      "weighted_rmse": 0.0
    },
    "sep_rate_surface_error": {
      "weighted_mae": 0.0,
      "weighted_rmse": 0.0
    },
    "decomposition_hire_rate": {
      "national_share": 0.0,
      "geography_share": 0.0,
      "industry_share": 0.0,
      "interaction_share": 0.0
    },
    "decomposition_sep_rate": {
      "national_share": 0.0,
      "geography_share": 0.0,
      "industry_share": 0.0,
      "interaction_share": 0.0
    }
  },
  "top_outlier_cells": [
    {
      "cell": "CA x Leisure and Hospitality",
      "measure": "hire_rate",
      "observed_or_allocated": "observed",
      "raw_residual": 0.0,
      "shrunken_residual": 0.0,
      "uncertainty": 0.0,
      "reliability": "reliable"
    }
  ],
  "warnings": []
}
```

---

## Test cases the agent must create

### Toy case 1: perfect benchmark match
- Provider flows and benchmark flows match exactly.
- Expect all distance and gap metrics to equal zero.
- Expect no meaningful outliers.

### Toy case 2: equal net growth, different churn
- Provider and benchmark both have net growth near zero.
- Provider has high hires and high separations while benchmark has low hires and low separations.
- Expect event and rate metrics to surface a mismatch despite similar net growth.

### Toy case 3: pure hire-composition industry tilt
- Provider hires are concentrated in one industry but geography is proportional.
- Expect `D_hires_industry` high and `D_hires_geography` near zero.

### Toy case 4: pure separation-composition geography tilt
- Provider separations are concentrated in one geography but industry mix is proportional.
- Expect `D_seps_geography` high and `D_seps_industry` near zero.

### Toy case 5: national over-intensity only
- Provider hire and separation rates are uniformly too high in every cell.
- Expect national rate gaps to be nonzero, but composition distances near zero.

### Toy case 6: sparse noisy cells
- Inject noise into tiny cells.
- Expect shrinkage to suppress tiny-cell outliers.

### Toy case 7: provider onboarding artifact
- Add new clients that ramp employees in over multiple months.
- Expect stabilization to block these from creating false hires.

### Toy case 8: provider offboarding artifact
- Remove clients administratively.
- Expect exits not to generate mass separations by default.

### Toy case 9: degraded geography
- State-only location plus large multi-state clients.
- Expect raw geographic diagnostics to be blocked or downgraded until redistribution occurs.

### Toy case 10: internal cross-state transfer
- Worker stays with same enterprise but moves from one state cell to another.
- Verify behavior under each transfer-handling rule.

### Toy case 11: mixed pay frequencies
- Weekly and biweekly workers produce apparent month-to-month timing noise.
- Expect warnings or smoothing behavior according to config.

### Toy case 12: continuing-employment identity check
- Verify `E_t_minus_1 = C + S` and `E_t = C + H` at every aggregation level.

---

## Reporting template the agent should target
Every executive summary should contain exactly:
- one benchmark-concept alignment statement
- one event-composition paragraph
- one rate-intensity paragraph
- one short list of top outlier cells
- one reliability/warnings paragraph

Suggested verbal structure:
1. `Hires are broadly / not broadly occurring in the right places.`
2. `Separations are broadly / not broadly occurring in the right places.`
3. `Provider hire/separation intensity is high / low relative to benchmark.`
4. `Mismatch is mostly geography / mostly industry / mostly a few interaction cells.`
5. `The most important outliers are ...`

---

## Future extensions
- Add recall-rate measurement explicitly.
- Add quit / layoff / discharge subclassification if pay or separation reason codes exist.
- Add hours-based flow analogs.
- Add wage-change overlays for job-stayers and movers.
- Add a joint employment-plus-flows dashboard so the user can see when stable net employment hides high churn.
