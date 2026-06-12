# Phase 1: National Model — Evaluation Infrastructure

**Source:** `national_model_spec.md` §2 (Phase 1) and §4.1 (implementation order)

**Objective:** Establish a leakage-safe evaluation baseline before any model changes. Every Phase 2 enhancement (`build_model()` changes) will be measured against this baseline. No modifications to `build_model()` in this phase.

**Training window:** 2003–present. QCEW treated as final for 2003–2016 (known limitation; QCEW revision history only available from 2017).

**Data directory layout (updated):**

```
data/
├── store/            # Canonical Hive-partitioned vintage store (CES 2003+, QCEW 2003+)
├── providers/        # Payroll provider parquets
├── downloads/        # Raw BLS inputs (CES CSVs, QCEW bulk, scraped HTML)
├── intermediate/     # Pipeline byproducts (vintage_dates, release_dates, revisions)
├── reference/        # Static BLS crosswalks (industry_codes.csv, geographic_codes.csv)
└── publication_calendar.parquet
```

Path constants live in `src/alt_nfp/config.py`: `STORE_DIR`, `DOWNLOADS_DIR`, `INTERMEDIATE_DIR`.

---

## Step 1.1 — Extend Historical Publication Dates to 2003

**Source:** `national_model_spec.md` §2.2.2, §4.1 step 1.1

**Why:** Leakage-safe backtesting requires knowing when each data point was actually published.

**Status:** Largely complete. Historical publication dates are already present in the scraped pipeline data:
- `data/intermediate/vintage_dates.parquet` has CES dates from 2003-01 and QCEW dates from 2007-09.
- `data/intermediate/release_dates.parquet` has CES from 2008-01, QCEW from 2007-09.
- `src/alt_nfp/lookups/publication_dates.py` contains only forward-looking dates (Oct 2024+) for the schedule scraper; historical dates are served by the pipeline parquets above.
- The vintage store (`data/store/`) has CES data from 2003-01 and QCEW from 2003-01 (~73K and ~584K rows respectively).

**Blocks:** Step 1.2, all Phase 2 evaluation.

### Tasks

- [x] **CES release dates (2003–present).** Already present in `data/intermediate/vintage_dates.parquet` (1,104 rows, 2003-01 to 2026-01). Scraped via `src/alt_nfp/ingest/release_dates/` pipeline.

- [x] **QCEW release dates (2007–present).** Already present in `data/intermediate/vintage_dates.parquet` (251 rows, 2007-09 to 2025-06). Pre-2007 QCEW observations in the vintage store use heuristic vintage dates from `get_qcew_vintage_date()` in `lookups/revision_schedules.py`.

- [ ] **Validate coverage gap.** `release_dates.parquet` CES starts 2008-01 (not 2003). Confirm that `vintage_dates.parquet` fills 2003–2007 via supplemental dates or the revision schedule heuristic. If there is a gap, extend the scraper or add supplemental entries.

### Files

- `src/alt_nfp/lookups/publication_dates.py` (forward-looking schedule only)
- `src/alt_nfp/ingest/release_dates/` (scraper pipeline package)
- `data/intermediate/vintage_dates.parquet`, `data/intermediate/release_dates.parquet`

---

## Step 1.2 — Validate and Fix Leakage-Safe Backtesting

**Source:** `national_model_spec.md` §2.2.1–2.2.4, §4.1 step 1.2

**Why:** The `as_of` parameter in `panel_to_model_data()` censors CES observations by `vintage_date`. QCEW censoring is partially implemented. Provider and cyclical indicator censoring need validation. Without confirmed leakage-free filtering, backtest results are unreliable.

**Blocks:** All Phase 2 evaluation.

### Tasks

- [ ] **Validate CES censoring.** Confirm `panel_to_model_data(as_of=...)` correctly filters CES observations by `vintage_date` for all vintage slots (first print through final).

- [ ] **Validate QCEW censoring.** Confirm QCEW observations are filtered by publication date (from Step 1.1). Check that the filtering works correctly for both 3-month and 12-month QCEW lags.

- [ ] **Validate provider censoring.** Provider observations should be excluded when `vintage_date > as_of`. Confirm this works correctly for all provider sources, including birth-rate data.

- [ ] **Validate cyclical indicator censoring.** The `_CYCLICAL_PUBLICATION_LAGS` dict in `panel_adapter.py` applies publication lags per indicator. Verify lag values are correct and that censoring logic properly zeroes out unavailable observations.

- [ ] **Unit tests.** Extend `tests/test_benchmark_backtest.py` (`TestAsOfCensoring`) with tests confirming that `panel_to_model_data(as_of=...)` correctly censors all sources (CES, QCEW, providers, cyclical indicators).

- [ ] **Backtesting smoke test.** Run the current model at two different `as_of` dates and verify that later dates produce tighter posteriors. This confirms the censoring is doing something meaningful.

### Files

- `src/alt_nfp/panel_adapter.py` (fix censoring bugs if found)
- `tests/test_benchmark_backtest.py` (extend `TestAsOfCensoring`)

---

## Step 1.3 — Add JOLTS to Cyclical Indicator Pipeline

**Source:** `national_model_spec.md` §2.3.1–2.3.5, §4.1 step 1.3

**Why:** The BD equation includes `φ_3` cyclical covariates (already in `build_model()`) but only three indicators are loaded. Adding JOLTS provides a fourth labor demand signal. JOLTS begins December 2000 so coverage is fine for the 2003+ training window.

**Can parallelize:** After Step 1.2.

### Tasks

- [ ] **JOLTS revision assessment.** Download JOLTS first-release vs. final values from ALFRED for 2003–present. Compare revision magnitudes to cross-sectional variation. If revisions are small relative to the centered covariate range, skip vintage tracking and use final values with publication-lag censoring only. Document the finding as an inline comment or short markdown note.

- [ ] **Evaluate JOLTS openings vs. hires.** Determine which series (JTSJOL or JTSHIL) has stronger covariance with the BD component. Pick the one with stronger out-of-sample BD prediction.

- [ ] **Add JOLTS to config.** Add a fourth entry to `CYCLICAL_INDICATORS` in `src/alt_nfp/config.py` with the correct FRED series ID and publication lag (~2 months).

- [ ] **Update data loading.** Update `_load_cyclical_indicators()` in `src/alt_nfp/panel_adapter.py` if needed to handle the new series.

- [ ] **Unit tests.** Add tests for `_load_cyclical_indicators()` covering all four indicators, verifying correct loading over the 2003–present window.

### Files

- `src/alt_nfp/config.py` (add JOLTS to `CYCLICAL_INDICATORS`)
- `src/alt_nfp/panel_adapter.py` (update `_load_cyclical_indicators()` if needed)
- `tests/` (new or extended tests for cyclical loading)

---

## Step 1.4 — Verify All Cyclical Indicator Data Loading and Censoring

**Source:** `national_model_spec.md` §2.3.3–2.3.4, §4.1 step 1.4

**Why:** The existing three indicators (initial claims, NFCI, business applications) need verification that data coverage extends to 2003 and that publication-lag censoring values are accurate. BFS begins July 2004 — partial coverage is acceptable since the model handles missing covariates gracefully (centered values default to zero).

**Can parallelize:** After Step 1.2.

### Tasks

- [ ] **Verify data coverage.** Confirm all four indicators have data back to 2003 (or their earliest available date). Document any gaps.

- [ ] **Verify publication-lag values.** Cross-check `_CYCLICAL_PUBLICATION_LAGS` in `panel_adapter.py` against actual release schedules for all four indicators:
  - Initial unemployment claims: ~1 week
  - Chicago Fed NFCI: ~1 week
  - Census BFS: ~2 months
  - JOLTS: ~2 months

- [ ] **End-to-end loading test.** Run `_load_cyclical_indicators()` over the full 2003–present window and confirm all indicators load, center, and censor correctly.

### Files

- `src/alt_nfp/panel_adapter.py` (fix lag values if incorrect)
- `tests/` (verification tests)

---

## Step 1.5 — QCEW Sigma Sensitivity Automation

**Source:** `national_model_spec.md` §2.4, §4.1 step 1.5

**Why:** QCEW noise is fixed at `σ_QCEW,M3 = 0.0005` and `σ_QCEW,M12 = 0.0015`. Understanding how model outputs respond to these values informs prior calibration for Phase 2.

**Status:** Complete. `src/alt_nfp/sensitivity.py` exists with `run_sensitivity()` that accepts QCEW noise scale factors, rebuilds and samples the model for each, extracts key parameters, and produces comparison plots.

**Independent:** Does not block Phase 2 but informs prior calibration.

### Tasks

- [x] **`src/alt_nfp/sensitivity.py`** — `run_sensitivity()` implemented with scale-factor sweep, parameter extraction, and comparison plots.

- [ ] **Test.** At least one test confirming the function runs without error on a short sample.

### Files

- `src/alt_nfp/sensitivity.py`
- `tests/` (smoke test)

---

## Step 1.6 — Precision Budget and Provider Weight Extraction

**Source:** `national_model_spec.md` §2.5, §4.1 step 1.6

**Why:** Understanding the effective precision share of each data source (QCEW, CES by vintage, each provider) reveals which sources are driving the posterior. Provider signal quality ranking (by `λ_p`) and bias summary (by `α_p`) are essential diagnostics.

**Status:** Partially complete. `print_source_contributions()` in `src/alt_nfp/diagnostics.py` computes precision-weighted information shares. Also used by `sensitivity.py`. Remaining work is formalizing the output as a structured DataFrame and wiring into a standalone diagnostic script.

**Independent:** Does not block Phase 2.

### Tasks

- [x] **`print_source_contributions(idata, data)`** — implemented in `src/alt_nfp/diagnostics.py`. Computes precision-weighted information shares per source.

- [ ] **Formalize as `compute_precision_budget(idata, data) -> pl.DataFrame`** returning a structured DataFrame with precision share, provider λ ranking, and provider α bias summary. Currently output is print-only.

- [ ] **Integration.** Wire into `scripts/benchmark_diagnostic.py` or a new diagnostic script so it can be run after any model fit.

### Files

- `src/alt_nfp/diagnostics.py` (extend `print_source_contributions` to return DataFrame)
- `scripts/benchmark_diagnostic.py` (integration)

---

## Step 1.7 — TODO Marker for Representativeness Correction

**Source:** `national_model_spec.md` §2.6, §4.1 step 1.7

**Why:** Representativeness correction requires provider microdata (cell-level signals at supersector × Census region) from a separate private repo. The national model will eventually consume a QCEW-weighted composite of cell-level signals rather than raw provider indices. No implementation here — just document the dependency and interface contract.

**Independent.**

### Tasks

- [ ] **Add TODO marker** in `src/alt_nfp/config.py` or `src/alt_nfp/panel_adapter.py` noting the dependency. Include the expected interface contract:

```python
# TODO: Representativeness correction (depends on private microdata repo)
# Expected input: a single repr-corrected national composite per provider per month.
# Computed externally as: y_p_t = Σ_c w_c_QCEW * y_p_c_t
# where c indexes supersector × Census region cells.
```

### Files

- `src/alt_nfp/config.py` or `src/alt_nfp/panel_adapter.py`

---

## Dependency Graph

```
Step 1.1 (historical pub dates)
  └─► Step 1.2 (validate backtesting)
        ├─► Step 1.3 (add JOLTS)
        ├─► Step 1.4 (verify cyclical indicators)
        ├─► Step 1.5 (QCEW sensitivity)  [independent]
        ├─► Step 1.6 (precision budget)   [independent]
        └─► Step 1.7 (repr correction TODO) [independent]
```

Steps 1.1–1.2 are sequential (dates needed before censoring validation). Steps 1.3–1.7 can be parallelized after 1.2.

---

## Success Criteria

**Source:** `national_model_spec.md` §7.1

- All backtesting infrastructure is leakage-safe across CES, QCEW, provider, and cyclical indicator sources.
- All four cyclical indicators (claims, NFCI, BFS, JOLTS) load correctly with proper censoring.
- Diagnostic tools (QCEW sensitivity, precision budget) produce actionable output.
- JOLTS revision assessment is documented.
- Representativeness correction interface is documented as a TODO.