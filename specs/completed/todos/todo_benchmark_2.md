# TODO: Benchmark Prediction — Phase 2 (Backtesting)

**Source spec:** `specs/benchmark_spec.md` §4
**Depends on:** Phase 1 (`benchmark.py`, `benchmark_revisions.py`) — complete.
**Scope:** Produce RMSE-by-horizon curves across benchmark years using only the
information available at each horizon. Requires `as_of` censoring in the data
pipeline, a horizon-to-date mapping, the backtest orchestrator, evaluation
metrics, comparative benchmarks, and plots.

**Success criteria:** Given a set of benchmark years and horizons, run the
full model at each `(year, horizon)` pair with appropriate data censoring,
compare posterior predictions against actual revisions, and produce an
RMSE-by-horizon table/plot that beats the Decker (2024) benchmark of 229K
at T-6 or earlier.


---

## Task 1: Historical publication date coverage

The `get_default_calendar()` in `revision_schedules.py` now prefers
`publication_calendar.parquet` (which merges scraped dates back to ~2003
with forward-looking hard-coded dates). This gives full CES/QCEW publication
date coverage for all backtest years.

- [x] **1.1** Updated `get_default_calendar()` to load from
  `data/publication_calendar.parquet` when available, falling back to
  hard-coded dicts.

- [ ] **1.2** Verify that `publication_calendar.parquet` has CES and QCEW
  dates back to at least 2013 (needed for the April 2013 → March 2014
  benchmark window). Run `scripts/build_publication_calendar.py` after the
  vintage pipeline completes to rebuild the calendar.

- [x] **1.3** Heuristic fallback already exists in `get_ces_vintage_date()`
  and `get_qcew_vintage_date()` — lag-based approximation kicks in when
  exact dates are missing.


---

## Task 2: `as_of` censoring in `panel_to_model_data()`

Added `as_of: date | None = None` parameter to `panel_to_model_data()` in
`panel_adapter.py`.

- [x] **2.1** When `as_of` is provided, filter the panel to
  `vintage_date <= as_of` before extracting growth arrays. This removes
  CES, QCEW, and provider observations not yet published.

- [x] **2.2** `as_of` supersedes `censor_ces_from` (the old CES-only
  censoring parameter). Both are retained for backward compatibility.

- [x] **2.3** Cyclical indicators (`claims_c`, `nfci_c`, `biz_apps_c`) are
  masked using publication lag constants: claims/NFCI ~1 month, business
  applications ~2 months.

- [x] **2.4** Unit tests verify:
  - `as_of` reduces CES observation count.
  - Earlier `as_of` has fewer QCEW observations than later `as_of`.
  - `as_of` supersedes `censor_ces_from`.

- [ ] **2.5** Leakage test (run after vintage store is built): at
  `as_of = 2024-04-15` (T-12 for March 2025), verify QCEW Q1 2024
  (published ~Sep 2024) is NOT present.


---

## Task 3: Horizon-to-date mapping

`horizon_to_as_of(march_year, horizon)` in `benchmark_backtest.py` maps
`(year, horizon)` → concrete `as_of` date.

- [x] **3.1** Five horizons defined per the spec:

  | Horizon | Calendar cutoff       | Information state |
  |---------|-----------------------|-------------------|
  | T-12    | April Y (15th)        | Window just closed. Providers + CES only. |
  | T-9     | July Y (15th)         | Pre-QCEW. Cyclical indicators informing BD. |
  | T-6     | October Y (15th)      | QCEW Q1 Y published (~September). |
  | T-3     | January Y+1 (15th)    | QCEW Q2 Y published (~December). |
  | T-1     | February Y+1 (15th)   | QCEW Q3 may be available. |

- [x] **3.2** Horizons are monotonically increasing (tested).
- [x] **3.3** Works for all years in `EXTENDED_YEARS` (tested).


---

## Task 4: Benchmark backtest orchestrator

`run_benchmark_backtest()` in `benchmark_backtest.py`.

- [x] **4.1** Signature:
  ```python
  def run_benchmark_backtest(
      years: list[int] | None = None,      # default: [2022, 2023, 2024, 2025]
      horizons: list[str] | None = None,    # default: all five
      sampler_kwargs: dict | None = None,   # default: LIGHT_SAMPLER_KWARGS
      checkpoint_dir: Path | None = None,   # save after each run for resumption
  ) -> pl.DataFrame:
  ```

- [x] **4.2** Inner loop: `build_panel()` once → for each `(year, horizon)`:
  `panel_to_model_data(as_of=...)` → `build_model()` → `sample_model()` →
  `extract_benchmark_revision()` → `summarize_revision_posterior()` → record.

- [x] **4.3** Uses `LIGHT_SAMPLER_KWARGS` (2000/2000 draws, 2 chains) by
  default. ~3-5 min per run on Apple Silicon.

- [x] **4.4** Checkpointing: saves intermediate results to parquet after
  each `(year, horizon)` run. Loads checkpoint on restart to skip completed
  pairs.

- [x] **4.5** Console output: per-run summary (posterior mean/std/HDI,
  actual, error) plus final RMSE-by-horizon table.

- [ ] **4.6** Run the presentation backtest: `years=[2022, 2023, 2024, 2025]`
  (4 years × 5 horizons = 20 runs, ~1 hour).

- [ ] **4.7** Run the extended backtest (follow-up): `years=EXTENDED_YEARS`
  (10 years × 5 horizons = 50 runs, ~3-4 hours).


---

## Task 5: Evaluation metrics and comparative benchmarks

- [x] **5.1** `compute_backtest_metrics(results)` → per-horizon: bias, MAE,
  RMSE, 90% HDI coverage, average interval width.

- [x] **5.2** `build_comparative_benchmarks(years)` → naive zero and
  prior-year revision predictions.

- [x] **5.3** `comparative_rmse(benchmarks)` → RMSE for each naive strategy.

- [x] **5.4** Literature benchmarks as constants: Decker (229K), Cajner (243K).

- [x] **5.5** All metrics tested (21/21 tests pass).


---

## Task 6: Plots

`benchmark_plots.py` with three plot functions and a `plot_all()` convenience.

- [x] **6.1** `plot_rmse_by_horizon()` — model RMSE line plot with naive and
  literature benchmark reference lines.

- [x] **6.2** `plot_revision_fan_chart(march_year)` — posterior mean + 50%/90%
  HDI bands across horizons, with actual as horizontal line.

- [x] **6.3** `plot_coverage_by_horizon()` — bar chart of 90% HDI coverage
  by horizon, with 90% nominal reference line.

- [x] **6.4** All plots saved to `output/benchmark/`.


---

## Task 7: Integration tests and acceptance criteria

- [x] **7.1** Censoring tests: `as_of` reduces obs, masks future QCEW,
  supersedes `censor_ces_from`.

- [ ] **7.2** Leakage test on real data (after vintage store build).

- [ ] **7.3** Determinism test: run `(2025, 'T-6')` twice with the same
  random seed → identical results.

- [ ] **7.4** Monotonicity check: for a given year, interval width decreases
  from T-12 to T-1.

- [ ] **7.5** Acceptance criteria (after running the backtest):
  - **Presentation gate (4 years):** completes for 2022-2025 × 5 horizons.
    RMSE at T-6 documented vs 229K. Interval width decreases. No leakage.
  - **Extended gate (10 years):** 90% HDI coverage 0.7-1.0. Provider-era vs
    pre-provider RMSE comparison.


---

## Sequencing

```
Task 1 (pub dates) — complete
  └──→ Task 2 (as_of censoring) — complete
         └──→ Task 3 (horizon mapping) — complete
                └──→ Task 4 (backtest orchestrator) — code complete, awaiting run
                       ├──→ Task 5 (metrics) — complete
                       ├──→ Task 6 (plots) — complete
                       └──→ Task 7 (acceptance) — awaiting run
```

All code is implemented. Remaining work is running the backtest (requires
vintage store to be built) and verifying acceptance criteria on the results.


---

## Files created / modified

### Created
- `src/alt_nfp/benchmark_backtest.py` — orchestrator, horizon mapping, metrics
- `src/alt_nfp/benchmark_plots.py` — RMSE-by-horizon, fan charts, coverage
- `tests/test_benchmark_backtest.py` — 21 tests (all passing)

### Modified
- `src/alt_nfp/panel_adapter.py` — added `as_of` parameter with vintage_date
  filtering and cyclical indicator masking
- `src/alt_nfp/lookups/revision_schedules.py` — `get_default_calendar()` now
  prefers `publication_calendar.parquet`
- `src/alt_nfp/__init__.py` — added Phase 2 exports


---

## Out of scope (deferred to Phase 3)

- Error-correction term (`φ_BM`) conditioning on prior-year surprise — §5.2
- Preliminary benchmark as observation at T-5 — §5.3
- Published monthly B/D factors — §6.4
- Historical CES vintage data (pre/post-benchmark prints) — using post-benchmark
  data with known conservative-bias caveat
