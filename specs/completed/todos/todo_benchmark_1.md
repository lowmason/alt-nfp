# TODO: Benchmark Prediction — Phase 1 (Extraction)

**Source spec:** `benchmark_prediction_alt_nfp_v2.docx` §3, §4.5, §7 Phase 1
**Scope:** Extract implied benchmark revisions from existing posteriors as a
derived quantity. No model changes. No backtesting loop. This is the foundation
that Phase 2 (backtesting) and Phase 3 (enhancements) build on.

**Success criteria:** Given an existing `InferenceData` object and the loaded data
dict, produce a posterior distribution over the implied benchmark revision (in
000s of jobs) for any March reference year within the estimation window. Validate
against the known 2024 and 2025 actuals as a smoke test.


---

## Task 1: Benchmark revision lookup table

The spec (§4.5) provides historical benchmark revisions (NSA total nonfarm, 000s).
This is static data that needs to live in the lookups module alongside the existing
revision schedules.

- [ ] **1.1** Create `src/alt_nfp/lookups/benchmark_revisions.py` with a dict
  mapping March reference year → revision in thousands. Include all years from
  the spec:

  ```python
  # March Y → NSA total nonfarm revision (000s)
  BENCHMARK_REVISIONS: dict[int, float | None] = {
      2009: -902,
      2010: -378,
      2011: 229,
      2012: 481,
      2013: 340,
      2014: 105,
      2015: -259,
      2016: -151,
      2017: 136,
      2018: -43,
      2019: -505,
      2020: None,  # COVID — excluded
      2021: None,  # COVID — excluded
      2022: 462,
      2023: -105,
      2024: -598,
      2025: -862,
  }
  ```

- [ ] **1.2** Add a helper function `get_benchmark_revision(year: int) -> float | None`
  that looks up the revision and raises `KeyError` for years outside the table.
  Return `None` for COVID years.

- [ ] **1.3** Add a constant for the CES employment level at each prior benchmark
  anchor (March Y−1 NSA total nonfarm, 000s). This is needed to convert
  cumulative growth-rate divergence into level-space revisions. These can be
  sourced from CES historical data (series CEU0000000001, NSA all employees
  total nonfarm). Store as:

  ```python
  # March Y-1 NSA total nonfarm employment level (000s)
  # Used as L_{Mar_{Y-1}} in the revision formula
  BENCHMARK_ANCHOR_LEVELS: dict[int, float] = {
      2008: ...,  # anchor for 2009 benchmark
      2009: ...,  # anchor for 2010 benchmark
      # ...through 2024 for the 2025 benchmark
  }
  ```

  **Input needed from Lowell:** Confirm whether to pull these programmatically
  from the CES data pipeline or hard-code from BLS archives. The values must be
  the post-benchmark (final) March NSA levels, not the pre-benchmark prints.

- [ ] **1.4** Export all public names from `__init__.py` in the lookups package.

- [ ] **1.5** Write unit tests in `tests/test_benchmark_revisions.py`:
  - All years 2009–2025 present in the dict.
  - COVID years return `None`.
  - `get_benchmark_revision` raises `KeyError` for year 2000.
  - Anchor levels dict has one fewer entry than revisions dict (shifted by 1 year).


---

## Task 2: Create `src/alt_nfp/benchmark.py` — extraction module

This is the core Phase 1 deliverable. The function
`extract_benchmark_revision()` takes an existing posterior and computes the
implied revision as a deterministic transformation — no additional MCMC needed.

### 2.1 Core extraction function

- [ ] **2.1.1** Create `src/alt_nfp/benchmark.py` with the main function:

  ```python
  def extract_benchmark_revision(
      idata: az.InferenceData,
      data: dict,
      march_year: int,
  ) -> np.ndarray:
      """Extract posterior samples of the implied benchmark revision.

      Follows spec §3.1: the revision is a deterministic transformation
      of the existing posterior over g_total_nsa.

      Parameters
      ----------
      idata : az.InferenceData
          Posterior inference data from the fitted model.
      data : dict
          The loaded data dict (from load_data()), containing model dates,
          CES NSA growth rates, and index metadata.
      march_year : int
          The March reference year for the benchmark (e.g. 2025).

      Returns
      -------
      np.ndarray
          1-D array of posterior samples of the implied revision in
          thousands of jobs. Length = n_chains × n_draws.
      """
  ```

- [ ] **2.1.2** Implement the five-step procedure from spec §3.1:

  1. **Identify the benchmark window.** Find time indices for April Y−1
     through March Y in `data['dates']`. Raise `ValueError` if the window
     is not fully within the estimation period.

  2. **Cumulate posterior latent growth.** Extract `g_total_nsa` from
     `idata.posterior`, sum over the benchmark window indices. Result is a
     (chains × draws) matrix of cumulative latent log-growth.

  3. **Cumulate observed CES NSA growth.** Sum the published CES NSA
     growth rates over the same window from `data`. Use the latest
     available vintage (pre-benchmark). This is a known scalar.

  4. **Convert to level-space revision.** Look up
     `BENCHMARK_ANCHOR_LEVELS[march_year - 1]`. Compute:
     `L * (exp(cum_latent) - exp(cum_ces))` for each posterior draw.
     Result in thousands of jobs.

  5. **Flatten.** Collapse chain and draw dims into a single 1-D vector.

- [ ] **2.1.3** Handle edge cases:
  - March year outside the estimation window → clear `ValueError`.
  - COVID years (2020, 2021) → warn but still compute if requested.
  - Missing anchor level → `KeyError` with informative message.

### 2.2 Decomposition function

- [ ] **2.2.1** Add `decompose_benchmark_revision()` per spec §3.3:

  ```python
  def decompose_benchmark_revision(
      idata: az.InferenceData,
      data: dict,
      march_year: int,
  ) -> dict[str, np.ndarray]:
      """Decompose revision into continuing-units divergence and BD accumulation.

      Returns
      -------
      dict with keys:
          'total' : 1-D array, same as extract_benchmark_revision()
          'cont_divergence' : 1-D array, L * [exp(Σ g_cont_nsa) - exp(Σ y_ces_nsa)]
          'bd_accumulation' : 1-D array, L * [exp(Σ BD_t) - 1]
      """
  ```

  This separates "CES sample is off on continuing-units growth" from
  "CES birth/death model is wrong." Per the spec, the BD term should
  dominate; large continuing-units divergence may signal provider
  representativeness issues.

### 2.3 Summary statistics helper

- [ ] **2.3.1** Add `summarize_revision_posterior()`:

  ```python
  def summarize_revision_posterior(
      samples: np.ndarray,
      actual: float | None = None,
  ) -> dict[str, float]:
      """Compute summary statistics for the revision posterior.

      Returns
      -------
      dict with keys:
          'mean', 'median', 'std',
          'hdi_5', 'hdi_95' (90% HDI bounds),
          'actual' (if provided),
          'error' (mean - actual, if provided)
      """
  ```


---

## Task 3: Identify CES NSA growth rates in the data dict

The extraction function needs to sum observed CES NSA growth rates over the
benchmark window. This requires understanding how CES NSA data is stored in
the data dict produced by `load_data()`.

- [ ] **3.1** Audit `data.py` / `load_data()` to identify which key(s) hold
  CES NSA growth rates and which vintage they represent. The spec requires
  the "pre-benchmark" CES NSA — i.e., the CES data as published before the
  benchmark revision is applied. Document findings.

- [ ] **3.2** If the data dict stores CES NSA growth by vintage
  (`g_ces_nsa_by_vintage`), the extraction function should use the final
  pre-benchmark vintage. If the vintage panel infrastructure is in place,
  use the `real_time_view` with appropriate `as_of` date. Document the
  chosen approach.

- [ ] **3.3** If CES NSA growth is not directly available in the data dict for
  the needed vintage, add a helper to compute it from the CES index data.
  The growth rate is `log(index_t / index_{t-1})` on the NSA series.

- [ ] **3.4** Verify that the `g_total_nsa` variable in the posterior matches
  the composite signal definition:
  `g_total_nsa = g_cont + BD_t + s_t`. Confirm the variable name in
  `model.py` and ensure it's stored in `idata.posterior`.


---

## Task 4: Diagnostic script — run on existing posterior

Phase 1 concludes by running the extraction on an existing posterior as a
diagnostic. This validates the implementation before Phase 2 adds backtesting.

- [ ] **4.1** Create `scripts/benchmark_diagnostic.py` (or add a section to an
  existing runner script) that:
  1. Loads a saved `InferenceData` from a prior model run.
  2. Loads the corresponding data dict.
  3. Calls `extract_benchmark_revision()` for each March year within the
     estimation window (e.g., 2019–2025, excluding COVID).
  4. Calls `decompose_benchmark_revision()` for the same years.
  5. Prints a summary table comparing posterior mean ± std against actuals.

- [ ] **4.2** Expected output format (per spec §4.7, adapted for Phase 1):

  ```
  Benchmark Revision Diagnostic (Phase 1 — full-sample posterior)
  ================================================================
  March Y | Posterior Mean | Std  | 90% HDI          | Actual | Error
  --------|----------------|------|------------------|--------|------
  2019    |     -XXX       | XXX  | [-XXX, -XXX]     |  -505  |  XXX
  2022    |     +XXX       | XXX  | [+XXX, +XXX]     |  +462  |  XXX
  2023    |     -XXX       | XXX  | [-XXX, -XXX]     |  -105  |  XXX
  2024    |     -XXX       | XXX  | [-XXX, -XXX]     |  -598  |  XXX
  2025    |     -XXX       | XXX  | [-XXX, -XXX]     |  -862  |  XXX

  Decomposition (2025):
    Continuing-units divergence:  XXX (XX%)
    BD accumulation:              XXX (XX%)
  ```

- [ ] **4.3** Produce a simple plot: posterior density for 2024 and 2025 with
  vertical lines at the actual revisions. Save to `output/`.

- [ ] **4.4** Sanity checks to verify in the diagnostic output:
  - The posterior for years with QCEW data available should be tight
    (std < 100K) — QCEW anchoring pins the revision.
  - The BD accumulation term should dominate the decomposition
    (per spec §3.3).
  - The sign of the posterior mean should match the actual for most years.
  - If the posterior is wildly off for the full-sample case, there's a bug
    in the extraction logic (since the full-sample posterior has all
    information including QCEW).


---

## Task 5: Unit tests for extraction

- [ ] **5.1** `tests/test_benchmark.py` — test `extract_benchmark_revision()`:
  - Synthetic test: construct a mock `InferenceData` with known
    `g_total_nsa` values and a mock data dict with known CES NSA growth.
    Verify the revision calculation matches hand-computed expected value.
  - Window identification: verify correct time indices for April 2024 →
    March 2025.
  - Edge case: March year outside estimation window raises `ValueError`.

- [ ] **5.2** Test `decompose_benchmark_revision()`:
  - Verify `total ≈ cont_divergence + bd_accumulation` (approximate
    because the decomposition is additive in log-space but the formula
    is multiplicative in level-space — the cross term should be small).

- [ ] **5.3** Test `summarize_revision_posterior()`:
  - Known input array → verify mean, median, HDI bounds.


---

## Sequencing

```
Task 1 (lookup table)
  └──→ Task 2 (extraction module) ←── Task 3 (CES NSA audit)
         └──→ Task 4 (diagnostic script)
         └──→ Task 5 (unit tests)
```

Tasks 1 and 3 can proceed in parallel. Task 2 depends on both. Tasks 4 and 5
depend on Task 2.


---

## Out of scope (deferred to Phase 2 and Phase 3)

These items are described in the spec but explicitly excluded from Phase 1:

- **Backtesting loop** (spec §4.4): iterating over years × horizons with
  information-set censoring. Requires `as_of` parameter in `load_data()`,
  QCEW censoring, and the full evaluation framework. → Phase 2.
- **RMSE-by-horizon curve** (spec §4.7): requires the backtest loop. → Phase 2.
- **Comparative benchmarks** (spec §4.6): naive zero, prior-year revision,
  CES-ex-BD regression, BED-based, preliminary benchmark. → Phase 2.
- **Error-correction term** (spec §5.1): conditioning on prior-year surprise.
  Requires model changes. → Phase 3.
- **Preliminary benchmark as observation** (spec §5.2): noisy observation at
  T−5. Requires model changes. → Phase 3.
- **Historical CES vintage data** (spec §6.2): pre- vs post-benchmark prints
  for backtesting. → Phase 2 data requirement.
- **Published monthly B/D factors** (spec §6.3): for the enhancement in §5.1.
  → Phase 3 data requirement.