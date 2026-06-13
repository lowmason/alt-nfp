# Benchmark Prediction Implementation Spec

**Alt-NFP Nowcasting System — Predicting the Annual CES Benchmark Revision**

Version: 1.0 | Date: 2026-02-26 | Author: Lowell Mason

---

## 1. Background and Motivation

Each February, BLS publishes the annual benchmark revision alongside the January Employment Situation report. CES re-anchors March employment levels to QCEW-derived near-census counts, distributing the revision linearly across the preceding 21 months of NSA data via the wedge-back procedure. Recent revisions have been exceptionally large: −862K (March 2025), −598K (March 2024), −505K (March 2019), −902K (March 2009). These revisions are overwhelmingly a birth/death model phenomenon — the B/D model contributes roughly one-third to two-thirds of reported CES job gains in any given year.

Predicting this revision before the preliminary (published ~August/September) is the highest-value application of the payroll-provider nowcasting system. Providers observe continuing-units employment in real time, giving a direct read on the gap between what CES reports and what is actually happening. The target is to beat RMSE = 229K (Decker 2024, BED-based) at horizon T−6 or earlier.

### 1.1 Relationship to Release 1

The benchmark prediction is a derived quantity from the existing Release 1 national accuracy model — it requires no additional model parameters. The model's posterior over `g_total_nsa` is deterministically transformed into a posterior over the implied benchmark revision. All three implementation phases build on the Release 1 infrastructure without modifying the core state-space specification (except for Phase 3 enhancements).


---

## 2. Implementation Phases

The implementation follows three phases of increasing complexity. Each phase is independently deployable and testable.

| Phase | Name | Model changes | New modules/functions | Key deliverable |
|-------|------|---------------|----------------------|-----------------|
| 1 | Extraction | None | `src/alt_nfp/benchmark.py` | Implied revision posterior from existing model output |
| 2 | Backtesting | None (data censoring only) | Extensions to `load_data()`, `run_benchmark_backtest()` | RMSE-by-horizon curve across historical benchmark years |
| 3 | Enhancements | Two optional model terms | Extensions to `build_model()` | Error-correction and preliminary-as-observation |

### 2.1 Dependencies between phases

Phase 1 is a prerequisite for Phases 2 and 3. Phase 2 is a prerequisite for evaluating Phase 3. Phase 3's two enhancements (error-correction term and preliminary-as-observation) are independent of each other and can be implemented in either order.

```
Phase 1 (Extraction)
    │
    ├──► Phase 2 (Backtesting)
    │        │
    │        ├──► Phase 3a (Error-correction term)
    │        └──► Phase 3b (Preliminary-as-observation)
    │
    └──► Ad-hoc diagnostic use on current posterior
```


---

## 3. Phase 1 — Extraction

### 3.1 Objective

Extract the implied benchmark revision as a derived posterior quantity from an existing model fit. No model changes. This phase produces a new `src/alt_nfp/benchmark.py` module and a static lookup table of historical benchmark revisions.

### 3.2 New module: `src/alt_nfp/benchmark.py`

#### 3.2.1 `extract_benchmark_revision()`

**Signature:**

```python
def extract_benchmark_revision(
    idata: az.InferenceData,
    data: dict,
    march_year: int,
) -> np.ndarray:
    """Extract posterior samples of the implied benchmark revision.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior inference data from the Release 1 model.
    data : dict
        Output of load_data(), containing dates, growth rates,
        and level information.
    march_year : int
        The benchmark reference year Y (revision is for March Y,
        published February Y+1).

    Returns
    -------
    np.ndarray
        1-D array of posterior samples of the implied revision
        in thousands of jobs. Negative values indicate CES
        overstated employment.
    """
```

**Algorithm (five steps):**

1. **Identify the benchmark window.** Locate time indices for April `Y-1` through March `Y` in `data['dates']`. The window `W_Y` contains 12 monthly observations. Raise `ValueError` if any month in the window falls outside the model's time range.

2. **Cumulate posterior latent growth.** Extract `g_total_nsa` from `idata.posterior`. This has shape `(chains, draws, T)`. Sum over the benchmark window indices to get cumulative latent NSA growth: a `(chains, draws)` matrix.

3. **Cumulate observed CES NSA growth.** Sum the published (pre-benchmark) CES NSA growth rates over the same window from `data['g_ces_nsa']`. This is a known scalar. Use the latest-available CES vintage (v3 if available, otherwise the aggregate series).

4. **Convert to level-space revision.** Look up the CES NSA employment level at March `Y-1` (the prior benchmark anchor) from `data['levels']`. The implied revision in thousands is:

    ```
    R_hat = L_prior * (exp(cum_latent) - exp(cum_ces)) / 1000
    ```

    where `L_prior` is the March `Y-1` NSA level in jobs.

5. **Flatten.** Collapse chain and draw dimensions into a single 1-D posterior sample vector via `.reshape(-1)`.

**Edge cases:**

- If the benchmark window extends beyond the model's fitted period (e.g., predicting at T−12 when only part of the window has data), use only the available months. Document this in the returned metadata.
- If CES NSA growth is NaN for censored months (backtest scenario), use only months where CES data is available and note the partial window.

#### 3.2.2 `decompose_benchmark_revision()`

**Signature:**

```python
def decompose_benchmark_revision(
    idata: az.InferenceData,
    data: dict,
    march_year: int,
) -> dict[str, np.ndarray]:
    """Decompose revision into continuing-units divergence and BD accumulation.

    Returns
    -------
    dict
        Keys: 'total', 'cont_divergence', 'bd_accumulation'.
        Each value is a 1-D posterior sample array (000s).
    """
```

**Decomposition logic:**

```
R_hat ≈ L * [exp(Σ g_cont_nsa) - exp(Σ y_ces_nsa)]     # continuing-units divergence
       + L * [exp(Σ BD_t) - 1]                            # model BD accumulation
```

The continuing-units divergence captures "CES sample is off on continuing-units growth." The BD accumulation captures "CES birth/death model is wrong." In practice the BD term should dominate. A large continuing-units divergence may indicate provider sample representativeness issues.

#### 3.2.3 `BENCHMARK_REVISIONS` lookup table

A module-level constant dict mapping March reference year to the actual revision (000s, NSA total nonfarm):

```python
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
    2020: None,   # COVID — exclude from evaluation
    2021: None,   # COVID — exclude from evaluation
    2022: 462,
    2023: -105,
    2024: -598,
    2025: -862,
}
```

Source: BLS Employment Situation benchmark revision tables. This is static data collected once.

#### 3.2.4 `summarize_revision_posterior()`

**Signature:**

```python
def summarize_revision_posterior(
    samples: np.ndarray,
    actual: float | None = None,
) -> dict:
    """Compute summary statistics for the revision posterior.

    Returns
    -------
    dict
        Keys: 'mean', 'median', 'std', 'hdi_5', 'hdi_95',
        'hdi_10', 'hdi_90', 'actual', 'error' (if actual provided).
    """
```

Uses `arviz.hdi()` for HDI computation. The 90% HDI is the primary credible interval.

### 3.3 Data requirements

All data for Phase 1 is already available in the current system:

- `g_total_nsa` posterior samples (from `idata.posterior`)
- CES NSA growth rates (from `data['g_ces_nsa']`)
- CES NSA employment levels (from `data['levels']`)
- Date index (from `data['dates']`)

No new data ingestion is needed.

### 3.4 Testing

- **Unit test:** Synthetic posterior with known `g_total_nsa` constant. Verify that `extract_benchmark_revision()` returns the expected revision analytically.
- **Smoke test:** Run on the current full-sample posterior and verify the 2024 and 2025 revision predictions are in a plausible range (within ±500K of actual).
- **Decomposition identity:** Verify that `total ≈ cont_divergence + bd_accumulation` for every posterior draw (up to the linearization approximation).

### 3.5 Acceptance criteria

- `extract_benchmark_revision()` runs without error on current posterior output.
- Posterior mean for March 2025 is within ±300K of −862K (the actual).
- The 90% HDI for March 2025 contains −862K.
- Decomposition shows BD accumulation as the dominant term.


---

## 4. Phase 2 — Backtesting

### 4.1 Objective

Produce RMSE-by-horizon curves across historical benchmark years using only information available at each horizon. This requires implementing data censoring for QCEW (and ideally provider data), running the model at each `(year, horizon)` pair, and comparing posterior predictions against actual revisions.

### 4.2 Information set construction

Valid backtesting requires preventing lookahead bias across all data sources. The current system supports CES censoring via `censor_ces_from` in `load_data()` but lacks QCEW and provider censoring.

#### 4.2.1 QCEW censoring (highest priority)

QCEW publication schedule determines which quarters are available at each horizon:

| Reference quarter | Approximate publication | Content |
|---|---|---|
| Q1 (Jan–Mar Y) | September Y | Contains March Y — the benchmark month |
| Q2 (Apr–Jun Y) | December Y | |
| Q3 (Jul–Sep Y) | March Y+1 | Often just before benchmark |
| Q4 (Oct–Dec Y) | June Y+1 | After benchmark |

**Implementation:** Add an `as_of` parameter to `load_data()` (or a new wrapper function). When `as_of` is set, QCEW observations are masked to only include data from quarters whose publication date falls on or before `as_of`. Use the existing `QCEW_RELEASE_DATES` lookup in `src/alt_nfp/lookups/publication_dates.py` for exact dates, extended with historical dates for backtest years.

```python
def load_data(
    providers: list[ProviderConfig] | None = None,
    censor_ces_from: date | None = None,
    as_of: date | None = None,  # NEW: universal censoring cutoff
) -> dict:
```

When `as_of` is provided:
- CES observations are masked for months whose first-print publication date > `as_of`. Uses `CES_RELEASE_DATES` or the vintage dates pipeline.
- QCEW observations are masked for quarters whose publication date > `as_of`.
- Provider data is masked for months whose reference period + provider lag > `as_of` (typically ~1 month lag).
- Cyclical indicators are masked analogously using their respective publication lags.

The `censor_ces_from` parameter is retained for backward compatibility with the existing backtest module but `as_of` is the preferred interface for new code.

#### 4.2.2 Historical publication date extensions

The current `publication_dates.py` module covers roughly Oct 2024 forward for CES and 2025–2026 for QCEW. Backtesting requires extending these lookups backward. Two approaches:

1. **Scraped dates (preferred).** The existing `ingest/release_dates/` pipeline scrapes BLS news release HTML files for vintage dates. Run this pipeline for all historical CES and QCEW releases covering 2009–2025 and persist the results.

2. **Heuristic fallback.** For CES, approximate the first-print publication date as the first Friday of the month following the reference month. For QCEW, approximate as 5 months + 2 weeks after quarter end. These are rough but sufficient for backtest censoring where exact dates matter less than getting the quarter right.

The scraped approach is strongly preferred because it prevents edge-case leakage (e.g., government-shutdown-delayed releases like Sep/Oct 2025).

#### 4.2.3 CES vintage data

Ideally, backtesting should use the CES vintage that was available at each horizon (i.e., the pre-benchmark print for months that have since been benchmarked). The current system supports vintage-specific CES data (`g_ces_sa_v1`, `g_ces_sa_v2`, `g_ces_sa_v3`) but only for the recent period.

For initial backtesting, use the final (post-benchmark) CES data for the pre-benchmark window. This is conservative — it slightly understates the model's advantage because the model would have been comparing against the noisier pre-benchmark CES in real time. Flag this as a known limitation and address in a follow-up with full vintage CES data.

### 4.3 Horizon definition

For benchmark year Y (March Y reference, published February Y+1), five horizons:

| Horizon | Calendar cutoff (`as_of`) | Key information state |
|---|---|---|
| T−12 | ~April Y | Benchmark window just opened. Providers + CES only. No QCEW for this year. |
| T−9 | ~July Y | Pre-QCEW. Cyclical indicators informing BD. |
| T−6 | ~October Y | QCEW Q1 published (~September). Contains March Y. Revision largely determined. |
| T−3 | ~January Y+1 | QCEW Q2 published (~December). Full-year provider divergence visible. |
| T−1 | ~February Y+1 | QCEW Q3 may be available. All CES through December in hand. |

Specific `as_of` dates for each `(year, horizon)` pair should be computed programmatically from the publication date lookups. A helper function should map `(march_year, horizon_label)` → `date`.

### 4.4 Backtest procedure

#### 4.4.1 `run_benchmark_backtest()`

**Signature:**

```python
def run_benchmark_backtest(
    years: list[int] | None = None,
    horizons: list[str] | None = None,
    sampler_kwargs: dict | None = None,
) -> pl.DataFrame:
    """Run the benchmark revision backtest.

    Parameters
    ----------
    years : list[int], optional
        Benchmark years to evaluate. Defaults to all years with
        both provider data and actual revisions (excluding COVID).
    horizons : list[str], optional
        Horizon labels to evaluate. Defaults to
        ['T-12', 'T-9', 'T-6', 'T-3', 'T-1'].
    sampler_kwargs : dict, optional
        Override sampling kwargs (lighter for backtesting).

    Returns
    -------
    pl.DataFrame
        Columns: march_year, horizon, as_of_date, posterior_mean,
        posterior_std, hdi_5, hdi_95, hdi_10, hdi_90, actual,
        error, squared_error.
    """
```

**Outer loop:** Iterate over benchmark years. Default set: years where both provider data and actual benchmark revisions are available. Exclude 2020–2021 (COVID) from standard evaluation. Based on current data availability (provider data starts ~2019), the initial backtest covers 2019, 2022–2025 (4 years). If historical provider data can be reconstructed or simulated, extend to 2009–2019.

**Inner loop:** For each year, iterate over horizons. At each `(year, horizon)` pair:

1. Compute the `as_of` date for this horizon.
2. Call `load_data(as_of=as_of_date)` with appropriate censoring.
3. Build and sample the Release 1 model (use lighter sampling for backtest: fewer draws, fewer chains).
4. Call `extract_benchmark_revision(idata, data, march_year)`.
5. Call `summarize_revision_posterior(samples, actual=BENCHMARK_REVISIONS[march_year])`.
6. Record all summary statistics.

**Sampling configuration for backtesting:**

```python
BACKTEST_SAMPLER_KWARGS = {
    'draws': 500,
    'tune': 500,
    'chains': 2,
    'target_accept': 0.9,
    'cores': 2,
}
```

This is lighter than production (1000 draws, 1000 tune, 4 chains) to keep total runtime manageable across ~20 `(year, horizon)` runs.

#### 4.4.2 Evaluation metrics

**Primary metrics** (computed per-horizon, pooled across years):

- Mean error (bias): average of `(posterior_mean - actual)` across years.
- MAE: mean absolute error.
- RMSE: root mean squared error.
- 90% HDI coverage: fraction of years where actual falls within 90% HDI. Target: ~0.9.
- Average interval width: mean `(hdi_95 - hdi_5)`. Should shrink from T−12 to T−1.

**Comparative benchmarks:**

| Benchmark | Description | Implementation |
|---|---|---|
| Naive zero | "No revision expected." Market default. | Constant prediction of 0. |
| Prior-year revision | R̂_Y = R_{Y-1}. Simple persistence. | Shift BENCHMARK_REVISIONS by 1 year. |
| Cumulative published B/D | Predict revision as linear function of summed monthly B/D factors. | Requires B/D factor data (Phase 3 dependency). |
| CES-ex-BD regression | Per Cajner et al.: RMSE = 243K on 2008–2017. | Literature benchmark, no implementation needed. |
| BED-based prediction | Per Decker: adj. R² = 73%, RMSE = 229K in-sample. | Literature benchmark, no implementation needed. |
| Preliminary benchmark | BLS's own Aug/Sep estimate. Only available at T−5. | Collect preliminary benchmark history. |

**Target:** Beat RMSE = 229K (Decker) at T−6 or earlier.

#### 4.4.3 Output: RMSE-by-horizon curve

The headline deliverable of Phase 2. A table and plot showing:

| Horizon | RMSE (000s) | Coverage (90% HDI) | Avg interval width (000s) |
|---|---|---|---|
| T−12 | — | — | — |
| T−9 | — | — | — |
| T−6 | — | — | — |
| T−3 | — | — | — |
| T−1 | — | — | — |

The key finding would be: T−12 RMSE meaningfully beats naive zero (demonstrating providers detect B/D drift before official data confirms it). At T−6 (post-QCEW Q1), RMSE should drop sharply. At T−1, RMSE should be very small.

### 4.5 Data requirements beyond Phase 1

- **Historical QCEW publication dates.** Extend `QCEW_RELEASE_DATES` backward to cover all backtest years (2009–2025). Approximately: Q1 published in September, Q2 in December, Q3 in March, Q4 in June. Exact dates from BLS schedule archives.
- **Historical CES publication dates.** Extend `CES_RELEASE_DATES` backward. Available from the scraper pipeline or BLS schedule archives.
- **Historical CES vintages** (pre- vs. post-benchmark prints). Desirable but not blocking — use post-benchmark data with the known-conservative-bias caveat.
- **QCEW vintage-matched data.** The system needs to know which QCEW quarters were published at each `as_of` date. This is derived from the publication date lookup — no separate data source needed.

### 4.6 Testing

- **Censoring unit tests:** Verify that `load_data(as_of=...)` correctly masks QCEW observations. For a known `as_of` date, assert that QCEW observations after the publication cutoff are NaN.
- **Determinism test:** Run the same `(year, horizon)` pair twice with the same random seed. Verify identical results.
- **Monotonicity check:** For a given year, RMSE should generally decrease (or at least not systematically increase) as horizon moves from T−12 to T−1. Interval width should shrink.
- **Leakage test:** At T−12 for year Y, verify that no QCEW data from Q1 of year Y (published ~September Y) is present in the data dict.

### 4.7 Acceptance criteria

- Backtest completes for at least 3 benchmark years × 5 horizons without errors.
- 90% HDI coverage is between 0.7 and 1.0 (plausible calibration given small sample).
- RMSE at T−6 is documented and compared against Decker (229K) and Cajner et al. (243K).
- Interval width decreases monotonically from T−12 to T−1.
- No lookahead bias: QCEW and CES data are verifiably censored at each horizon.


---

## 5. Phase 3 — Model Enhancements

### 5.1 Overview

Two targeted, independent enhancements that modify `build_model()`. Each is evaluated by re-running the Phase 2 backtest and comparing RMSE-by-horizon curves against the unenhanced baseline.

### 5.2 Enhancement 3a: Benchmark error-correction term

#### 5.2.1 Motivation

Benchmark errors appear autocorrelated (−105K → −598K → −862K). Conditioning on the prior year's surprise — the difference between the actual revision and the model's predicted revision — may improve predictions, particularly at early horizons before QCEW pins down the current year.

#### 5.2.2 Specification

Modify the structural BD equation to include an error-correction term:

```
BD_t_adj = BD_t + φ_BM * (R_{Y-1} / (12 * L) - BD_bar_{Y-1}^model)
```

where:
- `R_{Y-1}` is the known prior-year benchmark revision (from `BENCHMARK_REVISIONS`).
- `L` is total employment level (approximately 155M).
- `BD_bar_{Y-1}^model` is the model's average BD over the prior benchmark window (April Y-2 through March Y-1).
- `φ_BM` is a new learnable parameter: the error-correction loading.

**Prior on `φ_BM`:** `Normal(0, 0.5)`. Centered at zero (no error-correction by default) with moderate uncertainty allowing the data to inform the degree of autocorrelation.

#### 5.2.3 Implementation

**Data requirement:** Published monthly B/D factors from archived CES birth/death supplemental tables. These are needed to compute the "surprise" — the difference between the actual revision and the model's implied BD over the prior window.

In practice, `R_{Y-1}` is known after the February Y publication and the model's prior-window BD can be computed from the posterior. For the first iteration, use a simplified version: condition on `R_{Y-1}` directly as a covariate in the BD equation, scaled to a monthly rate.

**Changes to `build_model()`:**

```python
# In the BD block, after computing bd_t:
if data.get('prior_benchmark_revision') is not None:
    phi_bm = pm.Normal('phi_bm', mu=0, sigma=0.5)
    # Monthly rate of prior revision surprise
    prior_revision_rate = data['prior_benchmark_revision'] / (12 * data['employment_level'])
    bd_t = bd_t + phi_bm * prior_revision_rate
```

**Changes to `load_data()`:**

Add `prior_benchmark_revision` to the data dict. For benchmark year Y, this is `BENCHMARK_REVISIONS[Y-1]`. For years without a prior revision (or COVID years), set to 0.0 (no correction).

#### 5.2.4 Evaluation

Re-run the full Phase 2 backtest with the error-correction term enabled. Compare RMSE-by-horizon against baseline. The enhancement is adopted if it improves RMSE at T−12 and T−9 without degrading T−6 and later (where QCEW dominates).

### 5.3 Enhancement 3b: Preliminary benchmark as observation

#### 5.3.1 Motivation

BLS publishes a preliminary benchmark estimate ~6 months before the final (August/September for a February publication). Recent preliminary-to-final gaps (−818K → −598K for 2024; −911K → −862K for 2025) suggest the preliminary systematically overstates downward revisions by 100–220K. The preliminary is a noisy but informative signal of the final revision, available at T−5.

#### 5.3.2 Specification

At T−5 and later horizons, add a measurement equation treating the preliminary as a biased, noisy observation of the model's implied revision:

```
y_prelim ~ Normal(R̂_Y + bias_prelim, σ_prelim²)
```

where:
- `y_prelim` is the published preliminary benchmark revision (a known scalar at T−5).
- `R̂_Y` is the model's implied revision (the derived quantity from Section 3).
- `bias_prelim` captures the systematic overshoot. Prior: `Normal(-150, 75)` based on the 100–220K overshoot pattern.
- `σ_prelim` captures idiosyncratic preliminary-to-final noise. Prior: `HalfNormal(100)`.

#### 5.3.3 Implementation

This is more complex than 3a because the observation is a function of the derived revision quantity, creating a circular dependency (the revision is derived from the posterior, but the preliminary observation informs the posterior).

**Approach:** Add the preliminary as an observed likelihood term in the PyMC model. The model already produces `g_total_nsa` as a latent quantity. Compute the implied revision within the model graph (sum of `g_total_nsa` over the benchmark window minus summed CES, converted to level-space). Then add the normal likelihood term.

```python
# Inside build_model(), after g_total_nsa is defined:
if data.get('preliminary_benchmark') is not None:
    # Compute implied revision within model graph
    window_idx = data['benchmark_window_indices']
    cum_latent = pt.sum(g_total_nsa[window_idx])
    cum_ces = data['cum_ces_nsa_benchmark']  # known scalar
    L_prior = data['prior_benchmark_level']
    implied_revision = L_prior * (pt.exp(cum_latent) - np.exp(cum_ces)) / 1000

    bias_prelim = pm.Normal('bias_prelim', mu=-150, sigma=75)
    sigma_prelim = pm.HalfNormal('sigma_prelim', sigma=100)

    pm.Normal(
        'obs_preliminary',
        mu=implied_revision + bias_prelim,
        sigma=sigma_prelim,
        observed=data['preliminary_benchmark'],
    )
```

**Changes to `load_data()`:**

Add `preliminary_benchmark` to the data dict. This is populated only when (a) the `as_of` date is at or after the preliminary publication date, and (b) a preliminary value is available for the target benchmark year.

**Preliminary benchmark lookup:**

```python
PRELIMINARY_BENCHMARKS: dict[int, tuple[float, date]] = {
    # march_year: (preliminary_revision_000s, publication_date)
    2024: (-818, date(2024, 8, 21)),
    2025: (-911, date(2025, 8, 20)),
}
```

This is a small, slowly-growing lookup. Update annually when each new preliminary is published.

#### 5.3.4 Evaluation

Re-run backtest at T−5 and later horizons with the preliminary-as-observation term. Compare against baseline. The enhancement should tighten the posterior at T−5 and improve RMSE. Because the preliminary has limited history (2019–2025 with COVID gap), the prior on `bias_prelim` and `σ_prelim` does most of the work.

### 5.4 Testing for Phase 3

- **Prior predictive checks:** For both enhancements, run prior predictive simulation to verify that the new terms don't dominate the prior predictive distribution or create pathological behavior.
- **Posterior recovery:** On synthetic data with known revision, verify that the model with enhancements recovers the true revision at least as well as the baseline.
- **Comparative RMSE:** The primary test is the Phase 2 backtest comparison. Each enhancement should improve (or at worst not degrade) RMSE at relevant horizons.

### 5.5 Acceptance criteria

- Enhancement 3a: RMSE at T−12 improves by >5% relative to baseline, or there is clear evidence of autocorrelation exploitation in the posterior.
- Enhancement 3b: RMSE at T−5 improves relative to baseline when preliminary data is available.
- Neither enhancement degrades RMSE at T−1 (where QCEW anchoring should dominate regardless).
- Sampler diagnostics (R-hat, ESS, divergences) remain acceptable with enhancements enabled.


---

## 6. Data Requirements Summary

### 6.1 Already available (no new collection)

| Data | Source | Used in |
|---|---|---|
| CES index data (SA/NSA) | `data/ces_index.csv` | All phases |
| QCEW index data (NSA) | `data/qcew_index.csv` | All phases |
| Payroll provider indices | `data/*.csv` or `.parquet` | All phases |
| Provider birth-rate data | Provider-specific files | All phases |
| Cyclical indicators | `data/` directory | All phases |
| CES/QCEW publication dates (recent) | `src/alt_nfp/lookups/publication_dates.py` | Phase 2 |

### 6.2 Needed for Phase 1 (collect once)

| Data | Source | Format |
|---|---|---|
| Historical benchmark revisions (2009–2025) | BLS Employment Situation releases | Static dict in `benchmark.py` |

### 6.3 Needed for Phase 2

| Data | Source | Priority | Format |
|---|---|---|---|
| Historical QCEW publication dates (2009–2025) | BLS schedule archives or heuristic | High | Extend `QCEW_RELEASE_DATES` |
| Historical CES publication dates (2009–2025) | Scraper pipeline / BLS schedule | High | Extend `CES_RELEASE_DATES` |
| Historical CES vintages (pre/post-benchmark) | BLS LABSTAT API or archived releases | Medium | Vintage-tagged growth rates |
| QCEW vintage-matched availability | Derived from publication date lookup | High | No separate data |

### 6.4 Needed for Phase 3

| Data | Source | Enhancement | Format |
|---|---|---|---|
| Published monthly B/D factors | Archived CES B/D supplemental tables | 3a | Time series |
| Preliminary benchmark revisions | BLS preliminary benchmark publications | 3b | Static dict |
| Preliminary benchmark publication dates | BLS schedule | 3b | Static dict |


---

## 7. File and Module Layout

```
src/alt_nfp/
├── benchmark.py            # NEW: Phase 1 — revision extraction + lookup table
├── benchmark_backtest.py   # NEW: Phase 2 — backtest loop + evaluation
├── data.py                 # MODIFIED: Phase 2 — add as_of censoring
├── model.py                # MODIFIED: Phase 3 — error-correction + preliminary obs
├── config.py               # MODIFIED: add backtest sampling config
├── lookups/
│   ├── publication_dates.py    # MODIFIED: Phase 2 — extend historical dates
│   └── benchmark_revisions.py  # NEW (or in benchmark.py): static lookup
└── plots/
    └── benchmark_plots.py  # NEW: RMSE-by-horizon curves, revision fan charts
```

### 7.1 New files

**`src/alt_nfp/benchmark.py`** (~150 lines)
Core extraction logic. Functions: `extract_benchmark_revision()`, `decompose_benchmark_revision()`, `summarize_revision_posterior()`, `BENCHMARK_REVISIONS`.

**`src/alt_nfp/benchmark_backtest.py`** (~250 lines)
Backtest orchestration. Functions: `run_benchmark_backtest()`, `compute_backtest_metrics()`, `horizon_to_as_of()`, `build_comparative_benchmarks()`.

**`src/alt_nfp/plots/benchmark_plots.py`** (~200 lines)
Visualization. Functions: `plot_rmse_by_horizon()`, `plot_revision_fan_chart()`, `plot_revision_decomposition()`.

### 7.2 Modified files

**`src/alt_nfp/data.py`** — Add `as_of` parameter to `load_data()` with QCEW censoring logic. Backward compatible (`as_of=None` preserves current behavior).

**`src/alt_nfp/model.py`** — Phase 3 only. Add optional error-correction term and preliminary-as-observation likelihood, gated by data dict keys.

**`src/alt_nfp/lookups/publication_dates.py`** — Extend `QCEW_RELEASE_DATES` and `CES_RELEASE_DATES` backward to 2009.


---

## 8. Runtime Estimates

| Phase | Operation | Approx. runtime | Notes |
|---|---|---|---|
| 1 | `extract_benchmark_revision()` | <1 second | Pure NumPy posterior arithmetic |
| 2 | Single `(year, horizon)` backtest run | ~3–5 minutes | Light sampling (500 draws, 2 chains) |
| 2 | Full backtest (4 years × 5 horizons) | ~1–2 hours | 20 model fits, parallelizable by year |
| 2 | Full backtest (extended, 12 years × 5 horizons) | ~4–6 hours | 60 model fits |
| 3 | Re-run backtest with enhancement | Same as Phase 2 | One additional parameter; negligible overhead |


---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small backtest sample (4 years without COVID) | Unreliable RMSE estimates, wide confidence bands on metrics | Extend to pre-provider era using simulated provider signals or provider-free model variant |
| QCEW publication date uncertainty for historical years | Potential minor leakage in backtest censoring | Use conservative (later) publication date heuristics; validate against BLS archives |
| CES vintage data unavailable for backtest years | Slightly overstates model advantage (compares against post-benchmark CES) | Document as known conservative bias; collect vintage data as follow-up |
| Phase 3 overfitting with small training set | Error-correction term may capture noise rather than signal | Strong priors, leave-one-out cross-validation within backtest, comparison against prior predictive |
| Computational cost of full backtest | Developer iteration speed | Light sampling config, checkpoint intermediate results, parallelize across years |


---

## 10. Success Criteria

The benchmark prediction module is considered successful if:

1. **Phase 1:** The implied revision posterior is extractable from any Release 1 model fit, with the March 2025 prediction containing the actual (−862K) within the 90% HDI.

2. **Phase 2:** The RMSE-by-horizon curve demonstrates that T−12 RMSE meaningfully beats naive zero (proving providers detect B/D drift before official confirmation). At T−6, RMSE approaches or beats 229K (Decker).

3. **Phase 3:** At least one enhancement improves RMSE at its target horizon without degrading other horizons.

4. **Overall:** The system produces actionable benchmark revision predictions 6–12 months before BLS publication, with properly calibrated uncertainty, implementable as a recurring Bloomberg product for the annual benchmark cycle.