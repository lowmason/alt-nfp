# National-Level Model Improvements Implementation Spec

**Alt-NFP Nowcasting System — Evaluation Infrastructure & Model Enhancements**

Version: 1.0 \| Date: 2026-02-28 \| Author: Lowell Mason

------------------------------------------------------------------------

## 1. Overview

This spec covers all open national-level model improvements for the alt-NFP nowcasting system. It encompasses both the monthly NFP nowcasting model (Release 1) and the benchmark revision prediction model, prioritizing nowcast improvements.

### 1.1 Training Window

The model trains on **2003–present**. QCEW data is treated as final for 2003–2016 (QCEW revision history is only available from 2017). This is a known limitation for backtesting over the pre-2017 period — it introduces mild QCEW lookahead that slightly overstates model performance. The tradeoff is worthwhile: the 2003–present window captures the 2008 recession, which is the canonical example of BD model failure and essential for identifying cyclical parameters.

### 1.2 Phase Structure

| Phase | Scope                     | Model changes         | Prerequisite     |
|-------|---------------------------|-----------------------|------------------|
| 1     | Evaluation Infrastructure | None                  | —                |
| 2     | Model Enhancements        | Yes (`build_model()`) | Phase 1 baseline |

### 1.3 Relationship to Existing Specs

This spec builds on `specs/releases_spec.md` (Release 1 completion tasks §2.3) and `specs/benchmark_spec.md` (Phase 3 enhancements §5). It does not modify the Release 2/3 roadmap. All changes are confined to the national accuracy model.

### 1.4 Out of Scope

-   **Representativeness correction** (requires provider microdata from private repo). Tracked as a TODO; implementation happens elsewhere.
-   Industry decomposition (Release 2).
-   Geographic decomposition (Release 3).
-   High-frequency business registration data (back burner; revisit later).

------------------------------------------------------------------------

## 2. Phase 1: Evaluation Infrastructure

### 2.1 Objective

Establish a solid evaluation baseline before any model changes. Every Phase 2 enhancement gets measured against this baseline. No modifications to `build_model()`.

### 2.2 Leakage-Safe Backtesting

**Priority: Highest.** Blocks proper evaluation of all subsequent work.

#### 2.2.1 Current State

The `as_of` parameter in `panel_to_model_data()` censors CES observations by `vintage_date`. QCEW censoring is partially implemented — observations are filtered by publication date, but historical QCEW publication dates may be incomplete. Provider data censoring relies on `vintage_date` when populated.

#### 2.2.2 Required Changes

**Extend historical publication dates.** `lookups/publication_dates.py` currently has recent CES and QCEW release dates. Extend both backward to 2003:

-   CES: scrape from BLS Employment Situation news release archive, or use the existing scraper infrastructure in `ingest/release_dates.py`. Validate against known dates.
-   QCEW: quarterly publication dates follow a regular schedule (\~5 months after quarter end). Use heuristic for pre-2017, validate against BLS QCEW release calendar for 2017+.

**Validate provider censoring.** Provider observations should be excluded when `vintage_date > as_of`. Confirm that `panel_to_model_data()` handles this correctly for all provider sources, including birth-rate data.

**Validate cyclical indicator censoring.** The `_CYCLICAL_PUBLICATION_LAGS` dict in `panel_adapter.py` applies publication lags per indicator. Verify that the lag values are correct and that the censoring logic properly zeroes out unavailable observations.

#### 2.2.3 Deliverables

-   Extended `CES_RELEASE_DATES` and `QCEW_RELEASE_DATES` covering 2003–present.
-   Unit tests confirming that `panel_to_model_data(as_of=...)` correctly censors all sources.
-   A backtesting smoke test: run the current model at two different `as_of` dates and verify that later dates produce tighter posteriors.

#### 2.2.4 Files Modified

-   `src/alt_nfp/lookups/publication_dates.py`
-   `src/alt_nfp/panel_adapter.py` (if censoring bugs found)
-   `tests/test_benchmark_backtest.py` (extend `TestAsOfCensoring`)

### 2.3 Cyclical Indicator Data Loading

**Priority: High.** Both infrastructure (data pipeline) and a model improvement (the `φ_3` covariates are already in `build_model()` but depend on data availability).

#### 2.3.1 Indicator Suite

Four cyclical indicators, all entering the BD equation as centered covariates:

| Indicator | Source | Frequency | Publication lag | Vintage tracking |
|--------------|--------------|--------------|---------------|----------------|
| Initial unemployment claims | DOL via FRED | Weekly | \~1 week | Not needed (essentially unrevised) |
| Chicago Fed NFCI | Chicago Fed via FRED | Weekly | \~1 week | Not needed (not revised) |
| Census Business Formation Statistics | Census via FRED | Monthly | \~2 months | Not needed (minor revisions) |
| JOLTS openings or hires | BLS via FRED | Monthly | \~2 months | Check empirically; use ALFRED if material |

#### 2.3.2 Current State

`CYCLICAL_INDICATORS` in `config.py` defines three indicators (claims, NFCI, business applications). `_load_cyclical_indicators()` in `panel_adapter.py` loads and centers them. Publication-lag censoring is applied when `as_of` is provided.

#### 2.3.3 Required Changes

**Add JOLTS.** Add a fourth entry to `CYCLICAL_INDICATORS` for JOLTS openings (or hires — evaluate both series and pick the one with stronger BD covariance). Update `_load_cyclical_indicators()` to handle the new series.

**JOLTS revision assessment.** Before deciding on vintage tracking: download JOLTS first-release vs. final values from ALFRED for 2003–present. Compare revision magnitudes to cross-sectional variation. If revisions are small relative to the centered covariate range, skip vintage tracking and use final values with publication-lag censoring only. Document the finding.

**Verify data coverage.** Ensure all four indicators have data back to 2003. JOLTS begins December 2000, so coverage is fine. BFS begins July 2004 — the model handles missing covariates gracefully (centered values default to zero), so partial coverage is acceptable.

**Verify publication-lag values.** Cross-check `_CYCLICAL_PUBLICATION_LAGS` against actual release schedules. Current values may be approximate.

#### 2.3.4 Deliverables

-   JOLTS added to `CYCLICAL_INDICATORS` with correct FRED series ID and publication lag.
-   JOLTS revision assessment documented (inline comment or short markdown note).
-   Unit tests for `_load_cyclical_indicators()` covering all four indicators.
-   Verification that all indicators load correctly over the 2003–present window.

#### 2.3.5 Files Modified

-   `src/alt_nfp/config.py` (add JOLTS to `CYCLICAL_INDICATORS`)
-   `src/alt_nfp/panel_adapter.py` (update `_load_cyclical_indicators()` if needed)
-   `tests/` (new or extended tests for cyclical loading)

### 2.4 QCEW Sigma Sensitivity Automation

**Priority: Medium.** Diagnostic tool; does not block Phase 2 but informs prior calibration.

#### 2.4.1 Current State

QCEW noise is fixed at `σ_QCEW,M3 = 0.0005` and `σ_QCEW,M12 = 0.0015`. Sensitivity analysis (varying these by 0.5×/1×/2×) is currently manual.

#### 2.4.2 Required Changes

Create `src/alt_nfp/sensitivity.py` with a function that:

1.  Accepts a list of QCEW noise scale factors (e.g., `[0.5, 1.0, 2.0]`).
2.  For each scale, rebuilds the model with scaled QCEW noise and samples.
3.  Extracts key parameters: `φ`, `μ_g`, `σ_η`, provider loadings `λ_p`, provider biases `α_p`, BD coefficients.
4.  Produces a summary table of posterior means and stds across scales.
5.  Optionally produces comparison plots.

#### 2.4.3 Deliverables

-   `src/alt_nfp/sensitivity.py` with `run_qcew_sensitivity()`.
-   At least one test confirming the function runs without error on a short sample.

### 2.5 Precision Budget & Provider Weight Extraction

**Priority: Medium.** Diagnostic tool; does not block Phase 2.

#### 2.5.1 Required Changes

Add a function (in `checks.py` or a new `diagnostics.py`) that extracts from a fitted model:

1.  **Effective precision share** for each data source (QCEW, CES by vintage, each provider). Computed from posterior noise parameters: `precision_i = 1/σ_i²`, `share_i = precision_i / Σ precision_j`.
2.  **Provider signal quality ranking** by posterior mean of `λ_p` (closer to 1.0 = better signal).
3.  **Provider bias summary** by posterior mean of `α_p`.

Output as a Polars DataFrame and optionally print a formatted summary table.

#### 2.5.2 Deliverables

-   Function `compute_precision_budget(idata, data) -> pl.DataFrame`.
-   Integration into `scripts/benchmark_diagnostic.py` or a new diagnostic script.

### 2.6 TODO: Representativeness Correction

Representativeness correction requires provider microdata (cell-level signals at supersector × Census region), which lives in a separate private repository. The national model consumes a QCEW-weighted composite of cell-level signals rather than raw provider indices.

**Action:** Add a TODO marker in `config.py` or `panel_adapter.py` noting the dependency. No implementation in this repo. The interface contract is:

``` python
# Expected input: a single repr-corrected national composite per provider per month.
# Computed externally as: y_p_t = Σ_c w_c_QCEW * y_p_c_t
# where c indexes supersector × Census region cells.
```

------------------------------------------------------------------------

## 3. Phase 2: Model Enhancements

### 3.1 Objective

Improve both the nowcasting model and the benchmark prediction model through targeted additions to `build_model()`. Each enhancement is implemented, evaluated against the Phase 1 baseline via backtesting, and accepted or rejected independently.

### 3.2 Evaluation Protocol

For each enhancement:

1.  Implement the model change, gated by a data dict key so the baseline model is recoverable.
2.  Run the benchmark backtest (`run_benchmark_backtest()`) with and without the enhancement.
3.  Run nowcast backtesting (one-step-ahead or rolling-window RMSE against CES first print).
4.  Check sampler diagnostics: R-hat, ESS, divergences.
5.  Accept if: RMSE improves at target horizons without degrading other horizons, and sampler diagnostics remain clean.

### 3.3 Enhancement 2a: Era-Specific Latent State Parameters

**Priority: High (nowcast).** Addresses structural instability in the 2003–present training window.

#### 3.3.1 Motivation

Mean employment growth `μ_g` and AR(1) persistence `φ` differ across macroeconomic eras. The pre-GFC trend (\~150K/month) is substantially higher than the post-GFC trend. Forcing a single `μ_g` and `φ` across the full sample biases the latent state dynamics toward an average that doesn't represent any actual era.

#### 3.3.2 Specification

Replace scalar `μ_g` and `φ` with era-indexed vectors using known breakpoints:

| Era        | Index | Period                  | Rationale                      |
|----------------|----------------|-----------------|-----------------------|
| Pre-GFC    | 0     | 2003-01 through 2008-12 | Expansion + onset of recession |
| Post-GFC   | 1     | 2009-01 through 2019-12 | Recovery and expansion         |
| Post-COVID | 2     | 2020-01 through present | Pandemic disruption + recovery |

Breakpoints are fixed, not estimated.

``` python
N_ERAS = 3
ERA_BREAKS = [date(2009, 1, 1), date(2020, 1, 1)]

# In build_model():
era_idx = np.array([_date_to_era(d) for d in data['dates']])  # length-T int array

mu_g_era = pm.Normal('mu_g_era', mu=0.001, sigma=0.005, shape=N_ERAS)
phi_raw_era = pm.Beta('phi_raw_era', alpha=18, beta=2, shape=N_ERAS)

mu_g = mu_g_era[era_idx]
phi = phi_raw_era[era_idx]
```

The AR(1) transition becomes:

```         
μ_t^cont = mu_g[t] + phi[t] * (μ_{t-1}^cont - mu_g[t]) + σ_η * ε_t
```

Note: at era boundaries, `mu_g[t]` and `phi[t]` switch discretely. This is intentional — the model does not interpolate between eras. The latent state `μ_{t-1}^cont` carries forward continuously; only the dynamics parameters change.

#### 3.3.3 Priors

-   `mu_g_era ~ N(0.001, 0.005²)` — same prior per era, data-driven separation.
-   `phi_raw_era ~ Beta(18, 2)` — same prior per era; mode ≈ 0.94.
-   Alternative: hierarchical prior with shared hyperparameters across eras. This is cleaner but adds parameters. Start with independent priors; move to hierarchical if posteriors are implausibly separated.

#### 3.3.4 Implementation

**Changes to `build_model()`:**

-   Replace `mu_g = pm.Normal(...)` scalar with `mu_g_era = pm.Normal(..., shape=N_ERAS)`.
-   Replace `phi_raw = pm.Beta(...)` scalar with `phi_raw_era = pm.Beta(..., shape=N_ERAS)`.
-   Add `era_idx` array to the data dict in `panel_to_model_data()`.
-   Modify the `pytensor.scan` AR(1) step function to index into era-specific parameters.

**Changes to `panel_adapter.py`:**

-   Add `_date_to_era()` helper and `era_idx` computation in `panel_to_model_data()`.
-   Add `ERA_BREAKS` to `config.py`.

**Changes to diagnostics:**

-   Update `checks.py` to report era-specific posterior summaries for `mu_g` and `φ`.

#### 3.3.5 Evaluation

-   Compare nowcast RMSE (one-step-ahead against CES first print) with and without era-specific parameters.
-   Compare benchmark backtest RMSE by horizon.
-   Inspect era-specific posteriors: do they separate meaningfully? If `mu_g_era` posteriors substantially overlap across all eras, the enhancement adds complexity without value.
-   Check for boundary artifacts at era transitions (January 2009, January 2020).

#### 3.3.6 Computational Cost

Adds 4 parameters (2 `mu_g_era` + 2 `phi_raw_era` beyond the originals). No new sampling challenges. Negligible runtime impact.

#### 3.3.7 Files Modified

-   `src/alt_nfp/config.py` (add `ERA_BREAKS`, `N_ERAS`)
-   `src/alt_nfp/panel_adapter.py` (add `era_idx` to data dict)
-   `src/alt_nfp/model.py` (era-indexed `mu_g` and `phi`)
-   `src/alt_nfp/checks.py` (era-specific diagnostics)
-   `tests/test_model.py` (test model construction with era parameters)

### 3.4 Enhancement 2b: Cyclical Indicators Fully Wired

**Priority: High (nowcast).** Depends on Phase 1 §2.3 data loading being complete.

#### 3.4.1 Current State

The `build_model()` code already handles cyclical indicators via the `cyclical_keys` loop. If Phase 1 delivers all four indicators with correct data loading and censoring, this enhancement may require no additional model changes — just verification that the indicators flow through correctly and evaluation of their impact.

#### 3.4.2 Required Changes

If Phase 1 fully completes the data pipeline, the remaining work is:

1.  **Evaluate individual indicator contributions.** Run the model with each indicator individually and all together. Compare BD component accuracy.
2.  **Consider indicator-specific priors.** Currently all cyclical loadings share `φ_3 ~ N(0, 0.3²)`. If some indicators have known sign (e.g., higher claims → lower BD), consider informative priors.
3.  **JOLTS series selection.** Evaluate JOLTS openings vs. hires as the BD covariate. Pick the series with stronger out-of-sample BD prediction.

#### 3.4.3 Evaluation

-   BD component accuracy against annual benchmark revisions (the key metric for cyclical indicators).
-   Does adding JOLTS improve BD accuracy beyond the three existing indicators?
-   Nowcast RMSE improvement, particularly during turning-point months.

### 3.5 Enhancement 2c: National-Level Time-Varying Provider Bias

**Priority: Medium (nowcast).** Addresses the "static provider bias" limitation from Release 1.

#### 3.5.1 Motivation

Provider composition drifts over time as clients are gained and lost. A static `α_p` absorbs the average bias but cannot track this drift. Release 3 introduces full time-varying bias with QCEW error-correction at the cell level, but a simpler national-level version captures the main effect without industry/geography decomposition.

#### 3.5.2 Specification

Replace static `α_p` with a slow Gaussian random walk:

```         
α_p(t) = α_p(t-1) + σ_α_drift,p * ω_t,    ω_t ~ N(0, 1)
α_p(0) ~ N(0, 0.005²)
σ_α_drift,p ~ HalfNormal(0.0005)
```

The tight prior on `σ_α_drift` ensures the bias evolves slowly — on the order of 0.0005 per month, or \~0.006 per year. This prevents the random walk from absorbing signal that should go to the latent state.

The provider measurement equation becomes:

```         
y_{p,t}^G = α_p(t) + λ_p * g_t^{cont,NSA} + ε_{p,t}
```

#### 3.5.3 Implementation

**Changes to `build_model()`:**

Replace the static `alpha_p = pm.Normal(...)` with a GRW:

``` python
sigma_alpha_drift_p = pm.HalfNormal(f'sigma_alpha_drift_{pp_name}', sigma=0.0005)
alpha_p_innovations = pm.Normal(f'alpha_innovations_{pp_name}', 0, 1, shape=T)
alpha_p_0 = pm.Normal(f'alpha_0_{pp_name}', mu=0, sigma=0.005)
alpha_p = pm.Deterministic(
    f'alpha_{pp_name}',
    alpha_p_0 + pt.cumsum(sigma_alpha_drift_p * alpha_p_innovations)
)
```

For AR(1) error providers, the conditional mean uses `alpha_p[t]` instead of scalar `alpha_p`.

**Parameterization note:** Use non-centered parameterization (innovations + cumsum) to avoid the funnel geometry that centered random walks create. This is critical for NUTS efficiency.

#### 3.5.4 Evaluation

-   Does time-varying bias improve nowcast RMSE?
-   Inspect `α_p(t)` trajectories: do they show plausible slow drift, or do they overfit to noise?
-   Compare `σ_α_drift` posterior to prior: if the posterior concentrates near zero, the data doesn't support time-varying bias and the static version is preferred.
-   Check sampler diagnostics carefully — random walks can create sampling difficulties even with non-centered parameterization.

#### 3.5.5 Computational Cost

Adds `2 * P` parameters (one `σ_α_drift` and one `α_0` per provider) plus `T * P` innovation parameters. For 2 providers and T ≈ 270 months, this is \~544 new parameters. The non-centered parameterization should keep sampling efficient, but monitor ESS and runtime.

#### 3.5.6 Acceptance Criteria

-   Nowcast RMSE improves or is neutral.
-   `α_p(t)` trajectories are smooth (no high-frequency oscillation).
-   `σ_α_drift` posterior is bounded away from the prior upper tail (the data is informative).
-   No sampler pathologies (divergences, low ESS on drift parameters).

If the posterior on `σ_α_drift` concentrates near zero for all providers, reject the enhancement and keep static bias.

#### 3.5.7 Files Modified

-   `src/alt_nfp/model.py` (replace static `alpha_p` with GRW)
-   `src/alt_nfp/checks.py` (add `α_p(t)` trajectory plots)
-   `tests/test_model.py` (test model construction with time-varying bias)

### 3.6 Enhancement 2d: BD Error-Correction Term

**Priority: High (benchmark).** Phase 3a from `specs/benchmark_spec.md` §5.2.

#### 3.6.1 Specification

Add an error-correction term to the structural BD equation that conditions on the prior year's known benchmark revision:

```         
BD_t_adj = BD_t + φ_BM * (R_{Y-1} / (12 * L))
```

where:

-   `R_{Y-1}` is the known prior-year benchmark revision (from `BENCHMARK_REVISIONS`).
-   `L` is total employment level (\~155M).
-   `φ_BM ~ N(0, 0.5)` — centered at zero (no correction by default).

The term is a constant within each benchmark year (April Y through March Y+1), representing the model's prior belief about whether BD drift is persisting.

#### 3.6.2 Implementation

**Changes to `panel_to_model_data()`:**

Add `prior_benchmark_revision` to the data dict. For benchmark year Y, this is `BENCHMARK_REVISIONS[Y-1]`. For years without a prior revision, or COVID years, set to `0.0`.

``` python
# Compute monthly rate of prior benchmark revision
if data.get('prior_benchmark_revision') is not None:
    prior_revision_rate = data['prior_benchmark_revision'] / (12 * data['employment_level'])
else:
    prior_revision_rate = 0.0
```

**Changes to `build_model()`:**

``` python
if data.get('prior_benchmark_revision') is not None:
    phi_bm = pm.Normal('phi_bm', mu=0, sigma=0.5)
    prior_revision_rate = data['prior_benchmark_revision'] / (12 * data['employment_level'])
    bd_t = bd_t + phi_bm * prior_revision_rate
```

#### 3.6.3 Evaluation

Per `specs/benchmark_spec.md` §5.5:

-   RMSE at T−12 improves by \>5% relative to baseline.
-   No degradation at T−1 (where QCEW dominates).
-   Sampler diagnostics remain clean.

#### 3.6.4 Files Modified

-   `src/alt_nfp/model.py`
-   `src/alt_nfp/panel_adapter.py`
-   `tests/test_model.py`

### 3.7 Enhancement 2e: Preliminary Benchmark as Observation

**Priority: Medium (benchmark).** Phase 3b from `specs/benchmark_spec.md` §5.3.

#### 3.7.1 Specification

At T−5 and later horizons, treat the BLS preliminary benchmark estimate as a biased, noisy observation of the model's implied revision:

```         
y_prelim ~ N(R̂_Y + bias_prelim, σ_prelim²)
```

where:

-   `y_prelim` is the published preliminary benchmark revision.
-   `R̂_Y` is the model's implied revision (computed within the model graph).
-   `bias_prelim ~ N(-150, 75)` — captures systematic overshoot (100–220K pattern).
-   `σ_prelim ~ HalfNormal(100)` — idiosyncratic preliminary-to-final noise.

#### 3.7.2 Implementation

**New lookup:**

``` python
PRELIMINARY_BENCHMARKS: dict[int, tuple[float, date]] = {
    # march_year: (preliminary_revision_000s, publication_date)
    2024: (-818, date(2024, 8, 21)),
    2025: (-911, date(2025, 8, 20)),
}
```

**Changes to `panel_to_model_data()`:**

Add `preliminary_benchmark`, `benchmark_window_indices`, `cum_ces_nsa_benchmark`, and `prior_benchmark_level` to the data dict when `as_of >= preliminary_publication_date`.

**Changes to `build_model()`:**

``` python
if data.get('preliminary_benchmark') is not None:
    window_idx = data['benchmark_window_indices']
    cum_latent = pt.sum(g_total_nsa[window_idx])
    cum_ces = data['cum_ces_nsa_benchmark']
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

#### 3.7.3 Evaluation

Per `specs/benchmark_spec.md` §5.5:

-   RMSE at T−5 improves relative to baseline.
-   No degradation at T−1.
-   Limited backtest history (2024–2025 only for preliminary data), so prior does most of the work. Evaluate primarily via posterior predictive checks on the available years.

#### 3.7.4 Files Modified

-   `src/alt_nfp/lookups/benchmark_revisions.py` (add `PRELIMINARY_BENCHMARKS`)
-   `src/alt_nfp/model.py`
-   `src/alt_nfp/panel_adapter.py`
-   `tests/test_model.py`

------------------------------------------------------------------------

## 4. Implementation Order

### 4.1 Phase 1 (no model changes)

| Step | Task | Blocks |
|----------------------|----------------------|-----------------------------|
| 1.1 | Extend historical publication dates to 2003 | 1.2, all Phase 2 evaluation |
| 1.2 | Validate and fix leakage-safe backtesting for all sources | All Phase 2 evaluation |
| 1.3 | Add JOLTS to cyclical indicator pipeline | 2b |
| 1.4 | Verify all cyclical indicator data loading and censoring | 2b |
| 1.5 | QCEW sigma sensitivity automation | Independent |
| 1.6 | Precision budget / provider weight extraction | Independent |
| 1.7 | TODO marker for representativeness correction | — |

Steps 1.1–1.2 are sequential (dates needed before censoring validation). Steps 1.3–1.7 can be parallelized after 1.2.

### 4.2 Phase 2 (model changes, sequential)

| Step | Enhancement | Evaluation target |
|----------------|-----------------------|---------------------------------|
| 2a | Era-specific `μ_g` and `φ` | Nowcast RMSE |
| 2b | Cyclical indicators fully wired + JOLTS | BD accuracy, nowcast RMSE at turning points |
| 2c | Time-varying provider bias | Nowcast RMSE |
| 2d | BD error-correction term | Benchmark RMSE at T−12 |
| 2e | Preliminary benchmark as observation | Benchmark RMSE at T−5 |

Each step follows the evaluation protocol in §3.2: implement → backtest → accept/reject before moving to the next.

------------------------------------------------------------------------

## 5. Data Requirements

### 5.1 Already Available

| Data                                | Source                   | Used in     |
|-------------------|-------------------------|----------------------------|
| CES index data (SA/NSA)             | Vintage store            | All         |
| QCEW index data (NSA)               | Vintage store            | All         |
| Payroll provider indices            | Provider files           | All         |
| Provider birth-rate data            | Provider files           | All         |
| Claims, NFCI, BFS                   | FRED                     | Phase 1, 2b |
| Historical benchmark revisions      | `benchmark_revisions.py` | 2d          |
| CES/QCEW publication dates (recent) | `publication_dates.py`   | Phase 1     |

### 5.2 Needed

| Data | Source | Phase | Priority |
|-----------------|-----------------|-----------------|----------------------|
| Historical CES publication dates (2003–2016) | BLS archive scraper | 1.1 | High |
| Historical QCEW publication dates (2003–2016) | BLS schedule / heuristic | 1.1 | High |
| JOLTS openings and hires series | FRED (JTSJOL, JTSHIL) | 1.3 | High |
| JOLTS revision history (for assessment) | ALFRED | 1.3 | Medium |
| Preliminary benchmark revisions | BLS publications | 2e | Medium |
| Published monthly B/D factors | Archived CES B/D tables | 2d (optional) | Low |

------------------------------------------------------------------------

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
|------------------|----------------------|--------------------------------|
| Era breakpoints misspecified | Boundary artifacts in latent state | Inspect boundary behavior; consider soft transitions if artifacts appear |
| Time-varying bias absorbs latent state signal | Worse nowcast despite more flexible model | Tight prior on `σ_α_drift`; reject if `σ_α_drift` → 0 in posterior |
| JOLTS revisions material but ignored | Mild leakage in backtesting | Empirical revision assessment in Phase 1; add ALFRED vintages if needed |
| QCEW treated as final pre-2017 | Slightly overstated backtest performance for 2003–2016 | Document as known limitation; focus evaluation on 2017+ window |
| Small benchmark backtest sample | Unreliable RMSE estimates for 2d/2e | Strong priors; LOO-CV within backtest; extend to pre-provider era if feasible |
| Stacking multiple enhancements | Interaction effects obscure individual contributions | Sequential evaluation with accept/reject gates |

------------------------------------------------------------------------

## 7. Success Criteria

1.  **Phase 1:** All backtesting infrastructure is leakage-safe. All four cyclical indicators load correctly with proper censoring. Diagnostic tools (sensitivity, precision budget) produce actionable output.

2.  **Phase 2 (nowcast):** At least one of {era-specific parameters, cyclical indicators, time-varying bias} improves nowcast RMSE against CES first print relative to the Phase 1 baseline. Improvements should be most pronounced during turning-point months.

3.  **Phase 2 (benchmark):** BD error-correction improves benchmark RMSE at T−12 by \>5%. Preliminary-as-observation improves RMSE at T−5. Neither degrades T−1 performance.

4.  **Overall:** The national accuracy model is demonstrably better than the current implementation across both nowcasting and benchmark prediction use cases, with properly calibrated uncertainty and clean sampler diagnostics.