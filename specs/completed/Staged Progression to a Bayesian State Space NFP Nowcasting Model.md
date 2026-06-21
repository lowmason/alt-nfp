# Staged Progression to the Ultimate Bayesian State Space NFP Nowcasting Model

## Purpose

The uploaded specification describes the full **Ultimate Bayesian State Space NFP Nowcasting Model, Model 3** as one integrated production release. This document turns that single release into a practical staged build path: starting with the simplest defensible national nowcast and gradually adding the components needed to reach the full unified system.

The stages below are implementation and validation stages. They do not require the final product to be deployed piecemeal. The point is to reduce risk by proving one capability at a time before moving to the next layer.

The main progression is:

```text
naive baseline
  -> national Bayesian state-space model
  -> provider continuing-units layer
  -> structural birth/death layer
  -> representativeness-corrected provider composites
  -> national supersector narrative
  -> QCEW forecasting and dynamic provider bias
  -> full MinT-reconciled production system
```

The national nowcast remains the accuracy anchor throughout. The supersector narrative layer is added to explain the national estimate, not to weaken it.

---

## Stage Overview

| Stage | Name | Main question answered | Core output | Closest final-model capability |
|---:|---|---|---|---|
| 0 | Baseline and evaluation harness | Can we reproduce vintages and measure improvement honestly? | Backtest framework and naive NFP baseline | Validation infrastructure |
| 1 | Basic national Bayesian state-space model | Can we nowcast national NFP from official history alone? | National NFP nowcast with intervals | National latent state, CES/QCEW measurement |
| 2 | National provider continuing-units model | Do payroll providers improve the early national signal? | Provider-informed national continuing-units nowcast | Provider measurement equations |
| 3 | National structural birth/death model | Can we estimate the gap between continuing-units growth and total employment? | National NFP nowcast with explicit birth/death correction | Structural BD layer |
| 4 | Representativeness-corrected provider composite | Can we remove first-order provider composition bias? | QCEW-weighted national provider signals | Provider pipeline and reweighting |
| 5 | National supersector narrative layer | Can we explain the national nowcast by industry? | Supersector contributions and drivers | Industry decomposition |
| 6 | Forecasted QCEW and dynamic provider bias | Can we condition real-time supersector estimates before QCEW arrives and track provider drift? | Forecasted QCEW, bias trajectories, error correction | Real-time conditioning and bias correction |
| 7 | Full reconciled production Model 3 | Can all outputs be made coherent, monitored, and production-ready? | Full national and industry nowcast package | Ultimate unified release |

---

# Stage 0: Baseline and Evaluation Harness

## Goal

Build the testing foundation before building the model. This stage creates the vintage-aware data spine, naive baselines, and validation metrics that every later stage must beat.

## Data

- CES national total nonfarm history by vintage.
- CES first, second, and final national sample prints where available.
- QCEW national data after its publication lag.
- Benchmark and re-seasonalization dates.
- Calendar of publication regimes.

## Model

No advanced Bayesian model yet. Use simple baselines:

1. Last-month change.
2. Historical monthly average.
3. AR(1) on national NFP growth.
4. Seasonal naive model if NSA growth is used.

## Outputs

- `baseline_nowcast`
- `baseline_rmse`
- `baseline_mae`
- `baseline_bias`
- `vintage_calendar`
- `available_information_set`
- `backtest_cutoff_date`

## What is intentionally excluded

- Payroll providers.
- Birth/death modeling.
- Industry decomposition.
- MinT reconciliation.
- Full Bayesian hierarchy.

## Promotion gate

This stage is complete when historical backtests can be run without look-ahead bias. The model must know exactly which data were available on each historical forecast date.

---

# Stage 1: Basic National Bayesian State-Space Model

## Goal

Create the simplest serious version of the national nowcast. This stage estimates latent national employment growth using official data only.

## Data

- CES national total nonfarm, SA and optionally NSA.
- CES national vintages.
- QCEW national total employment growth.
- Basic seasonal calendar if NSA is modeled.

## Model

A minimal national latent state:

```text
latent national growth -> measured by CES vintages and QCEW
```

Core structure:

```text
mu_t = mu_g + phi * (mu_{t-1} - mu_g) + state_noise
```

Measurement equations:

```text
CES_vintage_t ~ latent national growth + vintage-specific noise
QCEW_t        ~ latent national growth + low census-anchor noise
```

Seasonality can be handled in one of two ways:

1. Start with seasonally adjusted data only.
2. Add a fixed monthly seasonal effect for NSA data.

The full Fourier time-varying seasonality is not required yet.

## Outputs

- `nfp_nowcast`
- `nfp_nowcast_std`
- `nfp_nowcast_ci`
- `latent_national_growth`
- `ces_vintage_noise`
- `qcew_anchor_residuals`

## What is intentionally excluded

- Provider data.
- Continuing-units versus total-employment split.
- Structural birth/death correction.
- Industry decomposition.
- MinT reconciliation.

## Promotion gate

The model should beat or match the Stage 0 baselines and produce credible intervals with reasonable historical coverage. If it cannot beat the naive baselines, later complexity will be difficult to justify.

---

# Stage 2: National Provider Continuing-Units Model

## Goal

Add private payroll providers as early signals of continuing-units employment growth. This creates the first true nowcasting edge before CES is released.

## Data

- Stage 1 data.
- One or more payroll provider national continuing-units growth signals.
- Provider availability dates.
- Provider vintage files if historical vintages exist.

## Model

Split national growth into continuing-units growth plus a simple birth/death term:

```text
national total growth = continuing-units growth + simple birth/death intercept
```

Provider observations load on continuing-units growth, not total growth:

```text
provider_growth_p,t ~ alpha_p + lambda_p * continuing_units_growth_t + provider_noise_p
```

At this stage, the birth/death component can be a constant mean or a very low-variance random effect. The goal is not yet to solve birth/death drift; the goal is to prove that provider data have signal.

## Outputs

- `g_cont`
- `provider_signal_quality`
- `provider_weights`
- `provider_bias`
- `provider_residuals`
- `early_provider_only_nowcast`

## What is intentionally excluded

- Provider representativeness correction.
- Provider birth-rate signals.
- Cyclical indicators.
- QCEW birth/death proxy.
- Time-varying provider bias.

## Promotion gate

Provider-informed nowcasts should improve the provider-only-period forecast, especially in Regime A. Diagnostics should show that providers add precision rather than simply duplicating noise.

---

# Stage 3: National Structural Birth/Death Model

## Goal

Add the key national value proposition: estimate the systematic gap between continuing-units payroll-provider growth and total employment growth caused by establishment births and deaths.

## Data

- Stage 2 data.
- Provider birth-rate or formation signals where available.
- Lagged QCEW-based birth/death proxy.
- Cyclical indicators such as claims, financial conditions, business applications, or other real-time demand-side variables.

## Model

Replace the simple birth/death intercept with a structural process:

```text
BD_t = phi_0
     + phi_1 * provider_birth_signal_t
     + phi_2 * lagged_QCEW_birth_death_proxy_t
     + phi_3 * cyclical_indicators_t
     + BD_noise_t
```

The national total-employment nowcast becomes:

```text
national NFP growth = continuing-units growth + structural BD_t
```

This is the first stage that directly targets the official birth/death vulnerability at turning points.

## Outputs

- `bd`
- `g_total_sa`
- `g_total_nsa`
- `birth_death_contribution`
- `birth_death_uncertainty`
- `cyclical_turning_point_diagnostics`
- `national_accuracy_layer_v1`

## What is intentionally excluded

- Industry-specific birth/death priors.
- Full provider composition correction.
- MinT reconciliation.

## Promotion gate

The model should improve against CES first prints and later benchmark-informed targets during periods where birth/death drift matters. It should also avoid overreacting when provider birth-rate signals are missing or noisy.

---

# Stage 4: Representativeness-Corrected Provider Composite

## Goal

Correct provider composition bias before provider signals enter the national measurement equation.

A provider that overweights healthcare, manufacturing, or leisure can produce a distorted national signal. This stage builds the pipeline that converts provider records into QCEW-weighted national composites.

## Data

- Stage 3 data.
- Provider continuing-units growth by supersector bucket.
- QCEW employment shares by supersector.
- Provider coverage indicators by supersector.
- Frozen rotating provider panels.

## Model and pipeline

Construct provider national signals as QCEW-weighted composites:

```text
corrected_provider_signal_p,t = sum over supersectors s of QCEW_weight_s,t * provider_growth_p,s,t
```

Where supersector buckets are missing, redistribute weights within the closest available supersector parent while preserving marginal distributions as much as possible.

This stage should also implement the frozen-panel logic:

```text
active provider panel k(t)
matched continuing units inside k(t)
exits removed from continuing-units growth
birth/death handled separately by the structural BD layer
```

Pseudo-establishment construction can start here if provider records are client-level rather than worksite-level.

## Outputs

- `representativeness_corrected_provider_signal`
- `provider_supersector_coverage`
- `composition_bias_before_after`
- `provider_residual_dispersion_before_after_correction`
- `coverage_adjusted_provider_weights`

## What is intentionally excluded

- Full MinT reconciliation.

## Promotion gate

Corrected provider signals should reduce residual dispersion versus raw provider aggregates, especially when sectoral growth dispersion is high.

---

# Stage 5: National Supersector Narrative Layer

## Goal

Add the first narrative layer: explain the national nowcast by industry supersector.

This stage keeps the national model as the accuracy anchor while adding industry contributions that sum toward the national story.

## Data

- Stage 4 data.
- CES national supersector data by vintage.
- QCEW national by sector.
- Provider supersector signals.
- Supersector employment weights.

## Model

Add supersector-level latent states:

```text
supersector continuing-units growth_s,t
supersector birth/death_s,t
supersector seasonal component_s,t
```

CES supersector observations anchor the industry layer:

```text
CES_supersector_s,t,v ~ supersector total growth_s,t + vintage-specific noise_s,v
```

QCEW sector data provide the census anchor:

```text
QCEW_sector_s,t ~ supersector total NSA growth_s,t + QCEW sector noise_s
```

Provider supersector signals measure continuing-units growth:

```text
provider_supersector_p,s,t ~ continuing-units supersector growth_s,t + provider noise_p,s
```

## Outputs

- `supersector_nowcasts`
- `supersector_contributions`
- `top_industry_drivers`
- `supersector_bd`
- `provider_supersector_coverage`
- `industry_residual_diagnostics`

## What is intentionally excluded

- MinT reconciliation across national and supersector nodes.

## Promotion gate

The industry narrative available around national CES first print should remain directionally stable through later vintages. Supersector estimates should improve against QCEW sector anchors versus simple share-based allocation.

---

# Stage 6: Forecasted QCEW and Dynamic Provider Bias

## Goal

Add the advanced real-time conditioning and provider-drift controls needed for the ultimate system.

This stage addresses two production realities:

1. QCEW is valuable but arrives with a 5-6 month lag.
2. Provider panels drift over time as client portfolios change.

## Data

- Stage 5 data.
- QCEW forecast features.
- National and sector forecast differentials.
- Provider-QCEW discrepancy histories.
- Provider panel composition history.

## Model

Forecast QCEW inside the publication lag, at the national and supersector levels:

```text
forecasted_QCEW_s,t = lagged_supersector_state
                    + sector_minus_national_adjustment
                    + forecast_covariates
                    + horizon_scaled_noise
```

The forecasted QCEW observation enters with much larger uncertainty than observed QCEW:

```text
forecasted_QCEW_s,t ~ supersector_total_growth_s,t + forecast_QCEW_noise
```

Add time-varying provider bias:

```text
alpha_p,s,t = alpha_p,s,t-1
            + random_walk_innovation_p,s,t
            - kappa_p * lagged_provider_QCEW_discrepancy_p,s,t-L
```

This allows provider bias to drift but prevents unbounded drift by pulling it back toward QCEW-consistent values.

## Outputs

- `qcew_forecast`
- `qcew_forecast_residuals_by_horizon`
- `bias_trajectories`
- `error_correction_speeds`
- `provider_drift_diagnostics`
- `forecasted_vs_observed_qcew_diagnostics`

## What is intentionally excluded

- Final production reconciliation and reporting layer.
- Full fallback/suppression rules.
- Production sampler configurations.

## Promotion gate

Forecasted QCEW must improve real-time conditioning without being treated as census truth. Provider bias trajectories should detect meaningful drift while avoiding overfitting monthly noise.

---

# Stage 7: Full Reconciled Production Model 3

## Goal

Integrate every layer into the final coherent production release.

This is the ultimate model described in the uploaded specification.

## Data

All prior-stage data:

- QCEW national and sector data.
- CES national and supersector vintages.
- Multiple payroll providers.
- Provider birth-rate signals.
- Cyclical indicators.
- QCEW forecasts.
- Historical revision data for covariance estimation.
- Full vintage calendar.

## Model

The final system contains:

1. National accuracy layer.
2. Supersector narrative layer.
3. Structural birth/death at national and industry levels.
4. Frozen-panel provider measurement pipeline.
5. Pseudo-establishment construction.
6. Representativeness-corrected provider national composites.
7. Fourier seasonality with evolving annual amplitudes.
8. Nested industry shrinkage.
9. Time-varying provider bias with QCEW error correction.
10. Forecasted QCEW real-time conditioning.
11. Discrepancy process for official aggregate reporting gaps.
12. Full-hierarchy MinT reconciliation across national and supersector nodes.
13. Information-regime-specific uncertainty reporting.
14. National standalone fallback.

The full hierarchy is reconciled across:

```text
national
  -> national supersectors
```

MinT reconciliation uses a covariance matrix built from direct revision histories where available and hierarchical shrinkage where direct supersector-level histories are unavailable.

## Outputs

All final outputs from the uploaded production specification, including:

- `nfp_nowcast`
- `nfp_nowcast_std`
- `nfp_nowcast_ci`
- `g_cont`
- `bd`
- `seasonal`
- `supersector_nowcasts_reconciled`
- `supersector_contributions`
- `top_industry_drivers`
- `provider_signal_quality`
- `provider_bias`
- `provider_weights`
- `precision_budget`
- `provider_supersector_coverage`
- `variance_components`
- `effective_shrinkage`
- `bias_trajectories`
- `qcew_forecast`
- `discrepancy_estimate`
- `information_regime`
- `regime_uncertainty`

## Production controls

- Vintage-aware backtesting.
- Regime A, B, and C labeling.
- Sampler configurations: `LIGHT`, `MEDIUM`, and `DEFAULT`.
- Non-centered parameterization for sparse hierarchical effects.
- Provider configuration system.
- Provider coverage diagnostics.
- Precision-budget reporting.
- Reconciliation diagnostics.
- Fallback rule: publish national nowcast even if supersector decomposition diagnostics fail.

## Promotion gate

The final system is production-ready when it passes:

1. National nowcast accuracy tests versus CES first, second, final, and benchmark-informed targets.
2. Credible-interval coverage tests.
3. Birth/death drift detection tests.
4. Supersector accuracy tests against QCEW sector data.
5. Regime B to Regime C stability tests.
6. Provider signal-quality and drift diagnostics.
7. QCEW forecast accuracy tests by horizon.
8. MinT reconciliation diagnostics.
9. Runtime and convergence standards for production sampling.

---

## Recommended Build Sequence by Workstream

The same stages can be organized into parallel workstreams.

| Workstream | Stages | Description |
|---|---:|---|
| Data and vintages | 0-7 | Build the real-time data spine, publication calendar, benchmark tracking, and no-look-ahead backtests. |
| National model | 1-4 | Build the national state-space, provider, birth/death, and representativeness-corrected accuracy layer. |
| Industry narrative | 5 | Add CES/QCEW/provider supersector decomposition. |
| Real-time conditioning | 6 | Add forecasted QCEW and time-varying provider bias. |
| Reconciliation and production | 7 | Add MinT, production diagnostics, reporting, and fallback logic. |

---

## Simplified Version Names

A clean naming convention would be:

| Version | Description |
|---|---|
| Model 0 | Naive baseline and vintage-aware evaluation harness. |
| Model 1A | Official-data-only national Bayesian state-space model. |
| Model 1B | National provider continuing-units model. |
| Model 1C | National structural birth/death model. |
| Model 2A | Representativeness-corrected provider national composite. |
| Model 2B | National supersector narrative model. |
| Model 3A | Supersector model with forecasted QCEW and dynamic provider bias. |
| Model 3B | Full MinT-reconciled unified production release. |

---

## Minimal-to-Ultimate Capability Matrix

| Capability | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vintage-aware backtest harness | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| National NFP nowcast | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Bayesian latent state | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| CES national vintages | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| QCEW national anchor | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Payroll provider national signals | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Continuing-units/total split | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Structural national birth/death | No | No | Minimal | Yes | Yes | Yes | Yes | Yes |
| Representativeness correction | No | No | No | No | Yes | Yes | Yes | Yes |
| Supersector decomposition | No | No | No | No | No | Yes | Yes | Yes |
| Nested industry shrinkage | No | No | No | No | No | Partial | Yes | Yes |
| Forecasted QCEW | No | No | No | No | No | No | Yes | Yes |
| Time-varying provider bias | No | No | No | No | No | No | Yes | Yes |
| Full MinT reconciliation | No | No | No | No | No | No | No | Yes |
| Production diagnostics and fallback | Basic | Basic | Basic | Basic | Medium | Medium | Advanced | Full |

---

## Practical Deployment Recommendation

The first externally useful product is likely **Stage 3 or Stage 4**, because that is where the model has a true national nowcasting edge from providers and structural birth/death correction.

The first externally useful narrative product is likely **Stage 5**, because it explains the national number by supersector using official CES supersector anchors.

The full research-grade and production-grade system is **Stage 7**.

---

## Summary

The natural staged path is:

1. Prove the vintage-aware backtest harness.
2. Build the national Bayesian state-space model.
3. Add payroll providers as continuing-units signals.
4. Add structural birth/death correction.
5. Correct provider composition bias using QCEW-weighted supersector composites.
6. Add industry decomposition.
7. Add forecasted QCEW and dynamic provider bias.
8. Reconcile the complete hierarchy and harden the system for production.

This progression preserves the core design principle of the ultimate model: **national forecast accuracy first, narrative decomposition second, coherent hierarchy last**.
