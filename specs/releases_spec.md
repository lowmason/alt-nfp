# alt\_nfp Implementation Specification

**Bayesian State Space NFP Nowcasting System — Releases 1–3**

February 2026 · Based on `bayesian_state_space_alt_nfp_v9`

---

## Table of Contents

1. [Overview and Design Principles](#1-overview-and-design-principles)
2. [Release 1: National Accuracy Model (Current Implementation)](#2-release-1-national-accuracy-model)
3. [Release 2: Industry Decomposition (Narrative Model — Supersector)](#3-release-2-industry-decomposition)
4. [Release 3: Geographic × Industry Decomposition](#4-release-3-geographic--industry-decomposition)
5. [Cross-Cutting Infrastructure](#5-cross-cutting-infrastructure)
6. [Testing and Validation Strategy](#6-testing-and-validation-strategy)
7. [Computational Requirements](#7-computational-requirements)
8. [Appendix: Release Comparison Matrix](#8-appendix-release-comparison-matrix)

---

## 1. Overview and Design Principles

This document specifies the implementation plan for the `alt_nfp` Bayesian state space NFP nowcasting system. The system produces national employment estimates 2–3 weeks before official BLS releases by fusing QCEW census data, CES survey data, and private payroll provider signals with structural birth/death modeling.

The system comprises two conceptual models delivered across three releases:

1. **National Accuracy Model (Release 1).** Produces a national NFP nowcast by fusing all data sources in a single latent state with structural birth/death modeling, Fourier seasonal adjustment, and vintage-aware CES measurement equations. This release is **currently implemented** in the `alt_nfp` codebase.

2. **Narrative Decomposition Model (Releases 2–3).** Decomposes the national estimate into industry and geographic components. Release 2 provides supersector decomposition; Release 3 adds state × supersector detail.

Each release is a fully functional system. The narrative model never degrades the national model — MinT reconciliation guarantees that the reconciled national estimate is at least as accurate as the standalone national estimate under correct covariance specification.

### 1.1 Design Principles

1. **Forecast accuracy before narrative:** Birth/death correction and QCEW anchoring are prioritized over decomposition.
2. **Multi-provider from the start:** The measurement error framework accommodates multiple providers as independent noisy observations. New providers require only a `ProviderConfig` entry.
3. **Industry before geography:** CES publishes national supersector detail on the same day as the national total. Geographic detail waits ~2 weeks for state CES.
4. **Coherent hierarchical forecasts:** Cell contributions sum exactly to national via MinT reconciliation.
5. **Information-regime awareness:** The system explicitly tracks which data sources are available at each point in the publication cycle.
6. **Representativeness-corrected provider signals:** Provider national signals are pre-processed via QCEW-weighted compositing of cell-level continuing-units growth from pseudo-establishments at the supersector × Census region level.

### 1.2 Data Sources

The system combines four classes of data:

- **QCEW:** Near-census of U.S. employment (>95% coverage). Published quarterly with ~5–6 month lag. Three monthly values per quarter (M3 contemporaneous, M1–M2 retrospective). Revision history from 2017Q1.
- **CES National:** Probability-based establishment survey (~670K worksites). First print ~3 weeks after reference period; two subsequent revisions. Publishes total nonfarm and supersector detail on the same day. Vintage data from May 2003.
- **CES State:** Published ~5 weeks after reference period. One revision before annual benchmark. States do not sum to national by design.
- **Payroll Provider Data:** Continuing-units employment growth from private payroll processors. Available ~3 weeks before CES first print. Constructed from rotating, frozen measurement panels.

### 1.3 Publication Timeline

For a given reference month (e.g., December, reference date December 12):

| Event | Approximate Date | Information Available |
|---|---|---|
| Reference period | Dec 12 | — |
| Provider data available | ~Jan 1 | Provider growth rates (national, by industry, by geography) |
| CES national 1st print | ~Jan 9 | National total nonfarm + supersectors (SA + NSA) |
| CES state 1st print | ~Jan 24 | State total nonfarm (SA + NSA) |
| CES national 2nd print | ~Feb 6 | National revised |
| CES state final print | ~Feb 28 | State revised |
| CES national 3rd (final) print | ~Mar 6 | National final sample-based |
| QCEW publication | ~Jun | Full quarter census data |

---

## 2. Release 1: National Accuracy Model

Release 1 is the core national nowcasting model. It is **currently implemented** in the `alt_nfp` codebase and produces a single national NFP nowcast by fusing all data sources through a Bayesian state space framework.

### 2.1 Current Codebase Architecture

The implementation lives in `src/alt_nfp/` with the following module structure:

| Module | Responsibility |
|---|---|
| `config.py` | `ProviderConfig` dataclass, `PROVIDERS` list, model constants (`SIGMA_QCEW_M3`/`M12`, `N_HARMONICS`, `BD_QCEW_LAG`), cyclical indicator specs, paths |
| `data.py` | `load_data()` — loads CES/QCEW/provider CSVs, computes growth rates, builds birth/death covariates, aligns to monthly calendar. Supports `censor_ces_from` for backtesting |
| `model.py` | `build_model()` — PyMC model: AR(1) latent state via `pytensor.scan`, Fourier seasonal (GRW on annual coefficients), structural BD, QCEW/CES/provider measurement equations |
| `sampling.py` | `sample_model()` — nutpie-preferred MCMC with fallback to PyMC NUTS. Three configs: `DEFAULT` (8K/6K), `MEDIUM` (4K/3K), `LIGHT` (2K/2K) |
| `checks.py` | Prior predictive checks, posterior predictive density overlays, test statistics (skewness, lag-1 ACF), LOO-CV with Pareto k̂ diagnostics |
| `backtest.py` | Expanding-window CES-censoring backtest over trailing months |
| `ingest/` | Data ingestion pipeline: panel builder, CES national/state ingestion, QCEW ingestion, payroll provider loading, vintage store (Hive parquet), BLS release date lookups |

The main entry point is `alt_nfp_estimation_v3.py` at the repo root, which orchestrates the full Bayesian workflow: data loading → prior predictive checks → sampling → diagnostics → posterior predictive checks → LOO-CV → plots → forecast → save.

### 2.2 Model Specification (As Implemented)

#### 2.2.1 Latent State: Continuing-Units Growth

Continuing-units employment growth follows a mean-reverting AR(1), implemented via `pytensor.scan` with non-centered innovations:

$$\mu_t^{cont} = \mu_g + \phi(\mu_{t-1}^{cont} - \mu_g) + \sigma_\eta \cdot \varepsilon_t, \quad \varepsilon_t \sim N(0,1)$$

**Priors:**

- $\mu_g \sim N(0.001, 0.005^2)$ — ~0.1%/mo mean growth, weakly informative
- $\phi \sim \text{Uniform}(0, 0.99)$ — allows high persistence; Cajner et al.: ρ ≈ 0.96
- $\sigma_\eta \sim \text{Half-N}(0, 0.005)$

**Implementation:** `model.py` defines `eps_g = pm.Normal('eps_g', 0, 1, shape=T)`, then uses `pytensor.scan` with an `ar1_step` function to propagate the latent state. The result is stored as `pm.Deterministic('g_cont', g_cont)`.

#### 2.2.2 Structural Birth/Death

Total employment growth decomposes into continuing-units growth and a birth/death offset:

$$\mu_t = \mu_t^{cont} + BD_t$$

The BD component is modeled as a time-varying structural process combining supply-side and demand-side information:

$$BD_t = \phi_0 + \phi_1 X_t^{birth} + \phi_2 BD_{t-L}^{QCEW} + \phi_3 X_t^{cycle} + \sigma_{BD} \cdot \xi_t, \quad \xi_t \sim N(0,1)$$

where:

- $X_t^{birth}$ — centered composite provider birth rate (averaged across providers that report it)
- $BD_{t-L}^{QCEW}$ — centered QCEW-based BD proxy ($g^{QCEW} - \bar{g}^{PP}$), lagged by $L = 6$ months (`BD_QCEW_LAG` in `config.py`)
- $X_t^{cycle}$ — vector of centered cyclical indicators (unemployment claims, NFCI, Census business applications; configured in `CYCLICAL_INDICATORS`)
- Missing covariates default to zero (centered), collapsing BD to $\phi_0 + \sigma_{BD} \cdot \xi_t$

**Priors:**

- $\phi_0 \sim N(0.001, 0.002^2)$ — mean BD contribution
- $\phi_1 \sim N(0.5, 0.5^2)$ — birth rate loading; positive = procyclical
- $\phi_2 \sim N(0.3, 0.3^2)$ — QCEW BD proxy persistence
- $\phi_3 \sim N(0, 0.3^2)$ — cyclical indicator loadings
- $\sigma_{BD} \sim \text{Half-N}(0, 0.001)$

#### 2.2.3 Fourier Seasonal

Truncated Fourier expansion with $K = 4$ harmonics (`N_HARMONICS` in `config.py`) and annually-evolving amplitudes via Gaussian random walks:

$$s_t = \sum_{k=1}^{K} \left[ A_k(y(t)) \cos\left(\frac{2\pi k \, m(t)}{12}\right) + B_k(y(t)) \sin\left(\frac{2\pi k \, m(t)}{12}\right) \right]$$

$$A_k(y) = A_k(y-1) + \omega_{A,k}(y), \quad \omega_{A,k} \sim N(0, \sigma_{\omega,k}^2)$$

**Priors:**

- $A_k(0), B_k(0) \sim N(0, 0.015^2)$ for $k = 1, \ldots, K$
- $\sigma_{\omega,k} \sim \text{Half-N}(0, 0.005/k)$ — tighter for higher harmonics

Annual-step drift is preferred over monthly drift. The model asks "how did this year's seasonal pattern differ from last year's?" without confounding seasonal evolution with the latent state dynamics.

**Implementation:** `model.py` computes Fourier basis vectors from `month_of_year`, then uses GRW-style annual coefficient evolution. The seasonal component `s_t` combines with the latent state to form the composite growth signals:

- $g_t^{cont,NSA} = \mu_t^{cont} + s_t$
- $g_t^{total,SA} = \mu_t^{cont} + BD_t$
- $g_t^{total,NSA} = \mu_t^{cont} + BD_t + s_t$

#### 2.2.4 Measurement Equations

**QCEW — Truth Anchor**

High-precision observation of total NSA growth with fixed noise differentiated by filing timing:

$$y_t^{QCEW} \sim N(g_t^{total,NSA}, \sigma_{QCEW}^2(t))$$

$$\sigma_{QCEW}(t) = \begin{cases} \sigma_{QCEW,M3} = 0.0005 & \text{if } t \text{ is a quarter-end month (M3)} \\ \sigma_{QCEW,M12} = 0.0015 & \text{if } t \text{ is a retrospective month (M1 or M2)} \end{cases}$$

Noise values are fixed (not estimated). The `qcew_is_m3` boolean array in `data.py` determines which months receive tighter noise. This is a **conditioning** approach (high-precision observation) rather than hard benchmarking.

**CES — Vintage-Specific**

Three prints per month with shared bias and signal loading but separate noise terms. Both SA and NSA series:

$$y_{t,v}^{CES,SA} \sim N(\alpha^{CES} + \lambda^{CES} \cdot g_t^{total,SA}, \sigma_{CES,SA,v}^2) \quad \text{for } v \in \{1,2,3\}$$

$$y_{t,v}^{CES,NSA} \sim N(\alpha^{CES} + \lambda^{CES} \cdot g_t^{total,NSA}, \sigma_{CES,NSA,v}^2)$$

**Priors:**

- $\alpha^{CES} \sim N(0, 0.005^2)$
- $\lambda^{CES} \sim N(1.0, 0.15^2)$
- $\sigma_{CES,SA,v} \sim \text{InverseGamma}(3, 0.004)$ for each vintage $v$
- $\sigma_{CES,NSA,v} \sim \text{InverseGamma}(3, 0.004)$ for each vintage $v$

**Implementation:** `model.py` loops over `v in range(3)`, indexing into `data['g_ces_sa_by_vintage'][v]` and masking to finite observations with `np.where(np.isfinite(...))`.

**Payroll Providers — Config-Driven**

Each provider $p$ gets independent measurement parameters with configurable error structure. Provider observations load on continuing-units NSA growth (not total growth):

*iid measurement error (default):*

$$y_{p,t}^G = \alpha_p + \lambda_p \cdot g_t^{cont,NSA} + \varepsilon_{p,t}, \quad \varepsilon_{p,t} \sim N(0, \sigma_{G,p}^2)$$

*AR(1) measurement error:*

$$y_{p,t}^G \mid y_{p,t-1}^G \sim N(\mu_{p,t}^{base} + \rho_p(y_{p,t-1}^G - \mu_{p,t-1}^{base}), \sigma_{G,p}^2)$$

where $\mu_{p,t}^{base} = \alpha_p + \lambda_p \cdot g_t^{cont,NSA}$.

**Per-provider priors:**

- $\alpha_p \sim N(0, 0.005^2)$
- $\lambda_p \sim N(1.0, 0.15^2)$
- $\sigma_{G,p} \sim \text{InverseGamma}(3, 0.004)$
- $\rho_p \sim \text{Beta}(2, 3)$ — AR(1) providers only; mode ≈ 0.25

**Implementation:** `model.py` loops over `pp_data`, reading each provider's `ProviderConfig`. For AR(1) providers, the conditional mean includes `rho_p * (y_obs[:-1] - mu_base[:-1])` and the initial observation uses marginal variance $\sigma_p^2 / (1 - \rho_p^2)$.

**Current provider configuration** (`config.py`):

```python
PROVIDERS: list[ProviderConfig] = [
    ProviderConfig(
        name='PP1',
        file='alt_nfp_index_1.csv',
        index_col='pp_index_1',
        error_model='ar1',
    ),
    ProviderConfig(
        name='PP2',
        file='alt_nfp_index_2.csv',
        index_col='pp_index_2_0',
        error_model='iid',
        births_file='alt_nfp_births_2.csv',
        births_col='pp2_births',
    ),
]
```

### 2.3 Release 1 Completion Tasks

The following items bring Release 1 to full specification per `bayesian_state_space_alt_nfp_v9`. These are refinements to the existing implementation, not architectural changes.

| # | Task | Module(s) | Status |
|---|---|---|---|
| 1 | Representativeness correction: implement QCEW-weighted compositing of cell-level provider signals at supersector × Census region level | `data.py`, new `repr_correction.py` | Not started |
| 2 | QCEW censoring in `load_data()` for proper backtesting (censor by publication date, not just CES) | `data.py` | Specified, not built |
| 3 | BLS publication date scraping and vintage date lookup (CES, QCEW, SAE schedules) | `ingest/release_dates.py`, `lookups/publication_dates.py` | Hard-coded dates present; scraper TBD |
| 4 | Leakage-safe backtesting: censor all sources (CES, QCEW, providers) to as-of date | `backtest.py`, `data.py` | CES censoring only |
| 5 | Vintage-aware CES data: load 3 vintages per month from vintage store or BLS API | `ingest/ces_national.py`, `vintage_store.py` | Infrastructure present; vintage loading partial |
| 6 | QCEW sigma sensitivity analysis automation | New `sensitivity.py` | Manual only |
| 7 | Precision budget output and provider weight extraction | New `diagnostics.py` or `checks.py` extension | Not started |
| 8 | Fix broken imports and centralize hard-coded constants | `config.py`, various | Ongoing |
| 9 | Expand unit test suite (model construction, data loading, prior ranges) | `tests/` | Minimal tests present |
| 10 | Benchmark prediction module (`extract_benchmark_revision`, benchmark backtest loop) | New `benchmark.py` | Specified in benchmark_prediction_alt_nfp_v2; not started |

### 2.4 Outputs

| Output | Description |
|---|---|
| `nfp_nowcast` | Point estimate (posterior mean of $\mu_T$) |
| `nfp_nowcast_std` | Uncertainty (posterior std of $\mu_T$) |
| `nfp_nowcast_ci` | 80% and 95% credible intervals |
| `g_cont` | Continuing-units growth trajectory |
| `bd` | Structural BD trajectory |
| `seasonal` | Fourier seasonal component |
| `provider_signal_quality` | Posterior mean of $\lambda_p$ for each provider |
| `provider_bias` | Posterior mean of $\alpha_p$ for each provider |
| `provider_weights` | Effective weight (precision share) of each provider in nowcast |
| `precision_budget` | Share of total precision from QCEW, CES (by vintage), each provider |

### 2.5 Diagnostics (As Implemented)

The following diagnostics are implemented in `checks.py` and the main estimation script:

- Standardized residuals by source (QCEW, CES SA/NSA, each provider)
- Prior predictive checks with density overlays and summary statistics
- Posterior predictive checks: density overlays + test statistics (skewness, lag-1 ACF) against replicated data
- LOO-CV with Pareto k̂ diagnostics flagging influential observations
- Divergence trace plots
- Source contribution decomposition

**Not yet implemented:**

- QCEW sigma sensitivity (parameter stability across 0.5×/1×/2× noise scales)
- Precision budget decomposition across noise scale configurations
- Representativeness correction impact comparison

### 2.6 Validation Metrics

- Out-of-sample RMSE against NFP releases (by CES vintage: first, second, final, benchmarked)
- Coverage of credible intervals (80% and 95%)
- Comparison to naive forecast (random walk on NFP)
- Comparison to post-2026 BLS NBDM (with WLR-regression improvement)
- Provider-specific signal quality rankings
- BD component accuracy against annual benchmark revisions

### 2.7 Limitations

- No geographic or industry decomposition
- Cannot explain drivers of national forecast
- Static provider bias (no drift accommodation)
- Representativeness correction removes first-order composition bias but within-cell heterogeneity can still create residual provider-specific deviations

---

## 3. Release 2: Industry Decomposition

Release 2 introduces the **dual-model architecture**: the unchanged Release 1 National Accuracy Model runs independently alongside a new Industry Narrative Model. MinT reconciliation produces coherent supersector estimates that sum to national.

Because CES publishes supersector detail on the same day as the national total, the industry narrative is **anchored to official data from Regime B onward** — no waiting for state CES.

### 3.1 Architecture Changes

Release 2 introduces the following structural additions to the codebase:

| New Module | Responsibility |
|---|---|
| `model_industry.py` | `build_industry_model()`: per-supersector latent states, BD, seasonal, measurement equations. Mirrors Release 1 structure but vectorized over $J$ supersectors. |
| `reconciliation.py` | MinT reconciliation: summing matrix $\mathbf{S}$, error covariance $\mathbf{W}$ estimation from revision cross-validation, reconciled forecast production. |
| `config.py` (extended) | `ProviderConfig` gains `cell_file` and `cell_level` fields. New `SUPERSECTORS` constant list. `InfoRegime` enum. |
| `ingest/ces_industry.py` | CES supersector ingestion (SA + NSA, 3 vintages) from BLS API or vintage store. |
| `ingest/qcew_industry.py` | QCEW by sector ingestion (national × sector level). |
| `repr_correction.py` | Extended from Release 1: cell-level signal construction from pseudo-establishments, both national composite (R1) and supersector aggregates (R2). |
| `pipeline.py` | Orchestrates dual-model estimation: (1) run national model, (2) run industry models, (3) reconcile via MinT. Produces unified output. |

### 3.2 Model Specification

#### 3.2.1 Supersector Latent States

For each supersector $j \in \{1, \ldots, J\}$ (~10 supersectors), an independent AR(1) latent state with exchangeable hyperpriors:

$$\mu_{j,t}^{cont} = \mu_{g,j} + \phi_j(\mu_{j,t-1}^{cont} - \mu_{g,j}) + \sigma_{\eta,j} \varepsilon_{j,t}, \quad \varepsilon_{j,t} \sim N(0,1)$$

**Exchangeable priors:**

- $\mu_{g,j} \sim N(\bar{\mu}_g, \tau_{\mu_g}^2)$ with $\bar{\mu}_g \sim N(0.001, 0.005^2)$, $\tau_{\mu_g} \sim \text{Half-N}(0, 0.003)$
- $\phi_j \sim \text{Beta}(\alpha_\phi, \beta_\phi)$ — common shape parameters; hierarchical on logit scale
- $\sigma_{\eta,j} \sim \text{LogNormal}(\mu_{\sigma_\eta}, \tau_{\sigma_\eta}^2)$

#### 3.2.2 Supersector Birth/Death

BD intensity varies systematically across industries — construction and hospitality have high birth/death rates; healthcare and utilities have low rates:

$$BD_{j,t} = \phi_{0,j} + \phi_{1,j} X_{j,t}^{birth} + \phi_{2,j} BD_{j,t-L}^{QCEW} + \phi_{3,j} X_t^{cycle} + \sigma_{BD,j} \xi_{j,t}$$

**Exchangeable priors (BDS-calibrated):**

- $\phi_{0,j} \sim N(\bar{\phi}_0, \tau_{\phi_0}^2)$ with $\bar{\phi}_0 \sim N(0.001, 0.002^2)$, $\tau_{\phi_0} \sim \text{Half-N}(0, 0.002)$
- $\sigma_{BD,j} \sim \text{LogNormal}(\mu_{\sigma_{BD}}, \tau_{\sigma_{BD}}^2)$

#### 3.2.3 Hierarchical Fourier Seasonal

Supersector seasonal patterns share innovation variance hyperparameters but each supersector identifies its own Fourier coefficients:

$$A_{k,j}(y) = A_{k,j}(y-1) + \omega_{A,k,j}(y), \quad \omega_{A,k,j} \sim N(0, \sigma_{\omega,k}^2)$$

The innovation variance $\sigma_{\omega,k}^2$ is **shared across supersectors** (hierarchical) — the prior says that the *rate* of seasonal evolution is similar across industries, even though the seasonal *patterns* differ.

#### 3.2.4 Measurement Equations

**QCEW by sector:**

$$y_{j,t}^{QCEW} \sim N(g_{j,t}^{total,NSA}, \sigma_{QCEW,j}^2(t))$$

Same M3/M12 differential as national. Industry-level noise is slightly larger due to employer industry misclassification.

**CES by supersector (vintage-specific):**

For supersector $j$ and vintage $v \in \{1, 2, 3\}$:

$$y_{j,t,v}^{CES,SA} \sim N(\alpha_j^{CES} + \lambda_j^{CES} \cdot g_{j,t}^{total,SA}, \sigma_{CES,j,SA,v}^2)$$

$$y_{j,t,v}^{CES,NSA} \sim N(\alpha_j^{CES} + \lambda_j^{CES} \cdot g_{j,t}^{total,NSA}, \sigma_{CES,j,NSA,v}^2)$$

Vintage-specific noise calibrated from national × sector CES revision data (May 2003+). Dunn et al. (2026) Table 1: national 2-digit NAICS has MAE of 0.0032 (vs. 0.0014 for national aggregate).

**Provider by supersector:**

$$y_{p,j,t}^G = \alpha_{p,j} + \lambda_{p,j} \cdot g_{j,t}^{cont,NSA} + \varepsilon_{p,j,t}$$

Error structure (iid or AR(1)) inherited from the provider's configuration. Provider × supersector bias drawn from exchangeable priors:

- $\alpha_{p,j} \sim N(\bar{\alpha}_p, \tau_\alpha^2)$ with $\bar{\alpha}_p \sim N(0, 0.005^2)$, $\tau_\alpha \sim \text{Half-N}(0, 0.003)$
- $\lambda_{p,j} \sim N(\bar{\lambda}_p, \tau_\lambda^2)$ with $\bar{\lambda}_p \sim N(1.0, 0.15^2)$, $\tau_\lambda \sim \text{Half-N}(0, 0.1)$

#### 3.2.5 Discrepancy Model

A random walk absorbs the structural gap between sum-of-supersectors and national:

$$\delta_t = \delta_{t-1} + \omega_t^\delta, \quad \omega_t^\delta \sim N(0, \sigma_\delta^2), \quad \sigma_\delta \sim \text{Half-N}(0, 0.002)$$

#### 3.2.6 MinT Reconciliation

Let $\hat{\boldsymbol{\mu}}_t = (\hat{\mu}_t^{nat}, \hat{\mu}_{1,t}, \ldots, \hat{\mu}_{J,t})'$ be the vector of base (unreconciled) forecasts from independent national and supersector models, and let $\mathbf{S}$ be the summing matrix:

$$\mathbf{S} = \begin{pmatrix} \mathbf{1}' \\ \mathbf{I}_J \end{pmatrix}$$

The MinT reconciled forecast is:

$$\tilde{\boldsymbol{\mu}}_t = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}\hat{\boldsymbol{\mu}}_t$$

where $\mathbf{W}$ is the covariance matrix of base forecast errors, estimated from historical revision-based cross-validation with QCEW vintages.

### 3.3 Implementation Plan

1. **Phase 1 — Data infrastructure.** Build CES and QCEW industry-level ingestion. Extend `ProviderConfig` with `cell_file` and `cell_level` fields. Construct supersector-level provider indices from cell-level signals.

2. **Phase 2 — Industry model.** Implement `build_industry_model()` with vectorized supersector latent states. Start with exchangeable priors, no hierarchical seasonal (use Release 1 seasonal structure per supersector).

3. **Phase 3 — MinT reconciliation.** Implement `reconciliation.py` with summing matrix construction, $\mathbf{W}$ estimation, and reconciled forecast extraction. Verify coherence (supersectors sum to national exactly).

4. **Phase 4 — Pipeline orchestration.** Build `pipeline.py` that runs both models and reconciles. Add information regime tracking (Regime A/B/C based on data availability).

5. **Phase 5 — Validation.** Confirm national RMSE does not degrade vs. Release 1. Evaluate supersector accuracy against QCEW by sector. Test industry narrative stability across regimes.

### 3.4 Additional Outputs

All outputs from Release 1, plus:

| Output | Description |
|---|---|
| `supersector_nowcasts_reconciled` | Reconciled supersector estimates (sum to national) |
| `supersector_contributions` | Employment-weighted contribution of each supersector to national change |
| `top_industry_drivers` | Supersectors contributing most to national change |
| `supersector_bd` | Industry-level BD trajectories |
| `provider_supersector_coverage` | Which providers inform which supersectors |
| `discrepancy_estimate` | Current $\delta_t$ |
| `information_regime` | Current regime (A, B, or C) |

### 3.5 Validation Metrics

- National RMSE (must not degrade vs. Release 1)
- Coherence: supersectors sum exactly to national
- Supersector-level accuracy against QCEW by sector
- Industry narrative stability: does the Regime B industry decomposition hold up when later vintages arrive?
- Provider ranking by signal quality, by supersector

### 3.6 Computational Scaling

Approximate parameters: ~50 + 10×P + 8×$J_s$ where P = providers, $J_s$ = number of supersectors (~10). Expected runtime: 10–30 minutes with nutpie.

### 3.7 Limitations

- No geographic decomposition (cannot answer "where?")
- Exchangeable priors across supersectors (no nested industry structure)
- Static provider bias

---

## 4. Release 3: Geographic × Industry Decomposition

Release 3 extends the narrative model to the full state × supersector grid (~550 cells), answering both "which industries?" and "where?" This is the most complex release, introducing nested hierarchical priors, time-varying provider bias, QCEW forecasting, and CES state measurement equations.

### 4.1 Architecture Changes

| New/Modified Module | Responsibility |
|---|---|
| `model_cell.py` | `build_cell_model()`: cell-level latent states with nested hierarchical priors. Extends `model_industry.py` to full state × supersector grid. |
| `hierarchy.py` | Nested random effects for geography (region → division → state) and industry (domain → supersector). Bias/loading decomposition. |
| `qcew_forecast.py` | QCEW forecast model using location quotients, sector forecasts, and horizon-dependent noise. Extends conditioning to real-time despite 5–6 month publication lag. |
| `reconciliation.py` (extended) | Full 3-level MinT: national → supersectors → state × supersector cells. Hierarchical shrinkage for $\mathbf{W}$ entries where revision data is unavailable. |
| `ingest/ces_state.py` (extended) | CES state ingestion: total nonfarm SA + NSA, 2 vintages (preliminary, final). Already partially implemented. |
| `ingest/qcew_state.py` | QCEW by state ingestion. M3/M12 differential applies at state level. |

### 4.2 Model Specification

#### 4.2.1 Cell-Level Latent States

For each cell $j$ in the state × supersector grid:

$$\mu_{j,t}^{cont} = \mu_{g,j} + \phi_j(\mu_{j,t-1}^{cont} - \mu_{g,j}) + \sigma_{\eta,j} \varepsilon_{j,t}, \quad \varepsilon_{j,t} \sim N(0,1)$$

$$\mu_{j,t} = \mu_{j,t}^{cont} + BD_{j,t}$$

#### 4.2.2 Nested Hierarchical Structure

Cell-level provider bias decomposes into geographic, industry, provider, and residual components:

$$\alpha_{p,j} = \alpha_p^{prov} + \alpha_{r(j)}^{region} + \alpha_{d(j)}^{division} + \alpha_{st(j)}^{state} + \alpha_{s(j)}^{supersector} + \alpha_{p,s(j)}^{prov \times ind} + \alpha_{p,j}^{resid}$$

**Geographic hierarchy:**

- $\alpha_r^{region} \sim N(0, \tau_{region}^2)$, $\tau_{region} \sim \text{Half-N}(0, 0.3)$
- $\alpha_d^{division} \sim N(0, \tau_{division}^2)$, $\tau_{division} \sim \text{Half-N}(0, 0.25)$
- $\alpha_{st}^{state} \sim N(0, \tau_{state}^2)$, $\tau_{state} \sim \text{Half-N}(0, 0.2)$

**Industry hierarchy:**

- $\alpha_{dom}^{domain} \sim N(0, \tau_{domain}^2)$, $\tau_{domain} \sim \text{Half-N}(0, 0.3)$
- $\alpha_s^{supersector} \sim N(\alpha_{dom(s)}^{domain}, \tau_{supersector|domain}^2)$, $\tau_{supersector|domain} \sim \text{Half-N}(0, 0.25)$

**Interaction and residual:**

- $\alpha_{p,s}^{prov \times ind} \sim N(0, \tau_{prov \times ind}^2)$, $\tau_{prov \times ind} \sim \text{Half-N}(0, 0.15)$
- $\alpha_{p,j}^{resid} \sim N(0, \tau_{resid}^2)$, $\tau_{resid} \sim \text{Half-N}(0, 0.1)$

Signal loading $\lambda_{p,j}$ follows the same nested decomposition. **Non-centered parameterization** is essential for sparse cells to avoid Neal's funnel.

#### 4.2.3 Nested BD Intensity

BD rates vary systematically by industry:

$$\phi_{0,j} = \bar{\phi}_0 + \phi_0^{domain(j)} + \phi_0^{supersector(j)} + \phi_0^{resid,j}$$

Analogous nesting for BD covariate loadings $\phi_{1,j}, \phi_{2,j}, \phi_{3,j}$.

#### 4.2.4 Hierarchical Fourier Seasonal with Industry-Parent Shrinkage

Sparse cells are shrunk toward their supersector's seasonal pattern:

$$A_{k,j}(y) = A_{k,s(j)}(y) + \delta_{A,k,j}(y)$$

where:

- $A_{k,s}(y) = A_{k,s}(y-1) + \omega_{A,k,s}(y)$, $\omega \sim N(0, \sigma_{\omega,k}^2)$
- $\delta_{A,k,j}(y) \sim N(0, \tau_{\omega,k}^2)$ — cell deviation from industry parent
- $\sigma_{\omega,k} \sim \text{Half-N}(0, 0.005/k)$
- $\tau_{\omega,k} \sim \text{Half-N}(0, 0.003/k)$

Data-rich cells identify departures (e.g., Alaska construction seasonality differs from national construction).

#### 4.2.5 CES State Measurement Equations

State CES publishes total nonfarm (not by industry). Two closings (preliminary, final) rather than three:

For vintage $v \in \{1, 2\}$:

$$y_{st,t,v}^{CES,SA} \sim N\left(\alpha_{st}^{CES} + \lambda_{st}^{CES} \cdot \sum_s g_{(st,s),t}^{total,SA}, \sigma_{CES,st,SA,v}^2\right)$$

State CES loads on the **sum of cell-level total SA growth across all supersectors within the state** — a partial observation constraining the geographic aggregate but not the industry composition within the state.

#### 4.2.6 Time-Varying Provider Bias

Provider bias drifts via RW1 with QCEW error-correction:

$$\alpha_{p,j,t} = \alpha_{p,j,t-1} + \omega_{p,j,t} - \kappa_p \cdot d_{p,j,t-L}$$

where:

- $\omega_{p,j,t} \sim N(0, \sigma_\omega^2)$ is the RW1 innovation, $\sigma_\omega \sim \text{Half-N}(0, 0.005)$
- $d_{p,j,t-L}$ is the discrepancy between provider $p$'s estimate and QCEW at lag $L$
- $\kappa_p \sim \text{Beta}(3, 3)$ is the provider-specific error-correction speed

The initial bias $\alpha_{p,j,0}$ retains the nested decomposition above.

#### 4.2.7 QCEW Forecast Model

During the 5–6 month QCEW publication lag, forecasted QCEW extends conditioning to the present:

$$\hat{y}_{j,t}^{QCEW} = \mu_{j,t|t-L} + \gamma_j^{LQ} \cdot LQ_j \cdot (\hat{y}_{s(j),t}^{sector} - \hat{y}_t^{national}) + X_{j,t}'\beta^{fcst} + \xi_{j,t}^{fcst}$$

$$\xi_{j,t}^{fcst} \sim N(0, \sigma_{fcst}^2 \cdot (1 + \rho \cdot h_t))$$

where $LQ_j$ is the location quotient of cell $j$ and $h_t$ is a forecast horizon variable that increases noise for longer-horizon forecasts.

**Conditioning on forecasted QCEW** uses $\sigma_{QCEW,fcst}^2 \gg \sigma_{QCEW}^2$, reflecting forecast uncertainty rather than census noise.

#### 4.2.8 Full-Hierarchy MinT

The summing matrix $\mathbf{S}$ encodes three levels: national, supersectors, and state × supersector cells. The $\mathbf{W}$ matrix uses:

- **Empirical estimates** where revision data exists (national, national × sector, state total nonfarm)
- **Hierarchical shrinkage estimates** for state × industry cells where revision data is unavailable

The Dunn et al. (2026) granularity-accuracy relationship ($-\ln(\text{RMSE}) \approx -\frac{1}{2}\ln(\text{cell size})$) provides structural form for interpolating expected noise at finer granularities.

**Note on state aggregation:** CES states do not sum to national by design — they are independently estimated. The structural aggregation constraint is imposed on the **latent state** (true employment does aggregate) while measurement equations reflect the independent estimation procedures.

### 4.3 Information Regimes

- **Regime A (~Jan 1 to Jan 9): Provider only.** Both industry and geographic decompositions rely on provider signals. Widest uncertainty bands.

- **Regime B (~Jan 9 to Jan 24): National CES with supersector detail, no state CES.** The industry narrative is **anchored**. The geographic narrative relies on provider signals and the nested hierarchical prior. This is the **highest-value regime**: the national number is known, the industry story is told, and the model provides early geographic intelligence before state CES arrives.

- **Regime C (~Jan 24 onward): Full information.** State CES arrives, geography is anchored. Full decomposition available.

**February–March benchmark asymmetry.** National benchmarks arrive in February; state benchmarks in March. During this window, the model's state-level estimates may be more accurate than official state CES for benchmarked months, because the model conditions on the QCEW that state benchmarks will eventually use.

### 4.4 Implementation Plan

1. **Phase 1 — Cell-level data.** Build state × supersector QCEW and provider cell-level data ingestion. Extend vintage store to handle state-level data.

2. **Phase 2 — Hierarchy module.** Implement `hierarchy.py` with nested random effects. Use non-centered parameterization throughout. Build geographic (region/division/state) and industry (domain/supersector) mappings.

3. **Phase 3 — Cell model.** Implement `build_cell_model()` with vectorized cell-level states, nested BD, industry-parent seasonal shrinkage. Start with static bias, add time-varying bias in Phase 5.

4. **Phase 4 — QCEW forecasting and state CES.** Implement `qcew_forecast.py` and CES state measurement equations. Wire up regime-aware data masking.

5. **Phase 5 — Time-varying bias.** Add RW1 bias dynamics with QCEW error-correction. This is computationally expensive ($P \times J \times T$ parameters) and may require variational initialization.

6. **Phase 6 — Full MinT and pipeline.** Extend reconciliation to 3 levels. Build full pipeline with regime-aware output. Validate coherence at all levels.

### 4.5 Additional Outputs

All outputs from Release 2, plus:

| Output | Description |
|---|---|
| `cell_nowcasts_reconciled` | Full state × supersector reconciled estimates |
| `geographic_contributions` | State-level contributions to national change |
| `top_geographic_drivers` | States contributing most to national change |
| `variance_components` | $\tau^2$ at each hierarchy level |
| `geographic_effects` | Region, division, state effects |
| `industry_effects` | Domain, supersector effects |
| `provider_industry_interactions` | Which providers excel in which industries |
| `effective_shrinkage` | Cell-level shrinkage toward each parent |
| `bias_trajectories` | $\alpha_{p,j,t}$ time series by provider and cell |
| `error_correction_speeds` | $\kappa_p$ by provider |
| `qcew_forecast` | Forecasted QCEW by cell |
| `regime_uncertainty` | How uncertainty bands evolve across Regimes A → B → C |

### 4.6 Validation Metrics

- National RMSE (must not degrade vs. Releases 1–2)
- Coherence: cells sum to supersectors sum to national
- Cell-level coverage against QCEW
- Regime-specific accuracy: geographic decomposition stability from Regime B to Regime C
- Variance decomposition by hierarchy level
- Time-varying bias detection: does the model identify known composition shifts?
- QCEW forecast accuracy at various horizons
- Sparse-cell shrinkage: do nested priors outperform exchangeable (Release 2) for small cells?

### 4.7 Computational Scaling

Approximate parameters: ~100 + 10×P + hierarchy + P×J + T×P×J (bias), where $J \approx 550$ cells and $T$ = time periods. Expected runtime: 3–6 hours with nutpie + JAX backend. Requires batched non-centered parameterizations and potentially variational initialization for MCMC.

---

## 5. Cross-Cutting Infrastructure

### 5.1 ProviderConfig Extensions

The `ProviderConfig` dataclass evolves across releases:

| Field | Release 1 | Release 2 | Release 3 |
|---|---|---|---|
| `name`, `file`, `index_col` | ✓ | ✓ | ✓ |
| `error_model` | `'iid'` \| `'ar1'` | `'iid'` \| `'ar1'` | `'iid'` \| `'ar1'` |
| `births_file`, `births_col` | Optional | Optional | Optional |
| `cell_file` | — | Required (parquet) | Required (parquet) |
| `cell_level` | — | `'supersector_x_region'` | `'state_x_supersector'` |

Adding a new provider at any release requires only a `ProviderConfig` entry:

```python
ProviderConfig(
    name='PP3',
    file='alt_nfp_index_3.csv',
    index_col='pp_index_3',
    error_model='ar1',
    births_file='alt_nfp_births_3.csv',
    births_col='pp3_births',
    cell_file='alt_nfp_cells_3.parquet',
    cell_level='supersector_x_region',
)
```

### 5.2 Vintage Data Management

All releases share a common vintage tracking infrastructure based on the Hive-partitioned vintage store in `ingest/vintage_store.py`. Key requirements:

- **CES vintages:** 3 prints (national), 2 prints (state). Track publication dates from BLS schedule scraping in `lookups/publication_dates.py`.
- **QCEW vintages:** Quarterly publications with revision history from 2017Q1. Publication dates keyed by `(year, quarter)` tuple.
- **Provider vintages:** Real-time (not revised). Panel construction dates define the effective vintage.
- **As-of reconstruction:** For any target date, the system must reconstruct the exact information set available, censoring all sources to their publication status. Currently implemented for CES only; QCEW and provider censoring are Release 1 completion tasks.

### 5.3 Pseudo-Establishment Pipeline

Shared infrastructure across releases, implemented in `repr_correction.py`:

1. **Geoclustering:** Cluster employees within each client by geographic proximity to construct pseudo-establishments.
2. **Cell assignment:** Assign pseudo-establishments to supersector × Census region cells. Industry from client NAICS (reliable at supersector level).
3. **Cell-level signals:** Compute continuing-units growth within each cell from the frozen measurement panel.
4. **National composite (Release 1):** QCEW-weighted average of cell signals: $y_{p,t}^G = \sum_c w_{c,t}^{QCEW} \cdot y_{p,c,t}^G$
5. **Supersector aggregates (Release 2):** Aggregate cell signals to supersector level.
6. **Raw cell signals (Release 3):** Pass through to cell-level measurement equations.

For cells where a provider has no coverage, weight is redistributed proportionally across covered cells within the same supersector or region, preserving the marginal industry and geographic distributions as closely as possible.

### 5.4 Information Regime Tracking

New module (or `config.py` extension) that determines the current information regime based on the reference month and current date:

```python
class InfoRegime(Enum):
    A = 'provider_only'           # ~Jan 1 to Jan 9
    B = 'national_ces_available'  # ~Jan 9 to Jan 24
    C = 'full_information'        # ~Jan 24 onward
```

The regime determines which measurement equations are active and which uncertainty bands to report. Regime determination uses the BLS publication dates from `lookups/publication_dates.py`.

### 5.5 Forecast Production Workflow

All releases follow the same four-step forecast production workflow:

1. **Historical estimation:** Run MCMC on all data through $T-1$
2. **BD forecast:** Estimate $BD_T$ using cyclical indicators, provider birth rates, and lagged QCEW BD proxy
3. **Continuing-units filtering:** Filter $\mu_T^{cont}$ from all payroll provider data for current month $T$
4. **Nowcast:** Posterior predictive distribution for $\mu_T = \mu_T^{cont} + BD_T$

Releases 2–3 add a reconciliation step after independent model estimation.

---

## 6. Testing and Validation Strategy

### 6.1 Unit Tests

- **Model construction:** Verify `build_model()`, `build_industry_model()`, `build_cell_model()` produce valid PyMC models with correct shapes.
- **Data loading:** Verify `load_data()` handles missing providers, CES censoring, and QCEW censoring correctly.
- **Prior ranges:** Sample from priors and verify parameters fall within physically meaningful ranges.
- **Reconciliation:** Verify MinT produces coherent forecasts (supersectors sum to national, cells sum to supersectors).
- **Vintage dating:** Verify as-of reconstruction correctly censors each source.
- **Provider addition:** Add a simulated provider via `ProviderConfig` and verify model builds without code changes.

### 6.2 Integration Tests

- **End-to-end pipeline:** Run full pipeline on synthetic data with known ground truth. Verify posterior concentrates near true values.
- **Provider addition:** Add a simulated provider via `ProviderConfig` and verify model adapts without code changes.
- **Regime transitions:** Verify output structure changes correctly as information regime shifts from A to B to C.

### 6.3 Backtesting Protocol

The backtesting protocol is **leakage-safe** and **expanding-window**:

- **Outer loop:** All months with provider data and subsequent CES releases for evaluation.
- **Inner loop:** For each month, censor all data sources to their as-of publication status.
- **Evaluation targets:** First-print CES (primary), benchmarked CES (secondary), QCEW (gold standard).
- **COVID exclusion:** 2020–2021 excluded from standard metrics but reported separately.
- **Benchmark backtest:** Specialized loop over benchmark years at 5 horizons (T−12, T−9, T−6, T−3, T−1) per `benchmark_prediction_alt_nfp_v2`.

### 6.4 Validation Gates

Each release must pass the following gates before deployment:

| Release | Gate Criterion | Metric |
|---|---|---|
| 1 | National RMSE below naive RW baseline | OOS RMSE vs. benchmarked CES |
| 1 | 80% CI coverage ≥ 75% | Empirical coverage rate |
| 2 | National RMSE does not degrade vs. Release 1 | Paired comparison |
| 2 | Coherence: supersectors sum exactly to national | Max absolute violation < 1e-10 |
| 2 | Industry narrative stability from Regime B to C | Rank correlation of supersector contributions |
| 3 | National RMSE does not degrade vs. Releases 1–2 | Paired comparison |
| 3 | Full coherence: cells → supersectors → national | Max absolute violation < 1e-10 |
| 3 | Sparse-cell shrinkage: nested priors outperform exchangeable | Cell-level RMSE comparison |

---

## 7. Computational Requirements

|  | Release 1 | Release 2 | Release 3 |
|---|---|---|---|
| **Parameters** | ~15 + 5×P | ~50 + 10×P + 8×$J_s$ | ~100 + 10×P + hierarchy + P×J + T×P×J |
| **Runtime** | Minutes | 10–30 min | 3–6 hours |
| **Sampler** | nutpie (default) | nutpie | nutpie + JAX; variational init |
| **Hardware** | Apple M4 Max (dev) | Apple M4 Max (dev) | M4 Max or cloud GPU |
| **Sampling config** | DEFAULT: 8K draws / 6K tune / 4 chains | MEDIUM: 4K/3K/4ch | LIGHT for dev; DEFAULT for production |

P = number of providers, $J_s$ = number of supersectors (~10), J = number of state × supersector cells (~550), T = time periods.

Sampler configurations are defined in `sampling.py` and remain unchanged across releases. Release 3 at full scale will require careful attention to sampler efficiency: batched non-centered parameterizations prevent Neal's funnel pathology in sparse cells.

---

## 8. Appendix: Release Comparison Matrix

| Feature | Release 1 | Release 2 | Release 3 |
|---|---|---|---|
| **Scope** | National | National + Supersectors | National + State × Supersector |
| **Multi-Provider** | Yes (repr.-corrected) | Yes | Yes |
| **Birth/Death** | Structural (national) | By supersector | Nested by industry |
| **Seasonal** | Fourier (time-varying) | Hierarchical Fourier | Hier. Fourier + industry-parent |
| **Hierarchy** | — | Exchangeable | Nested (geo + industry) |
| **MinT** | No | Yes (national ↔ supersectors) | Yes (national ↔ supersectors ↔ cells) |
| **QCEW** | Anchored (lagged) | Anchored (lagged) | Forecasted (real-time) |
| **CES Vintages** | Yes (national, 3 prints) | + supersector, 3 prints | + state, 2 prints |
| **Time-Varying Bias** | No | No | Yes (RW1 + QCEW error-correction) |
| **Info Regimes** | N/A | B: industry anchored | B: industry anchored; C: geography anchored |
| **Regime B Narrative** | National only | "Which industries?" | + preliminary "Where?" (provider-based) |

---

*This specification is a living document. Implementation details may evolve as empirical results from earlier releases inform design decisions for later ones. The staged release approach ensures each increment is independently valuable and backward-compatible.*