# Staged Implementation Plan: Bayesian NFP Nowcasting System

## Overview

This document describes a staged release plan for a Bayesian state space NFP nowcasting system. Each release is a fully functional system capable of producing NFP forecasts. Subsequent releases add complexity while maintaining backward compatibility with earlier functionality.

### Design Principles

1.  **Forecast accuracy before narrative:** Birth/death correction (the fundamental limitation of continuing-units data) is prioritized over geographic/industry decomposition.

2.  **Multi-provider from the start:** The measurement error framework naturally accommodates multiple providers as independent noisy observations of latent truth. Designing for this from Release 2 avoids retrofitting.

3.  **Coherent hierarchical forecasts:** Once cell-level estimation begins (Release 4), dual national/cell estimation with MinT reconciliation ensures cell contributions sum exactly to the national forecast.

# Release 1: National Measurement Error Model (Multi-Payroll Provider)

## Objective

Produce a national NFP nowcast by treating each payroll provider's national aggregate as a noisy signal of true employment growth.

## System Description

Multiple providers observe the same latent national employment growth $\mu_t$, each with their own bias, signal loading, and noise level. The model extracts the common signal while learning provider-specific characteristics.

Even with a single payroll provider initially, the framework is ready for additional payroll providers.

### Data Inputs

| Source                              | Frequency | Lag       | Description                                                                                                                                                                                                                |
|----------------------|-------------|----------------|----------------------|
| Payroll Provider *p* National Index | Monthly   | Real-time | Continuing-units employment growth constructed from provider microdata using a rotating, frozen measurement panel aligned to the CES reference week (stabilized clients only; exits removed without counting as job loss). |
| Official NFP                        | Monthly   | ~3 weeks | BLS CES headline number                                                                                                                                                                                                    |

### Provider Series Construction: Rotating, Frozen Measurement Panel

Notation: let k(t) denote the active panel at time t; the observed provider series $y_{p,t}$ is the panel-based continuing-units growth signal.

Within the frozen window, compute month-to-month growth using matched client observations only. Clients that administratively exit the provider during the window are removed from the panel and do not contribute negative change; the measurement target is the intensive-margin employment change of continuing units.

Panel mechanics: at fixed refresh intervals (e.g., quarterly), define an eligible set of clients that have satisfied a stabilization rule (e.g., ≥K consecutive reference periods or change-point-based stabilization). Freeze this set for the panel window; do not add new clients mid-window even if they become eligible later.

To align payroll-provider measures with CES-style continuing-units concepts---and to prevent provider onboarding/offboarding from masquerading as economic job flows---the provider's national index is computed on a rotating, internally frozen panel of clients.

### Complete Model Specification

**Latent State (True National Employment Growth):**

Here, $y_{p,t}$ refers to the rotating-panel continuing-units growth measure described above (not a raw level change that includes client entry/exit).

$$\mu_{t} = \mu_{t - 1} + \eta_{t},\quad\eta_{t} \sim N\left( 0,\sigma_{\eta}^{2} \right)$$

**Data Likelihood --- Official NFP:**

$$y_{t}^{NFP} = \mu_{t} + \varepsilon_{t}^{NFP},\quad\varepsilon_{t}^{NFP} \sim N\left( 0,\sigma_{NFP}^{2} \right)$$

**Data Likelihood --- Provider p National Aggregate:**

$$y_{p,t}^{G} = \alpha_{p} + \lambda_{p}\mu_{t} + \varepsilon_{p,t}^{G},\quad\varepsilon_{p,t}^{G} \sim N\left( 0,\sigma_{G,p}^{2} \right)$$

where for each payroll provider p:

- $\alpha_p$ is payroll provider-specific bias

- $\lambda_p$ is payroll provider-specific signal loading

- $\sigma^{2}_{G,p}$ is payroll provider-specific noise variance

**Hierarchical Priors over Providers:**

*Bias:*

$$\alpha_{p}|\mu_{\alpha},\tau_{\alpha} \sim N\left( \mu_{\alpha},\tau_{\alpha}^{2} \right)$$

$$\mu_{\alpha} \sim N\left( 0,{0.5}^{2} \right)$$

$$\tau_{\alpha} \sim \text{Half-N}(0,0.3)$$

*Signal Loading:*

$$\lambda_{p}|\mu_{\lambda},\tau_{\lambda} \sim N\left( \mu_{\lambda},\tau_{\lambda}^{2} \right)$$

$$\mu_{\lambda} \sim N\left( 1,{0.25}^{2} \right)$$

$$\tau_{\lambda} \sim \text{Half-N}(0,0.2)$$

*Observation Noise:*

$$\log\left( \sigma_{G,p} \right) \sim N\left( \mu_{\sigma},\tau_{\sigma}^{2} \right)$$

$$\mu_{\sigma} \sim N\left( - 3,{0.5}^{2} \right)$$

$$\tau_{\sigma} \sim \text{Half-N}(0,0.3)$$

**Other Priors:**

$$\mu_{0} \sim N\left( 0,{0.1}^{2} \right)$$

$$\sigma_{\eta} \sim \text{Half-N}(0,0.02)$$

$$\sigma_{NFP} \sim \text{Half-N}(0,0.005)$$

### Forecast Production

1.  **Historical Estimation:** Run MCMC on all available data through time T-1
2.  **Filtering:** Incorporate all payroll provider data for current month T
3.  **Nowcast:** Posterior predictive distribution for $\mu_T$

### Output Specification

| Output                    | Description                                          |
|---------------------------|---------------------------------------------|
| `nfp_nowcast`             | Point estimate (posterior mean of $\mu_T$)               |
| `nfp_nowcast_std`         | Uncertainty (posterior std of $\mu_T$)                   |
| `nfp_nowcast_ci`          | 80% and 95% credible intervals                       |
| `provider_signal_quality` | Posterior mean of $\lambda_p$ for each payroll provider      |
| `provider_bias`           | Posterior mean of $\alpha_p$ for each payroll provider      |
| `provider_weights`        | Effective weight of each payroll provider in nowcast |

### Validation Metrics

-   Out-of-sample RMSE against NFP releases
-   Coverage of credible intervals
-   Comparison to naive forecast (random walk on NFP)
-   Provider-specific signal quality rankings

### Limitations

-   **No birth/death adjustment:** Providers measure continuing units only
-   No geographic or industry decomposition
-   No QCEW anchoring
-   Cannot explain drivers of forecast

# Release 2: National Birth/Death Adjustment (Multi-Provider)

## Objective

Correct for the systematic gap between continuing units' data (what payroll providers measure) and total employment (what NFP captures) at the national level.

## Incremental Changes from Release 1

-   Decompose true employment into continuing-units and birth/death components

### Provider Series Construction: Rotating, Frozen Measurement Panel

This ensures that the provider likelihood loads on $\mu^{cont}_t$ and that administrative churn (client onboarding/offboarding) is excluded from the measurement equation rather than implicitly absorbed into the birth/death component.

In Release 2, payroll-provider observations are explicitly interpreted as continuing-units (intensive-margin) employment change. Accordingly, each provider's input series must be constructed using the rotating, frozen measurement-panel methodology.

-   Model BD as function of cyclical indicators
-   Anchor BD estimates with lagged QCEW birth/death data
-   Providers now explicitly measure continuing-units employment

## System Description

Each payroll provider measures employment change from continuing units only. The birth/death contribution is modeled separately and added to get total employment. This correction is critical at business cycle turning points.

### Data Inputs

| Source                            | Frequency | Lag        | Description                                                                                                                                                                                                                |
|----------------------|-------------|----------------|----------------------|
| Payroll Provider p National Index | Monthly   | Real-time  | Continuing-units employment growth constructed from provider microdata using a rotating, frozen measurement panel aligned to the CES reference week (stabilized clients only; exits removed without counting as job loss). |
| Official NFP                      | Monthly   | ~3 weeks  | Total employment (includes BD)                                                                                                                                                                                             |
| Cyclical Indicators               | Monthly   | Real-time  | GDP growth, unemployment rate, financial conditions                                                                                                                                                                        |
| QCEW National                     | Quarterly | 5-6 months | Near-census total employment                                                                                                                                                                                               |
| QCEW Birth/Death                  | Quarterly | 5-6 months | Actual BD contribution from QCEW microdata                                                                                                                                                                                 |

### Complete Model Specification

As in Release 1, $y_{p,t}$ is computed from the active panel k(t) and represents continuing-units growth only; client entry/exit are excluded by construction.

**Latent State Decomposition:**

$$\mu_{t} = \mu_{t}^{cont} + BD_{t}$$

where $\mu^{cont}_t$ is continuing-units change and $BD_t$ is net birth/death contribution.

**Latent Dynamics:**

$$\mu_{t}^{cont} = \mu_{t - 1}^{cont} + \eta_{t}^{cont},\quad\eta_{t}^{cont} \sim N\left( 0,\sigma_{\eta,cont}^{2} \right)$$

**Data Likelihood --- Official NFP (Total Employment):**

$$y_{t}^{NFP} = \mu_{t}^{cont} + BD_{t} + \varepsilon_{t}^{NFP},\quad\varepsilon_{t}^{NFP} \sim N\left( 0,\sigma_{NFP}^{2} \right)$$

**Data Likelihood --- Payroll Provider p (Continuing Units):**

$$y_{p,t}^{G} = \alpha_{p} + \lambda_{p}\mu_{t}^{cont} + \varepsilon_{p,t}^{G},\quad\varepsilon_{p,t}^{G} \sim N\left( 0,\sigma_{G,p}^{2} \right)$$

**Data Likelihood --- QCEW (Lagged, Total Employment):**

$$y_{t - L}^{QCEW} = \mu_{t - L}^{cont} + BD_{t - L} + \varepsilon_{t - L}^{QCEW},\quad\varepsilon_{t - L}^{QCEW} \sim N\left( 0,\sigma_{QCEW}^{2} \right)$$

**Birth/Death Model:**

$$BD_{t} = \phi_{0} + \phi_{1}X_{t}^{cycle} + \phi_{2}BD_{t - L}^{QCEW} + \xi_{t},\quad\xi_{t} \sim N\left( 0,\sigma_{BD}^{2} \right)$$

**Priors:**

*Hierarchical priors over payroll providers:* Same as Release 1

*Birth/death parameters:*

$$\phi_{0} \sim N\left( 0,{0.01}^{2} \right)$$

$$\phi_{1} \sim N\left( 0.5,{0.2}^{2} \right)\quad\text{[procyclical prior]}$$

$$\phi_{2} \sim N\left( 0.7,{0.15}^{2} \right)\quad\text{[persistence]}$$

$$\sigma_{BD} \sim \text{Half-N}(0,0.008)$$

*QCEW noise:*

$$\sigma_{QCEW} \sim \text{Half-N}(0,0.003)$$

### Forecast Production

1.  **Historical Estimation:** Run MCMC on all data through T-1
2.  **BD Forecast:** Estimate $BD_T$ using cyclical indicators and lagged QCEW BD
3.  **Continuing-Units Filtering:** Filter $\mu^{cont}_T$ from all payroll provider data
4.  **Total Employment Nowcast:** $\mu_T$ = $\mu^{cont}_T$ + $BD_T$

### Output Specification

All outputs from Release 1, plus:

| Output                     | Description                                   |
|----------------------------|--------------------------------------------|
| `bd_contribution`          | Estimated $BD_T$ contribution to current NFP    |
| `continuing_units_nowcast` | $\mu^{cont}_T$ (what payroll providers measure) |
| `cyclical_sensitivity`     | Posterior of $\phi_1$ (BD response to cycle)       |
| `bd_persistence`           | Posterior of $\phi_2$                              |

### Validation Metrics

-   RMSE improvement over Release 0, especially at turning points
-   BD estimates vs. realized QCEW BD (with lag)
-   Cyclical sensitivity $\phi_1$ should be positive and significant

### Limitations

-   National-level only---no geographic or industry decomposition
-   BD model doesn't capture industry heterogeneity
-   Time-invariant payroll provider bias
-   Cannot explain forecast drivers beyond BD split

# Release 3: Cell-Level Estimation with Dual Framework and MinT (Multi-Payroll Provider)

## Objective

Produce NFP nowcasts with geographic × industry decomposition, maintaining birth/death adjustment, multi-provider integration, and coherent national alignment.

## Incremental Changes from Release 2

-   Estimate latent employment growth for each cell (geographic unit × supersector)
-   **Independent national state** alongside cell-level states
-   **MinT reconciliation** ensures cells sum to national
-   Extend BD model to cell level with industry effects
-   Payroll Provider × cell parameters with hierarchical shrinkage
-   Add cell-level QCEW observations
-   Exchangeable hierarchical priors for partial pooling

## System Description

This release introduces cell-level estimation within the **dual-estimation framework**: we estimate both an independent national state (informed by NFP) and cell-level states (informed by payroll provider/QCEW data), then reconcile via MinT.

Each payroll provider now has cell-level observations, with payroll provider-specific and cell-specific parameters. Some payroll providers may have better coverage in certain industries or geographies.

### Data Inputs

| Source                              | Frequency | Lag        | Description                                    |
|----------------------|-------------|----------------|----------------------|
| Payroll Provider p Cell-Level Index | Monthly   | Real-time  | Employment growth by geo × supersector         |
| Official NFP                        | Monthly   | ~3 weeks  | National headline                              |
| Cyclical Indicators                 | Monthly   | Real-time  | GDP growth, unemployment, financial conditions |
| QCEW Cell-Level                     | Quarterly | 5-6 months | Near-census employment by cell                 |
| QCEW Birth/Death                    | Quarterly | 5-6 months | BD contribution by cell or national            |

### Complete Model Specification

**Latent States --- Dual Structure:**

*National State (Independent):*

$$\mu_{t}^{nat} = \mu_{t - 1}^{nat} + \eta_{t}^{nat},\quad\eta_{t}^{nat} \sim N\left( 0,\sigma_{\eta,nat}^{2} \right)$$

*Cell-Level States:* Let j index cells:

$$\mu_{j,t}^{cont} = \mu_{j,t - 1}^{cont} + \eta_{j,t},\quad\eta_{j,t} \sim N\left( 0,\sigma_{\eta}^{2} \right)$$

Total cell employment: $\mu_{j,t} = \mu_{j,t}^{cont} + BD_{j,t}$

**Data Likelihood --- Official NFP (Informs National State):**

$$y_{t}^{NFP} = \mu_{t}^{nat} + \varepsilon_{t}^{NFP},\quad\varepsilon_{t}^{NFP} \sim N\left( 0,\sigma_{NFP}^{2} \right)$$

**Data Likelihood --- Payroll Provider p, Cell j (Continuing Units):**

$$y_{p,j,t}^{G} = \alpha_{p,j} + \beta_{p,j}\mu_{j,t}^{cont} + \varepsilon_{p,j,t}^{G},\quad\varepsilon_{p,j,t}^{G} \sim N\left( 0,\sigma_{p,j}^{2} \right)$$

**Data Likelihood --- QCEW (Cell Level, Lagged):**

$$y_{j,t - L}^{QCEW} = \mu_{j,t - L} + \varepsilon_{j,t - L}^{QCEW},\quad\varepsilon_{j,t - L}^{QCEW} \sim N\left( 0,\sigma_{QCEW}^{2} \right)$$

**Birth/Death Model (Cell Level):**

$$BD_{j,t} = \phi_{0} + \phi_{s(j)}^{ind} + \phi_{1}X_{t}^{cycle} + \phi_{2}BD_{j,t - L}^{QCEW} + \xi_{j,t}$$

*Industry Effects (Exchangeable):*

$$\phi_{s}^{ind}|\mu_{\phi},\tau_{\phi} \sim N\left( \mu_{\phi},\tau_{\phi}^{2} \right)$$

**Discrepancy Model (National vs. Sum-of-Cells):**

$$\delta_{t} = \delta_{t - 1} + \omega_{t}^{\delta},\quad\omega_{t}^{\delta} \sim N\left( 0,\sigma_{\delta}^{2} \right)$$

**Hierarchical Priors --- Payroll Provider × Cell Parameters:**

The key innovation is decomposing payroll provider-cell parameters into payroll provider effects, cell effects, and residual:

*Bias:*

$$\alpha_{p,j} = \alpha_{p}^{prov} + \alpha_{j}^{cell} + \alpha_{p,j}^{resid}$$

$$\alpha_{p}^{prov}|\mu_{\alpha,prov},\tau_{\alpha,prov} \sim N\left( \mu_{\alpha,prov},\tau_{\alpha,prov}^{2} \right)$$

$$\alpha_{j}^{cell}|\mu_{\alpha,cell},\tau_{\alpha,cell} \sim N\left( \mu_{\alpha,cell},\tau_{\alpha,cell}^{2} \right)$$

$$\alpha_{p,j}^{resid} \sim N\left( 0,\tau_{\alpha,resid}^{2} \right)$$

*Signal Loading:*

$$\beta_{p,j} = \beta_{p}^{prov} + \beta_{j}^{cell} + \beta_{p,j}^{resid}$$

$$\beta_{p}^{prov}|\mu_{\beta,prov},\tau_{\beta,prov} \sim N\left( \mu_{\beta,prov},\tau_{\beta,prov}^{2} \right)$$

$$\mu_{\beta,prov} \sim N\left( 1,{0.2}^{2} \right)$$

$$\beta_{j}^{cell}|\tau_{\beta,cell} \sim N\left( 0,\tau_{\beta,cell}^{2} \right)$$

$$\beta_{p,j}^{resid} \sim N\left( 0,\tau_{\beta,resid}^{2} \right)$$

*Observation Noise:*

$$\log\left( \sigma_{p,j} \right) = log\left( \sigma_{p}^{prov} \right) + log\left( \sigma_{j}^{cell} \right) + \epsilon_{p,j}^{\sigma}$$

This decomposition allows: - Learning which payroll providers are systematically better/worse (payroll provider effects) - Learning which cells are harder to estimate (cell effects) - Capturing payroll provider-cell interactions (residual)

### Optimal Reconciliation (MinT)

**Base Forecasts:**

$${\widehat{y}}_{t} = \left\lbrack {\widehat{\mu}}_{1,t},...,{\widehat{\mu}}_{J,t},{\widehat{\mu}}_{t}^{nat} \right\rbrack'$$

**Reconciled Forecasts:**

$${\widetilde{y}}_{t} = S\left( S'W^{- 1}S \right)^{- 1}S'W^{- 1}{\widehat{y}}_{t}$$

**Constraint:**

$$\sum_{j}^{}w_{j}{\widetilde{\mu}}_{j,t} + \delta_{t} = {\widetilde{\mu}}_{t}^{nat}$$

### Forecast Production

1.  **Historical Estimation:** MCMC on all data through T-1
2.  **Base Forecast --- National:** Posterior mean of $\mu^{nat}_T$
3.  **Base Forecast --- Cells:** Combine all payroll provider signals, estimate $BD_{j,T}$, compute $\mu_{j,T}$
4.  **Covariance Estimation:** Compute W from historical forecast errors
5.  **MinT Reconciliation:** Coherent forecasts where cells sum to national

### Output Specification

All outputs from Release 2, plus:

| Output                     | Description                                          |
|----------------------------|--------------------------------------------|
| `nfp_nowcast_reconciled`   | Reconciled national forecast                         |
| `cell_nowcasts_reconciled` | Reconciled cell estimates (sum to national)          |
| `cell_contributions`       | $w_j$ × $\tilde{\mu}_{j,T}$ for each cell                         |
| `top_drivers`              | Cells contributing most to national change           |
| `provider_effects`       | $\alpha^{prov}_p$, $\beta^{prov}_p$ for each payroll provider |
| `provider_cell_coverage` | Which payroll providers inform which cells           |
| `discrepancy_estimate`     | Current $\delta_t$                                          |

### Validation Metrics

-   National RMSE
-   Coherence: cells sum exactly to national
-   Cell-level coverage against QCEW
-   Payroll Provider ranking by signal quality

### Limitations

-   Exchangeable priors ignore geographic/industry nesting
-   Time-invariant bias

# Release 4: Nested Hierarchical Structure (Multi-Payroll Provider)

## Objective

Improve cell-level estimation by exploiting nested structure of geography (region → division → state) and industry (domain → supersector).

## Incremental Changes from Release 3

-   Replace exchangeable priors with nested random effects
-   Bias/loading decomposes into geographic, industry, payroll provider, and residual components
-   BD intensity follows nested industry hierarchy
-   Enable finer geographic granularity (states) without overfitting

## System Description

The nested structure reflects true geographic and industry relationships: cells in the same division are more similar than cells in different divisions. Payroll Provider effects interact with this structure---some payroll providers may be better in certain regions or industries.

### Complete Model Specification

**Latent States, Data Likelihoods, Discrepancy, MinT:** Same as Release 3

**Hierarchical Priors with Nested Structure:**

*Bias --- Full Decomposition:*

$$\alpha_{p,j} = \alpha_{p}^{prov} + \alpha_{g(j)}^{geo} + \alpha_{s(j)}^{ind} + \alpha_{p,s(j)}^{prov \times ind} + \alpha_{p,j}^{resid}$$

*Payroll Provider Effects:*

$$\alpha_{p}^{prov} \sim N\left( 0,\tau_{\alpha,prov}^{2} \right)$$

*Geographic Component (Nested):*

$$\alpha_{g}^{geo} = \alpha_{r(g)}^{region} + \alpha_{d(g)}^{div|region} + \alpha_{g}^{state|div}$$

$$\alpha_{r}^{region} \sim N\left( 0,\tau_{region}^{2} \right)$$

$$\alpha_{d}^{div|region} \sim N\left( 0,\tau_{div|region}^{2} \right)$$

$$\alpha_{g}^{state|div} \sim N\left( 0,\tau_{state|div}^{2} \right)$$

*Industry Component (Nested):*

$$\alpha_{s}^{ind} = \alpha_{m(s)}^{domain} + \alpha_{s}^{supersector|domain}$$

$$\alpha_{m}^{domain} \sim N\left( 0,\tau_{domain}^{2} \right)$$

$$\alpha_{s}^{supersector|domain} \sim N\left( 0,\tau_{supersector|domain}^{2} \right)$$

*Payroll Provider × Industry Interaction:*

$$\alpha_{p,s}^{prov \times ind} \sim N\left( 0,\tau_{prov \times ind}^{2} \right)$$

This captures that some payroll providers are better in certain industries (e.g., one payroll provider may have better Manufacturing coverage).

*Residual:*

$$\alpha_{p,j}^{resid} \sim N\left( 0,\tau_{resid}^{2} \right)$$

**Signal Loading (Nested Industry + Payroll Provider):**

$$\beta_{p,j} = \beta_{0} + \beta_{p}^{prov} + \beta_{m(j)}^{domain} + \beta_{s(j)}^{supersector|domain} + \beta_{p,j}^{resid}$$

**Birth/Death Intensity (Nested Industry):**

$$\phi_{s}^{ind} = \phi_{m(s)}^{domain} + \phi_{s}^{supersector|domain}$$

**Variance Component Priors:**

$$\tau_{region} \sim \text{Half-N}(0,0.4)$$

$$\tau_{div|region} \sim \text{Half-N}(0,0.25)$$

$$\tau_{state|div} \sim \text{Half-N}(0,0.15)$$

$$\tau_{domain} \sim \text{Half-N}(0,0.4)$$

$$\tau_{supersector|domain} \sim \text{Half-N}(0,0.25)$$

$$\tau_{prov \times ind} \sim \text{Half-N}(0,0.15)$$

### Output Specification

All outputs from Release 3, plus:

| Output                             | Description                                       |
|----------------------------|--------------------------------------------|
| `variance_components`              | $\tau^{2}$ at each hierarchy level                        |
| `geographic_effects`               | Region, division, state effects                   |
| `industry_effects`                 | Domain, supersector effects                       |
| `provider_industry_interactions` | Which payroll providers excel in which industries |
| `effective_shrinkage`              | Cell-level shrinkage toward each parent           |

### Validation Metrics

-   Improved RMSE for sparse cells vs Release 3
-   Variance decomposition by hierarchy level
-   Payroll Provider × industry interaction patterns

### Limitations

-   Time-invariant bias
-   No QCEW forecasting for real-time conditioning

# Release 5: QCEW Conditioning and Time-Varying Bias (Multi-Payroll Provider)

## Objective

Allow payroll provider bias to evolve over time with QCEW error-correction and forecast QCEW for real-time conditioning.

## Incremental Changes from Release 4

-   Time-varying bias $\alpha_{p,j,t}$ with RW1 dynamics
-   QCEW error-correction anchors bias drift
-   Forecasted QCEW enables real-time conditioning
-   Location quotients and leading indicators predict QCEW

## System Description

Payroll Provider bias drifts over time due to client composition changes, market share shifts, or methodology updates. QCEW error-correction prevents unbounded drift. Forecasting QCEW to the present enables tighter conditioning even during the 5-6 month publication lag.

### Complete Model Specification

**Latent States, Nested Hierarchical Structure, Discrepancy, MinT:** Same as Release 4

**Time-Varying Bias:**

$$\alpha_{p,j,t} = \alpha_{p,j,t - 1} + \omega_{p,j,t} - \kappa_{p}\left( d_{p,j,t - L} \right)$$

where:

- $\omega_{p,j,t}$ ∼ N(0, $\sigma^{2}_\omega$) is RW1 innovation

- $d_{p,j,t-L}$ is discrepancy vs. QCEW at lag L

- $\kappa_p$ is payroll provider-specific error-correction speed

The initial bias $\alpha_{p,j,0}$ retains the nested decomposition from Release 4.

**Payroll Provider-Specific Error-Correction:**

$$\kappa_{p} \sim \text{Beta}(3,3)$$

Different payroll providers may have different bias persistence---some track QCEW more closely than others.

**Priors:**

$$\sigma_{\omega} \sim \text{Half-N}(0,0.005)$$

**QCEW Forecast Model:**

$${\widehat{y}}_{j,t}^{QCEW} = \mu_{j,t|t - L} + \gamma_{j}^{LQ} \cdot LQ_{j} \cdot \left( {\widehat{y}}_{s(j),t}^{sector} - {\widehat{y}}_{t}^{national} \right) + X'_{j,t}\beta^{fcst} + \xi_{j,t}^{fcst}$$

$$\xi_{j,t}^{fcst} \sim N\left( 0,\sigma_{fcst}^{2} \cdot \left( 1 + \rho \cdot h_{t} \right) \right)$$

**Conditioning on Forecasted QCEW:**

$${\widehat{y}}_{j,t}^{QCEW} = \mu_{j,t} + \varepsilon_{j,t}^{fcst},\quad\varepsilon_{j,t}^{fcst} \sim N\left( 0,\sigma_{QCEW,fcst}^{2} \right)$$

### Note on Temporal Dependence

The RW1 prior on $\alpha_{p,j,t}$ sets AR = 1 (unit root), capturing smooth drift. The QCEW error-correction provides bounded behavior. A GP would be theoretically superior but computationally prohibitive.

### Note on QCEW Treatment

The model conditions on QCEW (high-precision observation) rather than benchmarking (constraining sums). The tight $\sigma_{QCEW}$ prior is needed for identification.

### Output Specification

All outputs from Release 4, plus:

| Output                      | Description                                         |
|---------------------------|---------------------------------------------|
| `bias_trajectories`         | $\alpha_{p,j,t}$ time series by payroll provider and cell |
| `error_correction_speeds`   | $\kappa_p$ by payroll provider                             |
| `qcew_forecast`             | Forecasted QCEW by cell                             |
| `provider_bias_stability` | Which payroll providers have more stable bias       |

# Appendix A: Summary Release Comparison

| Release | Scope               | Multi-Payroll Provider | Birth/Death  | Hierarchy    | Dual + MinT | QCEW           | Time-Varying Bias |
|---------|----------|---------|---------|----------|---------|---------|---------|
| 1       | National            | **Yes**                | No           | ---          | No          | No             | No                |
| 2       | National            | Yes                    | **Yes**      | ---          | No          | Lagged         | No                |
| 3       | Geo × Supersector   | Yes                    | Yes (cell)   | Exchangeable | **Yes**     | Lagged         | No                |
| 4       | State × Supersector | Yes                    | Yes (nested) | **Nested**   | Yes         | Lagged         | No                |
| 5       | State × Supersector | Yes                    | Yes (nested) | Nested       | Yes         | **Forecasted** | **Yes**           |

# Appendix B: Multi-Payroll Provider Design Rationale

## Why Multi-Payroll Provider from the Start?

1.  **No retrofitting:** Adding payroll providers later requires restructuring; building it in from Release 1 is cleaner.

2.  **Graceful degradation:** With one payroll provider, hierarchical priors collapse to weakly informative; the framework works but doesn't overcomplicate.

3.  **Immediate benefits:** Even with one payroll provider, the framework reveals payroll provider-specific signal quality metrics.

4.  **Payroll Provider comparison:** When multiple payroll providers exist, the model automatically learns relative strengths by cell/industry.

## Payroll Provider × Cell Interaction

The decomposition $\alpha_{p,j} = \alpha^{prov}_p + \alpha^{cell}_j + \alpha^{prov \times ind}_{p,s(j)} + \alpha^{resid}_{p,j}$ captures:

-   **Payroll Provider main effect:** Some payroll providers are systematically biased
-   **Cell main effect:** Some cells are harder to estimate for all payroll providers
-   **Payroll Provider × industry:** Some payroll providers specialize in certain industries
-   **Residual:** Unexplained payroll provider-cell variation

## Combining Multiple Payroll Provider Signals

For a given cell j, the model combines signals from all payroll providers observing that cell via precision weighting. Payroll Providers with lower noise (higher $\beta_p$, lower $\sigma_{p,j}$) receive more weight.

# Appendix C: Implementation Notes {#appendix-c-implementation-notes-1}

## Non-Centered Parameterization

Essential for sparse cells and payroll providers with limited coverage:

    # Non-centered (recommended)
    α̃_{p,j} ~ N(0, 1)
    α_{p,j} = μ_α + τ_α × α̃_{p,j}

## Computational Scaling

| Release | Approx. Parameters                    | Typical Runtime |
|---------|---------------------------------------|-----------------|
| 1       | ~10 × P                              | Minutes         |
| 2       | ~15 × P                              | Minutes         |
| 3       | ~50 + 10×P + 5×J + P×J               | 30-60 min       |
| 4       | ~100 + 10×P + hierarchy + P×J        | 1-2 hours       |
| 5       | Release 4 + T×P×J (bias trajectories) | 3-6 hours       |

P = number of payroll providers, J = number of cells, T = time periods