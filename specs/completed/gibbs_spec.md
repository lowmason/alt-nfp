# Gibbs Sampler Implementation Spec

**Alt-NFP Nowcasting System — Block Gibbs / FFBS Backend**

Version: 1.0 \| Date: 2026-03-15 \| Author: Lowell Mason

------------------------------------------------------------------------

## 1. Background and Motivation

The current estimation backend uses NUTS (via nutpie / PyMC) on a non-centered parameterization of the full joint posterior. For Release 1 (single national latent state, T ≈ 300), this works well — the latent state dimension is manageable, and the PyMC ecosystem provides ArviZ diagnostics, LOO-CV, and prior/posterior predictive checks out of the box.

For Releases 2–3 the picture changes. Release 3 introduces \~700 cell-level AR(1) latent states (state × supersector), each of length T. The joint latent dimension grows to \~700 × T ≈ 210,000. NUTS must jointly navigate this space with gradient information, and even with non-centered parameterization, the posterior geometry at this scale is likely to produce divergences, low ESS, and prohibitive runtimes.

The alt_nfp model is almost entirely conditionally conjugate. A block Gibbs sampler with Forward Filtering Backward Sampling (FFBS) for the latent states exploits this structure:

1.  **FFBS draws the entire latent trajectory in O(T) per cell**, with no gradient computation. Cell-level states are conditionally independent given the hierarchical parameters, making the latent block embarrassingly parallel.
2.  **Most parameter blocks have closed-form conjugate conditionals** (Normal-Normal, Normal-InverseGamma), reducing each to a direct draw from a known distribution.
3.  **Only 3 + n_AR1_providers scalar parameters** require Metropolis-within-Gibbs steps. These are cheap (no matrix operations, just sums over T terms) and converge quickly with adaptive proposals.

### 1.1 Relationship to Existing Infrastructure

The Gibbs sampler consumes the same `data` dict produced by `panel_to_model_data()` and outputs `az.InferenceData` with the same variable names as the PyMC model. This means the entire downstream pipeline works unchanged: `print_diagnostics()`, `plot_growth_and_seasonal()`, `run_posterior_predictive_checks()`, `compute_precision_budget()`, and the benchmark extraction in `benchmark.py`.

### 1.2 Relationship to Releases

| Release | Recommended Backend | Rationale |
|------------------------|------------------------|------------------------|
| Release 1 (national) | NUTS (primary) + Gibbs (validation) | NUTS is working; Gibbs validates posterior agreement |
| Release 2 (industry) | Gibbs preferred | \~13 latent states × T; NUTS is feasible but Gibbs is faster |
| Release 3 (geo × industry) | Gibbs required | \~700 latent states × T; NUTS is infeasible |

### 1.3 Key Parameterization Difference

The PyMC model uses a **non-centered** parameterization (innovation variables ε_t \~ N(0,1), then g_cont_t reconstructed via `pytensor.scan`). This breaks the φ–σ ridge for NUTS but is incompatible with FFBS.

The Gibbs sampler uses a **centered** parameterization (actual state values g_cont_t). FFBS requires the Markov structure in centered form to exploit the recursive forward-backward factorization. The two parameterizations define the same model — they are related by a deterministic transformation — but the sampler geometries are different, and a given set of priors may behave differently under each parameterization.

### 1.4 Out of Scope

-   Modifications to `build_model()` or the PyMC/NUTS pathway. The Gibbs sampler is a parallel backend, not a replacement.
-   Release 2/3 hierarchical extensions (nested random effects, MinT reconciliation). Those are specified in `releases_spec.md` and will extend the Gibbs sampler once the Release 1 foundation is in place.
-   Particle Gibbs or SMC variants. The model's conjugacy structure makes vanilla FFBS sufficient; particle methods add complexity without benefit here.

------------------------------------------------------------------------

## 2. Implementation Phases

| Phase | Name | Scope | Key Deliverable |
|------------------|------------------|------------------|------------------|
| 1 | Core FFBS + Conjugate Blocks | Latent state, all conjugate parameter blocks | Working Gibbs sampler that reproduces NUTS posteriors on Release 1 |
| 2 | MH Blocks + Adaptation | Non-conjugate scalars (φ, σ_η, ρ_p) with adaptive proposals | Complete sampler with all blocks, tuned MH acceptance |
| 3 | Diagnostics + Validation | ArviZ output, convergence checks, NUTS cross-validation | Production-ready backend with formal agreement tests |
| 4 | Performance + Parallelism | Numba JIT for FFBS, multi-chain parallelism, cell-level vectorization | Release 3-ready performance |

### 2.1 Dependencies Between Phases

Phase 1 is a prerequisite for all subsequent phases. Phase 2 depends on Phase 1 (the MH blocks condition on FFBS output). Phase 3 depends on Phase 2 (diagnostics require the full sampler). Phase 4 is independent of Phase 3 (performance optimization can proceed in parallel with validation).

```         
Phase 1 (Core FFBS + Conjugate)
    │
    ├──► Phase 2 (MH Blocks + Adaptation)
    │        │
    │        ├──► Phase 3 (Diagnostics + Validation)
    │        └──► Phase 4 (Performance + Parallelism)
    │
    └──► Ad-hoc use with fixed φ/σ_η for debugging
```

------------------------------------------------------------------------

## 3. Model Specification in Gibbs Form

This section restates the Release 1 model specification from `releases_spec.md` §2.2 in the centered parameterization required by FFBS, and identifies the conjugacy class of each conditional distribution.

### 3.1 Latent State (Centered Parameterization)

$$g^{cont}_t = \mu_{g,e(t)} + \phi_{e(t)} \left( g^{cont}_{t-1} - \mu_{g,e(t)} \right) + \eta_t, \quad \eta_t \sim N(0, \sigma^2_\eta)$$

where $e(t) \in \{0, 1, 2\}$ indexes the era (Pre-GFC, Post-GFC, Post-COVID) at time $t$. The initial condition is drawn from the stationary distribution of era 0:

$$g^{cont}_0 \sim N\!\left(\mu_{g,0}, \; \frac{\sigma^2_\eta}{1 - \phi_0^2}\right)$$

### 3.2 Composite Signals (Deterministic Given Components)

Given $g^{cont}_t$, the birth/death component $BD_t$, and the seasonal component $s_t$:

| Signal | Formula | Feeds into |
|------------------------|------------------------|------------------------|
| Continuing-units NSA | $g^{cont,NSA}_t = g^{cont}_t + s_t$ | Provider likelihoods |
| Total SA | $g^{total,SA}_t = g^{cont}_t + BD_t$ | CES SA likelihood |
| Total NSA | $g^{total,NSA}_t = g^{cont}_t + BD_t + s_t$ | CES NSA + QCEW likelihoods |

### 3.3 Birth/Death Component

$$BD_t = \phi_0 + \phi_1 \cdot x^{births}_t + \phi_2 \cdot x^{qcew\_bd}_t + \sum_{j=1}^{n_c} \phi_{3,j} \cdot x^{cycle}_{j,t} + \sigma_{BD} \cdot \xi_t, \quad \xi_t \sim N(0, 1)$$

Write this as $BD_t = \mathbf{x}_t' \boldsymbol{\phi} + \sigma_{BD} \cdot \xi_t$ where $\mathbf{x}_t$ stacks the intercept, birth rate, QCEW BD proxy, and cyclical indicators.

### 3.4 Fourier Seasonal Component

$$s_t = \sum_{k=1}^{K} \left[ A_k(y(t)) \cos\!\left(\frac{2\pi k \, m(t)}{12}\right) + B_k(y(t)) \sin\!\left(\frac{2\pi k \, m(t)}{12}\right) \right]$$

where $y(t)$ is the year index and $m(t)$ is the month-of-year. The annual coefficients follow independent Gaussian random walks:

$$A_k(y) = A_k(y-1) + \sigma_{\omega,k} \cdot \epsilon^A_{k,y}, \quad B_k(y) = B_k(y-1) + \sigma_{\omega,k} \cdot \epsilon^B_{k,y}$$

### 3.5 Observation Equations

**QCEW (truth anchor):**

$$y^{QCEW}_t \sim N\!\left(g^{total,NSA}_t, \; \sigma^2_{QCEW}(t)\right)$$

where $\sigma^2_{QCEW}(t) = \sigma^2_{QCEW,mid}$ for M3+M1 months and $\sigma^2_{QCEW,boundary}$ for M2 months.

**CES (vintage-specific):**

$$y^{CES,SA}_{t,v} \sim N\!\left(\alpha_{CES} + \lambda_{CES} \cdot g^{total,SA}_t, \; \sigma^2_{CES,v}\right), \quad v \in \{1, 2, 3\}$$

$$y^{CES,NSA}_{t,v} \sim N\!\left(\alpha_{CES} + \lambda_{CES} \cdot g^{total,NSA}_t, \; \sigma^2_{CES,v}\right)$$

**Provider (iid):**

$$y^p_t \sim N\!\left(\alpha_p + \lambda_p \cdot g^{cont,NSA}_t, \; \sigma^2_p\right)$$

**Provider (AR(1)):**

$$y^p_t \mid y^p_{t-1} \sim N\!\left(\mu^{base}_{p,t} + \rho_p \left(y^p_{t-1} - \mu^{base}_{p,t-1}\right), \; \sigma^2_p\right)$$

where $\mu^{base}_{p,t} = \alpha_p + \lambda_p \cdot g^{cont,NSA}_t$.

### 3.6 Prior Specification (Gibbs-Adapted)

Priors are identical to the PyMC model except where conjugacy requires a distributional swap. Swaps are minimal and calibrated to match the original prior's effective location and spread.

| Parameter | PyMC Prior | Gibbs Prior | Conjugacy Class | Swap Rationale |
|---------------|---------------|---------------|---------------|---------------|
| $\mu_{g,e}$ | $N(0.001, 0.005^2)$ | Same | Normal-Normal | No change needed |
| $\phi_e$ | $\text{Beta}(18, 2)$ | Same | **Non-conjugate** (MH) | — |
| $\sigma_\eta$ | $\text{Half-N}(0, 0.005)$ | Same | **Non-conjugate** (MH) | — |
| $\boldsymbol{\phi}_{BD}$ | $N(\mu_0, \Sigma_0)$ | Same | Normal-Normal | No change needed |
| $\sigma^2_{BD}$ | $\text{LogNormal}(\mu_{ln}, \sigma_{ln})$ | $\text{IG}(\alpha_{BD}, \beta_{BD})$ | Normal-IG | Match mode ≈ 0.001², 90th pctile |
| $A_k(0), B_k(0)$ | $N(0, 0.015^2)$ | Same | Normal-Normal (via FFBS init) | No change needed |
| $\sigma^2_{\omega,k}$ | $\text{Half-N}(0, 0.005/k)$ | $\text{IG}(\alpha_\omega, \beta_{\omega,k})$ | Normal-IG | Match mode, scale as $1/k$ |
| $\sigma^2_{QCEW,mid}$ | $\text{LogNormal}$ | $\text{IG}(\alpha_{Q,mid}, \beta_{Q,mid})$ | Normal-IG | Match median ≈ 0.0005 |
| $\sigma^2_{QCEW,boundary}$ | $\text{LogNormal}$ | $\text{IG}(\alpha_{Q,bnd}, \beta_{Q,bnd})$ | Normal-IG | Match median ≈ 0.002 |
| $\alpha_{CES}$ | $N(0, 0.005^2)$ | Same | Normal-Normal | No change needed |
| $\lambda_{CES}$ | $N(1.0, 0.15^2)$ | Same | Normal-Normal | No change needed |
| $\sigma^2_{CES,v}$ | $\text{IG}(3, 0.004)$ | Same | Normal-IG | Already conjugate |
| $\alpha_p$ | $N(0, 0.005^2)$ | Same | Normal-Normal | No change needed |
| $\lambda_p$ | $N(1.0, 0.15^2)$ | Same | Normal-Normal | No change needed |
| $\sigma^2_p$ | $\text{IG}(3, 0.004)$ | Same | Normal-IG | Already conjugate |
| $\rho_p$ | $\text{Beta}(2, 3)$ | Same | **Non-conjugate** (MH) | — |

#### 3.6.1 InverseGamma Calibration Procedure

For each swapped prior, choose IG(α, β) parameters by matching two moments of the original prior on the variance scale:

1.  Compute the mode and 90th percentile of the original prior on $\sigma^2$.
2.  Solve for IG(α, β) such that $\text{mode}_{IG} = \beta / (\alpha + 1)$ matches the target mode, and $F^{-1}_{IG}(0.9; \alpha, \beta)$ matches the target 90th percentile.
3.  Verify via simulation that the IG prior places similar probability mass in the tails as the original.

Store the calibrated (α, β) pairs as constants in `config.py` alongside the existing `LOG_SIGMA_*` constants, with comments documenting the correspondence.

------------------------------------------------------------------------

## 4. Block Gibbs Sweep

One complete Gibbs iteration samples 9 blocks in sequence. Each block conditions on the current values of all other blocks. The sweep order is chosen to minimize serial dependency: the latent state is sampled first (it appears in all measurement equations), then the parameters that depend on the latent state.

### 4.1 Block Summary

| Block | Parameters | Method | State Dim | Cost per Sweep |
|---------------|---------------|---------------|---------------|---------------|
| 1 | $\{g^{cont}_t\}_{t=1}^T$ | FFBS (scalar Kalman) | 1 × T | O(T · n_obs_max) |
| 2 | $\{A_k(y), B_k(y)\}$ | FFBS (2K-dim Kalman) | 2K × n_years | O(K³ · n_years) |
| 3 | $\boldsymbol{\phi}_{BD}, \sigma^2_{BD}$ | Normal-IG conjugate | n_bd_covars + 1 | O(T · n_bd_covars²) |
| 4 | $\alpha_{CES}, \lambda_{CES}, \{\sigma^2_{CES,v}\}$ | Normal-Normal + IG | 2 + n_vintages | O(T · n_vintages) |
| 5 | $\{\alpha_p, \lambda_p, \sigma^2_p\}$ per provider | Normal-Normal + IG | 3 × n_providers | O(T · n_providers) |
| 6 | $\{\mu_{g,e}\}$ per era | Normal-Normal conjugate | n_eras | O(T) |
| 7 | $\sigma^2_{QCEW,mid}, \sigma^2_{QCEW,boundary}$ | IG conjugate | 2 | O(T_QCEW) |
| 8 | $\{\sigma^2_{\omega,k}\}$, $\sigma^2_{BD}$ | IG conjugate | K + 1 | O(K · n_years + T) |
| 9 | $\{\phi_e\}, \sigma_\eta, \{\rho_p\}$ | Metropolis-within-Gibbs | n_eras + 1 + n_ar1 | O(T) per scalar |

Total cost per sweep: dominated by Block 1 (FFBS), which is O(T) with small constants for scalar state. For Release 1 with T ≈ 300, a full sweep takes microseconds in compiled code.

### 4.2 Block 1: Latent State via FFBS

#### 4.2.1 State-Space Representation

At each time $t$, the scalar latent state $x_t \equiv g^{cont}_t$ evolves as:

$$x_t = c_t + F_t \, x_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$$

where $c_t = (1 - \phi_{e(t)}) \mu_{g,e(t)}$, $F_t = \phi_{e(t)}$, and $Q = \sigma^2_\eta$.

The observation vector $\mathbf{y}_t$ stacks all available observations at time $t$ (any subset of QCEW, CES SA/NSA vintages, and providers). Each observation has the linear form:

$$y_{i,t} = H_{i,t} \, x_t + d_{i,t} + \varepsilon_{i,t}, \quad \varepsilon_{i,t} \sim N(0, R_{i,t})$$

| Source | $H_{i,t}$ | $d_{i,t}$ | $R_{i,t}$ |
|------------------|------------------|------------------|------------------|
| QCEW | 1 | $BD_t + s_t$ | $\sigma^2_{QCEW}(t)$ |
| CES SA, vintage $v$ | $\lambda_{CES}$ | $\alpha_{CES} + \lambda_{CES} \cdot BD_t$ | $\sigma^2_{CES,v}$ |
| CES NSA, vintage $v$ | $\lambda_{CES}$ | $\alpha_{CES} + \lambda_{CES} \cdot (BD_t + s_t)$ | $\sigma^2_{CES,v}$ |
| Provider $p$ (iid) | $\lambda_p$ | $\alpha_p + \lambda_p \cdot s_t$ | $\sigma^2_p$ |
| Provider $p$ (AR(1)) | $\lambda_p$ | $\alpha_p + \lambda_p \cdot s_t + \rho_p(y^p_{t-1} - \mu^{base}_{p,t-1})$ | $\sigma^2_p$ |

Note: The AR(1) provider offset $d_{i,t}$ absorbs the conditional mean adjustment. At $t = 0$ for an AR(1) provider, $R_{i,0} = \sigma^2_p / (1 - \rho^2_p)$ (marginal variance).

#### 4.2.2 Forward Pass (Kalman Filter)

Initialize:

$$m_0^- = \mu_{g,0}, \quad C_0^- = \frac{\sigma^2_\eta}{1 - \phi_0^2}$$

For $t = 0, \ldots, T-1$:

**Predict** (for $t > 0$):

$$m_t^- = c_t + F_t \, m_{t-1}, \quad C_t^- = F_t^2 \, C_{t-1} + Q$$

**Update** (sequential processing of $n_t$ observations at time $t$):

For each observation $i = 1, \ldots, n_t$:

$$v_i = y_{i,t} - H_{i,t} \, m_t^{(i-1)} - d_{i,t}$$

$$S_i = H_{i,t}^2 \, C_t^{(i-1)} + R_{i,t}$$

$$K_i = H_{i,t} \, C_t^{(i-1)} / S_i$$

$$m_t^{(i)} = m_t^{(i-1)} + K_i \, v_i$$

$$C_t^{(i)} = (1 - K_i \, H_{i,t}) \, C_t^{(i-1)}$$

where $m_t^{(0)} = m_t^-$, $C_t^{(0)} = C_t^-$, and the final values are $m_t = m_t^{(n_t)}$, $C_t = C_t^{(n_t)}$.

Sequential updating is equivalent to batch updating when $R$ is diagonal (as it is here, since all observation errors are independent conditional on the state), and avoids forming and inverting the $n_t \times n_t$ matrix $S$.

#### 4.2.3 Backward Pass (Simulation Smoother)

Draw $x_T \sim N(m_T, C_T)$.

For $t = T-1, \ldots, 0$:

$$J_t = F_{t+1} \, C_t / C_{t+1}^-$$

$$\tilde{m}_t = m_t + J_t \left(x_{t+1} - m_{t+1}^-\right)$$

$$\tilde{C}_t = C_t - J_t^2 \, C_{t+1}^-$$

$$x_t \sim N(\tilde{m}_t, \tilde{C}_t)$$

Store $m_t^-$ and $C_t^-$ (the predictive moments) during the forward pass for use in the backward pass.

#### 4.2.4 Implementation Notes

-   All quantities are scalar for Release 1. Matrix operations become multiplications.
-   For numerical stability, clamp $C_t$ and $\tilde{C}_t$ to a small positive floor (e.g., $10^{-20}$) before taking square roots.
-   Store the full forward pass arrays $(m, C, m^-, C^-)$ of length $T$ before the backward pass. Memory is negligible for T ≈ 300.
-   At era boundaries, $c_t$ and $F_t$ change discontinuously. No special handling is needed — the filter simply uses the new values.

### 4.3 Block 2: Seasonal Coefficients via FFBS

#### 4.3.1 State-Space Representation

The state vector at year $y$ is $\boldsymbol{\theta}_y = [A_1(y), B_1(y), \ldots, A_K(y), B_K(y)]'$ of dimension $2K$.

**Transition:** $\boldsymbol{\theta}_y = \boldsymbol{\theta}_{y-1} + \boldsymbol{\omega}_y$, where $\boldsymbol{\omega}_y \sim N(\mathbf{0}, Q_\omega)$ and $Q_\omega = \text{diag}(\sigma^2_{\omega,1}, \sigma^2_{\omega,1}, \ldots, \sigma^2_{\omega,K}, \sigma^2_{\omega,K})$.

**Observation:** All months within year $y$ observe a linear combination of $\boldsymbol{\theta}_y$. For month $m$ (1-indexed) in year $y$, define the $1 \times 2K$ basis row:

$$\mathbf{f}_m' = [\cos(2\pi \cdot 1 \cdot m/12), \; \sin(2\pi \cdot 1 \cdot m/12), \; \ldots, \; \cos(2\pi K m/12), \; \sin(2\pi K m/12)]$$

The seasonal "observation" at month $t$ (with $y = y(t)$, $m = m(t)$) is derived by subtracting the non-seasonal components from the composite signals. The residuals that inform the seasonal are:

$$z_t = g^{cont,NSA}_t - g^{cont}_t = s_t + \text{noise}$$

where the noise comes from propagating the observation equations' noise through the subtraction. In practice, use the residuals from all observation equations where $s_t$ appears (QCEW, CES NSA, providers), weighted by their precision.

Stack all months in year $y$ into an observation matrix $F_y$ (up to 12 rows × $2K$ columns) with corresponding observation vector $\mathbf{z}_y$ and noise covariance $R_y$ (diagonal). Then run the standard $2K$-dimensional FFBS over the $n_{years}$ time steps.

#### 4.3.2 Implementation Notes

-   The $2K \times 2K$ matrix operations (with $K = 4$, so $2K = 8$) are small enough for direct dense linear algebra. No sparse structure is needed.
-   The initial distribution for $\boldsymbol{\theta}_0$ uses the prior: $N(\mathbf{0}, 0.015^2 \, I_{2K})$.
-   Within each year, the observation matrix $F_y$ has at most 12 rows (one per month). If some months lack observations that inform the seasonal (e.g., missing QCEW), the corresponding rows are omitted.

### 4.4 Block 3: Birth/Death Loadings

#### 4.4.1 Conditional Distribution

Given $g^{cont}$, $s$, and all observation data, the implied total NSA growth from QCEW and CES NSA observations defines a set of "BD-informative" residuals:

$$r_t = g^{total,NSA}_t - g^{cont}_t - s_t$$

In practice, $g^{total,NSA}_t$ is not directly observed — it is inferred from the observation equations. For the Gibbs conditional, treat the QCEW observations as informing BD:

$$y^{QCEW}_t - g^{cont}_t - s_t = BD_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_{QCEW}(t))$$

Since $BD_t = \mathbf{x}_t' \boldsymbol{\phi} + \sigma_{BD} \xi_t$, this gives:

$$y^{QCEW}_t - g^{cont}_t - s_t = \mathbf{x}_t' \boldsymbol{\phi} + \sigma_{BD} \xi_t + \varepsilon_t$$

which is a standard linear regression of $\boldsymbol{\phi}$ with total noise variance $\sigma^2_{BD} + \sigma^2_{QCEW}(t)$ per observation.

#### 4.4.2 Sampling Order

1.  Draw $\sigma^2_{BD}$ from its IG conditional (see Block 8).
2.  Draw $\boldsymbol{\phi}$ from its multivariate Normal conditional:

$$\boldsymbol{\phi} \mid \cdot \sim N(\boldsymbol{\mu}_{post}, \Sigma_{post})$$

$$\Sigma_{post}^{-1} = \Sigma_0^{-1} + \sum_t \frac{\mathbf{x}_t \mathbf{x}_t'}{\sigma^2_{BD} + \sigma^2_{QCEW}(t)}$$

$$\boldsymbol{\mu}_{post} = \Sigma_{post} \left( \Sigma_0^{-1} \boldsymbol{\mu}_0 + \sum_t \frac{\mathbf{x}_t \, r_t}{\sigma^2_{BD} + \sigma^2_{QCEW}(t)} \right)$$

where $r_t = y^{QCEW}_t - g^{cont}_t - s_t$ and $\Sigma_0, \boldsymbol{\mu}_0$ are from the prior.

#### 4.4.3 Covariates with Missing Data

When a BD covariate (birth rate, QCEW BD proxy, or cyclical indicator) has missing values at time $t$, set the corresponding element of $\mathbf{x}_t$ to zero. This is equivalent to excluding that covariate's contribution for that month, which matches the PyMC model's handling via masked arrays.

### 4.5 Block 4: CES Parameters

#### 4.5.1 Conditional Distribution

Given $g^{total,SA}$, $g^{total,NSA}$, and the CES observations, the measurement equations are standard linear regressions:

$$y^{CES,SA}_{t,v} = \alpha_{CES} + \lambda_{CES} \cdot g^{total,SA}_t + \varepsilon_{t,v}$$

Stack all observed CES SA and NSA data points across times and vintages into a single regression:

$$\mathbf{y}_{CES} = \alpha_{CES} \cdot \mathbf{1} + \lambda_{CES} \cdot \mathbf{g} + \boldsymbol{\varepsilon}$$

where $\mathbf{g}$ contains the appropriate composite signal ($g^{total,SA}_t$ for SA observations, $g^{total,NSA}_t$ for NSA observations) and $\boldsymbol{\varepsilon}$ has diagonal covariance with entries $\sigma^2_{CES,v(i)}$ for observation $i$ from vintage $v(i)$.

#### 4.5.2 Sampling Order

1.  Draw $(\alpha_{CES}, \lambda_{CES})$ jointly from their bivariate Normal conditional (Normal-Normal conjugate with heteroscedastic noise weighted by $1/\sigma^2_{CES,v}$).
2.  For each vintage $v$, draw $\sigma^2_{CES,v}$ from its IG conditional given the residuals $e_{t,v} = y_{t,v} - \alpha_{CES} - \lambda_{CES} \cdot g_t$:

$$\sigma^2_{CES,v} \mid \cdot \sim \text{IG}\!\left(3 + n_v/2, \; 0.004 + \tfrac{1}{2}\sum_{t \in v} e_{t,v}^2\right)$$

### 4.6 Block 5: Provider Parameters

Identical structure to Block 4, applied per provider. For iid providers, this is a direct Normal-Normal + IG conjugate draw.

For AR(1) providers, conditional on $\rho_p$ (sampled in Block 9), the pre-whitened observations are:

$$\tilde{y}^p_t = y^p_t - \rho_p \, y^p_{t-1}, \quad \tilde{\mu}^{base}_{p,t} = \mu^{base}_{p,t} - \rho_p \, \mu^{base}_{p,t-1}$$

This reduces to a standard regression of $(\alpha_p, \lambda_p)$ on the pre-whitened series with noise $\sigma^2_p$. The first observation uses the marginal variance $\sigma^2_p / (1 - \rho^2_p)$.

### 4.7 Block 6: Mean Growth Per Era

#### 4.7.1 Conditional Distribution

From the transition equation within era $e$:

$$g^{cont}_t - \phi_e \, g^{cont}_{t-1} = (1 - \phi_e) \mu_{g,e} + \eta_t$$

Rearranging: the "observation" for $\mu_{g,e}$ is:

$$z_t = \frac{g^{cont}_t - \phi_e \, g^{cont}_{t-1}}{1 - \phi_e}$$

with noise variance $\sigma^2_\eta / (1 - \phi_e)^2$. This is a Normal-Normal conjugate update:

$$\mu_{g,e} \mid \cdot \sim N\!\left(\frac{\mu_0 / \sigma^2_0 + n_e \bar{z}_e / \tau^2_e}{1/\sigma^2_0 + n_e / \tau^2_e}, \; \frac{1}{1/\sigma^2_0 + n_e / \tau^2_e}\right)$$

where $n_e$ is the number of transitions within era $e$, $\bar{z}_e$ is the mean of the $z_t$ values, and $\tau^2_e = \sigma^2_\eta / (1 - \phi_e)^2$.

#### 4.7.2 Boundary Handling

At era boundaries (January 2009, January 2020), the transition from $t-1$ to $t$ uses the new era's parameters. The observation $z_t$ uses $\phi_{e(t)}$ and $\mu_{g,e(t)}$, not the previous era's. This is consistent with the discrete-switch specification in `national_model_spec.md` §3.3.2.

### 4.8 Blocks 7–8: Variance Parameters (IG Conjugate)

All variance parameters with IG priors have closed-form IG conditionals. Given $n$ squared residuals $\{e_i^2\}$ from the relevant equation:

$$\sigma^2 \mid \cdot \sim \text{IG}\!\left(\alpha_0 + n/2, \; \beta_0 + \tfrac{1}{2}\sum_i e_i^2\right)$$

Specific applications:

| Parameter | Residuals | Source |
|------------------------|------------------------|------------------------|
| $\sigma^2_{QCEW,mid}$ | $y^{QCEW}_t - g^{total,NSA}_t$ for M3+M1 months | Block 7 |
| $\sigma^2_{QCEW,boundary}$ | $y^{QCEW}_t - g^{total,NSA}_t$ for M2 months | Block 7 |
| $\sigma^2_{\omega,k}$ | $A_k(y) - A_k(y-1)$ and $B_k(y) - B_k(y-1)$ for each $k$ | Block 8 |
| $\sigma^2_{BD}$ | $BD_t - \mathbf{x}_t' \boldsymbol{\phi}$ (BD innovation residuals) | Block 8 |

### 4.9 Block 9: Non-Conjugate Scalars (Metropolis-within-Gibbs)

#### 4.9.1 AR(1) Persistence $\phi_e$

**Target:** $p(\phi_e \mid g^{cont}, \mu_{g,e}, \sigma_\eta)$

**Log-conditional:**

$$\log p(\phi_e \mid \cdot) = \log \text{Beta}(\phi_e; 18, 2) - \frac{1}{2\sigma^2_\eta} \sum_{t: e(t)=e} \left(g^{cont}_t - \mu_{g,e} - \phi_e (g^{cont}_{t-1} - \mu_{g,e})\right)^2 + \text{const}$$

**Proposal:** Random-walk Metropolis on logit scale:

$$\text{logit}(\phi^*) \sim N(\text{logit}(\phi_e), \; s^2_\phi)$$

**Acceptance ratio:** includes the log-prior ratio, the log-likelihood ratio, and the Jacobian of the logit transform:

$$\log \alpha = \log p(\phi^* \mid \cdot) - \log p(\phi_e \mid \cdot) + \log[\phi^*(1-\phi^*)] - \log[\phi_e(1-\phi_e)]$$

Accept with probability $\min(1, \exp(\log \alpha))$.

#### 4.9.2 Innovation Scale $\sigma_\eta$

**Target:** $p(\sigma_\eta \mid g^{cont}, \mu_g, \phi)$

**Log-conditional:**

$$\log p(\sigma_\eta \mid \cdot) = \log \text{Half-N}(\sigma_\eta; 0, 0.005) - T \log \sigma_\eta - \frac{1}{2\sigma^2_\eta} \sum_t \left(g^{cont}_t - c_t - \phi_{e(t)} g^{cont}_{t-1}\right)^2 + \text{const}$$

**Proposal:** Random-walk Metropolis on log scale:

$$\log \sigma^*_\eta \sim N(\log \sigma_\eta, \; s^2_\sigma)$$

**Jacobian:** $\log \sigma^*_\eta - \log \sigma_\eta$ (from the log transform).

#### 4.9.3 Provider AR(1) Coefficient $\rho_p$

**Target:** $p(\rho_p \mid y^p, \alpha_p, \lambda_p, \sigma^2_p)$

**Log-conditional:** Sum of Beta(2,3) log-prior and the AR(1) conditional log-likelihood over provider $p$'s observations.

**Proposal:** Random-walk Metropolis on logit scale, same structure as $\phi_e$.

#### 4.9.4 Adaptive Proposal Scales

During warmup, adapt the proposal standard deviations $s_\phi$, $s_\sigma$, $s_\rho$ every 100 iterations using the Robbins-Monro rule:

$$\log s^{(k+1)} = \log s^{(k)} + \gamma_k (\bar{a}_k - a^*)$$

where $\bar{a}_k$ is the acceptance rate over the last 100 iterations, $a^* = 0.44$ is the optimal acceptance rate for scalar random-walk MH, and $\gamma_k = k^{-0.6}$ is the step size. Freeze adaptation at the end of warmup.

------------------------------------------------------------------------

## 5. Initialization

### 5.1 Strategy

Initialize from a rough approximation to the posterior rather than the prior, to avoid a long burn-in. The initialization uses fast, non-Bayesian estimates:

| Parameter | Initialization |
|------------------------------------|------------------------------------|
| $g^{cont}_t$ | CES SA first-print growth rates (available at all times) |
| $s_t$ | Residual from X-12-style seasonal extraction on CES NSA, or zero |
| $\mu_{g,e}$ | Mean of CES SA growth within each era |
| $\phi_e$ | Sample lag-1 autocorrelation of CES SA growth within each era, clipped to (0.5, 0.98) |
| $\sigma_\eta$ | Standard deviation of CES SA growth innovations |
| $\boldsymbol{\phi}_{BD}$ | OLS regression of QCEW-CES difference on BD covariates |
| $\alpha_{CES}, \lambda_{CES}$ | (0, 1) — CES is nearly unbiased with unit loading |
| $\alpha_p, \lambda_p$ | OLS of provider on CES growth |
| $\sigma^2_*$ | Prior mode for each variance parameter |
| $\rho_p$ | 0.25 (prior mode of Beta(2,3)) |
| Fourier coefficients | OLS fit of harmonics to CES NSA – CES SA difference |

### 5.2 Multi-Chain Initialization

For multiple chains, perturb the initialization by adding small Normal noise to each parameter, scaled to roughly 10% of the prior standard deviation. This ensures chains start in different regions of the posterior.

------------------------------------------------------------------------

## 6. Output and Diagnostics

### 6.1 ArviZ InferenceData Construction

Pack post-warmup draws into `az.InferenceData` using `az.from_dict()`. Variable names must match the PyMC model's naming convention:

| Gibbs State Field | ArviZ Variable Name | Shape |
|------------------------|------------------------|------------------------|
| `g_cont` | `g_cont` | (chain, draw, T) |
| `mu_g` | `mu_g_era` | (chain, draw, n_eras) |
| `phi` | `phi_raw_era` | (chain, draw, n_eras) |
| `sigma_eta` | `sigma_g` | (chain, draw) |
| `bd_phi` | `phi_0`, `phi_1`, `phi_2`, `phi_3` | (chain, draw) each |
| `sigma2_bd` | `sigma_bd` | (chain, draw) — store as σ, not σ² |
| `seasonal_ab` | `fourier_coeffs` | (chain, draw, n_years, 2K) |
| `alpha_ces` | `alpha_ces` | (chain, draw) |
| `lam_ces` | `lam_ces` | (chain, draw) |
| `sigma2_ces` | `sigma_ces_sa_*`, `sigma_ces_nsa_*` | (chain, draw) per vintage — store as σ |
| `alpha_pp[p]` | `alpha_{name}` | (chain, draw) per provider |
| `lam_pp[p]` | `lam_{name}` | (chain, draw) per provider |
| `sigma2_pp[p]` | `sigma_pp_{name}` | (chain, draw) per provider — store as σ |
| `rho_pp[p]` | `rho_{name}` | (chain, draw) per provider |

Note: The PyMC model stores standard deviations (σ) for some parameters and variances (σ²) for others. The Gibbs sampler internally works with variances. Convert to match PyMC conventions before packing into InferenceData.

### 6.2 Derived Quantities

After sampling, compute and store as `pm.Deterministic`-equivalent entries:

| Derived Variable | Formula | ArviZ Name |
|------------------------|------------------------|------------------------|
| Seasonal component | $s_t = \sum_k [A_k(y(t)) \cos(\cdot) + B_k(y(t)) \sin(\cdot)]$ | `seasonal` |
| BD component | $BD_t = \mathbf{x}_t' \boldsymbol{\phi}$ (mean, excluding $\xi_t$) | `bd` |
| Total SA growth | $g^{cont}_t + BD_t$ | `g_total_sa` |
| Total NSA growth | $g^{cont}_t + BD_t + s_t$ | `g_total_nsa` |

### 6.3 Convergence Diagnostics

Standard ArviZ diagnostics apply to the Gibbs output:

| Diagnostic | Target | Notes |
|------------------------|------------------------|------------------------|
| $\hat{R}$ (split-R-hat) | \< 1.01 | Multi-chain required |
| Bulk ESS / n_draws | \> 0.1 | Gibbs ESS may be lower than NUTS for highly correlated blocks |
| Tail ESS / n_draws | \> 0.05 | Check tail ESS for variance parameters |
| Trace plots | No trends, good mixing | Visual inspection for slow Gibbs mixing |
| Autocorrelation | ACF decays by lag \~50 | Gibbs chains are autocorrelated by construction; ACF decay rate indicates mixing speed |

### 6.4 Gibbs-Specific Diagnostics

In addition to standard ArviZ checks:

| Diagnostic | Purpose |
|------------------------------------|------------------------------------|
| MH acceptance rates | Verify $\phi$, $\sigma_\eta$, $\rho_p$ acceptance rates are 30–50% (optimal for scalar RW-MH) |
| Block-wise ESS | Compute ESS separately for FFBS blocks vs. MH blocks to identify bottlenecks |
| FFBS numerical stability | Monitor filtered variance $C_t$ for collapse (\< $10^{-15}$) or explosion |
| Cross-sampler validation | Compare Gibbs posterior summaries to NUTS (Phase 3) |

------------------------------------------------------------------------

## 7. Cross-Sampler Validation Protocol

### 7.1 Objective

Verify that the Gibbs sampler and NUTS produce statistically indistinguishable posteriors on the Release 1 model. This is the primary acceptance criterion for Phase 3.

### 7.2 Procedure

1.  Run NUTS on the Release 1 model with default settings (8K warmup, 6K draws, 4 chains).
2.  Run Gibbs on the same data with comparable settings (2K warmup, 6K draws, 4 chains — Gibbs needs less warmup).
3.  For each parameter, compare:

| Test | Criterion | Tolerance |
|------------------------|------------------------|------------------------|
| Posterior mean difference | $\|\bar{\theta}_{Gibbs} - \bar{\theta}_{NUTS}\| / \text{se}$ | \< 2 posterior standard errors |
| Posterior std ratio | $\text{sd}_{Gibbs} / \text{sd}_{NUTS}$ | Within (0.85, 1.15) |
| KS test on marginals | Two-sample KS p-value | \> 0.01 for all parameters |
| Credible interval overlap | 90% CI overlap coefficient | \> 0.80 |

### 7.3 Known Sources of Discrepancy

-   **Prior swaps** (IG vs. LogNormal/Half-Normal for variance parameters): calibration should minimize this, but some posterior sensitivity is expected. If discrepancy is attributable to the prior swap, document and accept.
-   **Centered vs. non-centered parameterization:** In principle these define the same posterior, but finite-sample MCMC behavior can differ. If one sampler has poor mixing in a region where the other does not, the effective posteriors may differ. Check trace plots in the disagreement region.

------------------------------------------------------------------------

## 8. File and Module Layout

The Gibbs sampler lives in a separate package under `packages/`, outside the main `src/alt_nfp/` tree. This keeps the pure-NumPy/SciPy Gibbs machinery cleanly separated from the PyMC-dependent model code — no PyMC or PyTensor imports leak into the Gibbs package. The two packages communicate through the shared `data` dict interface (produced by `panel_to_model_data()`) and the `az.InferenceData` output contract.

```         
packages/nfp-model-gibbs/
├── pyproject.toml             # Package metadata; depends on numpy, scipy, arviz
├── src/
│   └── nfp_model_gibbs/
│       ├── __init__.py        # Public API: run_gibbs()
│       ├── sampler.py         # Main Gibbs loop, GibbsConfig, GibbsState
│       ├── ffbs.py            # FFBS for latent state (Block 1) and seasonal (Block 2)
│       ├── conjugate.py       # Conjugate block samplers (Blocks 3–8)
│       ├── metropolis.py      # MH steps for φ, σ_η, ρ (Block 9) + adaptation
│       ├── initialization.py  # State initialization from data
│       ├── output.py          # Traces → az.InferenceData conversion
│       └── priors.py          # IG prior calibration constants + helper functions
└── tests/
    ├── test_ffbs.py
    ├── test_conjugate.py
    ├── test_metropolis.py
    └── test_integration.py

src/alt_nfp/
├── sampling.py                # MODIFIED: add method='gibbs' dispatch
├── config.py                  # MODIFIED: add IG prior constants
└── ...
```

### 8.1 New Files

**`packages/nfp-model-gibbs/pyproject.toml`** Package metadata. Dependencies: `numpy`, `scipy`, `arviz`. No dependency on `pymc`, `pytensor`, or `alt_nfp`. The package imports nothing from the main `alt_nfp` codebase — it receives the `data` dict at the `run_gibbs()` call boundary.

``` toml
[project]
name = 'nfp-model-gibbs'
version = '0.1.0'
dependencies = [
    'numpy>=1.26',
    'scipy>=1.12',
    'arviz>=0.18',
]

[build-system]
requires = ['hatchling']
build-backend = 'hatchling.build'

[tool.hatch.build.targets.wheel]
packages = ['src/nfp_model_gibbs']
```

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/ffbs.py`** (\~200 lines) Core FFBS implementation. Two public functions: - `ffbs_latent_state(data, state, rng)` → `np.ndarray` of shape (T,) - `ffbs_seasonal(data, state, rng)` → `np.ndarray` of shape (n_years, 2K)

Both operate on NumPy arrays. No PyMC or PyTensor dependency.

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/conjugate.py`** (\~250 lines) Conjugate block samplers. Generic helper functions: - `sample_regression_normal(y, X, prior_mean, prior_prec, sigma2, rng)` → coefficient vector - `sample_variance_ig(residuals, prior_alpha, prior_beta, rng)` → scalar variance

Plus model-specific wrappers: `sample_bd_params()`, `sample_ces_params()`, `sample_provider_params()`, `sample_mu_g()`, `sample_qcew_noise()`, `sample_fourier_noise()`.

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/metropolis.py`** (\~150 lines) Scalar MH steps with logit/log transforms and Robbins-Monro adaptation: - `mh_phi(phi_current, g_cont, mu_g, sigma_eta, era_mask, scale, rng)` → (phi_new, accepted) - `mh_sigma_eta(sigma_eta_current, g_cont, mu_g, phi, era_idx, scale, rng)` → (sigma_eta_new, accepted) - `mh_rho(rho_current, y_obs, mu_base, sigma2, scale, rng)` → (rho_new, accepted) - `adapt_scales(scales, accepts, counts, target)` → updated scales dict

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/sampler.py`** (\~150 lines) Main Gibbs loop: - `GibbsConfig` dataclass (n_warmup, n_draws, n_chains, seed, MH config) - `GibbsState` dataclass (all parameter arrays) - `run_gibbs(data, config)` → `az.InferenceData`

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/initialization.py`** (\~100 lines) - `initialize_state(data, rng)` → `GibbsState`

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/output.py`** (\~80 lines) - `traces_to_inferencedata(traces, data, config)` → `az.InferenceData`

**`packages/nfp-model-gibbs/src/nfp_model_gibbs/priors.py`** (\~60 lines) - IG prior constants calibrated from the PyMC priors - `calibrate_ig(target_mode, target_q90)` → (alpha, beta) helper

### 8.2 Modified Files

**`src/alt_nfp/sampling.py`** Add `method` parameter to `sample_model()`:

``` python
def sample_model(
    model: pm.Model | None = None,
    data: dict | None = None,
    *,
    method: str = 'nuts',
    sampler_kwargs: dict | None = None,
    gibbs_config: 'GibbsConfig | None' = None,
) -> az.InferenceData:
    if method == 'gibbs':
        from nfp_model_gibbs import run_gibbs
        return run_gibbs(data, gibbs_config)
    # ... existing NUTS path unchanged
```

This requires `nfp-model-gibbs` to be installed in the environment (e.g., `uv pip install -e packages/nfp-model-gibbs`). The import is deferred so that the Gibbs package is only needed when `method='gibbs'` is requested — the NUTS pathway has no dependency on it.

**`src/alt_nfp/config.py`** No changes required. The IG prior constants live in `packages/nfp-model-gibbs/src/nfp_model_gibbs/priors.py`, not in the main `alt_nfp` config, since the Gibbs package is self-contained. The main `config.py` retains the existing `LOG_SIGMA_*` constants for the PyMC model. If cross-sampler comparison tooling needs both prior specifications, import from both packages at the call site.

------------------------------------------------------------------------

## 9. Testing Strategy

### 9.1 Unit Tests

Unit tests live in `packages/nfp-model-gibbs/tests/` and have no dependency on the main `alt_nfp` package or PyMC.

| Test | Module | Validates |
|------------------------|------------------------|------------------------|
| FFBS on known AR(1) + Normal obs | `ffbs.py` | Forward filter recovers Kalman moments; backward samples have correct marginal |
| FFBS seasonal on known signal | `ffbs.py` | Recovers planted Fourier coefficients |
| Conjugate Normal-Normal update | `conjugate.py` | Matches analytical posterior mean and variance |
| Conjugate IG update | `conjugate.py` | Matches analytical IG posterior parameters |
| MH acceptance ratio on known target | `metropolis.py` | Acceptance probability matches hand-computed value |
| Logit/log transforms + Jacobian | `metropolis.py` | Correct Jacobian terms in acceptance ratio |
| State initialization | `initialization.py` | All parameters within prior support; no NaN/Inf |
| InferenceData variable names | `output.py` | All expected ArviZ variables present with correct shapes |

### 9.2 Integration Tests

Integration tests live in the main repo's `tests/` directory (e.g., `tests/test_gibbs_integration.py`) because they require both `nfp_model_gibbs` and `alt_nfp` (for `panel_to_model_data()` and the PyMC model).

| Test | Validates |
|------------------------------------|------------------------------------|
| Full Gibbs sweep on synthetic data with known ground truth | Posterior concentrates near true parameters after sufficient iterations |
| Gibbs on synthetic data reproduces FFBS-analytic posterior for fixed parameters | Isolates FFBS correctness from the rest of the sampler |
| Multi-chain R-hat \< 1.05 on real data | Chains mix and converge |
| Posterior predictive checks pass on real data | Model is correctly specified in Gibbs form |

### 9.3 Cross-Sampler Validation Tests

Per §7: automated comparison of Gibbs vs. NUTS posteriors on the Release 1 model with the full validation protocol. These tests live in `tests/test_gibbs_cross_validation.py` in the main repo and require both `alt_nfp` and `nfp_model_gibbs`.

------------------------------------------------------------------------

## 10. Computational Requirements

### 10.1 Release 1 Estimates

| Component | Time per Sweep | Sweeps | Wall Time (1 chain) |
|----|----|----|----|
| Block 1: FFBS latent (T=300) | \~50 μs | 6,000 | \~0.3 s |
| Block 2: FFBS seasonal (2K=8, n_years=23) | \~20 μs | 6,000 | \~0.1 s |
| Blocks 3–8: Conjugate draws | \~10 μs | 6,000 | \~0.1 s |
| Block 9: MH steps (\~5 scalars) | \~5 μs | 6,000 | \~0.03 s |
| **Total (pure Python/NumPy)** | \~85 μs | 6,000 | **\~0.5 s** |
| **Total with warmup (8,000 sweeps)** |  | 8,000 | **\~0.7 s** |

For comparison, NUTS on the same model takes \~2–5 minutes on Apple Silicon with nutpie. The Gibbs sampler should be roughly 200–500× faster per effective sample, though ESS per draw may be lower.

### 10.2 Release 3 Projections

With \~700 cells, Block 1 becomes 700 independent scalar FFBS passes. At \~50 μs each (sequential), this is \~35 ms per sweep. With numba JIT and parallel dispatch across 10 cores: \~3.5 ms per sweep. For 8,000 total sweeps: \~28 seconds. This is feasible on the M4 Max.

### 10.3 Phase 4 Optimization Targets

| Optimization | Expected Speedup | Effort |
|------------------------|------------------------|------------------------|
| Numba JIT on FFBS inner loop | 5–10× | Medium (rewrite inner loop as numba function) |
| Vectorized multi-cell FFBS | 10–50× for Release 3 | Medium (batch cells into array operations) |
| Multi-chain via `multiprocessing` | n_chains × | Low (embarrassingly parallel) |
| Pre-computed observation stacking | 2× on Block 1 | Low (avoid recomputing H, d, R each sweep) |

------------------------------------------------------------------------

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------------------------|------------------------|------------------------|
| IG prior swaps change posterior materially | Invalid cross-sampler comparison | Careful calibration (§3.6.1); run sensitivity analysis; option to keep LogNormal via MH at cost of one more scalar MH step |
| Gibbs mixing is slow for correlated blocks | Low ESS despite fast sweeps | Monitor block-wise ESS; consider collapsed Gibbs (marginalize out some parameters analytically) |
| Centered parameterization causes funnel for small σ_η | Poor FFBS behavior when σ_η → 0 | Beta(18,2) prior on φ prevents exact unit root; monitor filtered variance C_t for collapse |
| AR(1) provider pre-whitening introduces serial dependency in Block 5 | Incorrect conjugate conditional | Condition on ρ_p from previous sweep (standard Gibbs conditioning); verify with synthetic data test |
| QCEW noise estimated (not fixed) changes BD identification | Different BD loading posteriors vs. NUTS | Compare with QCEW noise fixed at PyMC values as a diagnostic |
| Numerical underflow in FFBS forward pass | NaN in latent state draws | Clamp filtered variance to floor; use log-space computations if needed |

------------------------------------------------------------------------

## 12. Success Criteria

1.  **Phase 1:** FFBS latent state and all conjugate blocks produce valid draws. Full Gibbs sweep runs without error on real data. Manual inspection of trace plots shows reasonable mixing.

2.  **Phase 2:** MH blocks converge with acceptance rates in 30–50%. Adaptation stabilizes by end of warmup. Full sampler produces sensible posteriors on real data.

3.  **Phase 3:** Cross-sampler validation (§7.2) passes all criteria. Gibbs posterior summaries match NUTS within tolerance for all parameters. Downstream outputs (nowcast, benchmark extraction, precision budget) are statistically equivalent.

4.  **Phase 4:** Gibbs sampler runs in \< 5 seconds for Release 1 (including multi-chain). Cell-level FFBS is parallelized and benchmarked for Release 3 scaling.

------------------------------------------------------------------------

## 13. Future Extensions

### 13.1 Release 2–3 Hierarchical Blocks

The Gibbs framework extends naturally to the hierarchical structure in Releases 2–3. New blocks would be added for:

-   Hierarchical variance parameters ($\tau^2_{region}$, $\tau^2_{division}$, etc.): IG conjugate conditional given the cell-level random effects.
-   Cell-level random effects ($\alpha^{region}_r$, $\alpha^{state}_{st}$, etc.): Normal conditional given the cell-level parameters and the hierarchical variance.
-   Cell-level FFBS: 700 independent scalar passes, conditionally independent given the hierarchical means.

### 13.2 Collapsed Gibbs

For parameters that appear in the FFBS observation equations but have Normal-Normal conjugate conditionals (e.g., $\alpha_{CES}$, $\lambda_{CES}$), it is possible to analytically marginalize them out of the FFBS step, producing a "collapsed" Gibbs sampler. This can substantially improve mixing by reducing the serial dependency between the latent state and the measurement parameters. The cost is a more complex observation covariance matrix in the FFBS step (no longer diagonal), which increases the per-step cost from O(n_obs) to O(n_obs²). Whether this tradeoff is worthwhile depends on the empirical mixing behavior of the uncollapsed sampler.

### 13.3 Sufficient Statistics Caching

Many conjugate blocks recompute the same sums ($\sum_t \mathbf{x}_t \mathbf{x}_t'$, $\sum_t \mathbf{x}_t y_t$, etc.) every sweep. When the design matrices are fixed across sweeps (as they are for the BD covariates and CES/provider loadings), the sufficient statistics can be precomputed once and reused, reducing conjugate block cost to O(1) per sweep (just the matrix solve). Only the terms involving the latent state (which changes every sweep) need recomputation.

------------------------------------------------------------------------

*This specification is a living document. Implementation details may evolve as empirical results from Phase 3 cross-validation inform design decisions.*