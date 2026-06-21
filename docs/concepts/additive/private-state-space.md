# The Private State-Space Model

The private nowcast is a Bayesian state-space model (`nfp_model.model.nfp_model`)
written in NumPyro. It estimates a single latent monthly growth path for private
employment (`industry_code` `'05'`) and ties it to several noisy observations of
that path — the near-census QCEW truth, the published CES prints, payroll-provider
signals, and cyclical leading indicators. The posterior over the latent path is
what the nowcast reads out.

This page documents the model exactly as it ships. Every prior constant below is
pinned in `nfp_model.config.ModelPriors`, and every likelihood is from
`nfp_model.model.nfp_model`. The model imports only `jax`, `numpyro`, and `numpy`
— it never sees a vintage date or touches the store; all censoring happens
upstream (see [Vintage data model & as-of censoring](../vintages-and-censoring.md)).

!!! note "Growth, not levels"
    Everything here is in **log month-over-month growth space**, the units the
    store derives at read time. The latent state is a growth rate; the nowcast
    layer turns a posterior growth draw back into a jobs-added number through the
    CES observation equation.

## The latent decomposition

The model builds three latent monthly paths and composes them into the growth
signals that each data source actually observes. The pieces are:

$$
g^{\text{cont}}_t \;\;(\text{continuing-units growth}),\qquad
s_t \;\;(\text{seasonal}),\qquad
\mathrm{bd}_t \;\;(\text{birth/death offset}).
$$

From these, the model forms exactly three composite signals, and **different
observations see different composites** — this is the heart of the design:

$$
\begin{aligned}
g^{\text{cont,nsa}}_t &= g^{\text{cont}}_t + s_t
  &&\text{(providers — continuing establishments, no births/deaths)}\\[4pt]
g^{\text{sa}}_t &= g^{\text{cont}}_t + \mathrm{bd}_t
  &&\text{(CES seasonally adjusted)}\\[4pt]
g^{\text{nsa}}_t &= g^{\text{cont}}_t + s_t + \mathrm{bd}_t
  &&\text{(QCEW and CES not-seasonally-adjusted)}
\end{aligned}
$$

The logic: payroll providers track *continuing* establishments, so they observe
continuing-units growth plus seasonality but **not** the birth/death term. The
seasonally adjusted CES print has seasonality removed but still carries the
structural birth/death adjustment. QCEW and the raw NSA CES print carry all three.
Getting this mapping right is what lets each source inform the latent state without
contaminating the others.

## Latent continuing-units growth — a non-centered AR(1)

The core state is a stationary first-order autoregression with an era-specific
mean. With $\mu_{g,t}$ the era mean for month $t$, $\phi$ the persistence, and
$\sigma_g$ the innovation SD,

$$
g^{\text{cont}}_t = \mu_{g,t} + \phi\,\bigl(g^{\text{cont}}_{t-1} - \mu_{g,t}\bigr)
  + \sigma_g\,\varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,1).
$$

The autoregression is parametrized by its **stationary** SD $\tau$ — which breaks
the well-known $\phi$–$\sigma$ ridge — and the innovation SD is recovered as

$$
\sigma_g = \tau\sqrt{1 - \phi^2}.
$$

Priors (`LatentPriors`):

$$
\tau \sim \mathrm{LogNormal}\!\bigl(\log 0.013,\ 0.5\bigr),\qquad
\phi_{\text{raw}} \sim \mathrm{Beta}(18, 2),\quad \phi = \min(\phi_{\text{raw}}, 0.99),
$$

$$
\mu_{g}^{(e)} \sim \mathcal{N}(0.001,\ 0.005)\quad\text{per era } e.
$$

The state is written **non-centered** (`eps_g ~ Normal(0, 1)`, scanned through the
AR(1) recursion) for better sampler geometry. The era index is a static covariate:
when present, each era $e$ gets its own mean $\mu_g^{(e)}$ (default `n_eras = 2`);
otherwise a single scalar mean is broadcast across the calendar.

## Seasonal — a Fourier block with annually-evolving amplitudes

Seasonality is a sum of $K = 4$ harmonics whose amplitudes drift slowly from year
to year (a Gaussian random walk across years). With $m_t$ the month-of-year and
year-indexed amplitudes $A_{k,y}, B_{k,y}$,

$$
s_t = \sum_{k=1}^{K} \left[
  A_{k,\,y(t)}\cos\!\frac{2\pi k\,m_t}{12}
  + B_{k,\,y(t)}\sin\!\frac{2\pi k\,m_t}{12}
\right].
$$

The amplitudes evolve as a random walk across years, implemented non-centered as a
cumulative sum of standardized innovations:

$$
\text{coef}_{k,y} = \text{coef}_{k,y-1} + \sigma^{\text{fourier}}_k\,z_{k,y},\qquad
z_{k,y}\sim\mathcal{N}(0,1),
$$

with the initial year scaled by `init_sd = 0.015`. The per-harmonic innovation SD
decreases with $k$ in log-space, so higher harmonics drift less:

$$
\sigma^{\text{fourier}}_k \sim
  \mathrm{LogNormal}\!\bigl(\log 0.0003 - \log k,\ 0.5\bigr).
$$

This is the centered `GaussianRandomWalk` of the frozen reference re-expressed
non-centered: an identical prior law with a friendlier posterior geometry.

## Structural birth/death — the shipping three-term form

The birth/death offset captures the net contribution of establishment births and
deaths that the continuing-units state cannot see. The model that **ships** is a
constant intercept, a Gaussian shock, and cyclical-covariate loadings:

$$
\mathrm{bd}_t = \varphi_0 + \sigma_{\text{bd}}\,\xi_t
  + \sum_{i}\varphi_{3,i}\,X^{\text{cycle}}_{i,t},
\qquad \xi_t \sim \mathcal{N}(0,1).
$$

Priors (`BirthDeathPriors`):

$$
\varphi_0 \sim \mathcal{N}(0.001,\ 0.002),\qquad
\sigma_{\text{bd}} \sim \mathrm{LogNormal}(\log 0.003,\ 0.5),\qquad
\varphi_{3,i} \sim \mathcal{N}(0,\ 0.3).
$$

!!! warning "Three terms, not five"
    Earlier design notes (and the frozen PyMC reference) describe a richer
    birth/death equation with extra $\varphi_1 X^{\text{birth}}$ and
    $\varphi_2\,\mathrm{BD}^{\text{QCEW}}$ regressors. **Those terms are not in
    the shipping model.** `nfp_model.model` samples only $\varphi_0$,
    $\sigma_{\text{bd}}$, and the cyclical loadings $\varphi_3$; `BirthDeathPriors`
    carries only those priors. The equation above is what the code computes.

### Cyclical indicators and their publication lags

The cyclical covariates $X^{\text{cycle}}_{i,t}$ are leading indicators centered
and standardized upstream (`nfp_ingest.model_data`), entering the birth/death term
through the loadings $\varphi_{3,i}$. The shipping default set has **two**
indicators (`ModelPriors.indicator_names = ("claims", "jolts")`,
`CYCLICAL_INDICATORS_DEFAULT`):

| Indicator | FRED series | Frequency | Publication lag |
|---|---|---|---|
| `claims` (initial jobless claims) | `ICNSA` | weekly → monthly mean | 1 month |
| `jolts` (job openings) | `JTSJOL` | monthly | 2 months |

The publication lags are enforced as part of Layer-2 as-of censoring (see
[as-of censoring](../vintages-and-censoring.md)): a month whose lag-offset
reference would only have been published after the horizon date is masked to
missing. Any indicator that is missing or all-zero for a given fit is **gated out**
before $\varphi_3$ is sampled, so censored backtests never introduce an
unidentified loading.

## Observation likelihoods

Each data source observes one of the three composite growth signals with its own
measurement model. Padded observation slots in batched (vmapped) fitting are
handled by boolean likelihood masks; the equations below describe a single real
observation.

### QCEW — the Student-t truth anchor

QCEW is the near-census of private employment and serves as the model's anchor. It
observes the full NSA growth signal through a Student-t likelihood ($\nu = 5$),
whose heavy tails keep a single noisy QCEW print from dominating:

$$
g^{\text{QCEW}}_t \sim \mathrm{StudentT}\!\left(5,\ g^{\text{nsa}}_t,\ \sigma^{\text{QCEW}}_t\right).
$$

The scale is a tiered base SD times a per-observation revision multiplier. The base
SD is one of two estimated LogNormal values depending on whether the observation is
a mid-quarter (M2) print or a boundary print:

$$
\sigma^{\text{QCEW}}_t = \sigma^{\text{base}}_t \cdot \text{mult}_t,\qquad
\sigma^{\text{base}}_t = \begin{cases}
  \sigma_{\text{mid}} & \text{M2 print}\\
  \sigma_{\text{boundary}} & \text{boundary print}
\end{cases}
$$

$$
\sigma_{\text{mid}} \sim \mathrm{LogNormal}(\log 0.0005,\ 0.15),\qquad
\sigma_{\text{boundary}} \sim \mathrm{LogNormal}(\log 0.002,\ 0.5).
$$

The tight M2 prior is deliberate: it prevents QCEW's extreme precision from
collapsing $\sigma$ toward zero and inducing a bimodal posterior. A `LogNormal`
(rather than `HalfNormal`) avoids the funnel that opens up as the scale shrinks.

### CES — best-available print, vintage-indexed noise

The CES likelihood observes whichever print survived as-of censoring for each month
(first / second / third), with a shared bias $\alpha$ and loading $\lambda$ and a
**per-vintage** noise SD, so older, more-settled prints carry tighter sigmas than
the noisy first print. For the seasonally adjusted series:

$$
g^{\text{CES,sa}}_t \sim \mathcal{N}\!\left(
  \alpha + \lambda\,g^{\text{sa}}_t,\ \sigma^{\text{ces,sa}}_{v(t)}\right),
$$

and identically for the NSA series against $g^{\text{nsa}}_t$. Priors (`CESPriors`):

$$
\alpha \sim \mathcal{N}(0,\ 0.005),\qquad
\lambda \sim \mathrm{TruncatedNormal}(1.0,\ 0.1;\ \text{low}=0.5),
$$

$$
\sigma^{\text{ces}}_v \sim \mathrm{LogNormal}(\log 0.002,\ 0.5)\quad\text{per vintage } v.
$$

### Providers — continuing-units growth, iid or AR(1) error

Each payroll provider observes the continuing-units NSA signal
$g^{\text{cont,nsa}}_t = g^{\text{cont}}_t + s_t$ (no birth/death term) with its own
bias $\alpha_p$, loading $\lambda_p$, and noise $\sigma_p$:

$$
\mu^{p}_t = \alpha_p + \lambda_p\,g^{\text{cont,nsa}}_t.
$$

Providers configured with `error_model = "iid"` use $y^p_t \sim \mathcal{N}(\mu^p_t,
\sigma_p)$; those configured `"ar1"` carry a persistent measurement error with
parameter $\rho_p$, conditioning each observation on its predecessor's residual and
inflating the first observation's variance to the stationary value
$\sigma_p/\sqrt{1-\rho_p^2}$. Priors (`ProviderPriors`):

$$
\alpha_p \sim \mathcal{N}(0,\ 0.005),\quad
\lambda_p \sim \mathcal{N}(1.0,\ 0.15),\quad
\sigma_p \sim \mathrm{InverseGamma}(3.0,\ 0.004),\quad
\rho_p \sim \mathrm{Beta}(2,\ 3).
$$

An unknown `error_model` raises rather than silently skipping the likelihood — a
deliberate deviation from the reference, which dropped it.

## Benchmark re-anchoring

CES levels are re-anchored once a year to the QCEW benchmark, and it is reasonable
to ask where that annual revision lives in this model. It is **not** a posterior
decomposition subsystem — the v2 model has no benchmark module. The benchmarking
force is split across two places:

1. **In the model — the QCEW Student-t anchor.** The continuous pull of the CES
   sample toward the near-census QCEW truth (the likelihood above) *is* the
   model's benchmarking mechanism. QCEW is the administrative ground truth; every
   posterior draw is shaped by how far the CES prints sit from it.

2. **At the data layer — the annual benchmark revision.** The discrete annual
   re-anchoring is handled before any array reaches the model. As-of censoring
   filters benchmark-revised CES rows out of the panel (they would leak a future
   re-anchoring into the past), and the benchmark wedge is spliced through the
   per-revision-cohort growth convention. Both are described in
   [Vintage data model & as-of censoring](../vintages-and-censoring.md). The
   historical published revision amounts live in
   `nfp_lookups.benchmark_revisions.BENCHMARK_REVISIONS`.

!!! note "No posterior benchmark decomposition in v2"
    The frozen reference repo carried a benchmark subsystem that extracted the
    revision from the posterior, decomposed it into continuing-units and
    birth/death pieces, and ran a horizon backtest. **None of that is in the
    shipping `nfp_model` package.** This page documents the model that runs.
