# TODO: Release 1 Model Upgrades

## Context

The methodology document (`docs/methodology_v8.md`) specifies three features
for the national accuracy model (Release 1) that are not yet implemented in
`src/alt_nfp/`.  The current model works end-to-end and produces valid
estimates.  These tasks upgrade it to the full Release 1 specification without
changing the data infrastructure (that is a separate TODO in
`TODO_data_infrastructure.md`).

**Priority order:** Tasks 1–3 are independent and can be done in any order.
Task 4 (forecast module) depends on Task 1.  Task 5 (diagnostics/plots)
depends on all of Tasks 1–3.

## References

- `docs/methodology_v8.md` — Release 1 specification (all equations and priors)
- `src/alt_nfp/model.py` — current PyMC model (modify in-place)
- `src/alt_nfp/data.py` — current data loading (modify for cyclical indicators)
- `src/alt_nfp/config.py` — constants and provider config (add new constants)
- `src/alt_nfp/forecast.py` — forward simulation (update for Fourier seasonal)
- `src/alt_nfp/sensitivity.py` — QCEW sigma sensitivity (update param specs)
- `src/alt_nfp/backtest.py` — expanding-window backtest
- `pp_estimation_v2.py` — main pipeline (plotting functions to update)

## Code Style

- Python 3.12, line length 100, formatter: black, linter: ruff
- Use **Polars** (not pandas) for all DataFrames
- Prefer **single quotes** over double quotes for strings
- **Two blank lines** after class and function definitions (PEP 8)
- Type hints on all public functions
- Docstrings on all public functions and classes (numpy-style)

---

## Task 1: Replace Static Seasonal with Truncated Fourier Expansion

### What exists

In `model.py`, the seasonal block is:

```python
s_raw = pm.Normal('s_raw', 0, 0.015, shape=11)
s_all = pt.concatenate([s_raw, (-pt.sum(s_raw)).reshape((1,))])
s_t = s_all[month_of_year]
pm.Deterministic('seasonal', s_all)
```

This is 11 free parameters with a sum-to-zero constraint, static across years.

### What the spec requires

A truncated Fourier expansion with K harmonics and annually-evolving
amplitudes:

```
s_t = Σ_{k=1}^{K} [ A_k(y(t)) · cos(2πk·m(t)/12) + B_k(y(t)) · sin(2πk·m(t)/12) ]
```

where `m(t) ∈ {0, ..., 11}` is the month index and `y(t)` is the year index.

Fourier coefficients evolve annually via Gaussian random walks:

```
A_k(y) = A_k(y-1) + ω_{A,k}(y),   ω_{A,k} ~ N(0, σ²_{ω,k})
B_k(y) = B_k(y-1) + ω_{B,k}(y),   ω_{B,k} ~ N(0, σ²_{ω,k})
```

### Priors

```
A_k(0), B_k(0) ~ N(0, 0.015²)    for k = 1, ..., K
σ_{ω,k} ~ Half-N(0, 0.005 / k)   [tighter for higher harmonics]
```

### Implementation steps

1. **Add config constants** in `config.py`:
   - `N_HARMONICS = 4` (default K; make configurable)

2. **Add year index to data dict** in `data.py`:
   - Compute `year_of_obs` as integer array mapping each `t` to a year index
     `y ∈ {0, ..., n_years - 1}`.  This is `(dates[t].year - dates[0].year)`.
   - Add `'year_of_obs'` and `'n_years'` to the returned dict.

3. **Replace seasonal block** in `model.py`:
   - Remove `s_raw`, `s_all`, `s_t` lines.
   - Add:
     ```python
     K = N_HARMONICS
     n_years = data['n_years']
     year_of_obs = data['year_of_obs']
     month_of_year = data['month_of_year']

     # Innovation std per harmonic (decreasing with k)
     sigma_fourier = pm.HalfNormal(
         'sigma_fourier', sigma=0.005 / pt.arange(1, K + 1), shape=K
     )

     # GRW on Fourier coefficients: shape (n_years, 2*K)
     # Columns 0..K-1 are A_k, columns K..2K-1 are B_k
     sigma_vec = pt.tile(sigma_fourier, 2)  # (2K,)
     fourier_coefs = pm.GaussianRandomWalk(
         'fourier_coefs',
         sigma=sigma_vec,
         init_dist=pm.Normal.dist(0, 0.015),
         shape=(n_years, 2 * K),
     )

     # Evaluate seasonal at each t
     k_vals = pt.arange(1, K + 1)  # (K,)
     cos_basis = pt.cos(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)
     sin_basis = pt.sin(2 * np.pi * k_vals * month_of_year[:, None] / 12)  # (T, K)

     A_t = fourier_coefs[year_of_obs, :K]   # (T, K)
     B_t = fourier_coefs[year_of_obs, K:]   # (T, K)

     s_t = pt.sum(A_t * cos_basis + B_t * sin_basis, axis=1)  # (T,)
     pm.Deterministic('seasonal_fourier', fourier_coefs)
     ```
   - The `s_t` variable name must remain `s_t` so downstream code
     (`g_cont_nsa = g_cont + s_t`, etc.) is unchanged.

4. **Update `pm.Deterministic('seasonal', ...)`** — store the full
   `fourier_coefs` array rather than a 12-element vector.  Downstream
   diagnostics reference `'seasonal'` so either rename consistently or
   keep both:
   ```python
   pm.Deterministic('seasonal', s_t)  # per-observation seasonal value
   pm.Deterministic('fourier_coefs', fourier_coefs)  # (n_years, 2K) for plotting
   ```

5. **Verify** the model still samples without divergences on the existing data.
   The Fourier model has fewer effective parameters per year than the current
   11-free specification, so it should be easier to sample, not harder.

### Tests

- Prior predictive check: `s_t` values should be within ±3% (0.03) for
  reasonable seasonal amplitude.
- Posterior: seasonal pattern should roughly match the current static
  estimate (bar chart in `growth_and_seasonal.png`).
- Run `sensitivity.py` and confirm QCEW sigma sensitivity results are stable.

---

## Task 2: CES Vintage-Specific Measurement Equations

### What exists

In `model.py`, CES enters as a single SA and single NSA observation per month:

```python
alpha_ces = pm.Normal('alpha_ces', mu=0, sigma=0.005)
lam_ces = pm.Normal('lam_ces', mu=1.0, sigma=0.15)
sigma_ces_sa = pm.HalfNormal('sigma_ces_sa', sigma=0.005)
sigma_ces_nsa = pm.HalfNormal('sigma_ces_nsa', sigma=0.005)

pm.Normal('obs_ces_sa', mu=..., sigma=sigma_ces_sa, observed=...)
pm.Normal('obs_ces_nsa', mu=..., sigma=sigma_ces_nsa, observed=...)
```

There is no vintage distinction: first print, second print, and final are
all loaded from the same column.

### What the spec requires

Each CES print (first, second, final; `v ∈ {1, 2, 3}`) enters as a
**separate observation** with vintage-specific noise:

```
y_{t,v}^{CES,SA} ~ N(α^CES + λ^CES · g_t^{total,SA}, σ²_{CES,SA,v})
y_{t,v}^{CES,NSA} ~ N(α^CES + λ^CES · g_t^{total,NSA}, σ²_{CES,NSA,v})
```

Shared `α^CES` and `λ^CES` across vintages.  Separate noise per vintage.

### Priors (per vintage)

```
σ_{CES,SA,v} ~ InverseGamma(3, 0.004)    for v = 1, 2, 3
σ_{CES,NSA,v} ~ InverseGamma(3, 0.004)   for v = 1, 2, 3
```

The InverseGamma(3, 0.004) has mode ≈ 0.001, keeping σ away from zero
(avoids Neal's funnel) while allowing data to pull toward larger values for
noisier first prints.

### Implementation steps

**This task has a data prerequisite and a model change.  If vintage-tracked
CES CSVs are not yet available, implement the model change with a fallback
that treats the current single-vintage data as vintage 3 (final).**

1. **Data loading** (`data.py`):
   - If vintage-tracked CSVs exist (columns like `g_ces_sa_v1`, `g_ces_sa_v2`,
     `g_ces_sa_v3`), load them into separate arrays.
   - If not, create the vintage structure from existing data:
     ```python
     # Fallback: treat existing CES as final vintage (v=3)
     g_ces_sa_by_vintage = [
         np.full(T, np.nan),  # v1 placeholder
         np.full(T, np.nan),  # v2 placeholder
         g_ces_sa,             # v3 = current data
     ]
     ```
   - Add `'g_ces_sa_by_vintage'`, `'g_ces_nsa_by_vintage'`, and per-vintage
     obs index arrays to the data dict.

2. **Model** (`model.py`):
   - Keep shared `alpha_ces`, `lam_ces`.
   - Replace single `sigma_ces_sa` / `sigma_ces_nsa` with vintage-indexed:
     ```python
     sigma_ces_sa = pm.InverseGamma(
         'sigma_ces_sa', alpha=3, beta=0.004, shape=3
     )
     sigma_ces_nsa = pm.InverseGamma(
         'sigma_ces_nsa', alpha=3, beta=0.004, shape=3
     )
     ```
   - Loop over vintages:
     ```python
     for v in range(3):
         g_sa_v = data['g_ces_sa_by_vintage'][v]
         obs_v = np.where(np.isfinite(g_sa_v))[0]
         if len(obs_v) > 0:
             mu_sa = alpha_ces + lam_ces * g_total_sa[obs_v]
             pm.Normal(
                 f'obs_ces_sa_v{v+1}',
                 mu=mu_sa,
                 sigma=sigma_ces_sa[v],
                 observed=g_sa_v[obs_v],
             )
         # Same for NSA
     ```

3. **Update `build_obs_sources()`** in `data.py` to include per-vintage
   entries for posterior predictive checks.

4. **Update `sensitivity.py`** `_build_param_specs()` to include the new
   per-vintage sigma parameters in the comparison table.

### Tests

- With fallback data (single vintage), results should be nearly identical
  to current model (only the prior on sigma changes from HalfNormal to
  InverseGamma).
- When vintage data is added, v1 noise should be estimated larger than v3.

---

## Task 3: Add Cyclical Indicators to Birth/Death Model

### What exists

In `model.py`, the BD equation uses two covariates:

```python
bd_t = phi_0 + phi_1 * birth_rate_c + phi_2 * bd_qcew_c + sigma_bd * xi_bd
```

where `birth_rate_c` is the centered composite provider birth rate and
`bd_qcew_c` is the centered QCEW-minus-provider-mean proxy (lagged 6 months).

### What the spec requires

Add a vector of centered cyclical indicators `X_t^{cycle}`:

```
BD_t = φ_0 + φ_1 · X_t^{birth} + φ_2 · BD_{t-L}^{QCEW} + φ_3 · X_t^{cycle} + σ_BD · ξ_t
```

Cyclical indicators (demand-side, complementing the supply-side provider
birth rate):

1. **Initial unemployment claims** (weekly → monthly average, SA)
2. **Financial conditions index** (e.g., Chicago Fed NFCI, weekly → monthly)
3. **Census business applications** (monthly, from Census Business Formation
   Statistics)

### Priors

```
φ_3 ~ N(0, 0.3²)   [per indicator, or a single vector prior]
```

### Implementation steps

1. **Data acquisition** — Place CSV files in `data/`:
   - `data/claims_weekly.csv` — initial unemployment claims (from FRED:
     ICSA series).  Columns: `date`, `claims`.
   - `data/nfci.csv` — Chicago Fed NFCI (from FRED: NFCI series).
     Columns: `date`, `nfci`.
   - `data/business_applications.csv` — Census Business Formation Statistics
     (from Census: series BA_BA).  Columns: `date`, `applications`.

   **If CSVs are not yet available, create placeholder loading that
   gracefully skips missing files** (same pattern as the existing
   `births_file` optional loading in `data.py`).

2. **Data loading** (`data.py`):
   - Add a function `_load_cyclical_indicators(dates)` that:
     - Loads each CSV if it exists, otherwise returns NaN.
     - For weekly series (claims, NFCI), computes the monthly average
       aligned to `dates`.
     - Centers each indicator (subtract mean of non-NaN values).
     - Returns a dict: `{'claims_c': np.ndarray, 'nfci_c': np.ndarray,
       'biz_apps_c': np.ndarray}`.
   - Add the cyclical indicator arrays to the data dict.
   - Print summary stats (N obs, mean) alongside existing BD covariate
     summary.

3. **Config** (`config.py`):
   - Add `CYCLICAL_INDICATORS` list of dicts:
     ```python
     CYCLICAL_INDICATORS: list[dict] = [
         {'name': 'claims', 'file': 'claims_weekly.csv', 'col': 'claims', 'freq': 'weekly'},
         {'name': 'nfci', 'file': 'nfci.csv', 'col': 'nfci', 'freq': 'weekly'},
         {'name': 'biz_apps', 'file': 'business_applications.csv', 'col': 'applications', 'freq': 'monthly'},
     ]
     ```

4. **Model** (`model.py`):
   - After the existing `phi_2 * bd_qcew_c` term, add cyclical loadings:
     ```python
     # Cyclical indicators (demand-side BD covariates)
     cyclical_data = []
     cyclical_obs_masks = []
     for key in ['claims_c', 'nfci_c', 'biz_apps_c']:
         arr = data.get(key)
         if arr is not None and np.any(np.isfinite(arr)):
             cyclical_data.append(arr)
             cyclical_obs_masks.append(np.isfinite(arr))

     n_cyclical = len(cyclical_data)
     if n_cyclical > 0:
         phi_3 = pm.Normal('phi_3', mu=0, sigma=0.3, shape=n_cyclical)
         cyclical_contribution = pt.zeros(T)
         for i, (arr, mask) in enumerate(zip(cyclical_data, cyclical_obs_masks)):
             safe = np.where(mask, arr, 0.0)
             cyclical_contribution = cyclical_contribution + phi_3[i] * safe
     ```
   - Add `cyclical_contribution` to the `bd_t` expression.
   - Use the same graceful-fallback pattern as existing covariates:
     centered values zero out when missing, so `bd_t` collapses to
     `phi_0 + sigma_bd * xi_bd` in the worst case.

5. **Update `sensitivity.py`** `_build_param_specs()` to include `φ_3`
   parameters in the comparison table.

### Tests

- With no cyclical CSV files present, model should produce identical
  results to current (graceful degradation).
- With data present, `phi_3` posteriors should be estimable and the BD
  component should show improved fit at turning points (2020, 2022).

---

## Task 4: Update Forecast Module for Fourier Seasonal

**Depends on:** Task 1 (Fourier seasonal in model.py)

### What exists

`forecast.py` uses the static 12-element seasonal vector:

```python
seasonal_post = idata.posterior['seasonal'].values  # (chains, draws, 12)
# ...
g_nsa_fwd[:, :, h] = g_cont_fwd[:, :, h] + seasonal_post[:, :, mi] + bd_fwd[:, :, h]
```

### What needs to change

With Fourier seasonal, the forecast must evaluate the Fourier expansion at
future month/year combinations using the posterior `fourier_coefs`.

1. **Extract Fourier coefficients** from posterior:
   ```python
   fourier_post = idata.posterior['fourier_coefs'].values  # (chains, draws, n_years, 2K)
   ```

2. **For each forecast month**, determine its month index `m` and year index
   `y`.  If the forecast year extends beyond the last estimated year, use the
   last year's coefficients (the GRW's forecast is its final value plus
   optional drift — for short horizons, holding constant is reasonable):
   ```python
   K = N_HARMONICS
   k_vals = np.arange(1, K + 1)
   cos_val = np.cos(2 * np.pi * k_vals * m / 12)  # (K,)
   sin_val = np.sin(2 * np.pi * k_vals * m / 12)  # (K,)

   A_k = fourier_post[:, :, y, :K]   # (chains, draws, K)
   B_k = fourier_post[:, :, y, K:]   # (chains, draws, K)

   s_fwd = np.sum(A_k * cos_val + B_k * sin_val, axis=-1)  # (chains, draws)
   ```

3. **For multi-year forecasts** (beyond 1–2 months), optionally propagate
   the GRW forward by adding innovations drawn from the posterior
   `sigma_fourier`.  For the typical 1–3 month nowcast horizon, holding
   coefficients at the last estimated year is adequate.

### Tests

- Forecast seasonal values should be close to the historical model seasonal
  for the same calendar months.
- Forecast fan charts should be visually similar to current output.

---

## Task 5: Update Diagnostics and Plots

**Depends on:** Tasks 1–3

### Seasonal visualization

The current seasonal bar chart in `pp_estimation_v2.py` (`plot_results()`)
shows a 12-bar comparison of model vs. empirical seasonal.  With Fourier
seasonal, this needs to show:

- **Time-varying seasonal**: a line plot of `s_t` over the full sample,
  colored by year or with a gradual color shift showing evolution.
- **Fourier coefficient trajectories**: small multiples of `A_k(y)` and
  `B_k(y)` for each harmonic, showing the GRW evolution.
- **Current-year seasonal pattern**: the familiar 12-month bar chart but
  using the last year's Fourier evaluation instead of the static parameters.

### CES vintage residuals

If vintage-specific CES data is loaded, add a residual subplot showing
standardized residuals by vintage (v1, v2, v3) in different colors.  This
visualizes whether earlier vintages are indeed noisier.

### BD covariate contributions

With cyclical indicators added, the BD decomposition plot should show the
contribution of each covariate to `BD_t`:
- `φ_1 · birth_rate_c_t`
- `φ_2 · bd_qcew_c_t`
- `φ_3[0] · claims_c_t`, `φ_3[1] · nfci_c_t`, `φ_3[2] · biz_apps_c_t`
- `σ_BD · ξ_t` (residual)

Stacked area or line chart showing how each demand-side and supply-side
covariate contributes to the BD estimate over time.

### Precision budget update

`sensitivity.py` `_precision_shares()` and `print_source_contributions()`
need to account for the new vintage-specific CES sigmas (three entries
instead of one for CES SA and CES NSA).

---

## Summary: Files Modified per Task

| File | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------|--------|--------|--------|--------|--------|
| `config.py` | K constant | — | cyclical config | — | — |
| `data.py` | year index | vintage arrays | cyclical loading | — | obs_sources |
| `model.py` | Fourier block | vintage loop + IG priors | cyclical φ_3 | — | — |
| `forecast.py` | — | — | — | Fourier eval | — |
| `sensitivity.py` | — | param specs | param specs | — | precision budget |
| `pp_estimation_v2.py` | — | — | — | — | plot functions |
| `backtest.py` | — | — | — | — | possibly |