# Government wedge forecast — Track B step 1 (design/spec)

Status: **design, 2026-06-19**. The implementation spec for the government side of Track B
(`specs/model_improvements.md` §2/§9/§11). Forecasts the government **wedge** so a Total-NFP
posterior can be assembled from the existing private nowcast and scored against the
Total-NFP consensus. Rationale and the full decision trail live in
`docs/government_design.md` (decision record — *not* implemented from); this spec is the
normative *what*. Companions: `specs/bloomberg_consensus.md` (consensus contract),
`specs/a5_real_competitors.md` (the eval harness this extends).

## TL;DR

1. **Target = the wedge `g = published_00 − published_05`**, first-print MoM change, SA,
   thousands, at release-eve. Then `Total = private_nowcast + wedge` reproduces published
   Total **by construction** (no SA-additivity residual, no MinT). The wedge target is
   derived from `00` & `05` first prints already in the store — **no new target acquisition**.
2. **Model = a thin Bayesian change-space STS** (NumPyro): constant drift + a single shrunk
   monthly-seasonal block + a deterministic, announcement-priored **intervention layer** +
   masked iid-Normal likelihood. It is a coherent-posterior generalization of the
   calendar-month-mean (~23k RMSE floor it must not underperform); the complexity budget is
   spent on interventions, the only place an edge over consensus's implicit government
   forecast exists.
3. **Assembly = independent draw-wise convolution** of the private nowcast posterior and the
   wedge posterior into a Total-NFP posterior, with the private leg converted growth→change
   via `change_draws_k`. A residual-coupling knob is specified but **default off**.
4. **Scoring = extend the A5/Tier-0 harness** to score assembled Total vs the Total `00`
   first print and vs consensus (pluggable, `—` until the Bloomberg file lands).
5. **Government `90/91/92/93` SA** are acquired as a lightweight published-SA artifact and
   used for **diagnostics + intervention-prior calibration only** — *not* likelihood
   regressors, *not* the full government store axis.
6. **Build here, validate on the port.** Build gate = convergence + sane posterior, **not**
   accuracy. The accuracy/keep-drop verdict vs consensus is deferred to the Bloomberg port.

## 1. Scope & non-goals

**In scope.** (a) A standalone Bayesian wedge model (`nfp_model/wedge.py`); (b) the Total
assembly helper (harness layer); (c) the Total-vs-consensus scoring scaffold (extends the
A5 harness); (d) the lightweight government-signal artifact + diagnostics.

**Non-goals / deferred.** Component-structured *likelihood* (modeling `91/92/93`
separately) — deferred to the port (§9). The government **store axis**
(`ownership='government'`, codes `90–93`) — **not** realized; lightweight artifact instead
(§3.2). Always-on residual coupling — specified, default off (§5). Census activation —
dormant until 2030 (§4.3). Lagged government as a regressor / drift-anchor — rejected (§7).
No change to `model.py` and **no new A3 parity baseline** (the wedge is a *separate* model).

## 2. Target — the first-print wedge

Define the wedge target per reference month from store first prints.
`nfp_ingest.first_print.first_print_changes` is **keyword-only after `store_path`**
(`store_path` defaults to `VINTAGE_STORE_PATH`) and returns columns
`ref_date, first_print_growth, first_print_change_k, vintage_date` (the first-print release
date, `fp_vintage`):

```python
fp00 = first_print_changes(industry_type="total", industry_code="00")  # → first_print_change_k
fp05 = first_print_changes(industry_type="total", industry_code="05")
wedge = (fp00.join(fp05, on="ref_date")            # join on ref_date
              # require contemporaneous release: fp_vintage(00) == fp_vintage(05) before differencing
              .with_columns(wedge_change_k = chg00 − chg05))
```

Both legs must come from the **same first-print release** — join on `ref_date` and assert
the two `vintage_date` (`fp_vintage`) values are identical (or within the release window)
before differencing, so the wedge is a single-vintage difference, not a cross-vintage one.
Verified to run on the store (n≈106; COVID-excluded n≈82; std ≈ 24k) — working read example
in `scripts/_wedge_diag.py` (which does `load_dotenv('.env')`, the store `.env` gotcha).
Units are thousands, SA, MoM net change. COVID (2020–21) and the Oct-2025 no-print hole are
**masked**, not dropped (§4.4). The as-of backtest reconstructs the target through M−1 by
filtering on the first-print `vintage_date ≤ release_eve(M)` (no lookahead; M's own first
print is the scored actual, revealed at release).

## 3. Data

### 3.1 Target (no new acquisition)
From the store via `first_print_changes` (§2). Load-bearing store dependency; nothing added.

### 3.2 Government signals (lightweight, diagnostics/calibration only)
Two **different** acquisition mechanisms — the FRED indicator pattern does not fit both:

**(a) CES `90/91/92/93` SA** (total govt / federal / state / local) — needed in v1 for
diagnostics. Added as `CYCLICAL_INDICATORS`-style FRED entries
(`nfp_lookups.provider_config.CyclicalIndicator` = `(name, fred_id, freq, pub_lag)`; fetched
by `nfp_ingest.indicators.download_indicators` → `fetch_fred_series`, read via
`read_indicator`). The candidate FRED ids (e.g. `CES9000000001` … `CES9300000001`) are
**plan-side verification, NOT asserted to exist here** — the plan must confirm they are
fetchable before wiring.

**(b) The BLS decennial Census NSA worker table** — *not on FRED*, so it cannot use the
indicator path. Specify a **separate mechanism**: a manual download committed under
`data/government/` with an explicit schema (ref-month, NSA level, OTM change), or a
dedicated `read_census_table()`. **Dormant until 2030** — built but unfit in v1.

Neither is read into the model likelihood. Consumers: the CES `90–93` SA artifact feeds
(i) a **decomposition diagnostic** (`wedge ≈ 90` up to the small SA residual `r`) and
(ii) **intervention-prior calibration** (how a past federal RIF landed in observed `91` →
sets the §8 prior sd). The Census table feeds only the dormant 2030 intervention slot. Do
**not** extend the reserved `ownership='government'` store axis (gold-plating;
`docs/government_design.md` §8.6). Ad-hoc reads must `load_dotenv('.env')` or they hit the
empty local store.

## 4. The model — `nfp_model/wedge.py`

A standalone NumPyro program importing **only** jax/numpyro/numpy (honors the model-layer
import boundary; imports no `nfp_*` package). It consumes a plain numpy/dict input (a
`ModelData`-style mapping) assembled by the harness, not any `nfp_ingest` object. Object =
the wedge MoM **change** `y_t` (thousands, SA), modeled directly in **change-space**.

**Input-dict contract** (the `data` mapping the harness passes, analogous to
`nfp_model.data.MODEL_ARRAY_KEYS`): `y` (T-length wedge MoM change, thousands), `month_of_year`
(T-length int in 1–12, preserved for **all** T including masked rows — §4.2), `T`, optional
`mask` (T-length bool, read via `data.get("mask")`). **Intervention handoff (pin this):**
interventions arrive as **pre-built change-space basis arrays** `X_intervention`
(shape `T × K`, one column per active mechanism, built by the harness from the
`KNOWN_INTERVENTIONS` table) plus per-column prior `(mean, sd)` magnitudes; the model samples
a `K`-vector of coefficients and adds `X_intervention @ coef` to `mu`. (This is the one
genuine ambiguity — basis-arrays-in, coefficients-sampled — and is pinned here; everything
else has one implementer and no parity gate.)

### 4.1 Mean equation
```
mu_t = drift + season[month_of_year[t]] + Σ_k intervention_k(t)
```
- `drift` — a single **constant**. **Units note (critical):** `y` is CES `change_k`, already
  *in thousands of jobs*, so its numeric magnitude is O(±100) with std ≈ 24 — NOT tens of
  thousands. Priors are in these thousands-units: `drift ~ Normal(0, 50)`.
  **No random-walk / AR on the change-mean** (that is I(2) on the wedge level; the RW
  baseline's 30.2k vs the seasonal mean's 23.0k confirms persistence hurts). An AR(1)
  residual is out of scope (guilty-until-proven).
- Seasonal block (§4.2); intervention layer (§4.3).

### 4.2 Seasonality — single shrunk monthly block
11 free monthly effects `season[1..11]`, hierarchical shrinkage `season[m] ~ Normal(0,
τ_season)`, `τ_season ~ HalfNormal(~30)` (thousands-units, §4.1), with the 12th month **pinned by sum-to-zero**:
`season[12] = −(season[1] + … + season[11])`, computed **deterministically** (a
`numpyro.deterministic` transform of the 11 sampled sites, *not* a 12th sampled site) so the
drift/seasonal ridge is removed and divergence debugging is unambiguous. **Dummy monthly
effects, not Fourier** (the dominant signal is sharp state/local education seasonality).
**No federal/state-local split** — with wedge-only observations it is non-identified (the
likelihood sees only the sum). Named calendar effects (Nov poll-worker, Dec USPS) are
absorbed into the relevant monthly effect.

**Identification under masking.** `month_of_year` is preserved for **all** T rows, including
masked ones; masked months contribute **zero likelihood** but their seasonal coefficient
retains its `N(0, τ_season)` prior (shrinkage prevents collapse). The 2017–2026 window
leaves ~7–8 unmasked obs per calendar month, so no month is empty in v1; add a post-fit
**per-calendar-month R-hat check** (warn, not fail) to catch a future zero-unmasked-obs
month.

### 4.3 Intervention layer (the edge)
Deterministic prior adjustments (**not** regime-switching), in change-space. **Shape is
fixed by config; only magnitude is sampled**, under an **informative announcement-derived
prior** (never flat). Level-space X-13 shapes map to change-space:

| Event | Level shape | Change-space encoding |
|---|---|---|
| Permanent RIF (no back-pay) | level shift | single negative pulse (cumsum steps down & stays) |
| Phased RIF | ramp | short negative box (−m/k over k months) |
| Census hiring | temporary change | positive pulse + smaller decaying-reversal pulses |
| Back-paid shutdown | additive outlier | +spike then −giveback, nets ≈ 0 |

Concrete v1 mechanisms over 2017–2026 (ex-COVID): (a) the **2025 federal RIF** — the one
live in-sample shock — negative pulse/box, magnitude `~Normal(announced_count, sd)` (§8);
(b) a **dormant Census slot** — built, unfit, activates 2030 from the §3.2 table;
(c) the **Oct-2025 back-paid shutdown** — AO ≈ null (handled by the §4.4 mask, no term).

**Operationalization (temporal-axis table — NOT a flat dict).** `nfp_lookups.benchmark_revisions`
is a flat `dict[int, float|None]` with **no as-of axis**; mirroring it would make the
lookahead guard vacuous (a single magnitude per ref-month necessarily encodes the realized
shock). Instead, `KNOWN_INTERVENTIONS` is a table with an **announcement-date axis**, rows:

```
(ref_month, intervention_name, shape, announced_magnitude_k, announced_magnitude_sd_k,
 announcement_date, source_url)
   shape ∈ {'pulse', 'box', 'tc'}   # the change-space encodings of §4.3's table
```

A helper `get_known_interventions_as_of(as_of)` returns only rows with
`announcement_date ≤ release_eve(as_of)` — that censored subset is what builds the
`X_intervention` basis and supplies the magnitude priors for that fit. The operator
interface `predict_wedge_change(..., interventions=[...])` takes the same announcement-dated
rows. **Frontier rule:** a brand-new end-of-series shock enters only when its
`announcement_date` is on/before release-eve; its *shape* is config, never inferred from the
1–2 frontier months (the guard in §10 enforces this by a **date** comparison, not a value
comparison).

### 4.4 Likelihood & masking
`y_t ~ Normal(mu_t, sigma)`, `sigma ~ HalfNormal(~30)` (thousands-units, §4.1; anchored to
the ≈23 cal-month-mean residual). The likelihood is wrapped in a mask context over COVID 2020–21 and
the Oct-2025 hole — **mask, never delete rows**, to keep drift/seasonal on a contiguous
calendar axis. **Import boundary:** `wedge.py` does **not** import `nfp_model.model._maybe_mask`
(that would break the no-`nfp_*` rule); it **reimplements the same idiom inline** via
`numpyro.handlers.mask(mask=…)` (the pattern at `model.py:53-57`). Non-centered
parametrization throughout (house style). **Reserve (off by default):** swap
Normal→StudentT(ν≈6) only if posterior-predictive checks show shock-month variance leakage.

### 4.5 Output
Posterior predictive for the nowcast month T: `mu_T + noise` draws, shape
`(num_chains, num_samples)`, in **change-k** — ready to convolve with the private nowcast.

## 5. Assembly — Total-NFP posterior (harness layer, not `nfp-model`)

An explicit, **tested** `assemble_total` helper (harness side; `nfp-model` stays
assembly-free). Default = **independent draw-wise convolution**:

```python
private_change_k = scoreboard.change_draws_k(nowcast_pred_draws,
                                             prev_index=base_index, idx_to_level=idx_to_level)
total_change_k   = private_change_k + wedge_change_k   # element-wise, both length N
```

- **Growth→change conversion.** Persisted private `nowcast_pred_draws` are **growth/index
  space**; they MUST pass through `nfp_vintages.scoreboard.change_draws_k` (flattens via
  `reshape(-1)`) before summation. Wedge draws are natively change-k.
- **`(base_index, idx_to_level)` anchor** comes from `nfp_ingest.model_data.levels_provenance(levels)`
  (the function `scripts/run_a5_backtest.py:109` already uses). **Highest-risk seam:** a wrong
  anchor reintroduces the `base_index` NaN class the project already hit — reuse the
  first-finite anchor and cover with a test asserting no NaN.
- **Draw-count alignment (pin this).** `N = num_chains_wedge × num_samples_wedge` — the
  **wedge fit is authoritative**; the private `change_draws_k` output is resampled/broadcast
  to length `N` before the element-wise add (no silent broadcast — define and test the
  resample). Private and wedge MCMC do **not** share a master seed (independent fits); the
  pairing is positional after resample.
- **As-of provenance.** The `nowcast_pred_draws` consumed here must come from an
  **as-of-censored** private fit — the harness passes `as_of` to both `build_panel` and
  `panel_to_model_data` before fitting (`run_a5_backtest.py:134-156`; draws extracted at
  `batch.py:314-322`). `assemble_total` must be fed the as-of-matched private draws for the
  same release-eve, never a final-vintage fit.
- **Adding-up is exact** (we forecast the wedge directly): `Total = private + wedge` ≡
  published `00`. No MinT/reconciliation.
- **Residual-coupling knob (default OFF, deferred to port).** When enabled, add `η·z_t` to
  the wedge mean, where `z_t = (private_draw − mean(private_draws)) / std(private_draws)` is
  the **standardized, mean-centered** per-draw private residual (source = the resampled
  `private_change_k` vector). Because `z` is mean-zero, the Total **point** forecast is
  invariant; only intervals widen. Enable only if port-side Total interval coverage is
  miscalibrated under independence **and** out-of-sample error-ρ is materially positive
  (measured outcome-ρ ≈ +0.19; error-ρ expected lower).

**Wedge fit primitive & persistence.** The wedge is sampled by a new `fit_wedge_batch`
(mirroring `nfp_model.sampling.fit_model_batch`'s signature pattern — `fit_model` hardcodes
`NUTS(nfp_model)` at `sampling.py:67,82-84`, so it is **not** reusable). Its predictive draws
persist under the key **`wedge_pred_draws`** alongside the private `nowcast_pred_draws`, so
`assemble_total` loads both with the §5 alignment rule. (Sampler knobs and the on-disk path
template are plan-side, §9; the **draw key** and **alignment requirement** are normative.)

**`assemble_total` unit test (normative):** `assemble_total(private (2,50), wedge (3,100))
→ (300,)`; element-wise checks for `η=0` (pure sum) and `η>0` (point-invariant, intervals
widen); no NaN from the anchor.

## 6. Scoring scaffold

Extend `scripts/run_a5_backtest.py` + `nfp_vintages.scoreboard` to score the assembled Total
change against (i) the Total `00` first print and (ii) consensus, reusing the existing
`change_draws_k` / `crps_sample` / `interval_coverage` machinery on the assembled Total
draws. Consensus is loaded via the existing pluggable `load_consensus()` / `Consensus`
(T-1-only, `None`-tolerant). COVID excluded from headline metrics; the Oct-2025 frontier
month flagged, not silently pooled. Record a venue tag (`full`/`public-only`) per row.

**Two committed fixtures** under `tests/fixtures/` (today `test_competitors.py` has only an
inline `Consensus(None)` test and a populated helper — no fixture file): (1) a
**null/absent** consensus fixture exercising the `None`-tolerant path that renders the
consensus column as `—`; (2) a **populated** fixture with named synthetic values that
exercises the join/scoring arithmetic and validates `Total = private + wedge` against the
scored consensus. (Literal fixture numbers are plan-side; the two paths are normative — §10.)

## 7. Government signals are NOT likelihood inputs

`90/91/92/93` SA are diagnostics + intervention-prior calibration only (§3.2). Rationale:
the only release-eve government info is *lagged* government, which is low-persistence (the
wedge change is near-white around its seasonal mean), and no proven external leading
indicator exists. Their genuine value (federal-shock attribution) is captured through prior
calibration, not regression. See `docs/government_design.md` §8.4.

## 8. Open input required from the maintainer

The **2025 federal RIF intervention entry** — one (or more) `KNOWN_INTERVENTIONS` rows with
the §4.3 schema: `ref_month` (effective month(s)), `shape` (`'pulse'` for a permanent
level-shift — the lean — or `'box'` for a phased ramp), `announced_magnitude_k` (announced
permanent-separation count, **no back-pay** ⇒ a real level shift per F8),
`announced_magnitude_sd_k` (honest sd), `announcement_date`, and `source_url`. These are
announcement-derived human inputs — they cannot be data-inferred without violating the
frontier rule, and the `announcement_date` is what the §10 guard censors on. v1 **scaffolds
the machinery with placeholder priors** so implementation is not blocked; the maintainer
supplies the real values. This is a **human-input gate**, not engineering work.

## 9. Phasing & venue

**v1 (build locally):** §4 core (drift + shrunk monthly block + masked iid-Normal); one live
intervention (2025 RIF) + dormant Census slot; `KNOWN_INTERVENTIONS` + operator interface
**with** the as-of-censored fixture and lookahead guard (§10); the lightweight signal
artifact + decomposition/calibration diagnostics; `assemble_total` with the explicit
`change_draws_k` conversion; the Total-vs-consensus scaffold. **Build gate = convergence +
sane posterior** (R-hat, divergences, posterior-predictive on the clean window) — **not**
accuracy. **Defer to the Bloomberg port:** the accuracy/keep-drop verdict vs consensus; the
residual-coupling knob; StudentT-by-default; the component-structured likelihood (only if a
residual-seasonality / shock-attribution diagnostic shows it earns its keep, and only with
component observations entered as separate observed series). **Defer to 2030:** Census
activation. Government data is public, so the wedge **baseline** is fully validatable
locally; only the intervention **edge** and the consensus head-to-head need the port.

## 10. Test plan (normative)

- **Target:** `wedge_change_k` join correctness vs `_wedge_diag` numbers (std ≈ 24k
  COVID-excluded); both `00` and `05` legs joined on `ref_date` with **identical resulting
  `vintage_date` (`fp_vintage`)** before differencing; as-of reconstruction excludes M's own
  first print.
- **Lookahead guard (critical — DATE comparison, not value):** for each backtest `as_of`,
  load the censored intervention subset via `get_known_interventions_as_of(as_of)` and assert
  every intervention magnitude used by that fit comes only from rows with
  `announcement_date ≤ release_eve` — never a realized post-hoc shock size. Without this the
  headline edge is fake (§4.3).
- **Convergence smoke + build gate:** one `wedge.py` fit on the **clean window** (2017–2026
  ex-COVID-2020/21 and ex-Oct-2025) converges: R-hat ≤ 1.01 (incl. a **per-calendar-month**
  R-hat warn check, §4.2), no divergences, and posterior-predictive within gate —
  **80% interval coverage in [60%, 95%]** and **ppc-mean RMSE ≤ 2× the 23k cal-month
  baseline**. `τ_season` funnel watched (non-centered).
- **Assembly:** `assemble_total((2,50),(3,100)) → (300,)` (§5) — growth→change conversion
  via `change_draws_k`, draw resample to `N=wedge`, no NaN from the `levels_provenance`
  anchor; `Total = private + wedge` identity on a fixture; `η=0` pure-sum and `η>0`
  point-invariant/interval-widening checks.
- **Scoring (two fixtures, §6):** (1) null/absent consensus → column renders `—`;
  (2) populated consensus → join/scoring arithmetic + `Total = private + wedge` validated
  against the scored consensus; COVID/frontier exclusion honored.
- **Intervention shapes:** change-space encodings (`pulse`/`box`/`tc`) cumulate to the
  intended level shapes (a permanent-RIF `pulse` → a sustained level step in cumsum).

## 11. Parity & boundaries

`nfp_model/wedge.py` imports only jax/numpyro/numpy. It is a **new, separate** model — it
does not touch `model.py`/`nowcast.py` and needs **no new A3 parity baseline** (it alters
nothing parity-gated). The assembly (`assemble_total`) and scoring live in the harness /
`nfp_vintages`, never inside `nfp-model`. If v1 later promotes government components to
modeled likelihood inputs (§9, port), that is a new model version behind its own baseline.

## 12. Risks

Carried from `docs/government_design.md` §9: (1) assembly-seam units mismatch (mitigate:
tested `assemble_total` + first-finite anchor); (2) intervention-table lookahead (mitigate:
as-of fixture + guard test — §10); (3) independence assumption mildly overconfident on Total
intervals (mitigate: ship independent, port-side coverage triggers the coupling knob);
(4) seasonal-vs-intervention confound at n≈82 (mitigate: shrinkage + informative
announcement prior); (5) `τ_season` funnel (mitigate: non-centered, HalfNormal, sum-to-zero);
(6) the intervention **edge** is only thinly testable locally (verdict belongs to the port).
