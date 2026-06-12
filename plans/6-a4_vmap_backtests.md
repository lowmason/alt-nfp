# Implementation Plan: A4 — vmapped backtests

Phase A4 of `plans/0-port_and_staged_plan.md`: batch the vintage-aware
nowcast backtest across as-of snapshots with `vmap`, so the full evaluation
harness runs in minutes instead of an hour — cheap enough to run on every
model change.

**Gate (from the plan of record):** full 24-month vintage-aware backtest in
minutes, results identical to the serial run.

"Identical to the serial run" is read as the A3 standard: the serial
baseline is 24 independent `fit_model` runs (the A3-proven path); the
batched run must match it within Monte Carlo error under the calibrated
criteria in `nfp_model.parity` (scalar z/SD tests, path bands, nowcast
draws + change_k bounds). Bit-identity across different computation graphs
is not a meaningful target; statistical parity under a different RNG stream
is the same evidence standard A3 used against the PyMC reference.

## Why padding + masking (the design crux)

`vmap` requires every batch element to share shapes, but each as-of date
has its own T (calendar length), observation counts, and vintage-bucket
count. The batch is built by padding to the per-key maximum and masking
padded likelihood terms out:

- **Likelihoods**: every obs site gets an optional boolean mask
  (`numpyro.handlers.mask`); padded slots contribute exactly zero log-prob.
  Padded obs values are finite (0.0) so gradients stay finite; padded
  indices point at t=0 (any valid slot — masked anyway).
- **Latents**: padded timesteps still sample their non-centered z's, which
  are prior-only N(0,1) dimensions touching no likelihood — the joint over
  everything observed is unchanged, so the posterior over all shared
  parameters and the real-T latent path is *exactly* the serial posterior.
  This is provable in closed form and tested via log-density equality, not
  just MCMC agreement.
- **Static structure is genuinely shared**: every date starts at 2012-01
  with a contiguous monthly calendar, so `month_of_year`, `year_of_obs`,
  `era_idx` (break 2020-01) agree on the overlap and extend to T_max —
  they're closed over as concrete numpy, not traced. Same for the provider
  list/error models and the cyclical-gating decision (asserted uniform
  across the batch; mixed presence would change `phi_3`'s dimension and is
  a build error, not a maskable condition).
- **AR(1) providers pad at the end**: the conditional mean uses the
  previous *real* observation for every real slot; padded tail terms are
  masked. End-padding keeps that exact (no real obs ever conditions on a
  padded one).

## Workstreams

### 6.1 Trace-safe model with optional masks (`model.py`)

Under `vmap` the per-date arrays are tracers, so the model must consume
them with `jnp` (the current code bakes them in via `np.asarray`, which
raises on tracers). Changes, all backwards compatible:

- obs values / index arrays / noise multipliers / cyclical arrays →
  `jnp.asarray`; gathers instead of numpy fancy indexing on constants.
- optional `qcew_mask` / `ces_sa_mask` / `ces_nsa_mask` / per-provider
  `pp_mask` keys; absent ⇒ today's exact behavior.
- optional `cyclical_active` static tuple overriding the `np.any(arr != 0)`
  gating (which can't run on tracers); absent ⇒ today's data-driven gating.

Regression evidence that the unbatched path is unchanged: the fast suite,
the MCMC smoke tests, and the A3 golden spot check
(`NFP_A3_PARITY=1 pytest …/test_parity_golden.py`) re-run green after the
refactor.

### 6.2 `nfp_model.batch` — padding and the vmapped fit

- `pad_model_inputs(inputs: list[dict]) -> BatchedInputs`: splits into a
  static dict (closed over) and a batched dict (leading date axis: padded
  values, masks, `c_idx`, `T_real`). Asserts uniform cyclical gating,
  uniform provider sets, no empty providers.
- `fit_model_batch(batched, settings, seed)`: `jax.vmap` over a function
  that builds `MCMC(NUTS(nfp_model), …, chain_method="vectorized",
  progress_bar=False)` and runs it on one date's slice. NumPyro's
  `MCMC.run` is traceable when the progress bar is off (the whole run is a
  `lax.scan`), which is what makes vmap-of-MCMC possible. Per-date keys
  from one split; warmup adaptation stays per-date/per-chain.
- **In-graph reduction**: the vmapped inner returns the A3 fixture schema
  (scalar/small-vector draws, path mean/SD over real T, nowcast predictive
  draws at `c_idx`, divergence count) instead of raw samples, so the
  z-arrays (T-length non-centered noise × draws × dates) never materialize
  batch-wide. Deterministics recomputed in-trace from the draws (the
  `Predictive` mechanics), nowcast arithmetic in `jnp` mirroring
  `nowcast.py`.
- Device strategy on CPU: plain `vmap` first (one XLA program, intra-op
  threading). If lock-step tree-doubling across 24×2 lanes eats the win,
  fall back to sharding the batch across forced host devices
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`, N ≈ performance
  cores) — measured on a pilot before the full run. The same code runs
  unmodified on a real GPU later.

### 6.3 Tests (synthetic, no store)

- `test_batch_unit.py`: padded+masked log-density equals the unpadded
  model's on every likelihood site (exact, via `substitute`/`trace` with
  shared params — the real proof that padding changes nothing); mask/shape
  bookkeeping; mixed-gating assertion fires; `c_idx`/`T_real` mapping.
- `test_batch_smoke.py` (`slow`): 2–3 synthetic dates, tiny preset:
  vmapped batch vs per-date `fit_model`, posterior means/SDs within
  tolerance; batch seed reproducibility.

### 6.4 `scripts/run_a4_backtest.py` — the 24-month run

Subcommands (all restartable):

- `snapshot` — build the as-of grid (last 24 months of the uncensored
  panel, day-12 convention, `end_year=2026`) via
  `nfp_ingest.snapshots.snapshot_model_data` into
  `data/backtests/snapshots/`, plus one uncensored actuals bundle
  (`g_ces_sa`, `ces_sa_index`, `base_index`, `idx_to_level`, dates) — the
  same truth convention as the reference backtest (best-available
  revision, not first print; first-print scoring is A5's dual-track
  concern). Unbuildable dates recorded, not fatal.
- `serial` — the baseline: per-date `fit_model` (light preset) on the
  snapshot's `from_snapshot` data, reduced via `collect_parity_arrays`,
  one npz per date. ~1 h wall; runs in the background.
- `batched` — `pad_model_inputs` over all snapshots → `fit_model_batch`
  → per-date npz in the same schema + wall-time record.
- `compare` — per date, `nfp_model.parity.compare_reduced`
  (serial as "ref", batched as "new") + the backtest results table
  (actual vs nowcast change_k, MAE/RMSE — serial and batched side by
  side) → `data/backtests/a4_report.md` + parquet. Exit 1 on any parity
  failure.

### 6.5 Gate + docs

Gate annotation in plans/0 quoting wall times and the parity verdict;
status block here; `nfp-model` CLAUDE.md gains the batch module and the
backtest commands; memory updated.

## Sequencing

6.2's risk (does vmap-of-MCMC trace at all?) is retired *first* with a
tiny synthetic prototype, before any refactor. Then 6.1 → 6.2 → 6.3 (fast
suite + A3 spot check green) → 6.4 snapshot grid → serial baseline in the
background while batched lands → compare → 6.5.

## Risks

- **vmapped NUTS lock-step**: every lane pays the max tree depth per step.
  Mitigation: measured pilot (4 dates) before the full run; host-device
  sharding fallback; worst case the gate is still met by sharded serial
  programs (the harness API doesn't change either way).
- **Memory**: reduced in-graph output keeps the batch footprint ~1 GB at
  light preset × 24 dates (36 GB available). Raw-sample collection is the
  thing to avoid, and is avoided by design.
- **2024 as-of buildability**: A3 proved 2025-02 … 2026-01; the 2024 grid
  half is new territory for `build_model_data`. The snapshot step
  surfaces failures cheaply (A1's negative-master pattern if any date is
  legitimately unbuildable).

## Finding: evaluation actuals are convention-laden (dual-track scoring)

The grid build surfaced that the two "actual" conventions disagree by
>150k on 5 of 24 window months, for three distinct reasons:

1. **Genuine large revisions** — e.g. the 2025-08-01 release's May/June
   2025 downward revision (best-available 2025-06 = −13k vs first print
   +164k under the store's basis).
2. **Annual benchmarks** — the 2024 benchmark lands *in* the 2025-01
   first print (rev-0 −469k vs best +112k); the 2025 benchmark (−911k
   preliminary) enters via the **2026-02 vintages now in the store** (the
   live capture is ongoing) and splices best-available growth at
   2025-10 (−1,194k) and 2026-01 (−900k).
3. **Store growth semantics at revision edges** — rev-0 growth rows are
   differenced against the *prior vintage's* previous-month level, so in
   big-revision months they sit off the same-day headline (2025-07 rev-0
   −186k vs the +73k headline: exactly the −258k May/June revision).

A4's report therefore scores against **both** conventions (plans/0
strategic question 1 endorses dual-track scoring through Phase A), flags
†-rows, and excludes them from best-available metrics. Defining the
scoring convention — and whether headline-convention first-print changes
need a within-release level difference in the data layer — is **A5's
evaluation question**, not A4's. None of this touches the A4 gate (the
gate is batched-vs-serial parity and wall time).
