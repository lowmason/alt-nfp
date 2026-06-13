# Port and Extend Plan for the JAX NFP Nowcasting Model

## Status of this document

This plan **supersedes** "Staged Progression to a Bayesian State Space NFP Nowcasting Model.md" as the build plan. That document remains useful as a capability map for the unbuilt layers (its Stages 5–7), but its core premise — building from nothing through eight promotion gates — does not match reality:

**A working PyMC/HMC implementation exists at `~/Projects/alt_nfp` and already covers Stages 0–4.**

| Staged-doc stage | Status in `alt_nfp` | Evidence |
|---|---|---|
| 0 — Vintage harness | **Built** | Hive-partitioned vintage store, two-layer as-of censoring, nowcast backtest (`nfp_models.backtest`), benchmark backtest at T-12/9/6/3/1 |
| 1 — National state-space | **Built** | AR(1) latent with era-specific `mu_g` (pre/post COVID), CES vintage-indexed sigmas, QCEW Student-t anchor |
| 2 — Provider continuing-units | **Built** | Provider G, config-driven loading, AR1 measurement errors, continuing-units split |
| 3 — Structural birth/death | **Built** | `bd_t = φ₀ + φ₁·X^birth + φ₃·X^cycle + σ_bd·ξ` (final form — see Decided Questions) |
| 4 — Representativeness correction | **Substantially built** | 44-cell (4 regions × 11 supersectors) QCEW-weighted compositing, weight redistribution, staleness tracking, pseudo-establishment filtering. Verify: frozen rotating panels |
| 5 — Supersector narrative | Not built | Store has SAE state-level data, but no industry decomposition in the model |
| 6 — Forecasted QCEW + dynamic provider bias | Not built | — |
| 7 — MinT reconciliation + production | Not built | — |

The reboot has three motivations, and they map to two different kinds of work:

1. **JAX / dynamax / GPU** → rewrite of the *model* package only.
2. **Agent-driven rework** → better inputs (old repo as reference, this doc, ported tests), not a blank slate.
3. **Separation of concerns** → already largely exists (five-package uv workspace); two specific seams need fixing, surgically.

**Core decision: this is a port-and-extend, not a rebuild.** Data packages cross by copy. Only the model layer is rewritten. The old repo stays frozen as the reference implementation until parity passes.

---

## Decided questions — do not reopen without evidence

These were settled empirically in attempt #1. Reopening any of them costs weeks. The per-package `CLAUDE.md` files and `specs/`+`archive/` in the old repo are the authoritative record; highlights:

- **Two-layer censoring is required.** Vintage-date-only filtering fails because CES rev-0/1/2 publish days apart (polluted diagonal). Ref-date-only fails on benchmark lookahead and missing frontier revisions. The solution: combined `vintage_date <= D` + `ref_date < D` filtering, then rank-based selection (`_select_ces_at_horizon`: rank 1→rev-0, rank 2→rev-1, rank 3+→rev-2 with `benchmark_revision=0`; `_select_qcew_at_horizon` with quarter-dependent max revision {Q1:4, Q2:3, Q3:2, Q4:1}), with frontier fallbacks and fail-fast validation.
- **CES best-available print.** One observation per month per SA/NSA at the highest available revision. CES vintages are correlated at ρ > 0.99; using all of them overcounts information.
- **Cyclical indicators: claims (ICNSA) and JOLTS openings (JTSJOL) only.** NFCI, business applications, and the lagged QCEW BD proxy (φ₂) were **removed** — posteriors indistinguishable from zero. The staged doc's Stage 3 equation is the pre-pruning form; the surviving form is `bd_t = φ₀ + φ₁·X^birth + φ₃·[claims, jolts] + σ_bd·ξ_t`, with covariate gating when data is all-zero (avoids unidentified parameters in backtest iterations).
- **COVID handling.** Era-specific `mu_g` (break at 2020-01), persistence `phi` and marginal SD shared across eras. 2020–2021 excluded from evaluation. Sample starts 2012.
- **QCEW likelihood.** Student-t (ν=5), two estimated LogNormal base sigmas (M2 vs M3+M1 boundary), revision multipliers from the publication schedule, post-COVID boundary-month era multipliers. LogNormal (not HalfNormal) sigma priors prevent funnel geometry. The M2 prior is deliberately tight to prevent QCEW precision dominance.
- **LOO-CV is a data-quality audit, not model evaluation.** Model evaluation is the vintage-aware backtest, full stop.
- **Units and join conventions.** CES in thousands; QCEW converted persons→thousands at processing. Panel uses day=12 (BLS convention); indicators use day=1; joins are month-truncated.
- **Publication lags.** Provider: 3 weeks. Claims: 1mo, JOLTS: 2mo (per-indicator `_CYCLICAL_PUBLICATION_LAGS`).

---

## Target architecture

Three concerns, physically separated:

```
acquisition   — hit BLS/FRED/provider endpoints, append-only raw vintage archive
                (network, credentials, rate limits; run rarely)
knowability   — pure function: raw archive → "the panel as knowable on date D"
                (no network, no model; deterministic; ruthlessly tested)
inference     — ModelData arrays in, posterior out
                (JAX-land; never sees a vintage_date)
```

Package layout in the new repo (uv workspace, same pattern as old repo):

| Package | Origin | Notes |
|---|---|---|
| `nfp-lookups` | **copy** | Schemas, ProviderConfig, revision schedules, benchmark revisions |
| `nfp-download` | **copy** | BLS/FRED clients, release-date scraper |
| `nfp-ingest` | **copy** | Vintage store, panel, compositing, indicators |
| `nfp-vintages` | **copy** | Pipeline + `alt-nfp` CLI |
| `nfp-model` | **new** | JAX/NumPyro model, sampling, diagnostics, backtests (named `nfp-model-jax` in early drafts; renamed at A3) |

Two known seams in the old layout, fixed in Phase A2 (deliberately *after* golden-master tests exist):

1. **Duplicate download layer.** `nfp-vintages/download/` duplicates `nfp-download`. Consolidate to one acquisition path.
2. **Knowability leaks into the model package.** `panel_adapter.py` (in the old `nfp-model-hmc`) owns half the censoring: provider publication lag, cyclical-indicator masking, best-available CES selection. Move all of it into the data side so **one function answers "what was knowable on date D"** and the model package consumes finished arrays.

**The boundary is an artifact, not a function call.** Phase A2 introduces a serialized `ModelData` snapshot (parquet/npz + content hash) per as-of date. Consequences: the GPU backtest loop never touches the network (pure `vmap` over stored arrays); every run pins to a snapshot hash; the model layer is developed offline against fixtures; failures localize to one side of the boundary.

Also carried over: per-package `CLAUDE.md` files, `specs/` + `archive/` (the written scar tissue), CI config, ruff/black/mypy settings. Left behind: `htmlcov/`, `site/`, `output/`, `archive/` monolith scripts, loose root scripts, `.venv`.

---

## Phase A — Port with parity gates

The staged doc's promotion gates ("beat AR(1)") are obsolete: a working reference implementation is a far stronger gate than a naive baseline. Every Phase A gate is a **parity gate** against `alt_nfp`. No new model features in Phase A — parity is the scope-creep firewall.

### A0 — Repo skeleton and package copy

Copy the four data packages, the rooted test suite for them, lookups data, and agent context (CLAUDE.md files, specs). Wire the uv workspace.

**Gate:** ported test suite green; `uv run alt-nfp build` reproduces the vintage store from raw downloads.

> **Gate status: ✅ PASSED (2026-06-12).** Suite: 361 passed / 1 intentional
> skip. Reproduction run: old repo's frozen raw downloads (323 MB) copied in,
> `alt-nfp process` → all three revision parquets **byte-identical** to the
> reference intermediates (release/vintage-dates identical modulo 4 additive
> run-date-projected future slots); `alt-nfp build --releases <frozen
> releases.parquet>` into a scratch S3 prefix → **770,506 rows, every
> derivable value identical to the reference store**. The reference store has
> 64 additional/different national-headline CES rows (62 extra keys + 2
> values) that exist in **no raw input** — they were live-captured by the old
> repo's monthly `current` runs (release-day vintages, Mar 2025–Jan 2026).
> Conclusion: the ported pipeline reproduces 100% of what is derivable from
> raw downloads; the canonical store's live-captured rows are by nature
> irreproducible, which is the vintage store's purpose. Two operational notes:
> (1) **never rebuild the canonical store in place** — rebuilds go to a
> scratch prefix (now also a hard rule in root CLAUDE.md); (2) BLS now 403s
> the calendar index scrape — the builder degrades gracefully to cached
> release pages (warning, not crash); fresh-page scraping needs a header/
> client fix before the next live `current`/`process` run.

### A1 — Golden-master censoring fixtures

Generate censored panels in the **old** repo for a set of as-of dates chosen to exercise the known edge cases: a January (benchmark month), a current-frontier month, each QCEW quarter-boundary rule, the COVID era break, a month with stale provider data. Commit them as fixtures. The new repo must reproduce them value-identical.

**Gate:** golden masters committed; new-repo panels match for every fixture date. This — not "backtests run without look-ahead bias" — is the real Stage 0 gate.

> **Gate status: ✅ PASSED (2026-06-12).** 9 censored panels + 1 provider
> fixture generated from the old repo (its venv, its local store, read-only)
> at as-of dates covering: COVID break (2020-05-12), mid-sample control,
> all four QCEW quarter rules, the January benchmark print (2025-02-12),
> stale-provider month (2025-11-12; staleness *behavior* gates in A2), and
> the frontier (2026-01-12). The new repo reproduces **every panel
> value-identical** (11/11 tests in
> `packages/nfp-ingest/tests/test_golden_masters.py`) across polars
> 1.38→1.41 and local→S3 store. One deviation: fixtures live in
> `s3://alt-nfp/golden/a1/` rather than git (public repo, proprietary
> provider values) — only the manifest is committed. One finding: the
> originally planned frontier 2026-02-12 is *correctly unbuildable* — the
> 2025 shutdown left Oct/Nov-2025 supersector detail unpublished until the
> 2026-02-16 make-up print, so the fail-fast censoring validator raises;
> this is pinned as a **negative master**. Details:
> `plans/3-golden_masters.md`.

### A2 — Seam fixes and the ModelData snapshot

Consolidate the download layers. Move all knowability logic (panel_adapter censoring, publication lags, best-available selection) into the data side behind a single `model_data(as_of=D)` entry point. Introduce the serialized snapshot artifact and precompute snapshots for the full backtest grid.

**Gate:** golden masters still pass; the model package imports nothing from acquisition; snapshots are hash-stable across regeneration.

> **Gate status: ✅ PASSED (2026-06-12).** Download layer consolidated into
> `nfp_download.bls.bulk`; processing modules renamed (collision-free);
> knowability ported to `nfp_ingest.model_data.build_model_data(as_of=D)`
> with **9/9 array-exact parity** against the old `panel_to_model_data` at
> the A1 dates (fixtures: `s3://alt-nfp/golden/a2/`); hash-pinned snapshots
> (`nfp_ingest.snapshots`, `alt-nfp snapshot`) with build-twice hash
> stability proven. A1 masters green throughout; no acquisition imports in
> the model-data path. **Finding:** the frozen reference has a latent
> indicators-path regression (default-config runs since the settings
> refactor silently dropped claims/jolts — `panel_adapter` resolves
> `indicators_dir` against the model package dir). Masters pin the intended
> behavior; **the A3 parity baseline must use the corrected config** or the
> reference posterior will lack φ₃. Details: `plans/4-a2_seams_snapshots.md`.

### A3 — `nfp-model` parity

Port the model to JAX. Pragmatic sequencing:

1. **Direct NumPyro translation first** (same likelihoods, same priors, NUTS on GPU). This is a mechanical port and the parity target is unambiguous.
2. **Kalman marginalization second**, where structure allows. The linear-Gaussian core (latent AR(1), CES observations, provider loadings) can be marginalized through a Kalman filter — dynamax's filtering primitives inside a NumPyro likelihood — so only static parameters are sampled. Note: the QCEW Student-t breaks exact Gaussianity; either keep QCEW as a sampled-latent branch, or use the scale-mixture-of-normals representation (per-obs auxiliary variance) which preserves conditional Gaussianity. Expect dynamax's packaged model classes to be insufficient for the hierarchical/regression structure — use its filters, not its models.

**Gate:** on identical snapshots, posterior parity with the HMC reference within Monte Carlo error (key params: era `mu_g`, `phi`, sigma hierarchy, `lambda_G`, `alpha_G`, BD path; criterion: |mean difference| small relative to pooled posterior SD and MCSE), plus matched nowcast distributions across a 12-month backtest window. If JAX can't match HMC, that's a bug found cheap; if it can, Stages 0–4 are banked.

> **Gate status: ✅ PASSED (2026-06-12). 14 fixtures, 476/476 criteria.**
> The package landed as **`nfp-model`** (direct NumPyro translation; Kalman
> marginalization deferred to A4-if-needed). Reference baseline: 14 seeded
> nutpie fits with the **corrected indicators config** (per the A2 finding),
> 2 default-preset + the 12-month light-preset window 2025-02 … 2026-01
> (fixtures: `s3://alt-nfp/golden/a3/`, manifest committed). Every sampled
> site, every latent path, and every window nowcast matched within MC error
> — worst |Δnowcast| 32k jobs, inside MCSE bounds; new side sampled with
> **zero divergences in all 14 fits** (ref: 0–4) at ~70% of the reference's
> wall time. One SD-band criterion was recalibrated kurtosis/ESS-aware after
> a reference-side low-ESS excursion (centered-GRW scale params, ESS ≈ 175;
> re-seed evidence in `plans/5-a3_model_parity.md`). The model layer imports
> nothing from the data packages (test-enforced); ModelData dicts and v2
> snapshots are the only interface. **Stages 0–4 are banked.** Details:
> `plans/5-a3_model_parity.md`.

### A4 — Speed: the actual GPU payoff

`vmap`/`pmap` the backtest across as-of snapshots. The GPU's value here is not making one fit faster — it's making the **evaluation harness** cheap enough to run on every change, which transforms the economics of every later gate.

**Gate:** full 24-month vintage-aware backtest in minutes, results identical to the serial run.

> **Gate status: correctness ✅ PASSED; speed scoped to GPU (2026-06-13).**
> The backtest now runs as **one vmapped NUTS program** over the 24-date
> grid (`nfp_model.batch.fit_model_batch`): pad each as-of snapshot to
> common shapes, mask padded likelihood slots (padded latent timesteps are
> prior-only — posterior-invariant, proven by exact log-density equality in
> `test_batch_unit.py`), reduce each date to the A3 fixture schema in-graph.
> **Results identical to the serial run: 24/24 dates, 816/816 parity
> criteria PASS** (the A3 instrument, serial baseline as reference);
> batched-vs-serial nowcast agreement ME −1k / MAE 8k / RMSE 11k against a
> hundreds-of-k scale; **0 divergences in all 24 batched fits**. **Finding:**
> "in minutes" is a **GPU** property — plain `vmap` on **CPU** is only
> **~1.6×** (batched 48.2 min vs serial 75.0 min) because vmapped NUTS
> lock-steps every lane to the deepest tree per iteration (free on parallel
> GPU hardware, pure overhead on CPU). On this CPU-only box we bank the
> correctness gate + a **GPU-ready** harness (identical code runs on GPU
> unmodified) + the lock-step tax quantified; the order-of-magnitude speed
> demonstration awaits GPU access or the host-device sharding lever, neither
> of which blocks A5. The grid build also surfaced that the evaluation
> *actuals* are convention-laden (first-print vs best-available diverge
> >150k on 5/24 months — large revisions, annual benchmarks, store growth
> semantics at revision edges); the report scores dual-track and flags
> splice rows, and **defining the scoring convention is an A5 question**.
> Full analysis: `specs/ces_growth_convention.md` (read-time revision-cohort
> growth vs headline, A5 options A/B/C); A4 view:
> `plans/6-a4_vmap_backtests.md`.

### A5 — Real competitors in the harness

Add the benchmarks that matter to every backtest report: **ADP prints** (FRED; mind the Aug-2022 methodology break) and **consensus survey median** (Bloomberg/Econoday history — sourcing this is a real acquisition task, plan for it). Naive baselines stay as sanity floors, not as gates.

**Gate:** every backtest report scores model vs. ADP vs. consensus vs. naive, at each information regime. This closes the staged doc's biggest omission: it never named the competition.

---

## Phase B — Extend (the genuine frontier)

Stages 5–7 of the staged doc, re-gated against real competitors. Do not start Phase B until the strategic questions below are answered — they determine its ordering.

- **B1 — Supersector narrative layer** (staged-doc Stage 5). CES supersector vintages into the store; supersector latent states; contributions that explain the national number. Gate: narrative stability across vintages + accuracy vs. QCEW sector anchors vs. share-based allocation.
- **B2 — Forecasted QCEW + time-varying provider bias** (Stage 6). Fill the 5–6 month QCEW lag with an explicitly-noisier forecast observation; random-walk provider bias with error-correction pullback toward QCEW.
- **B3 — MinT reconciliation + production hardening** (Stage 7). Reconciled hierarchy, regime-specific uncertainty, fallback rules.

## Strategic questions to answer before Phase B

Carried from the planning session; each changes Phase B's shape:

1. **What is the target: the CES first print, or benchmark-informed truth?** These diverged by 818k in 2024 and have different optimal forecasts (one models BLS's measurement process including its B/D errors; the other models reality). Affects every loss function and the value proposition itself. Cheap hedge: the harness already scores both (nowcast backtest vs. first/second/final prints; benchmark backtest vs. revised truth) — keep dual-track scoring through Phase A and decide before B1.
2. **Who consumes the output?** Trading signal → turning points and the BD layer are the product; B2 leads. Research narrative → the supersector story is the product; B1 leads. Personal research edge → A5's consensus comparison decides everything.
3. **Does the banked model actually beat consensus and ADP, and at which horizons?** A5 answers this empirically with the *existing* model before any Phase B investment. If the answer is no at all horizons, the edge most plausibly lives in benchmark-revision prediction — where consensus doesn't compete — and Phase B should be re-planned around that.

---

## Anti-goals

- **No rewriting the data packages** beyond the two A2 seam fixes. "While I'm porting, let me clean up `nfp-ingest`" is how a three-week port becomes a three-month rewrite.
- **No universal data platform.** ~Five sources; a module per source with a common raw-vintage schema is the whole acquisition layer.
- **No new model features during Phase A.** Parity defines done.
- **No deleting the old repo.** It is the reference implementation, the golden-master generator, and the spec-of-record until A3 passes — and the fallback production model after that.
