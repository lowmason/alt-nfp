# Port and Extend Plan for the JAX NFP Nowcasting Model

## Correction (2026-06-19) — parity is a port-fidelity floor, not correctness

**Read this before the rest of the document.** Everything below was written on the
premise that the frozen PyMC implementation at `~/Projects/alt_nfp` is a correctness
*oracle* — "port the data packages, gate the model against the reference, parity
defines done." That premise is wrong, and the maintainer flagged it as a foundational
flaw:

- **The reference is a work-in-progress with bugs, not validated truth.** It was never
  certified against reality; it was the best available prior attempt.
- **What parity actually bought** (and it was worth buying): it de-risked the
  PyMC→JAX/NumPyro rewrite. A green A1/A2/A3 gate proves the rewrite *reproduces* the
  reference, which isolates *translation* errors from *specification* errors — you can
  debug the port without simultaneously re-litigating the model. That is **necessary
  but not sufficient.**
- **Why it's not sufficient — and is actively dangerous if mistaken for correctness:**
  the goldens are generated *from* the reference, so a reference bug becomes a "correct"
  golden the v2 must match. Worse, a genuine correctness fix *diverges* from the
  reference and therefore *fails* the parity gate. Parity can entrench reference bugs.
- **Decision — validate-first.** Correctness is validated against **external ground
  truth** — published BLS first prints / ALFRED real-time vintages, benchmark identities,
  economic sanity over known episodes — *not* against the reference. Reference-divergence
  is **not** failure: when a correctness fix diverges, **re-baseline the golden** (the
  corrected behavior becomes the baseline). Parity ≠ correctness.

**First result (prong-1, 2026-06-19).** The ported data layer reconstructs the *published*
BLS first print **exactly for 14 of 15** regime-spanning months (2018→2025, incl. COVID)
vs ALFRED real-time PAYEMS — one 0.18% outlier at the Apr-2020 COVID crash (store −20,537k
vs published −20,500k; real, isolated, low-priority). **The data layer is validated against
reality.** And the method demonstrated the blind spot concretely: parity-vs-reference is
*structurally* unable to see a reconstruction-vs-published gap — a green gate would certify
the −20,537 as "correct." The open question is **prong-2**: does the *model* (not the data)
beat naive baselines over a clean window? That is what the A5 backtest now measures.

The rest of this document is the **build history**. It is accurate as a record of what was
done and gated; read every "parity" claim below as **port-fidelity**, not correctness.

---

## Status of this document

This plan **supersedes** "Staged Progression to a Bayesian State Space NFP Nowcasting
Model.md" as the build plan. That document remains a capability map for the unbuilt layers
(its Stages 5–7), but its premise — building from nothing through eight promotion gates —
did not match reality:

**A working PyMC/HMC implementation exists at `~/Projects/alt_nfp` and already covers
Stages 0–4.**

| Staged-doc stage | Status in `alt_nfp` | Evidence |
|---|---|---|
| 0 — Vintage harness | **Built** | Hive-partitioned vintage store, two-layer as-of censoring, nowcast backtest (`nfp_models.backtest`), benchmark backtest at T-12/9/6/3/1 |
| 1 — National state-space | **Built** | AR(1) latent with era-specific `mu_g` (pre/post COVID), CES vintage-indexed sigmas, QCEW Student-t anchor |
| 2 — Provider continuing-units | **Built** | Provider G, config-driven loading, AR1 measurement errors, continuing-units split |
| 3 — Structural birth/death | **Built** | `bd_t = φ₀ + φ₁·X^birth + φ₃·X^cycle + σ_bd·ξ` (final form — see Decided Questions) |
| 4 — Representativeness correction | **Substantially built** | 44-cell (4 regions × 11 supersectors) QCEW-weighted compositing, weight redistribution, staleness tracking, pseudo-establishment filtering |
| 5 — Supersector narrative | Not built | Store has SAE state-level data, but no industry decomposition in the model |
| 6 — Forecasted QCEW + dynamic provider bias | Not built | — |
| 7 — MinT reconciliation + production | Not built | — |

**The port-and-extend sequencing was right; treating the reference as truth was not.**
Rewriting the data plumbing *and* the model engine at once is too many moving parts, so the
plan held the data layer fixed (ported by copy), rewrote only the model, and gated the
rewrite against the known reference. That sequencing is sound. The error (corrected above)
was never scheduling the **correctness-validation** phase behind the port-fidelity gates,
and mistaking green gates for "done." Data packages crossed by copy; only the model layer
was rewritten; the old repo stays frozen as the **port target** — a reference, not an oracle.

---

## Decided questions — do not reopen without evidence

These were settled empirically in attempt #1. Reopening any of them costs weeks. (They are
*modeling/engineering* settlements; none of them is a claim that the reference is correct —
that is the validate-first track.) The per-package `CLAUDE.md` files and `specs/`+`archive/`
are the authoritative record; highlights:

- **Two-layer censoring is required.** Vintage-date-only filtering fails because CES
  rev-0/1/2 publish days apart (polluted diagonal). Ref-date-only fails on benchmark
  lookahead and missing frontier revisions. The solution: combined `vintage_date <= D` +
  `ref_date < D` filtering, then rank-based selection (`_select_ces_at_horizon`: rank
  1→rev-0, rank 2→rev-1, rank 3+→rev-2 with `benchmark_revision=0`;
  `_select_qcew_at_horizon` with quarter-dependent max revision {Q1:4, Q2:3, Q3:2, Q4:1}),
  with frontier fallbacks and fail-fast validation.
- **CES best-available print.** One observation per month per SA/NSA at the highest
  available revision. CES vintages are correlated at ρ > 0.99; using all of them overcounts
  information.
- **Cyclical indicators: claims (ICNSA) and JOLTS openings (JTSJOL) only.** NFCI, business
  applications, and the lagged QCEW BD proxy (φ₂) were **removed** — posteriors
  indistinguishable from zero. The surviving form is `bd_t = φ₀ + φ₁·X^birth + φ₃·[claims,
  jolts] + σ_bd·ξ_t`, with covariate gating when data is all-zero (avoids unidentified
  parameters in backtest iterations).
- **COVID handling.** Era-specific `mu_g` (break at 2020-01), persistence `phi` and marginal
  SD shared across eras. 2020–2021 excluded from *model evaluation* (not from data-layer
  reconstruction checks). Sample starts 2012 in the reference; the **rebuilt store starts
  2017** (see Target architecture).
- **QCEW likelihood.** Student-t (ν=5), two estimated LogNormal base sigmas (M2 vs M3+M1
  boundary), revision multipliers from the publication schedule, post-COVID boundary-month
  era multipliers. LogNormal (not HalfNormal) sigma priors prevent funnel geometry. The M2
  prior is deliberately tight to prevent QCEW precision dominance.
- **LOO-CV is a data-quality audit, not model evaluation.** Model evaluation is the
  vintage-aware backtest, scored against external truth (validate-first).
- **Units and join conventions.** CES in thousands; QCEW converted persons→thousands at
  processing. Panel uses day=12 (BLS convention); indicators use day=1; joins are
  month-truncated. (Note: the day=12 vs day=1 split is a live source of join bugs — it was
  behind the A5 first_print month-key fix.)
- **Publication lags.** Provider: 3 weeks. Claims: 1mo, JOLTS: 2mo (per-indicator
  `_CYCLICAL_PUBLICATION_LAGS`).

---

## Target architecture

Three concerns, physically separated:

```
acquisition   — hit BLS/FRED/provider endpoints, raw vintage archive
                (network, credentials, rate limits; run rarely)
knowability   — pure function: raw archive → "the panel as knowable on date D"
                (no network, no model; deterministic; ruthlessly tested)
inference     — ModelData arrays in, posterior out
                (JAX-land; never sees a vintage_date)
```

Package layout (uv workspace, same pattern as old repo):

| Package | Origin | Notes |
|---|---|---|
| `nfp-lookups` | **copy** | Schemas, ProviderConfig, revision schedules, benchmark revisions |
| `nfp-download` | **copy** | BLS/FRED clients, release-date scraper |
| `nfp-ingest` | **copy** | Vintage store, panel, compositing, indicators, `model_data` |
| `nfp-vintages` | **copy** | Pipeline + `alt-nfp` CLI |
| `nfp-model` | **new** | JAX/NumPyro model, sampling, batch, nowcast, parity (named `nfp-model-jax` in early drafts; landed as `nfp-model` at A3) |

Two seams in the old layout were fixed in Phase A2 (deliberately *after* golden-master tests
exist):

1. **Duplicate download layer.** `nfp-vintages/download/` duplicated `nfp-download`.
   Consolidated to one acquisition path (`nfp_download.bls.bulk`).
2. **Knowability leaked into the model package.** `panel_adapter.py` (old `nfp-model-hmc`)
   owned half the censoring. All of it moved into the data side so **one function answers
   "what was knowable on date D"** (`nfp_ingest.model_data.build_model_data(as_of=D)`) and
   the model package consumes finished arrays.

**The boundary is an artifact, not a function call.** A2 introduced a serialized `ModelData`
snapshot (`.npz` + content hash) per as-of date. Consequences: the GPU backtest loop never
touches the network; every run pins to a snapshot hash; the model layer is developed offline
against fixtures; failures localize to one side of the boundary.

**The vintage store is rebuilt and replaceable (not append-only/irreplaceable).** The
canonical `s3://alt-nfp/store` now holds the **rebuilt** schema — reconstructable public
CES/QCEW, 2017+, normalized — promoted from `…/store-rebuild` on 2026-06-18 (plans/10 T8;
prior canonical preserved at `…/store-prev-20260618`). The earlier "live-captured rows that
exist in no raw input / irreplaceable" framing is **retired** — it traced to a broken
`_fetch_ces_releases`, not true live-capture. The store is replaceable public data; the only
surviving rule is operational safety: **never `alt-nfp build` straight to `…/store`** —
rebuild to a scratch prefix, then promote deliberately (snapshot prior canonical first;
copy-then-delete per partition, since filenames encode vintage ranges). `is_canonical_store`
guards `build_store`/`mirror_store`.

---

## Phase A — Port with **port-fidelity** gates

The staged doc's promotion gates ("beat AR(1)") are obsolete: reproducing a working
reference is a far stronger *port* gate than a naive baseline. Every Phase A gate is a
**port-fidelity parity gate** against `alt_nfp` — it proves the rewrite reproduced the
reference. **It does not certify correctness** (that's the validate-first track below). No
new model features in Phase A — port-fidelity is the scope-creep firewall for the *port*.

### A0 — Repo skeleton and package copy

Copy the four data packages, their rooted test suite, lookups data, and agent context
(CLAUDE.md files, specs). Wire the uv workspace.

**Gate:** ported test suite green; `uv run alt-nfp build` reproduces the vintage store from
raw downloads.

> **Gate status: ✅ PASSED (2026-06-12).** Suite: 361 passed / 1 intentional skip.
> Reproduction run: old repo's frozen raw downloads (323 MB) copied in, `alt-nfp process` →
> all three revision parquets **byte-identical** to the reference intermediates;
> `alt-nfp build --releases <frozen releases.parquet>` into a scratch S3 prefix →
> **770,506 rows, every derivable value identical to the reference store**. The reference
> store had 64 additional/different national-headline CES rows that exist in no raw input.
>
> **[Superseded 2026-06-18.]** The then-conclusion ("those rows are live-captured and
> irreproducible — the store is irreplaceable, never rebuild it") was **overstated**. The
> missing rows traced to a **broken `_fetch_ces_releases`** (broken since initial), not true
> live-capture. The store was subsequently **rebuilt** from reconstructable public CES/QCEW
> (2017+) and **promoted to canonical** (plans/10–12, 2026-06-18). The store is
> **replaceable**; "never rebuild in place" survives only as an operational safety guard
> (rebuild to scratch, promote deliberately). One other A0 note still stands: BLS 403s the
> calendar index scrape — the builder degrades to cached release pages (warning, not crash).

### A1 — Golden-master censoring fixtures

Generate censored panels in the **old** repo for as-of dates exercising the known edge cases
(a January benchmark month, a current frontier, each QCEW quarter rule, the COVID break, a
stale-provider month). Commit as fixtures. The new repo must reproduce them value-identical.

**Gate:** golden masters committed; new-repo panels match for every fixture date.

> **Gate status: ✅ PASSED (2026-06-12).** 9 censored panels + 1 provider fixture from the
> old repo (read-only) at as-of dates covering the COVID break, mid-sample control, all four
> QCEW quarter rules, the January benchmark print, a stale-provider month, and the frontier.
> The new repo reproduces **every panel value-identical** (11/11 tests in
> `test_golden_masters.py`) across polars 1.38→1.41 and local→S3. Fixtures live in
> `s3://alt-nfp/golden/a1/` (public repo, proprietary provider values — only the manifest is
> committed). The originally planned 2026-02-12 frontier is *correctly unbuildable* (the 2025
> shutdown left Oct/Nov-2025 supersector detail unpublished) — pinned as a **negative
> master**. (A1 goldens were later re-baselined for the rebuilt store, plans/12.) Details:
> `plans/3-golden_masters.md`.

### A2 — Seam fixes and the ModelData snapshot

Consolidate the download layers. Move all knowability logic into the data side behind a
single `build_model_data(as_of=D)` entry point. Introduce the serialized snapshot and
precompute the backtest grid.

**Gate:** golden masters still pass; the model package imports nothing from acquisition;
snapshots are hash-stable across regeneration.

> **Gate status: ✅ PASSED (2026-06-12).** Download layer consolidated into
> `nfp_download.bls.bulk`; knowability ported to
> `nfp_ingest.model_data.build_model_data(as_of=D)` with **9/9 array-exact parity** against
> the old `panel_to_model_data` (fixtures: `s3://alt-nfp/golden/a2/`); hash-pinned snapshots
> with build-twice stability proven. No acquisition imports in the model-data path.
> **Finding:** the frozen reference has a latent indicators-path regression (default-config
> runs since the settings refactor silently dropped claims/jolts — `panel_adapter` resolves
> `indicators_dir` against the model package dir). Masters pin the *intended* behavior; the
> A3 baseline uses the **corrected** config. *(This is itself an instance of the correction
> above — a reference bug the port had to consciously fix, not reproduce.)* Details:
> `plans/4-a2_seams_snapshots.md`.

### A3 — `nfp-model` port

Pragmatic sequencing:

1. **Direct NumPyro translation** (same likelihoods, same priors, NUTS). A mechanical port;
   the port-fidelity target is unambiguous. **This is what shipped.**
2. ~~Kalman marginalization second~~ — **never pursued, and not in the v2 path.** An early
   idea was to marginalize the linear-Gaussian core (latent AR(1), CES observations, provider
   loadings) via a Kalman filter — dynamax filtering primitives inside a NumPyro likelihood —
   to sample only static parameters. It was never built. **Confirmed 2026-06-19:** there is
   no dynamax anywhere (deps are jax/numpyro/numpy; `dynamax` survives only as a dropped
   `pyproject` keyword), and the AR(1) latent is a hand-rolled `jax.lax.scan`. The QCEW
   Student-t breaks exact Gaussianity anyway, so exact marginalization never cleanly applied.
   If revisited, it is a pure speedup gated on its own baseline — not a correctness lever.

**Gate:** on identical snapshots, posterior parity with the HMC reference within Monte Carlo
error (era `mu_g`, `phi`, sigma hierarchy, `lambda_G`, `alpha_G`, BD path; |mean diff| small
vs pooled posterior SD and MCSE), plus matched nowcast distributions across a 12-month window.

> **Gate status: ✅ PASSED (2026-06-12). 14 fixtures, 476/476 criteria.** Landed as
> **`nfp-model`** (direct NumPyro translation; the Kalman idea was never pursued — see above).
> Reference baseline: 14 seeded nutpie fits with the **corrected indicators config**, 2
> default-preset + the 12-month light-preset window 2025-02 … 2026-01 (fixtures:
> `s3://alt-nfp/golden/a3/`). Every sampled site, latent path, and window nowcast matched
> within MC error — worst |Δnowcast| 32k jobs, inside MCSE bounds; **0 divergences in all 14
> fits** (ref: 0–4) at ~70% of the reference's wall time. One SD-band criterion recalibrated
> kurtosis/ESS-aware after a reference-side low-ESS excursion. The model layer imports nothing
> from the data packages (test-enforced). **Stages 0–4 are banked *as a port*** (faithful to
> the reference — correctness is the validate-first track). Details:
> `plans/5-a3_model_parity.md`.

### A4 — Speed: the GPU payoff

`vmap`/`pmap` the backtest across as-of snapshots. The GPU's value is not one faster fit —
it's making the **evaluation harness** cheap enough to run on every change.

**Gate:** full 24-month vintage-aware backtest in minutes, results identical to serial.

> **Gate status: correctness ✅ PASSED; speed scoped to GPU (2026-06-13).** The backtest
> runs as **one vmapped NUTS program** over the 24-date grid
> (`nfp_model.batch.fit_model_batch`): pad each snapshot to common shapes, mask padded
> likelihood slots (padded latent timesteps are prior-only — posterior-invariant, proven by
> exact log-density equality in `test_batch_unit.py`), reduce each date in-graph. **Results
> identical to serial: 24/24 dates, 816/816 criteria PASS**; **0 divergences in all 24 batched
> fits**. **Finding:** "in minutes" is a **GPU** property — plain `vmap` on CPU is only ~1.6×
> (vmapped NUTS lock-steps every lane to the deepest tree per iteration: free on GPU, overhead
> on CPU). We bank the correctness gate + a GPU-ready harness. The grid build also surfaced
> that evaluation *actuals* are convention-laden (first-print vs best-available diverge >150k
> on 5/24 months) — **the scoring convention is an A5 question** (`specs/ces_growth_convention.md`).
> A4 view: `plans/6-a4_vmap_backtests.md`.

### A5 — Real competitors, and the first reality gate

Add the benchmarks that matter to every backtest report. Naive baselines stay as sanity
floors, not gates. **A5 is the first gate that scores against external reality (the actual
first print) — i.e., the start of the validate-first track for the *model*.**

**Gate:** every backtest report scores model vs. consensus vs. naive, at each information
regime (first print = target).

> **Refinement (2026-06-13, `specs/a5_real_competitors.md`):** **ADP is dropped** — post-Aug-2022
> it publicly disclaims forecasting the BLS print, so it is not a fair first-print competitor.
> The competition is **consensus** (Bloomberg, T−1, staged) + naive floors (+ an optional smart
> bridge baseline). Information regimes: **T−7 and T−1** (BLS-release(M) − 7 / − 1 days), where
> payroll-provider inputs are highest-quality. Target = first print (SQ1).
>
> **Update (2026-06-19) — harness fixed + first reality results.** The harness is built and
> end-to-end-validated on branch `a5-rebuilt-integration`. Three eval-side bugs fixed
> (first_print drift+shutdown, month-join, consensus month-key). One model-column blocker —
> **every batched nowcast was NaN** — was diagnosed **not** as a model bug but as a **harness
> provenance bug**: `base_index = float(ces_sa_index[0])` is NaN on the rebuilt store (the
> 2017-01 panel start has no growth predecessor → `cum_level` leaves `ces_sa_index[0]=NaN`),
> which scaled the whole index path to NaN. **The model marginalizes missing months correctly**
> — A4, with a finite anchor, produces finite nowcasts on the *same* shutdown months. Fixed via
> `levels_provenance()` (first-finite anchor + `nanargmin`); committed. **prong-1 of
> validate-first passed** (data layer reconstructs published first prints 14/15 exact vs ALFRED;
> see top of this doc). **Open — prong-2:** run A5 over a clean window and ask *does the model
> beat naive baselines?* The single shutdown-frontier month scored so far (Jan-2026: −309k vs
> +130k actual, losing to a trailing-mean) is a frontier artifact on the light preset — the
> clean-window run is what decides it.

---

## Phase B — Extend (the genuine frontier)

Stages 5–7 of the staged doc, re-gated against real competitors. **(SQ1 + SQ2 answered
2026-06-13 → B1 leads: the product is an accurate first-print nowcast for Bloomberg plus a
supersector "why" narrative; SQ3 — does the model beat consensus — is the prong-2 question A5
measures.)**

- **B1 — Supersector narrative layer** (Stage 5). CES supersector vintages into the store;
  supersector latent states; contributions that explain the national number. Gate: narrative
  stability across vintages + accuracy vs. QCEW sector anchors vs. share-based allocation.
- **B2 — Forecasted QCEW + time-varying provider bias** (Stage 6). Fill the 5–6 month QCEW
  lag with an explicitly-noisier forecast observation; random-walk provider bias with
  error-correction pullback toward QCEW.
- **B3 — MinT reconciliation + production hardening** (Stage 7). Reconciled hierarchy,
  regime-specific uncertainty, fallback rules.

## Strategic questions

1. **Target: CES first print, or benchmark-informed truth?**
   > **Answered (2026-06-13): the first print.** A5 scores against the within-release headline
   > BLS announces (`specs/ces_growth_convention.md` Option A). Benchmark/revised-truth targeting
   > is a separate later model. Revised truth may be shown as an *unscored* reference.
2. **Who consumes the output?**
   > **Answered (2026-06-13): the research-narrative consumer.** Output is (1) the most accurate
   > NFP **first-print** nowcast (for Bloomberg publication) and (2) a **narrative** of *why* —
   > the national change decomposed into supersector contributions. So **B1 leads Phase B**. A5
   > stays national first-print scoring; supersector scoring is B1's extension.
3. **Does the model actually beat consensus/naive, and at which horizons?**
   > **This is prong-2 of validate-first** (above) — A5 answers it empirically with the existing
   > model before any Phase B investment. If the answer is "no at all horizons," the edge most
   > plausibly lives in benchmark-revision prediction (where consensus doesn't compete), and
   > Phase B should be re-planned around that.

---

## Validate-first — the current chapter (decided 2026-06-19)

The work going forward is **validate-first** (the fork chosen over "targeted fix-as-found" and
"full rebuild"): quantify where the ported mechanics are actually wrong against external ground
truth, then let the evidence pick targeted-fix vs rebuild per component.

- **Prong-1 — data layer vs published BLS.** ✅ Spot-check passed (2026-06-19): store
  first-print reconstruction = ALFRED real-time PAYEMS, 14/15 exact, one 0.18% COVID outlier.
  *Optional next:* promote the throwaway `scripts/_validate_alfred.py` into a committed 2017+
  ground-truth gate (move the ALFRED fetch into `nfp_download.fred` with realtime params + a
  test); investigate the Apr-2020 37k (likely the `first_print` partner selection / SA handling
  at the COVID extreme — low priority).
- **Prong-2 — model vs reality.** ⏳ Open and the high-value step. Run A5 over a **clean window**
  (ex-COVID, ex-shutdown) and score the nowcast vs first prints vs naive baselines. The
  `base_index` fix unblocked this. This is where the real signal is — the data layer looks sound;
  the model is the open question.
- **Methodology going forward.** Parity is a port-fidelity floor; correctness is validated
  against external truth; a correctness fix that diverges from the reference is expected, and
  re-baselines the golden rather than failing.

---

## Anti-goals

- **No rewriting the data packages** beyond the A2 seam fixes *and* validate-first-driven
  correctness fixes. "While I'm porting, let me clean up `nfp-ingest`" is how a port becomes a
  rewrite — but a *ground-truth-validated* bug fix is not scope creep, it's the point.
- **No universal data platform.** ~Five sources; a module per source with a common raw-vintage
  schema is the whole acquisition layer.
- **No new model *features* during the port.** Port-fidelity defined "ported"; it does **not**
  define "correct" — that's the validate-first track, where deliberate, ground-truth-validated
  divergence from the reference is allowed (and re-baselines the golden).
- **Don't delete the old repo** — it's the frozen **port target** and fixture generator. But it
  is **not** validated truth and **not** an assumed fallback production model: it's a buggy WIP.
  Treat it as a reference, never an oracle.
