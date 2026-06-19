# A5 — Real competitors in the harness (design)

Status: **design, revised 2026-06-19** (retargeted to **private** NFP). Implements
the A5 gate in `plans/0-port_and_staged_plan.md` ("Real competitors in the
harness") and resolves the evaluation-convention question left open by
`specs/ces_growth_convention.md` and the A4 dual-track finding
(`plans/6-a4_vmap_backtests.md`) — by choosing **Option A (the first
print)** as the first-print target over revised-CES targeting (plans/0 SQ1,
answered 2026-06-13). The private retarget (2026-06-19) adds a second scored
truth on a distinct axis — the **private QCEW-settled** administrative value,
made the **primary** comparison (§2c) — which SQ1 did not speak to. The
**model-side counterpart** (and the source of the private retarget) is
`specs/model_improvements.md`. Companion spec for the Bloomberg consensus source —
now a **Track B** (deferred Total-NFP) artifact, see below — is
`specs/bloomberg_consensus.md`.

**Private retarget (2026-06-19).** A5 scores the **private** nowcast
(`industry_code='05'`), not total nonfarm (`'00'`). The model's signal is
inherently private — it is QCEW-anchored (private in this store) and its richest
inputs are private payroll providers — so it must be scored against the **private**
first print and the **private** QCEW-settled truth. The run-path's `'00'` default
is a latent mismatch corrected in `model_improvements.md`; `'00'` total data
exists only for a future, unbuilt **Track B** ("private nowcast + government
forecast = total"). On the private track the **only** competitors are the **naive
floors**: ADP is removed entirely (competitor *and* regressor), and **consensus is
removed** — it is a Total object with no meaning against the private nowcast alone,
so it moves to the deferred Track B (which also needs a new government forecast).

## TL;DR

1. A5 is **evaluation-side only**. It adds *targets* (what we score against)
   and *competitors* (what we compare the model to) plus a scoreboard
   report. It touches **no** `nfp-model` code, **no** `transform_to_panel`,
   **no** `build_model_data`, and **no** A1/A2/A3 golden-mastered path. The
   model already emits nowcasts through the A4 batched harness; A5 wraps
   scoring around them. The parity firewall stays intact. Threading
   `industry_code='05'` into the data-layer reads that *feed* the model
   (`first_print_changes`, `panel_to_model_data`, the snapshot target) is a
   data-layer change, not a model change — the firewall holds.
2. **Targets: two PRIVATE truths** (`industry_code='05'`). The **private
   QCEW-settled** value is the **primary** comparison (§2c) — the model is
   QCEW-anchored, so the administrative private number is the closest truth it can
   be held to. The **private first print** is the second scored truth (plans/0
   SQ1, **answered 2026-06-13** — Option A, now read on `'05'`): the within-release
   private headline change BLS announces, an additive extractor over store levels.
   Both are scored at each regime.
   Benchmark-informed / "revised-truth" targeting is **deferred** to a separate
   later model. A `final_view` private revised-truth column may appear as an
   *unscored* reference (to show revision magnitude) but is **not** an A5 target.
3. **Two near-release regimes.** The model is scored at **T−7** and **T−1**
   (BLS-release(M) − 7 and − 1 days), both new grids. These sit in the window
   where the payroll-provider inputs are near-complete — provider series
   quality rises as more of month M's payrolls are processed, so day-12 of M
   (the A4 grid) starves the model of its best input. The naive floors run at
   both; the **T−7 → T−1 gap quantifies the value of the final week of provider
   and claims information** before the print. (Consensus, formerly the T−1
   head-to-head, is a Track-B Total object and no longer appears here.)
4. **Competitor set (private track) — naive floors only.** model · naive floors
   (random-walk, trailing-12-month mean), scored against the private first print
   and the **primary** private QCEW-settled truth. **ADP is removed entirely**
   (competitor *and* regressor). **Consensus is removed** from the private track
   — a Total object, deferred to Track B. The former vintage-censored
   claims/JOLTS **bridge baseline** is also out as a competitor ("naive floors
   only"); the same regression survives as the Tier-1 **Aruoba diagnostic** in
   `model_improvements.md` §4 — see §4 here.

## 1. Scope and non-goals

**In scope.** New evaluation-side library code (private targets + naive-floor
competitor adapters), two new near-release snapshot grids (T−7, T−1) on the
**private** target, and an `a5_report.md` scoreboard. All series are
`industry_code='05'` (private), threaded through the data-layer reads that feed
the model.

**Explicit non-goals (the firewall).**

- No change to `nfp-model`. The model's nowcast is an input to A5, not a
  subject of it.
- No change to `transform_to_panel`, `build_model_data`, the vintage store,
  or any golden-mastered selection/growth path. The first-print target is an
  **additive** read over store levels (per `ces_growth_convention.md` §5
  Option A) — it adds a new derived series, it does not alter the `growth`
  column A1/A2 pin. Passing `industry_code='05'` to the data-layer reads
  (`first_print_changes`, `panel_to_model_data`, `read_vintage_store`) selects
  the private series — it is the *data* fed to the model, not model code, and
  these functions already accept the parameter (default `'00'`).
- No new model features (Phase A's parity-is-done rule still holds; competitor
  comparison is measurement, not modeling).
- The **store is append-only and read-only** for A5 (root CLAUDE.md hard
  rule). A5 reads levels; it never writes.

## 2. Targets — the private first print and the private QCEW-settled truth

A5 scores against the **private** (`industry_code='05'`) actuals. Two truths are
scored: the **private QCEW-settled** value — the **primary** comparison, the
administrative number the QCEW-anchored model is closest to (§2c) — and the
**private first print**, the within-release headline change BLS announces (§2a).
plans/0 strategic question 1 is **answered (2026-06-13): Option A, the first
print** (now read on `'05'`). Benchmark-informed ("revised-truth") targeting is a
separate, later model's concern and is out of A5 scope.

### 2a. Private first print (Option A) — a scored target

The within-release **private** headline change BLS announces on release day:
`headline(p, R) = log L(p at R) − log L(p−1 at R)`, both private (`'05'`) levels
from the same release. Built by the existing extractor reading store **levels**
under the structural pairing rule from `ces_growth_convention.md` §5 (the pairing
mechanics are industry-agnostic; only the series read changes):

- partner of `(p, rev r, bmr 0)` is `(p−1, rev r+1, bmr 0)` for r ∈ {0, 1};
- at benchmark months, fall back to `p−1`'s latest `(rev 2, bmr ≥ 1)` row;
- **never** join on `vintage_date` equality (historical stamps misalign);
- for r = 2 outside benchmark season headline == cohort by construction.

On the rebuilt private store, coverage is **106/108 rev-0 months** (2017-01 →
2026-01; the two nulls are the 2017-01 history-start edge and the 2025-11
Oct-2025-shutdown ref-date hole — see memory `ces-oct2025-shutdown`). Verified
2026-06-19 against published **private** headlines: **+83k** for 2025-07, **+111k**
for 2025-01, **+172k** for 2026-01. (The earlier total-nonfarm figures
271/273-month coverage, a 2003 history edge, and +73k/+143k/+130k were `'00'`
facts and do not apply to the private series.)

**Home:** existing `nfp_ingest/first_print.py` (alongside `model_data.py`).
`first_print_changes(industry_code='05')` returns per-period
`(ref_date, first_print_growth, first_print_change_k, vintage_date)`. Unit-tested
against the private headlines above. Zero contact with pinned paths.

### 2b. Revised truth — unscored reference only (deferred)

The diff of one internally coherent latest-published **private** level path
(`nfp_vintages.views.final_view` semantics on `'05'`) is basis-consistent by
construction and remains trivially available. It MAY be rendered as an *unscored*
reference column so the magnitude of subsequent revision is visible beside the
scores — but it is **not** an A5 target, it gates nothing, and the model is not
scored on it. **NB:** this `final_view` revised truth (latest *published* CES) is
distinct from the §2c **QCEW-settled** truth (the administrative anchor); the
QCEW-settled value is a *scored* primary target, the `final_view` diff is only an
optional reference. Full revised-truth evaluation belongs to the future
benchmark-targeting model (plans/0 SQ1, resolved).

**Home (only if the reference column is wanted):** `nfp_vintages.evaluation`
gains a small `revised_truth_change_k(panel)` helper on the private `final_view`.
Defer if not needed.

### 2c. Private QCEW-settled truth — the primary scored target

The model's latent is pinned to **private QCEW** (`industry_code='05'`,
`model.py` Student-t anchor), so the **QCEW-settled private value** is the closest
administrative truth it can be held to and is the **primary** scoreboard
comparison (per `model_improvements.md` §3). The model is scored against this
QCEW-settled private number *and* the private first print at each regime; the
QCEW-settled column is the **primary** comparison (the closest administrative
truth to the QCEW-anchored model). Naive floors are scored against it identically.
**Basis caveat (open item, §8):** this store's QCEW `'05'` is **NSA-only** while
the CES first print and the model nowcast are **SA** — the QCEW-settled scoring
must reconcile that basis (via the model's existing QCEW-anchor basis handling, or
an explicit SA/NSA bridge) before the comparison is apples-to-apples; neither this
spec nor `model_improvements.md` §3 has pinned the mechanism. Home: the scoring
path in `run_a5_backtest.py` reads the QCEW-settled private series alongside
`first_print_changes('05')`; no pinned path is touched.

## 3. Information regimes — the harness shape

The model is scored at **two near-release horizons**: **T−7**
(BLS-release(M) − 7 days) and **T−1** (BLS-release(M) − 1 day, release-eve).
Both sit in the window where the model's inputs are at their best, and the
one-week gap isolates the value of the final week of information before the
print.

**Why not day-12 of M (the A4 grid)?** The model's signal is dominated by the
payroll-provider series, whose quality is a function of *how many of month
M's payrolls have been processed* when the model runs. At day-12 of M the
reference month is barely underway and the provider series for M is thin; by
T−7 / T−1 (early M+1) M's pay periods are essentially complete and the as-of
censoring automatically surfaces the higher-quality, later provider vintages.
Scoring at the near-release horizons therefore measures the model on the
information set it is actually designed to use — not an artificially starved
one.

| Regime | As-of date | Competitors present | Measures |
|---|---|---|---|
| **T−7** | BLS-release(M) − 7 days | model, naive | the model's read **one week before** the print, on near-complete provider data |
| **T−1** | BLS-release(M) − 1 day (release-eve) | model, naive | the model on the **full pre-release information set** — its best read before the print |

The T−7 → T−1 gap quantifies how much the **final week of provider payments and
claims is worth**: T−7 is the model's read a week early, T−1 is its read on the
complete pre-release information set, and the difference isolates the value of
that last week of high-frequency data. Both regimes are scored against the same
two private truths (QCEW-settled primary, first print). (The consensus
head-to-head that formerly defined T−1 is a Track-B Total contest — see the
TL;DR and §4 — and no longer lives in either private regime.)

**Grid construction.** Two new snapshot grids analogous to the A4 `snapshot`
command, with `as_of = _ces_publication_date(M) − {7, 1} days` (release date
from `release_dates/vintage_dates.py` `_ces_publication_date` /
`_first_friday`; use the scraped/actual release date where available, the
computed first-Friday rule otherwise). Censoring at either horizon still
excludes M's own print (its `vintage_date` is release day R > T−1, so
`vintage_date <= as_of` drops it), leaving M the unobserved nowcast target;
M−1's print and all of M's high-frequency data (claims, JOLTS at its lag,
provider data at its lag) are included. Both grids are **new compute** (the A4
day-12 grid is not reused for A5 scoring) but reuse `fit_model_batch`
**unchanged** — only the as-of dates differ; ~2 × 24 fits, tractable on the
batched harness. Buildability failures surface cheaply via the A1
negative-master pattern (record, don't crash).

## 4. Competitors — naive floors only

On the private (Track A) track the **only** competitors are the **naive floors**.
All competitors are reduced to a single comparable quantity per reference month:
**MoM change in private payrolls, thousands** (`change_k`, `industry_code='05'`),
the same unit the model nowcast and the private targets use.

- **Model.** Private nowcast `change_k` from the batched fits at each regime's
  as-of (produced by the T−7 / T−1 grids on the `'05'` target; A5 reads the
  reduction).
- **Naive floors.** random-walk (last published **private** print repeats) and
  trailing-12-month **private** mean. Sanity floors, scored against both private
  truths (QCEW-settled primary, first print).

**What is *not* here, and why.**

- **ADP — removed entirely (competitor *and* regressor).** Not a competitor, not
  a model input. Post-Aug-2022 ADP states it is "not intended to forecast" the
  BLS report, and the model's private-vs-private framing makes a separate
  private-employment proxy redundant rather than a fair scored competitor.
- **Consensus — removed from the private track → deferred to Track B.** The
  Bloomberg survey median forecasts the **Total** number; it has **no meaning**
  against the private nowcast alone (the private nowcast cannot see government
  employment). It moves to the deferred Track B (below), which also needs a new
  **government forecast**. The pluggable consensus adapter and
  `specs/bloomberg_consensus.md` become Track-B artifacts; nothing consensus
  ships on the private scoreboard or in the private MZ.
- **Bridge baseline — removed as a competitor.** The HARD RULE is "naive floors
  only," and the vintage-censored claims/JOLTS bridge regression is not a naive
  floor. The *same* regression is not lost: it survives as the **Tier-1 Aruoba
  diagnostic** in `model_improvements.md` §4 (first→third private-revision
  regression on the same vintage-censored cyclical arrays), feeding the §5A
  first-print offset. It is demoted from competitor to diagnostic, not deleted.

**Deferred Track B (spec-only, NOT built here).** The product that competes with
**consensus** is **Total NFP = private nowcast + government forecast**, scored
against the **Total-NFP consensus** + the Total first print. The **government
forecast is a new, undesigned component** (Track B's critical path; see
`model_improvements.md` §2, §9–11). Until it exists there is no valid consensus
comparison, because consensus is a Total object. A5 captures Track B as the
gateway to the consensus contest; it does not design or build it.

## 5. Report — the scoreboard

A new `scripts/run_a5_backtest.py` (sibling to `run_a4_backtest.py`, reusing
its `snapshot`/`batched` machinery for the T−7 and T−1 grids) emits
`data/backtests/a5_report.md` + `a5_results.parquet`:

- a **scoreboard** of `competitor × regime` scored against the **private
  QCEW-settled truth (primary)** and the **private first print**, with
  ME / MAE / RMSE (and, where dispersion exists, hit-rate / direction accuracy);
- per-date detail (each reference month: model + naive-floor `change_k`, the
  private QCEW-settled and first-print actuals, errors; plus the *unscored*
  private revised-truth reference column if enabled);
- COVID (2020–2021) excluded from headline metrics (decided-questions rule);
  shutdown-frontier months (e.g. 2025-11, 2026-01) flagged, not silently pooled.

The A4 report (`a4_report.md`) stays as-is — it is a parity-gate artifact
(batched-vs-serial). A5 is the evaluation layer above it; it does not modify
A4 outputs.

**Gate satisfied when** the report scores the **private** model against the
**naive floors** at each regime, against both private truths (QCEW-settled
primary + first print). This is a **reduced bar** relative to the literal plans/0
gate ("model vs ADP vs consensus vs naive"): ADP is removed entirely and consensus
moves to the deferred Track B (which needs the unbuilt government forecast), so on
the private track the **private QCEW-settled administrative truth replaces the
competitor contest** as the real benchmark. The ADP/consensus contest is met by
Track B when it is built. This amends the gate; the amendment is **to be
recorded** in plans/0's gate log (§7 step 8; mirror `model_improvements.md` §8).

**Forward compatibility (B1).** The output consumer is decided (plans/0 SQ2,
2026-06-13): an accurate first-print nowcast for Bloomberg publication, plus
a supersector "why" narrative — so **B1 (the supersector narrative layer)
leads Phase B**. A5 itself stays national, but the scoreboard and competitor
protocol are keyed on a **series identifier** (national now; supersectors
later) so B1 can score supersector nowcasts through the same harness without
a rebuild. This is a structural choice (one key column), **not** added scope
— no supersector logic ships in A5.

## 6. Module layout

```
nfp_ingest/first_print.py            # first-print extractor (store levels, industry_code='05')
nfp_vintages/evaluation.py           # + revised_truth_change_k(final_view '05'), competitor protocol
nfp_vintages/competitors/
    __init__.py                      # Competitor protocol + registry (naive floors)
    naive.py                         # random-walk + trailing-mean floors (private)
scripts/run_a5_backtest.py           # T−7 / T−1 grids + scoreboard; reuses A4 batched harness
```

Track-B-only (deferred, NOT built here): `consensus.py` (the `load_consensus()`
adapter, see `bloomberg_consensus.md`) and the government-forecast component live
with the Total assembly, not on the private track. The vintage-censored bridge
regression lives in `model_improvements.md` §4 as the Tier-1 Aruoba diagnostic.

Tests: `nfp_ingest/tests/test_first_print.py` (extractor vs published **private**
headlines, §2a), `nfp_vintages/tests/test_competitors.py` (naive-floor units /
alignment), `nfp_vintages/tests/test_a5_*` (scoreboard assembly, regime alignment,
both private truths).

## 7. Sequencing

Retire risk first, then build outward from the cheapest pieces:

1. **Private retarget** — thread `industry_code='05'` through the snapshot
   target, `first_print_changes`, `panel_to_model_data`, and the a5 index;
   confirm a `'05'` fit converges before any `'05'` eval is trusted
   (`model_improvements.md` §8, §11).
2. **First-print extractor** (`first_print.py`) — pure store-levels read,
   testable immediately against the **private** headlines (§2a). No grid needed.
3. **Private QCEW-settled truth** — wire the primary-truth column into the
   scoring path (§2c), alongside the first print.
4. **Revised-truth reference** (optional, *unscored* — `evaluation.py` on
   private `final_view`) — small; build only if wanted, else skip.
5. **Near-release grids (T−7, T−1)** on the `'05'` target — the new compute;
   reuse `fit_model_batch`. Build both snapshot grids, fit them batched, surface
   unbuildable dates.
6. **Naive-floor adapters** (`naive.py`) — the only private-track competitors.
7. **Scoreboard** (`run_a5_backtest.py`) wiring model + naive floors × regimes
   scored against both private truths; emit `a5_report.md`.
8. Gate annotation in plans/0 (record the reduced bar + Track-B deferral); spec
   → archive on completion; memory updated.

## 8. Open items and risks

- **`'05'` is untested in the model.** A1–A3 goldens and A4 were all on `'00'`.
  Before trusting any `'05'` A5 eval, confirm `first_print_changes('05')` yields a
  sane private series (✓ verified 2026-06-19: 108 months 2017→2026, 106 valid)
  **and** that one `'05'` model fit converges (`model_improvements.md` §8, §11).
- **QCEW-settled basis (NSA vs SA).** The store's QCEW `'05'` is NSA-only; the CES
  first print and the model nowcast are SA. The primary-truth comparison (§2c) must
  reconcile the basis — reuse the model's QCEW-anchor handling or add an explicit
  SA/NSA bridge — before it is apples-to-apples. Unresolved here and in
  `model_improvements.md` §3; pin the mechanism before trusting the primary
  scoreboard.
- **`bloomberg_consensus.md` is now stale.** It still frames consensus as a live
  A5 competitor; under this revision consensus is a Track-B (Total) artifact.
  Out of scope to edit here — flag as a follow-up to recontextualize for Track B.
- **Track B is undesigned.** The consensus contest needs a **government forecast**
  (Track B's critical path; `model_improvements.md` §11). Captured here, not built;
  consensus + `bloomberg_consensus.md` are Track-B artifacts until then.
- **Near-release buildability (T−7, T−1)** — frontier/shutdown months may be
  unbuildable at either horizon; same negative-master handling, recorded not
  fatal. The 2025-11 hole and 2026-01 frontier are flagged, not pooled (§5).
- Carry forward the `ces_growth_convention.md` §3 model-input observations
  (triple benchmark wedge at February as-ofs; permanent November rev-2
  outliers) into the Phase-B model-evidence agenda — not an A5 task.

## Appendix: gate restatement (plans/0)

> **A5 — Real competitors in the harness.** Add ADP prints (FRED; mind the
> Aug-2022 methodology break) and consensus survey median (Bloomberg/Econoday
> history — sourcing is a real acquisition task). Naive baselines stay as
> sanity floors, not gates. **Gate:** every backtest report scores model vs.
> ADP vs. consensus vs. naive, at each information regime.

**This design meets a corrected, reduced bar.** The private retarget
(2026-06-19) splits the literal gate across two tracks:

- **Track A (now):** the **private** model vs **naive floors only**, scored
  against the **private QCEW-settled truth (primary)** and the **private first
  print**, across the **T−7 and T−1** regimes. On the private track the
  QCEW-settled administrative truth *replaces* the ADP/consensus competitor
  contest as the real benchmark — ADP is removed entirely (a private-employment
  proxy, redundant here) and consensus is a Total object with no meaning against
  the private nowcast.
- **Track B (deferred, spec-only):** the ADP/consensus contest, recast as **Total
  NFP = private nowcast + government forecast** vs the **Total-NFP consensus** +
  Total first print. Its critical path is the new **government forecast**
  (undesigned; `model_improvements.md` §11). This is the only valid consensus
  comparison and is built later.

This amends the literal plans/0 gate ("model vs ADP vs consensus vs naive"); the
amendment is **to be recorded** in plans/0's gate log (§7 step 8) and mirrors
`model_improvements.md` §8.
