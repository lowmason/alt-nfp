# CES growth conventions in the vintage store — analysis for A5

Status: **analysis only, no code changes**. Feeds the A5 evaluation-convention
decision flagged in `plans/6-a4_vmap_backtests.md` ("Finding: evaluation
actuals are convention-laden") and plans/0 strategic question 1 (first print
vs benchmark-informed truth). All numbers below were verified read-only
against the canonical store (`s3://alt-nfp/store`) on 2026-06-12; national
CES SA, `industry_code='00'`.

## TL;DR

1. Growth is computed at **read time** in
   `nfp_ingest.vintage_store.transform_to_panel` (step 3,
   `vintage_store.py:476-499`): `log(employment).diff()` over
   `(source_tag, geo, industry, revision, benchmark_revision)` groups sorted
   by `ref_date`. The store itself holds **levels only** — growth never
   touches the append path.
2. Each growth row therefore differences a month's level against the prior
   month's level **from the same print ordinal**, which is a *different
   release* ("revision-cohort" convention). It is **not** the within-release
   headline change. The convention is intentional, documented, inherited
   verbatim from the frozen reference, and pinned by the A1/A2 golden
   masters.
3. Consequence, now quantified: rev-0 growth differs from the same-release
   headline by >100k in **51 of 271 months (19%)**, >50k in 45%. Every
   annual benchmark splices its full wedge into one month of *each* cohort
   (rev-0 at Jan, rev-1 at Dec, rev-2 at Nov), so a February as-of feeds the
   sampler up to **three consecutive wedge observations** (verified at
   `as_of_ref=2025-02-12`: −449k, −354k, −467k).
4. A headline-convention ("within-release") first print is fully
   reconstructible from existing store levels via a structural pairing rule
   — `(p, rev r) ↔ (p−1, rev r+1)`, benchmark-row fallback — with 271/273
   coverage. It should live as an **additive evaluation-side extractor**,
   not as a change to `transform_to_panel` (A1 pins panel columns *and*
   values exactly; A2 pins `build_model_data` arrays exactly).

## 1. Where per-vintage growth is computed (Q1)

There is exactly one live computation. `build_panel`
(`panel.py:26`) → `transform_to_panel` (`vintage_store.py:403`), step 3:

```python
# vintage_store.py:476-499
# Growth must be computed within a consistent vintage: same source tag,
# geography, industry, and (revision, benchmark_revision) pair.
growth_group = [
    "source_tag", "geographic_type", "geographic_code",
    "industry_type", "industry_code", "revision", "benchmark_revision",
]
lf = (
    lf.sort(*growth_group, "ref_date")
    .with_columns(pl.col("employment").log().diff().over(growth_group).alias("growth"))
    ...
)
```

Notes:

- `VINTAGE_STORE_SCHEMA` has no growth column; `append_to_vintage_store`
  anti-joins **level** rows on
  `(ref_date, …, revision, benchmark_revision)`. "How release files are
  diffed against the store" only controls *which level rows exist*, never
  growth values.
- Growth runs **before** rank selection (step 3b) — the CLAUDE.md note
  "preserves per-vintage measurement error semantics" refers to this
  ordering: diffing *after* selection would difference the censored
  diagonal's levels (July rev-0 against June rev-1 from the same release),
  i.e. would yield the headline convention at the frontier ranks. The
  before/after ordering *is* the convention choice.
- `ces_national.py` contains two legacy paths with the same cohort grouping
  (`load_ces_vintages`, lines 147-163) and a within-single-snapshot variant
  (`fetch_ces_current`, lines 64-94). Neither has any caller in v2;
  `build_panel` reads only the store.

## 2. The mechanism, on the three A4 episodes

A CES release publishes three prints: month *p* first print (rev-0), *p−1*
second (rev-1), *p−2* third (rev-2); each February's release additionally
re-bases the published level history to the annual benchmark. The cohort
convention pairs same-ordinal prints across adjacent releases, so any level
revision to *p−1* between the two releases lands in *p*'s "growth".

**2025-07 rev-0 (vintage 2025-08-01) — ordinary big revision.**
L(Jul, rev0) = 159,539. Cohort partner: Jun rev-0 (2025-07-03) = 159,724 →
**−185k**. Same-release partner: Jun rev-1 (2025-08-01) = 159,466 →
**+73k = the published headline**. The −258k gap is exactly June's level
revision between the two releases (the May+June downward revision — level
revisions accumulate).

**2026-01 rev-0 (vintage 2026-02-11) — annual benchmark in the first
print.** L(Jan-26, rev0) = 158,627. Cohort partner: Dec rev-0 (2026-01-09)
= 159,526, pre-benchmark basis → **−899k** (the 2025 benchmark, −911k
preliminary). Same-release partner: Dec's benchmark row
(rev2, bmr1, 2026-02-11) = 158,497 → **+130k**, the published headline.
(Dec-2025 has *no* rev-1 row — see §4 capture artifacts.)

**2025-10 rev-2 (vintage 2026-02-16, shutdown make-up print).**
L(Oct, rev2) = 158,408, post-benchmark basis. Cohort partner: Sep rev-2
(2026-01-09) = 159,593, pre-benchmark → **−1,185k**. Basis-consistent
partner: Sep (rev2, bmr1, 2026-02-11) = 158,548 → **−140k**. Same
mechanism as Jan's rev-0, shifted onto the rev-2 cohort by the
shutdown-compressed schedule.

These growths are exactly what `build_panel` emits and what
`run_a4_backtest.py` consumed: the rev-0 track via the panel's
`revision_number == 0` rows, the best-available track via
`model_data._ces_best_available` (max revision in {0,1,2} per period,
`model_data.py:310-334`) — which is how −449k (2024-11, rev-2), −1,194k
(2025-10) and −900k (2026-01) became *evaluation actuals*.

## 3. How big and how often

Store-wide comparison of rev-0 cohort growth vs the same-release diff
(271 months with both rows, 2003–2026):

| threshold | months | share |
|---|---|---|
| \|gap\| > 25k | 194 | 72% |
| \|gap\| > 50k | 123 | 45% |
| \|gap\| > 100k | 51 | 19% |

The tail is dominated by benchmark Januaries (gap ≈ the annual wedge:
2010-01 −1,356k, 2023-01 +815k, 2025-01 −609k) plus revision-heavy episodes
(2020-05 −640k, 2025-07 −258k).

The annual splice hits **all three cohorts**, one ref-month each
(as-published prints change benchmark basis with the February release):

| benchmark year | rev-0 @ Jan Y+1 | rev-1 @ Dec Y | rev-2 @ Nov Y |
|---|---|---|---|
| 2022 (+506k prelim) | +1,324k | +1,033k | +1,029k |
| 2024 (−598k final) | −468k | −354k | −450k |
| 2025 (−911k prelim) | −902k | (rev-1 missing) | −1,189k @ Oct* |

\* shutdown-shifted schedule. Consequences:

- **Censored panels:** at `as_of_ref=2025-02-12` the last three CES SA
  growth observations fed to the sampler are −449k (2024-11 rev-2), −354k
  (2024-12 rev-1), −467k (2025-01 rev-0) — the same wedge counted three
  times at the frontier. This is inherited reference behavior (A3 parity
  covered 2025-02) but worth restating: the *model input* convention and
  the *evaluation actual* convention are the same `growth` column.
- **Deep history:** rank-3+ selection takes original third prints
  (`bmr=0`), so every November (nominally) carries a permanent
  wedge-sized rev-2 observation in *every* as-of panel, absorbed by
  `sigma_ces[2]` under a Normal likelihood.
- **A4 window:** 5 of 24 months flagged `† SPLICE` (>150k disagreement):
  2024-11, 2025-01, 2025-06, 2025-07, 2025-10. 2026-01 is *not* flagged
  only because both tracks show the same −900k (rev-0 is the only print so
  far, so best-available == rev-0).

## 4. Intentional or artifact? (Q2)

Three layers with different answers:

**(a) The cohort convention — intentional, inherited, pinned.** The old
repo's `vintage_store.py:477-500` is line-identical (same comment); the
archived reference spec (`alt_nfp/archive/vintage_pipeline_spec.md` §3.2)
prescribes "for each (source, industry_code, revision) group, sort by
ref_date, compute log(emp_t/emp_{t-1})"; A1 golden masters pin the censored
panels value-identically. The rationale is real: a cohort is the diff of an
actually-published constant-ordinal level series, so the measurement error
of g(p, r) is e(p, r) − e(p−1, r) — one error process per revision rank,
which is what gives `sigma_ces[vintage_idx]` (rank-indexed observation
noise, `revision_schedules.py` noise multipliers 3.0/2.0/1.5) its meaning.
The headline alternative mixes ordinals inside one observation
(e(p, 0) − e(p−1, 1)). It is **not** an artifact of release-file diffing:
growth is never stored, and the append path moves levels only.

**(b) The benchmark splice — inherent, structural, previously
unquantified.** Given (a) plus the fact that BLS re-bases the level history
once a year *inside* the normal print cycle, each cohort must swallow the
wedge at exactly one ref-month per year. Nothing in the store is mislabeled:
the post-benchmark third prints really are the as-published third prints.
A4's contribution is measuring how much this contaminates both evaluation
tracks.

**(c) Genuine capture artifacts — small, separable.** (i) Dec-2025's rev-1
slot was never stored: the release-day tagger took *independent* maxes of
(vintage_date, revision, benchmark_revision) per ref_date, so on benchmark
day the calendar's (rev2, bmr1) row shadowed the same-day second print. The
live-capture path is `releases._latest_ces_vintage_dates`
(`alt-nfp current` → `build_releases` → `_fetch_ces_releases` →
`releases.parquet` → `build_store`); `tagger.latest_vintage_lookup`
(`tagger.py:39-51`) is a now-legacy mirror with no live callers. **Fixed**
(2026-06-13): both functions now key on `(ref_date, benchmark_revision)`,
emitting each track as a coherent `(vintage_date, revision)` pair so both
prints reach the store; regressions in `test_new_ingest.py`
(`test_benchmark_day_emits_both_prints`, `TestLatestCesVintageDates`).
Tagging only changes which rows *future* captures emit — it never
reinterprets the append-only store, so A1/A2 golden masters are untouched.
(ii) Historical vintage stamps for rev-1/rev-2
are schedule-derived, off the true release date by −3…+5 days in most
months (only 20/271 rev-0 rows share an exact stamp with their same-release
partner); the live-capture era stamps are true release dates. Both affect
*which level rows exist / how they're stamped*, not the growth math, but
they constrain how a headline reconstruction must pair rows (§5).

## 5. What a headline-convention first print requires (Q3)

**Definition.** headline(p, release R) = log L(p at R) − log L(p−1 at R),
both levels from the same release's published history.

**Pairing rule (store terms).** Same-release pairs are structural, not
temporal: for r ∈ {0, 1}, partner of (p, rev r, bmr 0) is
(p−1, rev r+1, bmr 0); at benchmark months, where the partner print is
itself the benchmark print, fall back to p−1's latest
(rev 2, bmr ≥ 1) row. Do **not** join on `vintage_date` equality (stamps
misalign historically, §4c). For r = 2 the same release does not republish
p−1 outside benchmark season, so headline == cohort there by construction —
rev-2 only needs the benchmark-season fallback.

**Verified coverage.** 271/273 rev-0 months have the (p−1, rev-1) partner.
The two misses: 2003-05 (history edge — no partner exists, emit null) and
2026-01 (Dec rev-1 shadowed, §4c-i — the bmr-1 fallback row exists and
yields the correct +130k). Ordinary benchmark Januaries need no fallback:
the Dec rev-1 row is post-benchmark as published (e.g. Jan-2025: 159,069 −
158,926 → +143k, the published headline). Shutdown make-up releases put two
same-ordinal prints in one release (Oct+Nov 2025 rev-1 both at 2026-01-09);
the structural rule handles them unchanged.

**Where it lives — constraints first.** A1
(`test_golden_masters.py:81-99`) asserts censored-panel **columns and
values** are identical to the frozen fixtures; A2
(`test_model_data_golden.py`) asserts exact array equality on
`build_model_data` output; A3/A4 pin model behavior on those arrays. So:
no new PANEL_SCHEMA column, no change to the `growth` column's values, no
change to selection — by default.

- **Option A (recommended for A5 scoring): additive evaluation-side
  extractor.** A small function in the data layer (natural home:
  `nfp_ingest`, alongside `model_data.py`; alternatively
  `nfp_vintages.evaluation`) that reads store *levels* via
  `read_vintage_store` and returns per-period
  `(first_print_growth, first_print_change_k, vintage_date)` under the
  pairing rule above. Backtest scripts consume it for the first-print
  track. Zero contact with golden-mastered paths; unit-testable against
  the published headlines (+73k, +143k, +130k above).
- **Option B (only if A5 decides the *model* should observe headline
  growth): opt-in convention switch on `transform_to_panel`**, e.g.
  `growth_convention="revision_cohort" (default) | "within_release"`.
  Default preserves every pinned value. This is a likelihood change, not an
  evaluation change: per-rank sigmas and the calibrated noise multipliers
  were estimated under cohort semantics, and benchmark wedges currently
  enter as (mis-modeled) observations rather than being absent. Requires
  new golden fixtures plus an A3-style parity/regression campaign — a
  Phase-B-shaped decision, out of A5's evaluation scope.
- **Option C (best-available track): score against one coherent level
  path.** The −453k/−1,194k "best-available actuals" are cohort splices in
  the *target*. The clean revised-truth definition is the diff of a single
  latest-published level series (`nfp_vintages.views.final_view`
  semantics), which is internally basis-consistent by construction —
  at the cost of a target that moves when benchmarks land (plans/0 SQ1's
  documented trade-off). This would replace the current † exclusions with
  a well-defined target instead of a censored metric.

## 6. Recommendation for A5

Score both tracks, each under a convention that matches its question:
first-print track via Option A (does the nowcast predict *what BLS will
announce*); revised-truth track via Option C (does it predict *reality as
later measured*). Keep the store and `transform_to_panel` untouched.
Of the §4c capture items, the rev-1 benchmark-day shadowing is **fixed**
(2026-06-13, §4c-i); the remaining item — true release-date stamps for
historical rev-1/rev-2 captures — stays open as a small fix that improves
future reconstructions without touching pinned history. Carry the §3
model-input observation (triple wedge at February as-ofs; permanent
November rev-2 outliers) into the Phase-B model-evidence agenda.

## Appendix: reproduction

Read-only; env from the repo-root `.env` (`load_dotenv`), store via
`read_vintage_store(source="ces", seasonally_adjusted=True,
geographic_type="national", industry_code="00")`. Cohort growths reproduce
`transform_to_panel` output exactly; headline counterparts are level diffs
of the partner rows named in §2. Key raw levels (thousands): Jun-25
rev0 159,724 / rev1 159,466; Jul-25 rev0 159,539; Dec-25 rev0 159,526 /
(rev2,bmr1) 158,497; Jan-26 rev0 158,627; Sep-25 rev2 159,593 / (rev2,bmr1)
158,548; Oct-25 rev2 158,408.
