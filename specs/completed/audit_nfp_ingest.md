# Package Review — `nfp-ingest`

**Scope this turn (read this first).** Project-knowledge retrieval was
unavailable while writing this, so I could only review the `nfp-ingest` source
already pulled into our session during the system pass. That covers the three
modules where the package's hard logic and risk actually live, at the level of
detail a deep dive needs:

- **`vintage_store.py`** — near-complete (the two selection helpers, the source
  tagger, the writers, the reader, and the censored-`transform_to_panel`
  dispatch in full; missing only the head/tail of `transform_to_panel` and the
  *body* of `_validate_censored_selection`, both of which the tests pin down).
- **`model_data.py`** — most of it (`ModelDataConfig`, `build_model_data`, the
  bulk of `panel_to_model_data` including the CES vintage remap, the provider
  loop, the QCEW noise/era logic, the BD-covariate block, the cyclical censor,
  and the return dict). Missing the *bodies* of `_ces_best_available`,
  `_load_cyclical_indicators`, `_build_levels_from_growth`, and the calendar
  construction.
- **`snapshots.py`** — complete.
- Tests: `test_vintage_store.py`, `test_snapshots.py`, `test_golden_masters.py`
  (substantial coverage of the selection helpers and the snapshot round trip).

**Deferred (not seen this turn, no findings asserted):** `panel.py`,
`payroll.py`, `compositing.py`, `ces_national.py`, `ces_state.py`, `qcew.py`
(ingest-side), `indicators.py`, `releases.py`, `base.py`, `aggregate.py`,
`tagger.py`, `release_dates/config.py`, `release_dates/vintage_dates.py`, and
those modules' tests. These are mostly mechanical (parsing, joins, weighting,
HTTP-fed transforms) but `payroll.py` (provider auto-detection) and
`compositing.py` (QCEW weighting / staleness) are the two most likely to hide
their own bugs — flag them for the next pass.

Findings new to this review are marked **[NEW]**; ones carried from the system
review are marked **[SYS]** and sharpened with package detail.

---

## `vintage_store.py`

The censoring engine. Public surface: `read_vintage_store`,
`transform_to_panel`, `append_to_vintage_store`, `compact_partition`, plus the
`VINTAGE_STORE_SCHEMA` and the private selection helpers. This is the best code
in the package — and it has the package's two sharpest latent bugs, both on the
*write* side rather than the (excellent) read/selection side.

### Correctness

**[NEW] I-1 (High within package) — `append_to_vintage_store` can overwrite a
prior fragment and silently lose rows.** The output filename is derived purely
from the vintage-date range of the batch being written:

```python
vmin = partition_df['vintage_date'].min()
vmax = partition_df['vintage_date'].max()
fname = f'v_{vmin}_{vmax}.parquet'
partition_df.drop(['source', 'seasonally_adjusted']).write_parquet(
    str(partition_dir / fname), storage_options=storage_options_for(store_path)
)
```

`write_parquet` is a single-file writer; it *replaces* the target. The anti-join
dedup guarantees the rows being written are not already present, but it does
**not** guarantee the *filename* is unique. Two appends whose surviving rows
span the same `[min, max]` vintage range but are otherwise disjoint produce the
same filename — and the second silently overwrites the first, destroying the
first batch's rows.

- The canonical workflow is safe: monthly live-capture appends carry a single
  vintage_date, so `vmin == vmax` is that month, and successive months get
  distinct filenames. Re-running the same append is also safe (anti-join → 0
  rows → `continue`, no write).
- The dangerous pattern is a **bulk/backfill** append of multi-vintage batches:
  batch A = rows at vintages {Jan, Mar}, batch B = different rows at vintages
  {Jan, Mar}. Both → `v_2025-01-01_2025-03-01.parquet`; B clobbers A.

Given the package's "the store only ever takes appends" invariant (the safety
mechanism behind the C-1 footgun in `build_store`), the *append* path is exactly
the one that must not lose data. **Fix:** make the filename unique per write —
e.g. append a short content/row hash or a monotonic counter, or detect an
existing target and pick a fresh name:

```python
stem = f'v_{vmin}_{vmax}'
fname = f'{stem}_{_short_hash(partition_df)}.parquet'   # collision-proof
```

(`compact_partition` already merges fragments, so extra files are harmless.)

**[NEW] I-2 (Medium) — `_select_ces_at_horizon`'s fallback contradicts its own
"never select benchmark rows" invariant.** The docstring is explicit: *"Benchmark-revised
rows `(revision=2, benchmark_revision>0)` are never selected. Benchmark-quality
information enters the model through QCEW observations instead."* But the
fallback for any `(series, ref_date)` not matched by the rank rules does:

```python
remainder = (
    remainder.sort(['revision', 'benchmark_revision'], descending=True)
    .unique(subset=sk + ['ref_date'], keep='first')
)
```

— which picks `max(revision), max(benchmark_revision)`, i.e. it *will* select a
benchmark-revised row when one is the most-revised option for an unmatched
ref_date. This is not a temporal look-ahead (rows are already filtered to
`vintage_date ≤ D`), but it is a **model-specification leak**: benchmark-quality
CES reaches the model directly for those months, which the design deliberately
routes through QCEW only. In a normal triangular store the fallback rarely fires
(every ref_date has rev-0/1/2 at bmr=0), and the test that exercises it
(`test_fallback_when_revision_missing`) uses a benchmark-free triangle, so the
benchmark-selection path is **untested**. **Fix:** add `& (benchmark_revision ==
0)` to the fallback's candidate set (or restate the invariant if parity with the
reference requires the current behaviour — worth checking the old repo here,
since the golden masters would otherwise catch a divergence).

**[NEW] I-3 (Low–Medium) — `_QCEW_MAX_REVISION` is a dead, duplicated source of
truth.** It's defined and documented at module top:

```python
_QCEW_MAX_REVISION: dict[int, int] = {1: 4, 2: 3, 3: 2, 4: 1}
```

…but in the code I can see it is never read. `_select_qcew_at_horizon` hardcodes
the same schedule inline in the `r1…r4` rank rules, and `test_vintage_store.py`
redefines `max_rev = {1: 4, 2: 3, 3: 2, 4: 1}` locally rather than importing it.
So the QCEW revision schedule is encoded in three places that must be kept in
sync by hand. **Fix:** either drive the rank rules and the test from the
constant, or delete it. (Hedge: I can't see the entire file this turn, so
confirm it's truly unreferenced before deleting.)

**Reads that are correct and worth not "fixing":**

- The rank-based selection (CES and QCEW) is sound. Growth is log-diffed within
  a revision cohort *before* selection, then one row per `(series, ref_date)` is
  picked by recency rank, reconstructing the triangular diagonal. The
  quarter-dependent QCEW rules (`r3`/`r4` branching on `_q == 4`, `_q == 3`,
  `~is_in([3,4])`) correctly encode the asymmetric `{Q1:4,Q2:3,Q3:2,Q4:1}`
  saturation, with a `rank ≥ 5 → max(revision)` tail and a fallback. This is the
  hard part of the whole system and it's done carefully.
- The `vintage_date ≤ D` + `ref_date < D` pre-filter plus the fail-fast
  `_validate_censored_selection` (rejects dup ref_dates, calendar gaps,
  null/zero employment, null growth before the sampler) is the right posture.
  This is the discipline I argued in the system review should propagate
  *outward* to the inputs.

### Design / clarity

- **[Low] The `2017-01-12` regime boundary is a magic constant.** It's
  documented in the docstring but lives as a literal `cutoff = date(2017, 1,
  12)` inside `_select_qcew_at_horizon`, unlike `era_breaks` which is a config
  knob. If the QCEW monthly-data start ever moves, this is an easy thing to
  miss. Consider lifting it to module/config level next to `_QCEW_MAX_REVISION`.
- **[SYS, M-6] `append_to_vintage_store` vs `compact_partition` disagree on the
  vintage_date tie-break.** Both dedup on a uniqueness key that *excludes*
  `vintage_date`; append's anti-join keeps the **existing** row (first-seen
  wins — correct for a vintage store), while compact keeps `max(vintage_date)`
  (last wins). They resolve a `(ref_date, geo, industry, revision,
  benchmark_revision)` collision with differing vintage_dates *oppositely*. In
  practice append prevents the second write so they shouldn't coexist, but the
  rule should be explicit and identical in both (prefer earliest vintage_date)
  and documented on the key.
- **[Low] `read_vintage_store` passes the full `VINTAGE_STORE_SCHEMA` (including
  the partition columns `source`/`seasonally_adjusted`) to a `hive_partitioning`
  scan**, while the written parquet files have those columns *dropped*. It works
  because Polars injects partition columns from the path, but it couples the
  read to that injection behaviour — a Polars version bump is the kind of thing
  that could break it. Minor, but worth a regression test that reads a freshly
  written store back and checks the schema/row count (see test gap below).

### Test coverage

Strong on the selection helpers (`TestSelectCesAtHorizon`,
`TestSelectQcewAtHorizon`, `TestValidateCensoredSelection`) and on the basic
`transform_to_panel` happy path. The gap is the **write path**: I see no unit
tests for `append_to_vintage_store` or `compact_partition` — `plans/2` describes
a *manual* MinIO smoke run, not pytest coverage. That gap is exactly where I-1
(filename collision) and the M-6 tie-break ambiguity hide. A `tmp_path` test
that appends two disjoint multi-vintage batches and asserts the row count is
conserved would have caught I-1 immediately, and would run in CI (it needs no
S3).

---

## `model_data.py`

The layer-2 transform: panel → model-ready arrays. Public surface:
`build_model_data` (layer-1 + layer-2), `panel_to_model_data` (layer-2 only),
`ModelDataConfig`, `build_obs_sources`. Dense, careful, and mostly correct — the
findings are about a bypassable censoring path, the dead covariates, and silent
degradation.

### Correctness

**[NEW] I-4 (Medium) — `panel_to_model_data(as_of=D)` does *not* apply layer-1
censoring, so calling it directly on a non-horizon-censored panel silently
produces look-ahead-contaminated model data.** `build_model_data` is safe: it
runs `build_panel(as_of_ref=D)` (the rank-based selection) and *then*
`panel_to_model_data(as_of=D)`. But `panel_to_model_data` is also a documented
public entry point ("for callers that already hold a panel"), and on its own it
applies only the `vintage_date ≤ D` cutoff plus `_ces_best_available`'s
"highest available revision per month." On a raw (all-vintages) panel, that
"highest available revision" for old months resolves to the **benchmark-revised**
value — precisely the look-ahead the rank-based selection exists to prevent. The
docstring doesn't warn that the input panel must already have been horizon-
selected via `build_panel(as_of_ref=D)`. **Why it matters:** the two entry
points have *different censoring semantics*, and the weaker one is the one that
looks like a convenience. **Fix:** have `panel_to_model_data` require/detect a
horizon-censored panel (e.g. a marker column or a cheap consistency check), or
at minimum document the precondition loudly and route external callers through
`build_model_data`. (Hedge: I'm inferring `_ces_best_available`'s revision
selection from its docstring — *"vintage_idx[t] = revision number (0/1/2)
selected, or -1 if missing"* — not its body. The exact benchmark handling there
is worth confirming in the next pass; if it already excludes `bmr > 0`, this
softens to "different-but-both-safe semantics," still worth documenting.)

**[SYS, H-3] The BD covariate arrays are dead but ship anyway.** `birth_rate`
(via `vstack`/`nanmean` over provider births), `bd_proxy = g_qcew - g_pp_avg`,
and the `bd_qcew_lagged` lag loop are computed in `panel_to_model_data`, returned
in the dict, and serialized into every snapshot (`GLOBAL_ARRAY_KEYS`), but the
model never reads them (`plans/5` confirms `φ₁·X^birth` was never wired and
`plans/0` says `φ₂·BD^QCEW` was pruned). Package-local fix: stop computing/
returning them here and drop them from `snapshots.GLOBAL_ARRAY_KEYS` (+
`SCHEMA_VERSION` bump), or move them behind a clearly-labelled diagnostics flag.
The waste is concentrated in this file.

**[SYS, H-4a] Silent degradation in the cyclical path, two ways.** (1) If the
indicator parquets are missing/mispathed, `_load_cyclical_indicators` yields
all-zero arrays, the model's gating drops them, and `φ_3` is silently never
sampled — `plans/5` records this exact footgun. (2) The censoring sentinel is
`arr[i:] = 0.0`, which collides with a legitimately-zero centred indicator
value, so "censored" and "present-but-zero" are indistinguishable. Births use
`np.nan` for the same job; cyclicals should too. **Fix:** warn when a configured
covariate loads all-zero; use a NaN sentinel for censored cyclicals so it can't
masquerade as a real observation.

**Reads that are correct:**

- **The CES contiguous vintage remap is a genuinely nice piece of design.**
  `_all_vintages = sorted(set(sa_idx) | set(nsa_idx))` then `{v: i for i, v in
  enumerate(...)}` sizes `sigma_ces_sa`/`sigma_ces_nsa` to exactly the distinct
  vintages present, handling gaps (e.g. only rev-0 and rev-2) gracefully. The
  `if not _all_vintages: _all_vintages = [2]` fallback is harmless (no CES obs ⇒
  the CES likelihood is skipped model-side anyway).
- The provider birth-rate and cyclical censoring loops use a `break` on the
  first month past the publication horizon — correct, because `dates` is
  ascending and `_offset_month(d, lag)` is monotonic, so zeroing/NaN-ing from
  the first violating index onward is exact.
- The QCEW noise model (`base_revision_mult × post-COVID era boundary mult`,
  applied in-place only to boundary months in era ≥ 1) matches the documented
  spec.

### Design / clarity

- **[NEW, Low] The industry-scope comment contradicts the documented default.**
  The inline comment reads *"no fallback 05→00 so we stay private-only when
  matching legacy"*, but the docstring and default are `industry_code='00'`
  (total nonfarm). 00 ≠ private. A reader trying to set the industry scope gets
  conflicting signals; one of the two is stale (likely the comment, from when the
  default was private). Reconcile.
- **[NEW, Low] Dead branch in `_qcew_series_with_meta`.** The
  `when(revision_number == -1).then(999)` sort-key remap exists to push CES
  benchmark rows last, but this selector is filtered to `source == 'qcew'`, and
  QCEW rows never carry `revision_number == -1` (that marker is set only for CES
  `benchmark_revision > 0` in `transform_to_panel`). So the branch never fires
  here — copy-paste residue from a CES-shaped selector. Harmless; delete for
  clarity.
- **[Low] The layer-2 `vintage_date ≤ as_of` re-filter is redundant on the
  `build_model_data` path** (layer-1 already applied it) but correctly defensive
  for the direct-`panel_to_model_data` path, and the `elif … warnings.warn(...)`
  for a panel lacking `vintage_date` is good. No change needed — just note the
  redundancy is intentional, and it does *not* substitute for the missing
  layer-1 selection (see I-4).

### Test coverage

This is the weak spot. I don't have a `test_model_data.py` in context, and the
real correctness gate for this file is `test_golden_masters.py` — which is
**S3-gated** (`skipif(not _golden_available())`). So in CI, `model_data.py`'s
many branches (the vintage remap edge cases, the censoring monotonic breaks, the
boundary multiplier, the CES best-available logic) ride almost entirely on the
golden masters that CI cannot run. The only CI-visible exercise is
`test_snapshots.py::TestHashStability` — which is *also* store-gated. This is the
package-level instance of the system review's H-2: **the heart of the data→model
transform is effectively unverified in CI.** A few synthetic, store-free unit
tests over `panel_to_model_data` (hand-built `PANEL_SCHEMA` frames exercising the
remap, a censored-cyclical month, a post-COVID boundary month) would close most
of it and run everywhere.

---

## `snapshots.py`

The artifact boundary. Public surface: `collect_snapshot`, `content_hash`,
`save_snapshot`, `load_snapshot`, `snapshot_model_data`, `snapshots_location`.
This module is in good shape — correct, well-tested, no real bugs.

### Correctness — clean

- `content_hash` hashes sorted `(name, dtype, shape, raw bytes)` + canonical
  meta JSON, explicitly **not** the npz bytes (zip timestamps). `save`/`load`
  are symmetric (the `content_hash` key is added after hashing on write and
  popped before re-hashing on read), and `load_snapshot` re-verifies and raises
  on mismatch. The documented same-endianness caveat is fine for this repo.
- `np.load(..., allow_pickle=False)` — correct (no pickle execution surface),
  and meta travels as JSON.
- The `from_snapshot` round trip (model side) and the v1→v2 `error_model`
  fallback are covered by `test_data.py`; `TestContentHash` /`TestRoundTrip`
  run store-free in CI. Good.

### Nits only

- **[NEW, Low] `collect_snapshot` doesn't assert provider names are `__`-free**,
  even though the `f'{name}__g_pp'` keying and `from_snapshot`'s `k.split('__',
  2)` depend on it. `batch.py` *does* assert it; mirror that assertion here for
  symmetry, since this is where the keys are minted.
- **[NEW, trivial] `snapshot_model_data` computes `content_hash` twice** — once
  for the filename, once inside `save_snapshot` — with identical inputs.
  Correct, just redundant; pass the digest into `save_snapshot` to skip the
  recompute.

---

## Prioritized findings (package-scoped)

| ID | Sev | Location | One-liner |
|----|-----|----------|-----------|
| **I-1** | High | `vintage_store.append_to_vintage_store` | Filename derived from vintage range only ⇒ a second multi-vintage append can overwrite a prior fragment and silently drop rows. **[NEW]** |
| **H-3** | High | `model_data.panel_to_model_data` (+ `snapshots.GLOBAL_ARRAY_KEYS`) | `birth_rate`/`bd_proxy`/`bd_qcew_lagged` computed + serialized but unread by the model. **[SYS]** |
| **H-4a** | High | `model_data` cyclical load/censor | Missing indicators ⇒ `φ_3` silently dropped; `0.0` censor sentinel collides with real zeros. **[SYS]** |
| **I-4** | Medium | `model_data.panel_to_model_data` | Direct call applies only layer-2; bypasses rank-based layer-1 ⇒ look-ahead-contaminated data, undocumented precondition. **[NEW]** |
| **I-2** | Medium | `vintage_store._select_ces_at_horizon` | Fallback selects `max(benchmark_revision)`, contradicting the "never select benchmark rows" invariant; path untested. **[NEW]** |
| **M-6** | Medium | `vintage_store` append vs compact | Opposite `vintage_date` tie-break on the same uniqueness key. **[SYS]** |
| **TestGap** | Medium | write path + `model_data` | No CI-visible unit tests for `append`/`compact` or for `panel_to_model_data` internals (golden masters are S3-gated). **[NEW/sharpened]** |
| **I-3** | Low–Med | `vintage_store._QCEW_MAX_REVISION` | Dead, triplicated revision schedule. **[NEW]** |
| Low cluster | Low | various | Industry-scope comment contradicts default; dead `999` QCEW branch; `2017-01-12` magic date; `read_vintage_store` schema/partition coupling; `collect_snapshot` missing `__` guard; double-hash in `snapshot_model_data`. **[NEW]** |

---

## Synthesis

**What's genuinely strong.** The censoring *selection* (`_select_*_at_horizon`,
`_validate_censored_selection`) and the snapshot boundary (`content_hash` and the
verified round trip) are the two best things in the package — correct, carefully
reasoned, and (for the selection helpers and snapshots) well tested. The CES
contiguous vintage remap is a small piece of real cleverness. Nothing here is
over-engineered.

**The pattern under the findings.** Two recurring shapes, both consistent with
the system-level read:

1. **The risk has migrated to the *write* and *entry-point* edges, not the
   math.** The selection logic is bullet-proofed with a fail-fast validator, but
   the writers next to it (`append_to_vintage_store`'s non-unique filename, the
   append/compact tie-break disagreement) and the second public entry point
   (`panel_to_model_data` bypassing layer-1) have sharp, mostly-silent edges. The
   careful part and the careless part sit in the same file.
2. **The correctness gate the package relies on is the one CI can't run.** Both
   `model_data` and the store write path lean on golden-master / hash-stability
   tests that are S3-gated, so the fast CI suite verifies the selection helpers
   and the snapshot codec but not the full transform or the writers — which is
   exactly where I-1 and I-4 live.

**Top changes for `nfp-ingest`, in order.**

1. **Make `append_to_vintage_store` filenames collision-proof (I-1)** and add a
   `tmp_path` test that appends two disjoint multi-vintage batches and asserts
   row-count conservation. Small change, prevents silent data loss on the one
   write path the architecture promises is safe.
2. **Add store-free unit tests for `panel_to_model_data`** (the remap, a censored
   cyclical month, a post-COVID boundary month) **and for `append`/`compact`** —
   this closes both the I-1 blind spot and the H-2 "CI can't see the transform"
   gap at the package level.
3. **Fix the two correctness leaks:** add `benchmark_revision == 0` to the CES
   fallback (I-2, after checking parity against the old repo), and make
   `panel_to_model_data` refuse or warn on a non-horizon-censored panel (I-4).
4. **Drop the dead BD covariates (H-3)** from this module and the snapshot
   schema, and give censored cyclicals a NaN sentinel + an all-zero-load warning
   (H-4a).
5. **Delete the dead `_QCEW_MAX_REVISION`** (or wire the rank rules to it) and
   clear the low-cluster nits (stale industry comment, dead `999` branch,
   `__`-guard, double-hash).

**Bottom line.** `nfp-ingest` is the most load-bearing data package and its core
logic earns the trust the rest of the system places in it — but the write path
and the second model-data entry point have silent failure modes that the
package's own (S3-gated) test posture won't catch. The highest-leverage work is
small and CI-shaped: make the safe-by-design append actually safe, and put the
transform under tests that run without credentials.

---

## Deferred to next pass (retrieval permitting)

`panel.py` (the CES/QCEW/provider join — verify the join keys and null
handling), `payroll.py` (the cell-level-vs-national auto-detection and
`_growth_series`/`load_provider_series` — auto-detection is bug-prone),
`compositing.py` (QCEW weighting, weight redistribution, staleness — numerical
correctness), `ces_national.py`/`ces_state.py`/`qcew.py` (ingest transformers,
unit conversions, industry hierarchy), `indicators.py`, `releases.py`
(`build_releases` — directly relevant to the live-capture/C-1 story), `base.py`,
`aggregate.py`, `tagger.py`, `release_dates/*`, plus the bodies of
`_ces_best_available`, `_load_cyclical_indicators`, `_build_levels_from_growth`,
`_validate_censored_selection`, and the `transform_to_panel` head/tail. Re-run
project-knowledge retrieval and I'll complete these.