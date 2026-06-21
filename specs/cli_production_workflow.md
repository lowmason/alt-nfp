# CLI Production Workflow — month-T updates, feed-driven automation, bootstrap separation

**Status:** design (approved 2026-06-20; revised after adversarial review; pre-implementation)
**Branch:** `a5-rebuilt-integration`
**Scope owner:** maintainer
**Supersedes the everyday surface of:** `nfp_vintages.__main__` (the stage-shaped `download`/
`process`/`current`/`build` pipeline)

## 1. Goal

Reshape the `alt-nfp` CLI from a **batch bootstrap tool** into a **production month-T
workflow**. For a given month `T`, the everyday commands capture the data that just became
knowable, fold it into the vintage store, and bake the model-data handoff — driven, if
desired, by a daily cron that watches the BLS release feed. The one-time historical load
moves to a script.

One sentence: **bootstrap once with a script; thereafter each month is a real-time
current-print capture appended to the store, optionally triggered by the BLS RSS feed.**

## 2. Background — the audit (why this is needed)

The current CLI mirrors the internal pipeline *stages*, not a workflow, and carries **two
disjoint store-build lineages**:

- **Legacy lineage** — `download → process → current → build` (`build_store`, old schema,
  full per-partition overwrite). The only **downstream reader** of `releases.parquet` is
  `build_store` (the `current`/`build` CLI wiring in `__main__.py` also references
  `RELEASES_PATH`); `revisions.parquet` is read only by `build_store`. In the rebuilt world
  this lineage is **dead**: it writes the superseded schema and nothing downstream reads its
  output.
- **Rebuild lineage** — `build-rebuild` (`write_rebuild_store`) is the **only** path that
  produced today's promoted canonical store. Also a full per-partition overwrite.

**Neither builder is incremental.** Yet incremental primitives already exist —
`append_to_vintage_store` / `compact_partition` (`nfp_ingest/vintage_store.py:678`, `:758`) —
with their only consumer (`tagger.tag_and_append`) **dead**. So the month-T machinery is
*latent, not missing*. The bare `alt-nfp` (no args) ends in `build(None)`, the most natural
invocation triggering a store rebuild, and it calls Typer-decorated functions directly
(fragile). See `specs/completed/audit_nfp_vintages.md` for the full audit (note: its "Critical
loaded gun" B-1 severity rests on the retired "append-forever/irreplaceable" store framing;
structurally the findings stand, but `build` is now simply **legacy/superseded**).

## 3. Design principles

1. **Production = API current-print capture + append.** Each month's vintage is the print
   BLS publishes, captured in real time and appended. We never re-run the triangular/bulk
   extract in production.
2. **Bootstrap = bulk/triangular reconstruction, run once, from a script.** Not the everyday
   CLI.
3. **The store is the source of truth.** Coverage, "is this captured?", and idempotence are
   decided by reading the store, not a side state-file.
4. **The BLS feed is the ground-truth release signal.** It answers "is the release out
   *now*?" — which the calendar can only predict and shutdowns can delay.
5. **The release calendar is a production dependency, not legacy.** `vintage_dates.parquet`
   (which assigns each captured print its `vintage_date`/`revision`/`benchmark_revision`) must
   be **advanced to `T` before every capture** — it feeds the CES tag step, the QCEW
   knowability guard, and `status`. Its refresh moves *into* the production surface; it is not
   deleted with the legacy `process` command (see §5.0, §10).
6. **Firewall.** No changes to `nfp-model`, `transform_to_panel`, `build_model_data`, or the
   A1/A2/A3 golden paths. The model handoff is `snapshot`; wiring `nfp-model` into the CLI is
   explicitly out of scope.

## 4. Command surface (final)

| Command | Role | Network |
|---|---|---|
| `alt-nfp update --as-of T [--only ces\|qcew\|indicators]` | Advance calendar, capture knowable prints for `T`, append to store | ✓ BLS / FRED |
| `alt-nfp snapshot --as-of T [--grid-end E]` | Bake hash-pinned ModelData (exists; day-12 fix §4a) | — |
| `alt-nfp status [--as-of T] [--store URI]` | Store coverage + "what's uncaptured" | — |
| `alt-nfp watch [--source ces\|qcew\|all] [--snapshot]` | Poll BLS feed; trigger `update` on a new release (cron) | ✓ BLS |
| `scripts/bootstrap_store.py` | One-time historical rebuild + promote (**not** a CLI command) | ✓ BLS |

The legacy `download`/`download-indicators`/`process`/`current`/`build`/`build-rebuild`
commands are **retired** from the everyday surface; their reusable bodies move into the
bootstrap script (§10) and — for the calendar scrape — into `update` (§5.0).

**§4a — `snapshot` day-12 fix (specified).** Today the day-12 validation only fires when
`--grid-end is None` (`__main__.py:308-311`), and the grid loop seeds at the 12th of the start
month regardless of `--as-of.day` (`__main__.py:316-320`) — so
`snapshot --as-of 2026-03-05 --grid-end …` silently snapshots `2026-03-12`, a date *later*
than the requested cutoff. **Fix:** require `as_of.day == 12` in **both** the single and grid
paths (reject any other day), so a snapshot never silently shifts the cutoff.

## 5. `update --as-of T` — internals

The three sources have three *different* update semantics because they have different vintage
structure. `update` orchestrates a shared **calendar advance** (§5.0) then all three sources
(or one, with `--only`); `--as-of T` is the knowability cutoff (no row with `vintage_date > T`
is appended). When invoked by `watch`, `T`/the as-of is the feed `pubDate` (the actual release
day).

### 5.0 Calendar advance (runs first, every update)

Before any capture, `update` **advances `vintage_dates.parquet` to `T`** by running the
release-calendar scrape (`_build_release_calendar`, today inside `__main__.py:61-168` — lifted
into a callable in `nfp_vintages` or `nfp_ingest`). This is mandatory: the CES tag step
(§5.1.2) and the QCEW/status knowability (§5.2, §8) all read this calendar, and an un-advanced
calendar makes the tag join return **null** → step-4 censoring drops every row → a **silent
empty capture** (the failure §8 warns is unrecoverable). The scrape already degrades
gracefully on a BLS 403 by falling back to cached release pages (`__main__.py:97-127`). When
`update` is feed-driven (`watch`), the feed `pubDate` is *also* threaded into the tag step so a
fresh release the scrape hasn't yet reflected is still tagged with the correct
`vintage_date`. A `--no-refresh-calendar` escape hatch skips the scrape when the operator knows
the calendar already covers `T`.

### 5.1 CES — the real month-T capture (new adapter `nfp_ingest/capture.py`)

The central gap: `_fetch_ces_releases` (`releases.py:101`) emits the **old `COMBINED_SCHEMA`**
(`releases.py:38`, 11 cols, legacy `industry_type` = `'national'/'domain'`, no
`ownership`/`size_class_*`), but `append_to_vintage_store` hard-rejects anything missing a
`VINTAGE_STORE_SCHEMA` column (`schemas.py:125-144`; raise at `vintage_store.py:700-702`). So
the net-new piece is a **capture-to-store adapter**, `capture_ces_print(as_of, store_path)`:

1. **Fetch** via the BLS JSON API (`fetch_ces_national_via_api`,
   `nfp_download/bls/ces_national.py:161`) — whose docstring notes it is the path for "current
   month data not yet in flat files." **`BLS_API_KEY` is a hard prerequisite for `update`**
   (fail-fast with a clear error, §13) — *not* a soft fallback: the current
   flat-file-first/API-fallback order returns an **empty frame** without a key
   (`releases.py:127-129`), and the flat file can lag the freshest print, so a soft fallback
   converts a missing secret into silent data loss. If a flat-file path
   (`fetch_ces_national`, `nfp_download/bls/ces_national.py:54`) is kept as a secondary, it
   must **assert month `T` is present and error** when it is missing.
2. **Tag** vintage_date/revision/benchmark_revision via the vintage calendar
   (`tagger.latest_vintage_lookup`, `tagger.py:19` / `_latest_ces_vintage_dates`,
   `releases.py:55`; both read `VINTAGE_DATES_PATH`, advanced in §5.0), **carrying the
   IND-IMD-1 rev1/bmr0 drop forward** (`releases.py:159-185`, `specs/ces_growth_convention.md`
   §5): on a benchmark release a ref_date is published as both `(rev1,bmr0)` and `(rev2,bmr1)`;
   the flat file carries only the post-benchmark level, so the `(rev1,bmr0)` row must be
   dropped or it is mis-stamped.
3. **Remap** to `VINTAGE_STORE_SCHEMA` using the **rebuilt** taxonomy: `'00'`→(`total`,`total`),
   `'05'`→(`total`,`private`), `size_class_*`=null. The values come from
   `nfp_lookups.industry.INDUSTRY_TAXONOMY:270-271` (via `ownership_for`, `industry.py:286`);
   reuse the call site `ces_builder._taxonomy_for` (`ces_builder.py:102`). **Do not** reuse
   `releases.py:193-199`'s legacy `'national'/'domain'` mapping.
4. **Censor** `vintage_date ≤ T`, then `append_to_vintage_store` → `compact_partition` on the
   touched `(source=ces, seasonally_adjusted)` partitions. Before the anti-join, **compare each
   incoming row's `employment` against the stored row for the same ukey** and emit a
   `CORRECTED-LEVEL` warning (+ a `status` row) when they differ — this is the runtime signal
   for the §6.3 silent-drop case (the append's ukey excludes both `vintage_date` and
   `employment`, so a corrected same-revision level is otherwise dropped invisibly).

The capture writes the **same supersector set `build_ces_panel` writes** (NSA + SA, the
bootstrap store's coverage) — not a narrowed `{'00','05'}` — so `update` and the bootstrap
store are coverage- and schema-consistent (the precondition the §7 overlap check compares
against).

"Current print for month `T`" is defined by the calendar, **not** a fixed offset (shutdowns
break `{T-1 rev0, T-2 rev1, T-3 rev2}`): the rows whose calendar `vintage_date` equals this
release's date.

### 5.2 QCEW — conditional, only when a new quarter is knowable

Most months this is a **no-op** (QCEW is quarterly). The capture is `capture_qcew_quarter(
as_of, store_path)`. **Knowable test:** iterate candidate quarters and pick the most recent
whose rev-0 `vintage_date ≤ T` — `get_qcew_vintage_date(ref_quarter, ref_year, revision=0)`
takes `ref_quarter` as the **string `'Q1'..'Q4'`** (`revision_schedules.py:299`); the §5.0
calendar must be present so this uses real release dates, not the day-1 lag fallback
(`revision_schedules.py:358-365`).

**Acquire-helper relocation (firewall fix).** The acquire helpers `_acquire_qcew_levels`
(`rebuild_store.py:175`) and `_acquire_qcew_size_native` (`:373`) currently live in
**`nfp-vintages`**, which sits *above* `nfp-ingest`; `capture.py` (in `nfp-ingest`) importing
them would be an **illegal upward import of private names**. They import only
httpx/polars/`nfp_lookups`/`nfp_download.client` (all legal for `nfp-ingest`), so **relocate
them to a new public `nfp_ingest/qcew_acquire.py`** (`acquire_qcew_levels`,
`acquire_qcew_size_native`); both `capture.py` and the bootstrap script (and
`rebuild_store.py`, which updates its imports) consume them from there. The helpers loop over
full years/all quarters, so the single-quarter wrapper fetches the containing year then
**filters to the one quarter** (and Q1-only for the size branch). Tag `revision=0`,
`seasonally_adjusted=False` (QCEW is NSA-only), then append → compact (with the same
corrected-level check as §5.1.4).

### 5.3 Indicators — a refresh, **not** an append

Indicators live **separate** from the vintage store (`data/indicators/<name>.parquet`,
`paths.py:55`; the store recognizes only `ces`/`qcew`/`sae`), with no vintage dimension. So
`update`'s indicator step **calls the existing `download_indicators()`** (`indicators.py:59-116`,
full FRED refresh/overwrite) — one line of orchestration, **no** `append_to_vintage_store`.
Routing indicators through the store would be incorrect.

This is as-of-correct for publication **timing** (the model masks future-knowable indicator
values via `ref_date + pub_lag` vs `as_of`, `model_data.py:456-466`). It does **not**
vintage-track indicator *revisions*: a current refresh stores latest-revised values, so a
historical backtest sees later-revised indicator levels — a known limitation (§12).

*Flag (out of scope, §12):* the in-code default `CYCLICAL_INDICATORS_DEFAULT`
(`provider_config.py:55-58`) is only `claims` + `jolts`; `nfci`/`biz_apps` from the docs are
not wired.

## 6. Store dedup & correctness

### 6.1 Decision A — ukey under-keying fix (in scope)

`append_to_vintage_store` and `compact_partition` dedup on a 7-column key
(`ref_date, industry_type, industry_code, geographic_type, geographic_code, revision,
benchmark_revision`; `vintage_store.py:709-717`, `:797-805`) that **excludes** `ownership`
and `size_class_type`/`size_class_code` — both real axes in the rebuilt schema
(`schemas.py:128,140-141`). For QCEW Q1 **size-class rows** (same industry/quarter/revision,
differing only by `size_class_code`), the current key would **collapse distinct size buckets
into one** — silent data loss on append.

Fix: **add `ownership`, `size_class_type`, `size_class_code` to both writers' ukey lists.**
A local edit to the two incremental writers in `vintage_store.py` — verified by the review to
**not** touch `transform_to_panel` (which uses its own `_CES_SERIES_KEY`/`_QCEW_SERIES_KEY`/
`dedup_key` and drops non-headline rows before censoring, `vintage_store.py:491-492`) or the
A1/A2/A3 goldens (all of which *read* the existing store and never call append/compact). The
existing store (written by `write_rebuild_store`, which doesn't use this key) is unaffected;
only future appends dedup more correctly. `test_vintage_store.py` expectations update with the
change (TDD).

### 6.2 append → compact policy

`compact_partition` keeps `MIN(vintage_date)` per ukey (`vintage_store.py:807-815`) — i.e. the
**first real-time appearance** of a given revision, which is exactly the as-of-correct row.
(This corrects the stale `audit_nfp_vintages.md:53` "keep max" claim.) `append_to_vintage_store`
keeps the **first-written** row; the two coincide when appends arrive in ascending-vintage
order, which is steady monthly production. **`update` always runs `compact_partition` after
`append_to_vintage_store`** — append alone leaves order-sensitive, fragment-accumulating
partitions. For crash-safety, `update` compacts **any partition with >1 fragment regardless of
whether new rows were appended** (cheap, idempotent), so a crash between append and compact is
self-healed on the next run (§9).

### 6.3 Decision B — corrected-level on an existing ukey (runtime warning, no auto-replace)

Because the ukey excludes both `vintage_date` **and** `employment`, if a capture re-stamps an
existing `(ref_date, revision, benchmark_revision)` key with a **corrected level**, the
anti-join **silently drops it** — the store keeps the stale level, and `first_print_change_k`
(the A5 scored actual) goes wrong for that month *and* the next (it is the `L_prev` partner).
This is rare (BLS usually corrects via a *new* revision = a new ukey, which appends cleanly).
**Decision:** the §5.1.4/§5.2 **runtime corrected-level comparison** (incoming vs stored
`employment` per ukey, *before* the anti-join) emits a `CORRECTED-LEVEL` warning + `status`
row when they differ — that is the actual detection signal, independent of any reconstruction.
**Auto-replacement is not built** in this iteration (follow-on). Note: the §7
overlap-divergence test does **not** cover this case at runtime (after promotion no per-month
bootstrap reconstruction is persisted to diff against), and the §7 first-print-unchanged test
*cannot* distinguish a legitimate no-op from a dropped correction — the runtime warning is the
only reliable detector.

## 7. Guardrail & tests

The "append ≡ rebuild byte-for-byte" intuition is too strong (the store is "replaceable, not
*identical*"). The guardrail is properties + fixtures that exercise the dangerous edges:

1. **Idempotence** — `update --as-of T` twice yields the same `(ukey → employment)` relation;
   re-running `append` returns 0 and a second `compact` is a no-op. Must include a
   same-ukey/different-vintage row to exercise the first-written-vs-min-vintage flip.
2. **Overlap-window divergence (fixture-only diagnostic)** — over a synthetic
   bootstrap∩capture window, compare **levels on the `rev0/bmr0` and `rev1/bmr0` rows** (the
   score-relevant rows; sentinel rows excluded — see below). Flagged, not asserted-zero (per
   "replaceable not identical"). This is a *fixture* test; it is **not** a runtime monitor
   (no persisted reconstruction post-promotion — §6.3).
3. **First-print-unchanged** — `first_print_changes()` (`first_print.py:53`) and
   `wedge_first_print_changes()` (`wedge_data.py:24`) are pinned across an update for all
   already-present months. *Caveat (documented):* this proves a capture didn't perturb
   existing months, but it **cannot** catch a dropped correction (§6.3) — that's the runtime
   warning's job.
4. **Calendar-not-advanced → loud failure** — `update --as-of T` with the calendar *not*
   advanced to `T` must **error loudly**, not silently append zero rows (regression test for
   the §5.0 dependency).

**Required fixtures** (note these are **synthetic store rows**, not capture output):
- a **February benchmark double-row** (`(rev1,bmr0)` + `(rev2,bmr1)` for one ref_date);
- a **shutdown "no-print" sentinel** — a row with literal `employment = -1.0` at the Oct-2025
  ref slot (the value the rebuilt store writes for shutdown-skipped slots, dropped downstream
  by `employment > 0`, `first_print.py:79-84`). Distinct from the *date* constant
  `CES_OCT_2025_RELEASED_WITH_NOV_REF = date(2025,10,12)` (`vintage_dates.py:45`), which is a
  calendar quirk, not the sentinel value. The capture path itself never produces `-1`, so the
  overlap test (property 2) must **exclude** sentinel rows or it will false-flag a future
  real-level capture against the bootstrap `-1`. The spec must also decide how `update` handles
  a shutdown month (write the delayed real level vs skip).

*Tie-break landmine to document:* the consumers disagree on which vintage wins — `append`
(first-written), `compact` (min-vintage), and `first_print_changes` itself mixes **two**
rules: the first-print level via max-vintage `.last()` (`first_print.py:94-98`) and the
prior-month partner via min-vintage `.first()` (`first_print.py:123-127`); `release_date_for`
uses min-vintage (`a5.py:33-57`). So a backfill/out-of-order append can perturb *both* legs of
a change in different directions. Steady ascending production is safe; out-of-order/backfill
appends are not (and must go through the bootstrap path, not `update`).

## 8. `status` (new `nfp_vintages/store_status.py`, ~100–150 lines)

A cheap, read-only health + knowability report. Built on `read_vintage_store`
(`vintage_store.py:336-406`, partition-prune + projection pushdown, LazyFrame) — **never**
`transform_to_panel` (the expensive growth/censoring path) and **not** `views.py` (panel-grain,
post-transform). Reports:

- **Header:** resolved store URI + `REMOTE/LOCAL/CANONICAL` flags; on a local store
  loudly warn "LOCAL FALLBACK" (cause-agnostic — either the `.env` gotcha of an unset
  `NFP_STORE_URI` or an explicit local `--store`) with the `NFP_STORE_URI` remote hint.
- **Per `(source, seasonally_adjusted)`:** latest/earliest `ref_date`, row count, last capture
  (`max(vintage_date)`), distinct vintage count.
- **Forward "UNCAPTURED" alarm:** per source, the latest ref_month BLS should have published as
  of `--as-of` (calendar `get_ces_vintage_date`/`get_qcew_vintage_date` rev0 ≤ as-of, or the
  feed §9); if the store lags, flag it. Load-bearing — the BLS API has no memory, so a missed
  monthly capture is gone. CES grid is monthly, QCEW grid quarterly (per-source, not shared).
- **Missing-month list** over the headline series (`geo 00`, `industry 00`/`05`), computed on
  **raw row presence** (no `employment > 0` filter) so the Oct-2025 `-1` sentinel counts as
  present; annotate known-shutdown months rather than flagging them.
- **`CORRECTED-LEVEL` rows** surfaced by the §5.1.4/§6.3 capture-time comparison.

All imports deferred inside the command body so `load_dotenv()` runs before
`VINTAGE_STORE_PATH` resolves (`paths.py:155` binds at import).

## 9. `watch` — feed-driven automation (new `nfp_download/release_dates/feed.py` + `alt-nfp watch`)

A thin trigger on top of `update`, designed for a daily cron:

1. **Fetch + parse** the BLS RSS feed (`https://www.bls.gov/feed/empsit.rss` for CES,
   `https://www.bls.gov/feed/cewqtr.rss` for QCEW). `www.bls.gov` intermittently 403s a plain
   httpx GET (memory `bls-akamai-blocking-intermittent`), so `feed.py` lives beside the
   existing `www.bls.gov` scraper (`nfp_download/release_dates/scraper.py`) and **reuses its
   curl_cffi Chrome-impersonating session** (`create_session(impersonate='chrome')`,
   `scraper.py:191-199`; the impersonation client is `nfp_download/client.py`, **not** the
   httpx-based `bls/_http.py`, which only serves the non-fingerprinted `data.bls.gov` API). The
   implementer captures a real feed fixture in red-phase TDD; RSS items carry `pubDate` + title.
2. **Decide "is this new?" from the store**, per `(source, ref-grain)`: the feed item gives a
   candidate release + `pubDate`; `status`-style coverage says whether that month/quarter is
   already captured. The store is the single source of truth; the feed contributes only "the
   release is out *now*." A same-day CES + QCEW co-release triggers **both** source updates.
3. **Trigger** `update --as-of <pubDate>` for the matching source. The `pubDate` (the release
   day) is the capture `vintage_date` and the `update` as-of. If `--snapshot`, also run
   `snapshot` — but at the **day-12 anchor for the captured ref-month** (e.g. `<refmonth>-12`),
   **not** the raw `pubDate`, because `snapshot --as-of` enforces the day-12 convention (§4a)
   and would otherwise reject the release date.

Idempotence/self-healing: a clean no-op on days with nothing new; `update` is anti-join
idempotent and **always re-compacts a fragmented partition** (§6.2), so a crash between append
and compact is repaired on the next `watch` run (the partial-append month reads as "present"
but the next run still compacts it). `watch` is a **CLI command** (cron invokes `alt-nfp
watch`), not a script — automation is a production concern.

## 10. Bootstrap script + legacy retirement

`scripts/bootstrap_store.py` lifts the **rebuild** lineage (not the legacy one), ordered:

1. `download_ces()` (`bls/bulk.py:62`) — extract `cesvinall/` triangular CSVs (the command
   body of `build-rebuild` assumes these are on disk; the script must prepend this).
2. `build_ces_panel()` (`ces_builder`) — CES NSA+SA store-schema rows; vintage dates from
   `revision_schedules` (run with `vintage_dates.parquet` **present** so bootstrap and the
   §5.1 capture agree on `(revision, benchmark_revision)` for overlap months — §7 property 2).
3. `acquire_qcew_levels(start, end)` → `build_qcew_panel`; `acquire_qcew_size_native(start,
   end)` (Q1-only) → `build_size_class_panel` — live CEW API (the helpers relocated to
   `nfp_ingest/qcew_acquire.py`, §5.2).
4. `compose_rebuild_panel` → `write_rebuild_store(allow_canonical=False)` → **scratch prefix**
   (`NFP_STORE_URI=s3://alt-nfp/store-rebuild`).
5. **Promote** via the **`_t8_promote.py` backup → cutover → verify** flow (copy-then-delete
   per partition; `_t8_promote.py` cutover copies rebuild files in then `fs.rm`s the old
   orphans). **Do not** use `scripts/mirror_store.py` here — it is **overwrite-only** and,
   because store filenames encode vintage ranges, an overwrite-mirror leaves stale fragments
   and corrupts the store (the exact hazard CLAUDE.md warns about). Keep the `is_canonical_store`
   refusal. (Generalize `_t8_promote.py:cutover` into the bootstrap's promote step.)

Scope is **national-only, 2017+** — the intended canonical scope, not restored to
state/region/pre-2017. Of the three `download_*` fns, the bootstrap needs **only**
`download_ces` (QCEW is fetched live from the CEW API, not the bulk ZIPs); wiring
`download_qcew_bulk` would re-create the dead legacy QCEW path.

**Retire** (delete from the CLI as commands): `build` (`build_store`), `current`, `process`,
the whole-history `download`, the bare-run chain, and the orphaned artifacts
(`revisions.parquet`, `releases.parquet`, `*_revisions.parquet`). **Keep** the release-calendar
scrape (`_build_release_calendar` / `build_vintage_dates`) — it is **retained as a callable**
invoked by `update` (§5.0) and usable by the bootstrap script; only the `process` *command*
wrapper goes away. (See §13 for the consumer-scope correction.)

## 11. Firewall & file map

**New:**
- `nfp_ingest/capture.py` — `capture_ces_print(as_of, store_path)`, `capture_qcew_quarter(as_of, store_path)` (fetch → tag → remap → censor → corrected-level check → append → compact).
- `nfp_ingest/qcew_acquire.py` — `acquire_qcew_levels` / `acquire_qcew_size_native`, relocated public (was private in `nfp-vintages/rebuild_store.py`).
- `nfp_vintages/store_status.py` — the `status` report helper.
- `nfp_download/release_dates/feed.py` — RSS fetch + parse (reuses `scraper.create_session` curl_cffi impersonation).
- A callable release-calendar advance (lift `_build_release_calendar` out of `__main__.py` into a function `update` and the bootstrap can call).
- `scripts/bootstrap_store.py` — one-time rebuild + promote (generalizes `_t8_promote.py` cutover).
- Tests: `nfp_ingest/tests/test_capture.py`, `test_qcew_acquire.py`, `nfp_vintages/tests/test_store_status.py`, `test_cli_update.py`, `test_update_guardrail.py`, `nfp_download/.../test_feed.py`, and extensions to `test_vintage_store.py` (ukey).

**Edited:**
- `nfp_vintages/__main__.py` — add `update`/`status`/`watch`; split the bare-run into plain `_fns`; fix `snapshot` day-12 (§4a); drop legacy commands; update the docstring banner.
- `nfp_ingest/vintage_store.py` — the 3-column ukey extension (§6.1) in `append_to_vintage_store` + `compact_partition` only.
- `nfp_vintages/rebuild_store.py` — import the relocated acquire helpers from `nfp_ingest.qcew_acquire` (not firewall paths).
- `packages/*/CLAUDE.md`, root `CLAUDE.md` — command-banner + map updates.

**Untouched (firewall):** `nfp-model/*`, `transform_to_panel`, `build_model_data`, the
A1/A2/A3 golden paths, `model_data.py`, `first_print.py`/`wedge_data.py`/`a5.py` (consumed
read-only).

## 12. Out of scope (explicit)

- Wiring `nfp-model` into the CLI (a `run-model` command). `snapshot` is the handoff.
- The D-1 full three-writer dedup unification (only the targeted ukey extension, §6.1).
- Corrected-same-revision-level **auto-replacement** (§6.3 — runtime warning only).
- Vintage-tracking indicator *revisions* (§5.3 — current refresh stores latest-revised values).
- Extending the indicators default list to `nfci`/`biz_apps` (§5.3 — flagged only).
- Restoring state/region/division or pre-2017 QCEW coverage.

## 13. Open questions / risks carried to planning

- **Calendar is a production dependency** (resolved into the design, §5.0): `vintage_dates.parquet`
  feeds the CES capture tag, the QCEW knowability guard, and `status` — its refresh moves into
  `update`, **not** deleted with `process`. The old "confirm no *model-side* reader" framing was
  mis-scoped (the model read path never calls `get_*_vintage_date`); the binding consumers are
  the new vintages-side capture/knowability/status code.
- **Runtime keys are hard prerequisites:** `update` fails fast without `BLS_API_KEY` (CES JSON
  API, §5.1.1) and needs `FRED_API_KEY` (indicators); `watch` needs network to `www.bls.gov`.
- **Remote compact semantics:** `compact_partition` uses `glob('*.parquet')` + `unlink()`;
  the remote S3/MinIO delete path is not exercised by the local-dir tests — verify against the
  real store before production.
- **`INDICATORS_DIR` is local-only** (`paths.py:55`, no `storage_options_for`/`is_remote`).
  A Bloomberg/S3 deployment that wants indicators in object storage needs net-new plumbing
  (an `INDICATORS_URI` + UPath handling) — flagged, not in this scope.
- **Feed schema:** confirm `empsit.rss`/`cewqtr.rss` item fields against a captured fixture in
  TDD; confirm publication cadence/back-fill behavior.
- **Shutdown-month capture behavior:** decide whether `update` writes a delayed real level or
  skips a shutdown month, so the §7 overlap test treats it consistently (§7, §6.3).

## 14. Suggested build order (for the plan)

1. **Relocate QCEW acquire helpers** to `nfp_ingest/qcew_acquire.py` (public) + update
   `rebuild_store.py` imports + tests — unblocks `capture.py` legally (§5.2).
2. **ukey fix** (§6.1) + `test_vintage_store.py` extension — the correctness floor, isolated.
3. **Calendar-advance callable** (lift `_build_release_calendar`) — the §5.0 dependency.
4. `capture.py` CES adapter (§5.1) incl. the corrected-level check + `test_capture.py`.
5. `update` CLI command (calendar-advance + CES + indicators) + the §7 guardrail tests
   (incl. the calendar-not-advanced loud-failure test) + the `snapshot` day-12 fix.
6. QCEW conditional capture (§5.2) wired into `update`.
7. `status` (§8) + `store_status.py`.
8. `watch` + `feed.py` (§9), incl. the day-12 snapshot anchor + self-healing compaction.
9. `scripts/bootstrap_store.py` (generalize `_t8_promote.py` promote) + legacy retirement
   (§10) + docs.
