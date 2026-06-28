# ALFRED frontier patch for CES vintages

Status: **design / approved for spec (2026-06-28).** Use the FRED **ALFRED** real-time
vintage API to **forward-fill the recent CES window** (`vintage_date > 2026-02-11`)
that the `cesvinall` bulk-file path cannot supply, and **append** it to the existing
`source='ces'` partition via the store's existing incremental-append machinery. This
is **not** a replacement: the `cesvinall`/`ces_builder` path remains the source for
all history ≤ 2026-01, and ALFRED only patches the frontier gap.

**Why a patch, not a replacement** (decided 2026-06-28 after verification): the BLS
`cesvinall` triangular CSVs simply **don't carry data for `2026-02-12+`** (a source
gap, consistent with the Oct-2025 shutdown tail), so the store dead-ends at ref
2026-01-12. A *full* ALFRED replacement was rejected because the probe proved domain
`08` and 12 fine sectors have **no ALFRED archive before 2011-03** — replacing
everything would *delete* 2003–2010 history the `cesvinall` path already has (§9). The
patch keeps that history and fills only the hole. Companion:
`specs/completed/store_rebuild.md` (schema + `ces_builder`), `capture.py` spec (the
append path this reuses).

All series-ID, depth, revision-convention, and value facts below are from **live
ALFRED probes + read-only store/calendar inspection on 2026-06-27/28**, not assumption.

## TL;DR

1. **Frontier patch.** ALFRED supplies only the cohorts the calendar says should exist
   (`vintage_date > 2026-02-11`, ≤ today) and the store lacks; everything ≤ 2026-01
   stays `cesvinall`-sourced and untouched.
2. **Reuse the append path.** A new ALFRED *source* feeds the existing
   `append_to_vintage_store()` (7-col anti-join ⇒ idempotent) + `_detect_corrected_levels()`
   (flags any seam value discrepancy). **Minimal append** at the seam — add only
   missing `(ref, revision)` rows; existing cesvinall rows stay, discrepancies flagged
   for review, never silently clobbered.
3. **Coverage = all 30 keys × {SA, NSA}** for the window (2026 is past every series'
   archive floor, so even the fine sectors + domain 08 are fully available).
4. **Cohorts**: `(0,0)/(1,0)/(2,0)` (first/second/third print) only. No new annual
   benchmark falls in this window (the Feb-2026 benchmark is already captured at the
   store frontier; next is Feb-2027), so `(2,1)` is out of scope here.
5. **Series-ID resolution is alias-first and title-verified** — required even for a
   2026-only patch because the aggregates (`00/05/06`/supersectors) have **no**
   systematic `CES…01` id; their archives live only under aliases (PAYEMS/USPRIV/…).
6. **Vintage dates come from the existing calendar** (`vintage_dates.parquet`,
   schedule-derived) — values from ALFRED, dates from the calendar — so the patched
   rows match the store's existing `vintage_date` convention exactly (no downstream
   ripple).

## 1. Decisions (locked 2026-06-27/28)

| # | Decision | Choice |
|---|---|---|
| 1 | Purpose | Forward-fill the 2026-02-12+ CES window the cesvinall files lack |
| 2 | Scope | **Patch, not replacement**; cesvinall/ces_builder keep ≤2026-01 + all history |
| 3 | Cohorts | `(0,0)/(1,0)/(2,0)` only (no new benchmark in-window) |
| 4 | Coverage | All 30 keys × SA/NSA for the window |
| 5 | Seam handling | **Minimal append** — only missing `(ref, rev)` rows; flag discrepancies |
| 6 | Vintage dates | **Schedule-derived from the existing calendar** (not ALFRED-actual) |
| 7 | Append mechanism | Reuse `append_to_vintage_store` + `_detect_corrected_levels` |
| 8 | Code home | ALFRED primitives → `nfp_download/alfred.py`; window builder → `nfp_ingest/ces_alfred.py` |

## 2. Current state (read-only inventory, 2026-06-27/28)

- `source='ces'`: **30 keys × {SA, NSA}**, ref **2017-01-12 → 2026-01-12**, vintage max
  **2026-02-11** (produced by `ces_builder` from `cesvinall`).
  - `total`: `00`,`05` · `domain`: `06`,`08` · `supersector`: `10 20 30 40 50 55 60 65 70 80`
  - `sector` (NAICS, 16): `21 22 31 32 42 44 48 52 53 54 55 56 61 62 71 72`
- **Calendar** (`vintage_dates.parquet`): CES ref **2003-01-12 → 2026-05-12**, vintage
  → **2026-06-08** — i.e. the schedule already covers the gap; this is a **frontier
  gap, not a missing-schedule problem**.
- **The gap** = cohorts with `vintage_date > 2026-02-11`: Dec-2025 `(2,0)`; Jan-2026
  `(1,0)/(2,0)`; Feb/Mar/Apr/May-2026 `(0,0)/(1,0)/(2,0)` as the calendar allows.
- ALFRED currency confirmed: all sampled series carry vintages through **2026-06-05/08**.

> **Out of scope (separate concern):** the store starts at **2017-01**, not 2003. If
> the store should also reach back to 2003, that is a `ces_builder` re-run with a 2003
> ref floor (it already supports the full history) — *not* an ALFRED job, and not this
> patch.

## 3. Series-ID resolution table (probe-verified, SA)

Alias-first (the aggregates have no systematic id); systematic `CES…01` fallback for
the fine sectors. All are current through 2026-06, so all serve the window.

| store key (type, code) | concept | ALFRED SA id |
|---|---|---|
| total `00` | Total Nonfarm | `PAYEMS` |
| total `05` | Total Private | `USPRIV` |
| domain `06` | Goods-Producing | `USGOOD` |
| domain `08` | Private Service-Providing | `CES0800000001` |
| supersector `10` | Mining & Logging | `USMINE` |
| supersector `20` | Construction | `USCONS` |
| supersector `30` | Manufacturing | `MANEMP` |
| supersector `40` | Trade/Transp/Util | `USTPU` |
| supersector `50` | Information | `USINFO` |
| supersector `55` | Financial Activities | `USFIRE` |
| supersector `60` | Prof & Business Svcs | `USPBS` |
| supersector `65` | Private Ed & Health | `USEHS` |
| supersector `70` | Leisure & Hospitality | `USLAH` |
| supersector `80` | Other Services | `USSERV` |
| sector `31` | Durable Goods | `DMANEMP` |
| sector `32` | Nondurable Goods | `NDMANEMP` |
| sector `42` | Wholesale Trade (NAICS 42) | `USWTRADE` |
| sector `44` | Retail Trade (NAICS 44) | `USTRADE` |
| sector `21` | Mining (NAICS 21) | `CES1021000001` |
| sector `22` | Utilities | `CES4422000001` |
| sector `48` | Transp & Warehousing | `CES4300000001` |
| sector `52` | Finance & Insurance | `CES5552000001` |
| sector `53` | Real Estate | `CES5553000001` |
| sector `54` | Prof/Sci/Tech Svcs | `CES6054000001` |
| sector `55` | Mgmt of Companies | `CES6055000001` |
| sector `56` | Admin & Waste Svcs | `CES6056000001` |
| sector `61` | Educational Svcs | `CES6561000001` |
| sector `62` | Health Care | `CES6562000001` |
| sector `71` | Arts/Entertain/Rec | `CES7071000001` |
| sector `72` | Accommodation & Food | `CES7072000001` |

**Title-verification is a hard build gate**: the probe mis-mapped `USSERV`→08 (it is
Other Services = 80) and `SRVPRD`→80 (it is domain 07). Each id's `/series` title +
`seasonal_adjustment_short` must match the store concept before use. Repo tables
`sae_states.INDUSTRIES`/`industry._CES_SECTOR` are SM/EN-oriented and **wrong** for
national CES ids — the probe is truth.

**NSA.** Fine-sector `CEU…01` ids exist (verified `CEU4142000001`, `CEU5552000001`);
NSA aggregates have no systematic id and need NSA aliases (`PAYNSA` family), resolved +
title-verified in the build discovery phase.

## 4. Architecture — ALFRED source into the existing append path

```
nfp_download/alfred.py            NEW. ALFRED vintage primitives (lifted from
  get_vintage_dates()             sae_states.py) + resolve_series_id()/title-verify.
  get_observations_for_vintages()
        │
        ▼
nfp_ingest/ces_alfred.py          NEW. build_ces_alfred_window(through, *, store_frontier)
  resolve §3 ids → fetch all vintages (output_type=2) → §5 extract (0,0)/(1,0)/(2,0)
  with real-time guard → CES→NAICS translate → join calendar vintage_date
  → VINTAGE_STORE_SCHEMA rows for the window (SA+NSA)
        │
        ▼
append_to_vintage_store(rows)     REUSED. 7-col anti-join (idempotent) +
  + _detect_corrected_levels()    _detect_corrected_levels (flags seam discrepancies)
        │
        ▼
scripts/patch_ces_alfred.py       NEW thin driver: dry-run report → review → append.
```

Nothing in `ces_builder` / `cesvinall` / the rank-based censoring / `first_print`
changes. The patch is additive.

## 5. Extraction algorithm (window-scoped)

Per resolved series, fetch **all** vintage dates (no tail cap), `output_type=2` wide:

1. Order vintage columns by date; each ref month `p` appears in its release vintage and
   every later one.
2. **`(0,0)/(1,0)/(2,0)` = the 1st/2nd/3rd *appearance* of `p`, as-published.** No
   benchmark exclusion, **no value-dedup** (CES is in thousands; a small revision can
   leave the rounded value unchanged, so deduping would drop a print and shift the
   index). Reproduces `output_type=4` automatically (§9).
3. **Real-time guard (mandatory — verified necessary, §9).** A series' first archived
   vintage carries full back-history, so a naive first-appearance mislabels a years-late
   value as a first print (`CES0800000001` ref-2003-01 first appearance = the 2011-03
   vintage, a **2,984-day** gap vs PAYEMS's genuine 37). **Keep `(0,0)` only when its
   vintage lands within ~70 days of `ref_date`**; equivalently, bound emitted ref months
   by `output_type=4`. For the 2026 window this is always satisfied, but the guard stays
   in the shared extractor.
4. **Window filter**: keep only cohorts with calendar `vintage_date > store_frontier`
   (2026-02-11) and `≤ today`. The `append_to_vintage_store` anti-join makes a re-run a
   no-op regardless, but window-filtering minimizes fetch + append volume.
5. **Value source = ALFRED; date source = calendar.** Attach each `(ref, revision)`'s
   schedule `vintage_date` from `vintage_dates.parquet` (which already covers the
   window), keeping the store's existing convention.

## 6. Store mapping

`VINTAGE_STORE_SCHEMA` rows: `source='ces'`, `geographic_type='national'`,
`geographic_code='00'`, `size_class_*` null, `ownership` from
`industry.ownership_for` (00→total, else private), **CES→NAICS sector translation** via
`industry.CES_SECTOR_TO_NAICS` (the §3 table is already NAICS-keyed),
`benchmark_revision=0`, `vintage_date` from the calendar (§5.5). Reuse
`capture._remap_ces_to_store_schema` where it fits.

## 7. Build → dry-run → append (reuse the safe path)

1. **Discover/resolve** §3 (+ NSA aliases), title-verify.
2. **Build** the window rows (§5) → `VINTAGE_STORE_SCHEMA`.
3. **Dry run**: compute `CaptureResult(appended, corrected, skipped)` **without
   writing**; review the `corrected` list (seam discrepancies vs cesvinall) and the
   `appended` count (expect the Feb–May 2026 cohorts + Dec-2025/Jan-2026 tail).
4. **Validate against ALFRED's own oracle**: the window's `(0,0)` matches FRED
   `output_type=4` (the §9 method) before writing.
5. **Append** via `append_to_vintage_store` (to a scratch store copy first if touching
   canonical — cf. memory `store-write-test-safety`; the wipe incident makes a
   scratch-first dry pass cheap insurance). Then run against canonical.
6. **Confirm**: gap filled (ref now reaches 2026-05), `first_print_changes` extends
   cleanly, a re-run appends 0 (idempotent).

## 8. Risks / open items

- **NSA aggregate aliases** (`PAYNSA` family) are the one unresolved id leg; the build
  discovery phase resolves + title-verifies them like the SA side.
- **Seam value discrepancies**: where cesvinall and ALFRED disagree on an existing
  frontier key, `_detect_corrected_levels` flags it (not auto-applied). Expect near-
  exact agreement (§9: ~98% on the 2017+ overlap); investigate any flagged row rather
  than ignore.
- **2003 history is a *separate* concern** (the store starts 2017, see §2 note) — a
  cesvinall `ces_builder` re-run, not this patch.
- **Rate limits**: sequential per-series fetch with backoff/checkpoint (~1s/series); a
  ~30-series × SA/NSA window pull is minutes, not a parallel workflow.
- **Ongoing capture**: if cesvinall keeps lagging, the regular `alt-nfp update` CES
  source could later move to ALFRED too — a follow-up, out of this patch's scope.

## 9. Verification (extraction PoC, 2026-06-27)

The §5 rule was implemented and run against live ALFRED across every level, validated
against two independent oracles: FRED `output_type=4` (initial-release) and the current
store's own `(0,0)/(1,0)/(2,0)` over the 2017+ overlap.

| level | series | genuine 3-print floor | rev0 vs FRED ot4 | vs store 2017+ (r0/r1/r2) |
|---|---|---|---|---|
| national `00` | PAYEMS | 2003-01 | 281/281 | 107/109 · 104/108 · 103/107 |
| national `05` | USPRIV | 2003-01 | 281/281 | 107/109 · 104/108 · 103/107 |
| domain `06` | USGOOD | 2003-01 | 281/281 | 108/109 · 106/108 · 106/107 |
| domain `08` | CES0800000001 | 2011-01 † | 183/183 | 107/109 · 104/108 · 103/107 |
| supersector `30` | MANEMP | 2003-01 | 281/281 | 108/109 · 107/108 · 106/107 |
| supersector `60` | USPBS | 2003-06 | 276/276 | 107/109 · 104/108 · 103/107 |
| sector `31` | DMANEMP | 2003-01 | 281/281 | 108/109 · 106/108 · 106/107 |
| sector `44` | USTRADE | 2003-01 | 281/281 | 108/109 · 106/108 · 106/107 |
| sector `21` | CES1021000001 | 2011-01 † | 183/183 | 108/109 · 106/108 · 106/107 |
| sector `52` | CES5552000001 | 2011-01 † | 183/183 | 108/109 · 106/108 · 106/107 |

† shallow tier — genuine prints only from 2011 (the real-time guard drops the pre-2011
back-history artifacts). **Irrelevant to this patch** (the window is 2026), but it is
exactly why full replacement was rejected: those keys would lose 2003–2010 history.

**Findings:** (1) rev-0 reproduces FRED's initial-release oracle **exactly** wherever
genuine prints exist. (2) The ALFRED extraction matches the *independently-built* store
**~98% on values** over 2017+; the residual is benchmark-boundary + COVID months. (3)
The real-time guard is necessary in general (kept in the shared extractor) though
trivially satisfied for the 2026 window. ⇒ ALFRED reliably reconstructs `(0,0)/(1,0)/
(2,0)` for the patch window across the full hierarchy, SA (NSA pending the alias probe).
