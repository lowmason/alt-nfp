# ALFRED as the CES vintage source of record

Status: **design / approved for spec (2026-06-27).** Replace the `cesvinall`-file
CES builder (`nfp_ingest/ces_builder.py::build_ces_panel`, which the store rebuild
calls) with the FRED **ALFRED** real-time vintage API as the sole producer of the
`source='ces'` partition of the vintage store. Motivated by the current store
dead-ending at **ref 2026-01-12** (read-only inventory, 2026-06-27): the bulk
`cesvinall` triangular files are stale/unrefreshed past that edge (compounded by the
Oct-2025 shutdown gap), so ~5 monthly prints (Feb–Jun 2026) are missing. ALFRED is
API-driven (never stale) and returns the literal published BLS vintage.

**Framing: ALFRED reproduces `ces_builder`'s exact row structure — from the API
instead of `cesvinall` files — extended back to 2003 and forward past 2026-01.** This
is a producer swap with an unchanged downstream (`write_rebuild_store` → store →
`_select_ces_at_horizon` → `first_print_changes` → panel). Companion:
`specs/completed/store_rebuild.md` (the schema + scratch→promote cutover this reuses).

All series-ID, depth, alias, and revision-convention facts below are from **live
ALFRED probes + read-only store inspection on 2026-06-27**, not assumption — the
naive plan ("fetch `CES{code}01` for all keys") was empirically falsified, and the
extraction rule is matched to the store's *observed* convention, not derived a priori.

## TL;DR

1. **Full replacement.** ALFRED becomes the sole source for `source='ces'`. A new
   ALFRED builder is a **drop-in for `ces_builder.build_ces_panel`** in
   `bootstrap_store.py` — same `VINTAGE_STORE_SCHEMA` output, no `cesvinall_dir` arg.
   Build to **scratch**, validate, promote via the existing copy-then-delete cutover,
   then retire `ces_builder`'s `cesvinall` dependency + the dead `ces_triangular.py`.
2. **Coverage = full parity**: all **30 industry keys × {SA, NSA}** (total `00`/`05`,
   domain `06`/`08`, 10 supersectors, 16 sectors).
3. **Per series, reproduce `ces_builder`'s cohorts**: `(0,0)/(1,0)/(2,0)` (first/
   second/third print, current benchmark basis) **plus** `(2,1)` — one row per
   distinct annual-benchmark basis (deduped by value-change). All from ALFRED vintages.
4. **History floor 2003-01** where the archive allows; **depth is ragged** (§5).
   2003 is also the principled floor — NAICS-basis CES begins ~2003.
5. **Series-ID resolution is alias-first and title-verified.** The deep real-time
   archives live under FRED *friendly aliases* (PAYEMS, USPRIV, MANEMP, …), not the
   systematic `CES…01` IDs. Every alias is verified against its `/series` title before
   use — the probe caught a USSERV(=Other Services)→08 mis-map that would have poisoned
   the store.
6. **Extraction = ordinal among *all* vintages, as-published** (no benchmark
   exclusion, no value-dedup for the monthly prints). PAYEMS `output_type=4` is the
   rev-0 oracle; the store itself is the cohort-convention oracle.

## 1. Decisions (locked 2026-06-27)

| # | Decision | Choice |
|---|---|---|
| 1 | Purpose | Replace the `cesvinall`-file CES path; ALFRED = source of record |
| 2 | Scope | Full replacement of `source='ces'`; retire the `cesvinall` CES builder |
| 3 | Cohorts captured | Reproduce `ces_builder`: `(0,0)/(1,0)/(2,0)` + per-benchmark `(2,1)` |
| 4 | Industry coverage | Full parity — all 30 keys × SA/NSA (subject to ALFRED availability) |
| 5 | History floor | 2003-01 request; **accept ragged per-series depth**, no synthesis |
| 6 | Code home | ALFRED primitives → `nfp_download/alfred.py`; builder → `nfp_ingest/ces_alfred.py` |
| 7 | Promotion | Build→scratch→validate→promote (the repo hard rule) |

## 2. The current partition (read-only inventory, target of the replacement)

`source='ces'`, ref 2017-01-12 → 2026-01-12, **30 keys × {SA, NSA} = 60**, produced
by `ces_builder.build_ces_panel` from `cesvinall`:

- `total`: `00`, `05` · `domain`: `06`, `08` · `supersector`: `10 20 30 40 50 55 60 65 70 80`
- `sector` (NAICS-coded, 16): `21 22 31 32 42 44 48 52 53 54 55 56 61 62 71 72`

**Revision cohorts per (key, ref_date)** — verified at the 2023→2024 benchmark
boundary (total `00`, SA):

- `(0,0)/(1,0)/(2,0)` — first/second/third print, as published in months p+1/p+2/p+3,
  on the benchmark basis *at the time* (`benchmark_revision=0`).
- `(2,1)` — the benchmark-restatement **history**: one row per distinct annual
  benchmark (Feb 2024, Feb 2025, Feb 2026, …), emitted only when the restated value
  **changes** from the previously emitted benchmark value. Up to **9** distinct `(2,1)`
  vintages per ref month observed (deeper history → more benchmarks). *This is a
  multi-vintage history, not a single "final" value.*

Observed nuance: a month whose 3rd print lands on a benchmark release carries the
**same value+date** as both `(2,0)` and the first `(2,1)` (e.g. ref 2023-11, vintage
2024-02-02, 157014). And **rev0 can sit on a benchmark release** — ref 2024-01's
`(0,0)` vintage is 2024-02-02, the Feb-2024 benchmark itself. The builder never
excludes benchmark vintages from the monthly-print count.

`55` appears at **both** `supersector` (Financial Activities) and `sector` (Mgmt of
companies); keys are the `(industry_type, industry_code)` pair.

## 3. Series-ID resolution table (probe-verified, SA)

Alias-first for depth; systematic `CES…01` fallback where no alias archives exist.
**Floor** = earliest real-time vintage observed on 2026-06-27.

| store key (type, code) | concept | ALFRED SA id | floor |
|---|---|---|---|
| total `00` | Total Nonfarm | `PAYEMS` | 1955-05 |
| total `05` | Total Private | `USPRIV` | 1971-09 |
| domain `06` | Goods-Producing | `USGOOD` | 1971-09 |
| domain `08` | Private Service-Providing | `CES0800000001` † | 2011-03 |
| supersector `10` | Mining & Logging | `USMINE` | 1961-11 |
| supersector `20` | Construction | `USCONS` | 1961-11 |
| supersector `30` | Manufacturing | `MANEMP` | 1961-11 |
| supersector `40` | Trade/Transp/Util | `USTPU` | 1961-11 |
| supersector `50` | Information | `USINFO` | 2003-06 |
| supersector `55` | Financial Activities | `USFIRE` | 1961-11 |
| supersector `60` | Prof & Business Svcs | `USPBS` | 2003-06 |
| supersector `65` | Private Ed & Health | `USEHS` | 2003-06 |
| supersector `70` | Leisure & Hospitality | `USLAH` | 2003-06 |
| supersector `80` | Other Services | `USSERV` | 1961-11 |
| sector `31` | Durable Goods | `DMANEMP` | 1961-11 |
| sector `32` | Nondurable Goods | `NDMANEMP` | 1961-11 |
| sector `42` | Wholesale Trade (NAICS 42) | `USWTRADE` | 1961-11 |
| sector `44` | Retail Trade (NAICS 44) | `USTRADE` | 1961-11 |
| sector `21` | Mining (NAICS 21) | `CES1021000001` | 2011-03 |
| sector `22` | Utilities | `CES4422000001` | 2011-03 |
| sector `48` | Transp & Warehousing | `CES4300000001` | 2011-03 |
| sector `52` | Finance & Insurance | `CES5552000001` | 2011-03 |
| sector `53` | Real Estate | `CES5553000001` | 2011-03 |
| sector `54` | Prof/Sci/Tech Svcs | `CES6054000001` | 2011-03 |
| sector `55` | Mgmt of Companies | `CES6055000001` | 2011-03 |
| sector `56` | Admin & Waste Svcs | `CES6056000001` | 2011-03 |
| sector `61` | Educational Svcs | `CES6561000001` | 2011-03 |
| sector `62` | Health Care | `CES6562000001` | 2011-03 |
| sector `71` | Arts/Entertain/Rec | `CES7071000001` | 2011-03 |
| sector `72` | Accommodation & Food | `CES7072000001` | 2011-03 |

† `08` has no deep alias; `CES0800000001` floors at 2011. (Derivation as `05−06` was
offered and declined — a derived value isn't a literally-published print.)

**Title-verification is a hard build gate**: the probe's mnemonic guesses mis-mapped
`USSERV`→08 (it is Other Services = 80) and `SRVPRD`→80 (it is total Service-Providing
= domain 07, not even in the store). Each resolved id's `/series` title +
`seasonal_adjustment_short` must match the store concept and `SA` before any fetch.
The repo tables `sae_states.INDUSTRIES` / `industry._CES_SECTOR` are SM/EN-oriented
and **wrong** for national CES ids — the probe is the source of truth.

**NSA.** Systematic `CEU…01` ids exist for the fine sectors (verified
`CEU4142000001`, `CEU5552000001` at 2011-03) but **not** for the aggregates
(`CEU0000000001` is absent). NSA aggregates need NSA aliases (`PAYNSA` family),
resolved in the build discovery phase under the same title-verification gate; expect
an ~2011 floor for most NSA keys.

## 4. Architecture — producer swap

```
nfp_download/alfred.py            NEW. ALFRED vintage primitives lifted out of
  get_vintage_dates()             nfp_vintages/processing/sae_states.py into the
  get_observations_for_vintages() download layer (boundary: fetching only). Adds a
  resolve_series_id()+title check /series title-verification helper. sae_states imports them back.
        │
        ▼
nfp_ingest/ces_alfred.py          NEW — drop-in for ces_builder.build_ces_panel
  build_ces_panel_alfred(*, as_of) resolve ids (§3) → fetch all vintages ≥2003 (output_type=2)
                                  → extract cohorts (§5) → CES→NAICS translate
                                  → VINTAGE_STORE_SCHEMA rows (SA+NSA)
        │
        ▼
bootstrap_store.py  (build_ces_panel → build_ces_panel_alfred)
  → write_rebuild_store(panel, scratch)        UNCHANGED downstream
```

The new builder returns the **same `VINTAGE_STORE_SCHEMA` rows** `ces_builder` does
(national geography, null size class, `source='ces'`, both adjustments), so
`write_rebuild_store` / `_select_ces_at_horizon` / `first_print_changes` / panel
construction are untouched. After promotion, retire `ces_builder`'s `cesvinall`
lineage and delete the already-dead `ces_triangular.py` + `cesvinall.zip` download in
`nfp_download.bls.bulk`.

## 5. Extraction algorithm

For each resolved series (`observation_start=2003-01-01`, **all** vintage dates — no
tail cap; `output_type=2` wide):

1. **Order vintages by date.** Every monthly release re-states the ref months it
   covers, so each ref month `p` appears in its release vintage and every later one.
2. **Monthly prints `(0,0)/(1,0)/(2,0)` = the 1st/2nd/3rd *appearance* of `p`,
   as-published.** No benchmark exclusion (Jan's rev0 legitimately sits on a Feb
   benchmark release — §2), and **no value-dedup** (CES is in thousands; a small
   revision can leave the rounded number unchanged, so deduping by value would drop a
   real print and shift the index). This rule reproduces `output_type=4` automatically.
3. **Benchmark cohort `(2,1)`**: walk the annual-benchmark vintages (each Feb release;
   value+date both from that vintage), emit one `(2,1)` row per benchmark basis whose
   restated value **differs** from the previously emitted one (first always emitted).
   Mirrors `ces_builder`'s `bench_year` dedupe-by-change.
4. **`vintage_date`** = ALFRED's actual real-time date (more accurate than
   `ces_builder`'s schedule-derived `get_ces_vintage_date` stamp; all revisions in a
   release share one date). **Deliberate divergence** from the legacy stamp — see §6/§7
   for the consumers to re-verify.
5. **Frontier / shallow series**: emit only the cohorts that exist (a recent month may
   have only `(0,0)`; a 2011-floor series simply starts later). **No synthesis** — an
   absent print is correct, recorded in the coverage report. Honor `as_of` for the
   frontier `(2,1)` cutoff exactly as `ces_builder` does.

**Oracles (PAYEMS first, before scaling):** `output_type=4` (initial-release-only)
returns every ref month's true first print in one call — confirm step 2 reproduces it.
Then read the **current store** at a benchmark boundary (ref 2023-11/12, 2024-01) and
confirm the ALFRED cohorts match `ces_builder`'s `(revision, benchmark_revision)`
tagging value-for-value. Frame as "which rule reproduces the store's cohorts," not
"confirm positional works."

## 6. Store mapping

Rows are written under `VINTAGE_STORE_SCHEMA` with `source='ces'`,
`geographic_type='national'`, `geographic_code='00'`, `size_class_*` null,
`ownership` from `industry.ownership_for(type, code)` (00→total, else private), and
**CES→NAICS code translation** for sectors via `industry.CES_SECTOR_TO_NAICS`. The §3
table is already keyed in the store's NAICS sector codes.

**`vintage_date` semantics decision:** use ALFRED-**actual** dates (the real
`realtime_start`), accept the shift vs `ces_builder`'s schedule-derived stamps, and
re-verify the consumers that assumed the old convention: rank-based
`_select_ces_at_horizon` (ranks by recency — should be robust), `first_print.py`'s
`_RELEASE_WINDOW_DAYS=15` window (was tuned to *staggered* stamps; ALFRED's same-day
stamps should resolve the rev-1 partner *better* — verify, don't assume), and
`rebuild_gates.py`'s `(2,1)` checks (lines ~355/547).

## 7. Build → scratch → validate → promote

1. **Discover/resolve** §3 (+ NSA aliases), title-verify, write a coverage report
   (per-key floor, cohorts available).
2. **Fetch + build** §5 → `VINTAGE_STORE_SCHEMA` rows (rebuild scratch → `tempfile`;
   sequential per-series with backoff/checkpoint, ~1s/series; **no parallel workflow** —
   it fights the rate limiter; cf. memory `prefer-cheap-inline-audits`).
3. **Write to scratch store** `NFP_STORE_URI=s3://alt-nfp/store-rebuild` (guarded by
   `is_canonical_store`; never write `…/store` directly; cf. memory
   `store-write-test-safety`).
4. **Gates:**
   - **PAYEMS oracle** — step-2 rev-0 reproduces `output_type=4`.
   - **Cohort match** vs the current store at benchmark boundaries (§5) — the
     `(revision, benchmark_revision)` tagging agrees.
   - **Overlap-diff is a real value-level correctness test, not "ALFRED wins ties":**
     the current SA `source='ces'` is the *published* CES drop-in (plans/11), so
     ALFRED (also published CES) should match it **~exactly on values**, with
     `vintage_date` **expected to shift** (actual vs schedule-derived). Compare on
     `(key, ref_date, revision, benchmark_revision)` values; a value divergence is an
     **extraction bug to fix**, not a residual to wave through. Benchmark-boundary
     months are where it will light up — investigate there. (Only series the old store
     *reconstructed* rather than published would invoke "external truth wins" — confirm
     which, if any, on the SA side; the aggregates are published-CES.)
   - **`first_print_changes`** resolves the rev-1 partner under ALFRED's same-day
     stamps (§6).
   - **`rebuild_gates.py`** `(2,1)` checks pass under the new builder.
   - **A1/A2 golden re-baseline** — goldens move under full replacement; re-baseline
     deliberately.
5. **Promote** via the copy-then-delete cutover (`bootstrap_store.py`), snapshot prior
   canonical first.
6. **Retire** `ces_builder`'s `cesvinall` lineage + the dead `ces_triangular.py` +
   `cesvinall.zip` download.

## 8. Risks / open items

- **NSA alias coverage** is the least-probed leg; NSA aggregates need `PAYNSA`-family
  aliases (build discovery phase). Worst case NSA floors at 2011 for aggregates too.
- **`vintage_date` semantics change** (ALFRED-actual vs schedule-derived) ripples to
  `_select_ces_at_horizon`, `first_print`, and `rebuild_gates` — §6/§7 list the
  re-verification. If any gate proves brittle, the fallback is to *re-derive* the
  legacy schedule stamps via `get_ces_vintage_date` for drop-in parity (values from
  ALFRED, dates from the schedule) — a documented escape hatch, not the default.
- **Ragged depth** (2003 deep aggregates / 2011 fine sectors + NSA) is accepted and
  surfaced in the coverage report; downstream tolerates ragged history.
- **Benchmark-vintage identification** back to 2003 must be reliable for the §5 step-3
  `(2,1)` walk — derive from the annual benchmark schedule (Feb release / Jan-Y first
  print, `ces_builder`'s convention), cross-check against observed ALFRED cadence.
- **Pre-2003 depth left on the table**: aliases reach 1955–1971, but 2003 is the clean
  NAICS floor; deeper is a deliberate future extension, not this spec.
