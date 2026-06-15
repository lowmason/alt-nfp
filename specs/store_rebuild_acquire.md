# T0 acquisition spike — findings (2026-06-15)

Read-only spike for [`plans/10`](../plans/10-store_rebuild.md) T0. Method:
inspected the cached CES triangular files (`data/downloads/ces/cesvinall`, fresh
2026-06-15), probed the live QCEW open-data API slices, and cross-checked against
the canonical store. Resolves T0's three listed unknowns and surfaces one new
acquire risk (QCEW vintaging) that gates the QCEW build.

## Unknown 1 — CES triangular coverage & `(rev,bmr)` reconstruction — ✅ GO

- **Coverage complete.** 113 NSA `tri_{code}_{NSA}.csv` files: `00/05/06/07/08`,
  all supersectors, and detailed sectors. Layout: **rows = vintages**
  (2003-05 → 2026-01), **columns = reference months** (`Jan_39` … `Jan_26`),
  cell = level of that ref-month as published in that release.
- **`cesvinall` alone reconstructs all four `(rev,bmr)` combos** — verified
  against the store to the unit. Jun-2023, `00`, NSA:

  | combo | triangle | store |
  |---|---|---|
  | `(0,0)` | 156963 | 156963 |
  | `(1,0)` | 156945 | 156945 |
  | `(2,0)` | 156905 | 156905 |
  | `(2,1)` | 156701 | 156701 |

  rev-0/1/2 = the first three values down a ref-month column; benchmark
  re-basings appear as steps at each **February** vintage row. The "separate bulk
  benchmarked file" named in spec §4.1 is just the triangle's latest vintage
  row — **not a distinct source.**

- **Finding for T2 + T6 — `(2,1)` is per-benchmark.** Each February vintage
  re-bases history, so a ref-month gets a *distinct* `(2,1)` per benchmark year,
  each with that February's release as `vintage_date`. `cesvinall` carries them
  all (Jun-2023: 2023-benchmark → 156842 @ vintage ≈2024-02; 2024-benchmark →
  156701 @ ≈2025-02). **The existing store collapses these to one `(2,1)` per
  ref-month and mis-pairs the *latest value* (156701) with the *earliest*
  benchmark date (2024-02-02)** — value and date from different benchmarks, a
  latent as-of lookahead. Spec §4.1 ("`vintage_date` = that year's February
  benchmark release") already implies per-benchmark rows; building them fixes the
  lookahead. **T6 history gate** must then compare only the *latest-benchmark*
  `(2,1)` against the old store (the rebuild legitimately has extra historical
  `(2,1)` rows the old store lacks).

## Unknown 2 — QCEW size-class coverage — ✅ GO

- **Size endpoint:** `data.bls.gov/cew/data/api/{year}/1/size/{1-9}.csv` (Q1
  only). National rows are **`own_code=5` (private)** — exactly the rebuild's
  scope. `size_code` 1–9 present; size agglvls `{21–28}`; industry codes include
  the supersector pulls (`1012`–`1027`, agglvl 22) and NAICS sectors (agglvl 23),
  with `month{1,2,3}_emplvl` → ÷1000. Small targeted slices, **not** the ~280 MB
  singlefile.

## Acquire source — API slices beat the singlefiles (informs T5)

- **US000 area slice** `data.bls.gov/cew/data/api/{year}/{qtr}/area/US000.csv`
  (`own_code=5`) carries **all national agglvls `{11–18}`**, every quarter,
  ~2083 rows/qtr — including **agglvl 13 (supersector pulls `1012`–`1027`)** and
  **agglvl 16 (Logging `1133`)**, the exact rows T3's crosswalk needs.
- So the QCEW acquire can be **area slices (per quarter) + size slices (per Q1)** —
  tiny and targeted. The cached `qcew_bulk.parquet` (singlefile-derived) lacks
  agglvl `13`/`16` and `size_code` (confirmed); the slice path sidesteps the gap
  *and* the 280 MB-per-year download.
- *Alternative* if slices prove insufficient: widen
  `download_qcew_bulk._WANTED_AGGLVL` to add `{13,16,21–28}` and keep `size_code`
  in `_KEEP_COLUMNS`.

## Unknown 3 — NAICS vintage — ✅ GO (low risk)

- NAICS-2022-for-all is acceptable at the supersector/sector aggregation the
  rebuild targets; the 2017→2022 reclassifications hit detailed industries, not
  2-digit sectors / supersectors. **Spot-check** the 3-digit subsectors feeding
  the durable/nondurable (`31`/`32`) split. Spec already defers vintage-aware
  crosswalks.

## ⚠ OPEN RISK (new) — QCEW historical per-industry vintaging

This is the real gate on the QCEW build, and it is *not* one of T0's three listed
unknowns:

- T3's `build_qcew_panel(raw)` expects `raw` **already tagged with a `revision`
  column** — it does not synthesize revisions.
- The live QCEW API serves **current (latest-revision) data only.**
  `qcew-revisions.csv` gives Initial→Final revision values but **only at the
  total US/state level**, not per industry.
- So per-industry **rev-0..4 history (2017+) is not re-downloadable from scratch**
  via the API. The existing store's QCEW vintage structure is also murky on quick
  inspection (national `05` Q1-2017 shows only rev-0; unfiltered `00` shows
  rev 0–4 across geographies) and is of unverified provenance.
- **Decision needed before the QCEW build (T5):** how to source per-industry QCEW
  vintages —
  (a) reuse the existing store's QCEW captures (re-crosswalk, don't rebuild
  history);
  (b) reconstruct vintages by applying the total-level revision ratios
  (`qcew-revisions.csv`) to current per-industry levels;
  (c) capture-forward only (rev accumulates over future releases; history
  approximate);
  (d) check `~/Projects/alt_nfp` for saved per-release singlefile captures.
  Recommend a short dedicated spike (call it **T0.5**) before T5's QCEW path.

## Go/no-go

| Task | Verdict |
|---|---|
| **T2** CES builder | **GO** — `cesvinall` suffices; build per-benchmark `(2,1)` rows + correct `vintage_date`. |
| **T4** size-class | **GO** — source = QCEW size endpoint (Q1, national=private). |
| **T5** acquire/orchestration | **GO for levels** via API slices; **QCEW vintaging (T0.5) is the gating open question** — spike first. |
| **T6** gates | unaffected; refine the history-consistency gate for per-benchmark `(2,1)`. |
