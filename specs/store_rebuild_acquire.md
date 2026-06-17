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

### CONFIRMED via live probe (2026-06-15) — informs the T5 size crosswalk

- **The size tree is the area tree shifted `+10`.** Verified against
  `2024/1/size/*.csv` (`own_code=5`): size agglvl `21`=national-total,
  `22`=domain, `23`=supersector, `24`=2-digit sector, `25`=3-digit, `26`=4-digit
  (incl. `1133` Logging), `27`/`28`=5/6-digit — i.e. `size_agglvl = area_agglvl
  + 10` for the by-industry levels. Every code T3's crosswalk pulls is present:
  `321`/`311` (3-digit mfg, for durable `31`/nondurable `32`) at agglvl `25`;
  `1133` (Logging→sector `11`) at agglvl `26`; 2-digit sectors at `24`;
  supersectors at `23`.
- **Design unlocked — reuse `build_qcew_panel`, don't reimplement.** Remap size
  agglvl `−10` and feed straight through the T3-tested `build_qcew_panel`, run
  **per `size_code`** (a correctness requirement — its `_VINTAGE_GROUP` has no
  `size_code`, so one combined call would sum across size classes), then attach
  `size_code` → that is the `native` frame `build_size_class_panel` consumes. No
  new pull-maps. The size hierarchy is then *identical by construction* to the
  levels hierarchy.
- **Duplicate family at agglvl `61`–`64`** carries the same industry codes as
  `21`–`24` (e.g. `1012` at both `23` and `63`) — including both would
  double-count. The `−10` remap drops it for free (`61`–`64`→`51`–`54`, never
  pulled by `build_qcew_panel`), but the fetch should filter to `21`–`28`
  explicitly.
- **Disclosure suppression — contained, policy = drop `'N'`.** `disclosure_code`
  is `''` (disclosed) or `'N'` (withheld, value zeroed in the open API).
  Suppression is **zero at agglvl `23`/`24`** (supersectors, 2-digit sectors,
  hence domains/`05`) and appears only at `25`/`26` (3-digit/4-digit) — affecting
  only sectors `31`/`32`/`11`, the §10 best-effort tier (`1133` is even absent in
  the largest size bucket). **Policy:** keep only disclosed cells
  (`disclosure_code ∈ {'', null}`) before crosswalking — never sum a withheld
  cell as a real zero. Hard-gate levels stay exact; `31`/`32`/`11` size
  breakdowns are disclosed-only (a small, flagged undercount). Log the disclosure
  distribution + dropped-row counts per year during the real fetch.

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

## QCEW per-industry vintaging — ✅ RESOLVED (T0.5, decision A, 2026-06-15)

Investigated and settled. **Per-industry QCEW vintages do not exist** as a public
source:

- BLS publishes QCEW revision history **only at the national total** —
  `qcew-revisions.csv` and the entire `bls.gov/cew/revisions/` page are area×field
  with no industry/size breakdown (verified by fetching the page: the sole data
  file is the total-level CSV; "industry" mentions are nav chrome). The open-data
  API serves current values only; historical singlefiles are overwritten.
- Confirmed in the store: only `industry_code='00'` carries rev 0–4; **every
  per-industry private code is rev-0** (`n_rev=1`). The pipeline hardcodes
  `industry_code='00'` for the revisions CSV and tags all bulk per-industry rows
  `revision=0`.
- The `qcew_vintages.parquet` / `load_qcew_vintages` / `ingest_qcew` path (the
  `QCEW_VINTAGE_SCHEMA` mechanism that *would* hold per-industry revisions) is a
  **dead stub** — nothing writes it, no live callers; the file exists in neither
  repo. The live QCEW data comes from `nfp_vintages/processing/qcew_bulk.py`
  (per-industry rev-0 + national-`00` rev 0–4).

**Decision A:** store per-industry QCEW as a single `revision=0` row (current
value); carry revision uncertainty **model-side** via the `QCEW_REVISIONS` noise
schedule (M3/M12 multipliers; nominal depth Q1=4/Q2=3/Q3=2/Q4=1). No per-industry
reconstruction. Trade-off accepted: the rev-0 row holds a benchmarked level tagged
at its initial publication date (a small as-of lookahead). **Rejected B**
(proportional synthesis from total-level ratios — manufactures unpublished data,
assumes uniform per-industry revision). Spec §5 corrected to match.

So T5's QCEW path is simply **acquire current private slices (API) → T3 crosswalk
→ rev-0 rows** — no vintage-reconstruction step, no separate gate.

## Go/no-go

| Task | Verdict |
|---|---|
| **T2** CES builder | **GO** — `cesvinall` suffices; build per-benchmark `(2,1)` rows + correct `vintage_date`. |
| **T4** size-class | **GO** — source = QCEW size endpoint (Q1, national=private). |
| **T5** acquire/orchestration | **GO** — API slices for levels; QCEW = current rev-0 → T3 crosswalk (T0.5 resolved A, no reconstruction). Blocked only on T2. |
| **T6** gates | unaffected; refine the history-consistency gate for per-benchmark `(2,1)`. |
