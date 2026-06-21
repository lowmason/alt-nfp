# Canonical store audit — findings (2026-06-14)

Read-only audit of `s3://alt-nfp/store` (MinIO stand-in for Bloomberg enterprise
S3), run to ground the rebuild decision. No writes. Grounds
[`store-replaceable-and-rebuild-backlog`] and supersedes the "irreplaceable /
ongoing capture" assumptions in the root `CLAUDE.md`.

## Structure & schema

- Partitions present: `source=ces` (`seasonally_adjusted` true + false),
  `source=qcew` (false only). **No `source=sae`.**
- CES and QCEW carry the **same 9 columns**: `geographic_type, geographic_code,
  industry_type, industry_code, ref_date, vintage_date, revision,
  benchmark_revision, employment`.
- Volume / currency: **CES 39,729 rows, 2003-01 → 2026-01**; **QCEW 691,110 rows,
  2003-01 → 2025-06**.
- CES `(revision, benchmark_revision)` population: `(0,0)`≈9.9k, `(1,0)`≈9.9k,
  `(2,0)`≈9.9k, `(2,1)`≈10.0k.
- QCEW revision depth by quarter (all `benchmark_revision=0`): **Q1=4, Q2=3,
  Q3=2, Q4=1** subsequent revisions (months 1-3 → rev 0-4, 4-6 → 0-3, 7-9 → 0-2,
  10-12 → 0-1).

## ① CES benchmark months — failure mode is MISSING ROWS, not wrong levels

National `'00'` December rows across all years:

- **2003–2023: complete and clean.** All four `(rev,bmr)` slots present; the
  ordinary second print `(1,0)` is always *distinct* from the benchmark reprint
  `(2,1)` (`r2b1 − r1b0` wedge ∈ [−104k, +78k]) — never a fanned duplicate.
  Triangular-derived; correct.
- **2024-12: missing the benchmark reprint `(2,1)`.** Present: `r0b0`=159,536 /
  `r1b0`=158,926 / `r2b0`=158,942. No `(2,1)` row.
- **2025-12: missing the second & final prints.** Only `r0b0`=159,526 and
  `r2b1`=158,497 — the §4c-i shadowing (matches `ces_growth_convention.md` §2's
  "Dec-2025 has no rev-1 row").
- **No IND-IMD-1 fanning anywhere.** Consistent with `_fetch_ces_releases` having
  been import-broken since repo inception (it never ran in v2). The live-capture
  era *dropped* rows rather than fanning wrong ones.
- **Labeling subtlety:** Dec-2024 `r1b0` (vintage 2025-02-10) was captured *after*
  the Feb-2025 benchmark, so it is a post-benchmark level tagged `bmr=0`. The
  rebuild must define how to label a second print whose release coincides with
  the annual benchmark.

## ② `size_class_type` / `size_class_codes` — ENTIRELY ABSENT (not null)

Neither column exists in the CES or QCEW stored schema. Rebuild must **add** them:
populated for QCEW **Q1** establishment-size-class data, null for CES and QCEW
Q2/Q3/Q4.

## ③ Provider `'00'` vs `'05'` — confirmed mislabeled

Providers are not in the store. `data/providers/g/g_provider.parquet` (84 rows,
from 2019-02; columns `geographic_*`, `industry_*`, `ref_date`, `employment`) is
labeled `national '00'` (total nonfarm) but represents **private** payroll. CES
already carries a `domain '05'` (total private) series, so the `'05'` target
exists. Rebuild should relabel provider national output to `'05'`.

## Strategic takeaway

The store is **reconstructable**: history (≤2023) is complete and correct, and the
live-era gaps (2024-12, 2025-12) would be filled correctly by a rebuild from
current triangular files + a fresh `alt-nfp current` (import now fixed on
`audit-independent`/`main`). Rebuild scope: regenerate cleanly to a scratch
prefix; **add** `size_class`; **relabel** providers `'05'`; **handle
benchmark-month `(rev,bmr)` labeling** deliberately; validate against the
correct historical rows before promoting.
