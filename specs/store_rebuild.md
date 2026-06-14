# Vintage-store rebuild — private, NSA, vintage-aware

**Status:** design spec for a clean rebuild of the vintage store. Supersedes the
"irreplaceable / never rebuild" premise in the root `CLAUDE.md`. Companions:
`store_audit_findings.md` (current state), `ces_qcew_industry.md` (the QCEW→CES
private crosswalk), `size_classes.md` (the size-class dimension),
`ces_growth_convention.md` (vintage/print semantics), `bloomberg_consensus.md`
(the downstream `00` target).

---

## 1. Premise — the store is replaceable; rebuild it deliberately

The store holds only **public CES + QCEW** data, reconstructable from the BLS
triangular/bulk files. The historical core (≤2023) is complete and correct; the
live-capture era has gaps (2024-12 missing the benchmark reprint; 2025-12 missing
the second/final prints — see `store_audit_findings.md`). So:

- **Retire "never rebuild in place."** Rebuilds run to a **scratch prefix**
  (`s3://alt-nfp/store-rebuild`), are **validated**, then **promoted** to canonical
  as a deliberate cutover. The canonical store never takes an unvalidated in-place
  build.
- **Parity-vs-frozen-reference is retired** for this layer — the rebuild
  *intentionally* diverges (private-only, NSA, QCEW-mapped). A1/A2 goldens are
  **re-baselined** against the rebuilt store, not held.
- **Everything is materialized in the store** (no on-the-fly reconstruction): the
  QCEW→CES private series is computed at build time and stored.

---

## 2. Scope

**In scope.** A national, **not-seasonally-adjusted**, **private-only**,
vintage-aware store with two sources keyed on the **same CES private industry
codes**:

| `source` | role | content |
|---|---|---|
| `ces` | target / actuals | CES published **private** prints (1st/2nd/3rd) from triangular + bulk files |
| `qcew` | input | QCEW employment **mapped to CES private codes**, monthly, vintage-tagged |

**Coverage:** **2017-01 →** present (for now). **Units:** thousands of persons.

**Out of scope (handled elsewhere / deferred — see §11).** Government and total
codes; seasonal adjustment; geography beyond national; live capture; the provider
store (separate repo/object-store).

---

## 3. Industry — private only

Use the CES private hierarchy from `ces_qcew_industry.md` §1 (CES `industry_code`
convention as already used in `source=ces`):

| Level | codes |
|---|---|
| Top aggregates | `05` total private, `06` goods-producing, `08` private service-providing |
| Supersectors | `10 20 30 40 50 55 60 65 70 80` |
| Sectors | `11 21 22 23 31 32 42 44 48 51 52 53 54 55 56 61 62 71 72 81` |

**Dropped entirely:** `00` (total nonfarm), `07` (service-providing, folds in
government), `90/91/92/93` (government). The model nowcasts **private**; the
`00` headline is reconstructed **downstream** as private + government (a separate
step, outside this store — there is no payroll-provider signal for government).

The CES-internal vs NAICS-2-digit sector reconciliation (`ces_qcew_industry.md`
§2) and the two structural sums — Mining-and-Logging (`10` = `21` + Logging
`1133`) and Durable/Nondurable (`31`/`32` from 3-digit subsectors) — are applied
at build time per that spec.

---

## 4. The two sources

### 4.1 `source=ces` — CES published private prints (target/actuals)

- **Triangular files (`cesvinall`)** recover the real-time **1st / 2nd / 3rd
  prints** → the `(revision, benchmark_revision)` cohort structure
  (`ces_growth_convention.md`). **Bulk files** carry the **benchmarked** levels.
- **Benchmark status is by date.** Whatever is in the bulk files is benchmarked,
  **except the current tail**: months **2026-01 → 2026-05** are not yet
  benchmarked (the next annual benchmark lands **2027-01-12**) → tagged
  `benchmark_revision = 0`.
- Private `industry_code`s only (§3); `00/07/90…` never written.

### 4.2 `source=qcew` — QCEW mapped to CES codes (input)

- **Crosswalk applied at build time** (`ces_qcew_industry.md`): `own_code=='5'`,
  national (`area_fips=='US000'`), aggregate the listed `(industry_code,
  agglvl)` cells into the CES private codes of §3. `industry_code` is therefore
  the **CES code** (`05/06/08/10-80/sector`); **raw NAICS provenance is dropped**.
- **Monthly levels:** use `month1_emplvl`, `month2_emplvl`, `month3_emplvl` of each
  quarter → three monthly rows, each **÷ 1000** (persons → CES-comparable
  thousands). Sum the measure column over cells; never sum a rate.
- **Per-vintage aggregation (critical, see §5).** Each CES-code aggregate is summed
  only over leaf cells sharing one QCEW `(revision, vintage_date)`; the result
  inherits that `(revision, vintage_date)`.

---

## 5. Vintage & as-of semantics

Both sources are vintage-aware so the real-time censoring (`vintage_store.py`
Layer-1 + `panel_adapter` Layer-2) works unchanged.

- **CES:** `(revision ∈ {0,1,2}, benchmark_revision ∈ {0,1}, vintage_date)` per the
  cohort convention; growth is computed within a consistent cohort downstream.
- **QCEW:** carries its **own** vintage structure — `revision 0-4` with
  quarter-dependent depth (**Q1=4, Q2=3, Q3=2, Q4=1**), a distinct `vintage_date`
  per `(ref_date, revision)`, and `benchmark_revision = 0` always (QCEW has no
  annual benchmark). The crosswalk sum **never crosses a QCEW vintage** — mixing
  vintages inside one aggregate would corrupt the level/growth, exactly as mixing
  ordinals would for CES.
- **Publication lag.** QCEW lands ~5–6 months after the reference quarter; the
  reconstructed series inherits QCEW's `vintage_date`, so as-of knowability is
  honored automatically.

---

## 6. Representation — NSA, log-growth

The store holds **NSA levels** (`employment`, thousands). Growth is computed
per-cohort downstream and the model observes **`log(growth)`** — per-period
`log(emp_t / emp_{t-1})` (the expected choice) — **or**, alternatively,
`log(cumprod(growth))` (reconstructed log levels). The two are mutually
exclusive; the store commits to neither (it stores levels). The **SA `00`
headline** is composed downstream from the private nowcast + government.

---

## 7. Schema

`source=ces` and `source=qcew` share one schema (Hive-partitioned by
`(source, seasonally_adjusted)`; `seasonally_adjusted=false` only):

| column | type | notes |
|---|---|---|
| `geographic_type` | str | `national` (only, for now) |
| `geographic_code` | str | `00` |
| `industry_type` | str | `domain` / `supersector` / `sector` |
| `industry_code` | str | **private CES codes only** (§3) |
| `ref_date` | date | day-12 convention |
| `vintage_date` | date | per §5 |
| `revision` | u8 | CES 0-2; QCEW 0-4 |
| `benchmark_revision` | u8 | CES 0/1; QCEW 0 |
| `employment` | f64 | thousands |
| `size_class_type` | str? | `size_classes.md` — null for CES + QCEW Q2/Q3/Q4 |
| `size_class_code` | str? | **singular**; null where `size_class_type` is null |

**Removed vs today:** raw-NAICS provenance, government/total `industry_code`s.
**Added:** `size_class_type`, `size_class_code`.

---

## 8. Size-class dimension

Per `size_classes.md`: `size_class_type ∈ {total, small, medium, large}` (1/3/5/9
nested buckets over native QCEW `size_code` 1–9), `size_class_code` singular.
Lives in `source=qcew`, populated **only for QCEW Q1 — ref_dates with month ∈
{01, 02, 03}** (size is a Q1 establishment-size product, assigned by March
employment); **null for CES and for QCEW Q2/Q3/Q4** (months 04–12).

**Full cross-product on Q1.** For those Q1 months, every industry cell carries
every size scheme — the rows are the cross-product `industry_type ×
size_class_type`. A given `industry_code` at a given `(ref_date∈Q1, rev,
vintage_date)` expands to its `total` (`'0'`), `small` (`'1'-'3'`), `medium`
(`'1'-'5'`), and `large` (`'1'-'9'`) rows: `total` is the all-sizes value, the
rest the nested breakdowns. `geographic_type == 'national'` for now. Size-class
rows inherit their industry parent's QCEW `(rev, vintage_date)`.

The **provider store** (separate repo/object-store) bins its microdata to the
*same* scheme (March/third-month employment) so the two line up.

---

## 9. Build pipeline (to scratch)

1. **Acquire** (public): CES triangular (`cesvinall`) + CES bulk current; QCEW bulk
   (all quarters, national, private). 2017+.
2. **CES** → emit private prints with `(rev, bmr, vintage_date)`; tag the current
   tail (≥2026-01) `bmr=0`.
3. **QCEW** → per `(rev, vintage_date)`: apply the `ces_qcew_industry` crosswalk
   (`own_code=='5'`), explode `month1/2/3_emplvl` to monthly, ÷1000, write the
   CES-coded private aggregates (+ the Q1 size cross-product) with QCEW's
   `(rev, vintage_date)`.
4. **Write** both to `NFP_STORE_URI=s3://alt-nfp/store-rebuild` (the canonical guard
   refuses `…/store`; `is_canonical_store` from the audit branch).

## 10. Validation & promotion

**Acceptance gates** (replace frozen-reference parity):
- **History consistency:** rebuilt `source=ces` private prints match the current
  store's private rows where both exist (≤2023, the known-good core).
- **Gap fill:** the 2024-12 / 2025-12 December cohorts are now complete.
- **Reconstruction accuracy:** rebuilt `source=qcew` `05/06/08/…` vs *published*
  CES private at **benchmark months / annual averages**, within the residuals
  `ces_qcew_industry.md` §8 predicts (sample-vs-census, religious orgs, etc.) —
  not exact equality.
- **Vintage integrity:** `_validate_censored_selection`-style fail-fast checks pass
  on an as-of slice (no dups, no cross-vintage sums, no nulls/zeros).

**Promotion:** once gates pass, cut over deliberately (repoint `NFP_STORE_URI` to
the validated prefix, or copy scratch→canonical with the explicit
`--allow-canonical` escape hatch). Document the cutover; keep the prior canonical
snapshot until the new one is confirmed in the model.

---

## 11. Deferred / future

- **Live capture (D4):** cron a daily read of the BLS feed (`https://www.bls.gov/feed/`)
  to keep the frontier fresh. Until then a **stale store is acceptable** (rebuild
  periodically). `_fetch_ces_releases` is now import-fixed (on `main`) but not yet
  load-bearing.
- **NAICS-vintage drift:** use the NAICS-2022 crosswalk for all years initially
  (2017+ coverage limits exposure); add vintage-aware crosswalks later.
- **Geography** beyond national; **seasonal adjustment** (downstream, if ever).
  **Births-deaths:** unchanged by this rebuild.

---

## 12. Open decisions

None outstanding — §8 is resolved (full `industry_type × size_class_type`
cross-product, QCEW Q1 only, national). The spec is ready for an implementation
plan.
