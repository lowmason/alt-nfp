# Replicating CES National Published (Private) Series from QCEW

**Scope:** Reconstruct the CES National employment estimates for all *private* published industries —
3 top aggregates (`05`, `06`, `08`), 10 supersectors, and 20 sectors — using QCEW industry,
ownership, and aggregation-level codes.

**Source of truth used to write this spec:** BLS `ce.industry` (CES National industry/NAICS map),
QCEW *High-Level Industry Titles & Crosswalk*, QCEW *Ownership Titles (NAICS)*, and QCEW
*Aggregation Level Codes (NAICS)*. All NAICS-vintage facts below are stated for NAICS 2022.

---

## 1. What's in scope

| Level | CES codes |
|---|---|
| Top aggregates | `05` Total private, `06` Goods-producing, `08` Private service-providing |
| Supersectors | `10 20 30 40 50 55 60 65 70 80` |
| Sectors (NAICS 2-digit IDs) | `11 21 22 23 31 32 42 44 48 51 52 53 54 55 56 61 62 71 72 81` |

**Out of scope and why:** `00` (Total nonfarm) and `07` (Service-providing) both fold in government,
so they cannot be reproduced from a private-only (`own_code == '5'`) pull. The government supersector
(`90`) and its sectors are excluded by request.

---

## 2. `[DOUBLE CHECK]` results — sector list reconciliation

The requested sector identifiers are **NAICS 2-digit codes** (the same convention as the `sector`
column in `industry_codes.csv`). These are **not** the CES internal sector codes shown in the prompt's
`CES SECTOR` block — the two systems disagree inside Trade/Transportation/Utilities. Reconciliation:

| Requested (NAICS 2-dig) | CES internal sector | CES 8-digit industry_code | Title |
|---|---|---|---|
| `42` | `41` | `41420000` | Wholesale trade |
| `44` | `42` | `42000000` | Retail trade (NAICS 44–45) |
| `48` | `43` | `43000000` | Transportation and warehousing (NAICS 48–49) |
| `22` | `22` | `44220000` | Utilities |

Two requested codes need explicit handling (confirmed against `ce.industry`):

- **`11` is not a CES sector.** The *only* part of NAICS 11 that CES carries is **Logging (NAICS 1133)**,
  published inside the Mining-and-Logging supersector (`ce.industry` row `10113300`). Treat `11` ≡ Logging.
- **`31` / `32` (Durable / Nondurable) are CES conventions, not NAICS sectors.** They split NAICS 31–33
  by 3-digit subsector and must be aggregated (Section 6.4).

Everything else in the list maps 1:1 to a NAICS sector.

---

## 3. The three QCEW code systems

### 3.1 Ownership — always filter `own_code == '5'`

| Code | Title |
|---|---|
| 0 | Total Covered |
| 1 | Federal Government |
| 2 | State Government |
| 3 | Local Government |
| 5 | **Private** |
| 8 | Total Government |
| 9 | Total U.I. Covered (excl. Federal) |

The `own_code == '5'` filter is doing real work: it is what removes the government-owned establishments
that sit *inside* otherwise-private NAICS sectors — U.S. Postal Service (in NAICS 48–49), public schools
(61), public hospitals (62), and municipal utilities (22). CES routes those to its government series; the
ownership filter reproduces that split automatically. **No separate carve-out is needed for them.**

### 3.2 Aggregation level — national, NAICS-coded

Filter national records (`area_fips == 'US000'`) at the level matching the cell being built:

| `agglvl_code` | Level |
|---|---|
| 11 | National, total — by ownership |
| 12 | National, by Domain — by ownership |
| 13 | National, by Supersector — by ownership |
| 14 | National, by NAICS Sector — by ownership |
| 15 | National, by NAICS 3-digit — by ownership |
| 16 | National, by NAICS 4-digit — by ownership |

### 3.3 Industry codes used here

- **QCEW high-level supersector aggregates:** `1012 1013 1021 1022 1023 1024 1025 1026 1027` (agglvl 13).
- **NAICS sectors** (agglvl 14): 2-digit, except the three range-sectors coded with a hyphen —
  `31-33`, `44-45`, `48-49`.
- **NAICS 3-digit** (agglvl 15) for the manufacturing durable/nondurable split.
- **NAICS 4-digit** `1133` (agglvl 16) for Logging.

> The high-level **domain** codes `101` (Goods-Producing) and `102` (Service-Providing) are **deliberately
> not used** — see Section 4.

---

## 4. Core principle + the two structural exceptions

Every series: `own_code == '5'`, national, summed over the listed `(industry_code, agglvl_code)` cells.
A cell is a **direct pull** when one QCEW industry code equals the CES cell, and a **sum** otherwise.
Two things force sums:

**Exception A — Agriculture (affects `10`, and `06`/`05` above it).**
QCEW's *Natural Resources and Mining* aggregate (`1011`, agglvl 13) is defined as **NAICS 11 + 21**, so it
carries *all* UI-covered agriculture (crop, animal, forestry support, fishing/hunting). CES carries only
**Logging** from NAICS 11. Therefore **do not use `1011`** (and do not use `101` / total `10`, which inherit
it). Build Mining-and-Logging from `Mining (21)` + `Logging (1133)`.

**Exception B — Durable/Nondurable.** No NAICS sector exists for these; sum the 3-digit subsectors (6.3).

A secondary reason to avoid the domain aggregates: `102` (and total `10`) also fold in **Unclassified**
(`1029`) and **Public Administration** (`1028`), neither of which belongs in a CES private aggregate.
Building `08` from its seven named service supersectors sidesteps both.

---

## 5. Mapping — top aggregates

All `own_code == '5'`. These are pure roll-ups (no new leaf pulls):

| CES | Title | Definition |
|---|---|---|
| `06` | Goods-producing | `10` + `20` + `30` |
| `08` | Private service-providing | `40` + `50` + `55` + `60` + `65` + `70` + `80` |
| `05` | Total private | `06` + `08` |

---

## 6. Mapping — supersectors and sectors

### 6.1 Supersectors (`own_code == '5'`)

| SS | Title | QCEW pull | agglvl | Method |
|---|---|---|---|---|
| `10` | Mining and logging | `21` + `1133` | 14 + 16 | **sum** (Exception A) |
| `20` | Construction | `1012` *(= NAICS 23)* | 13 | direct |
| `30` | Manufacturing | `1013` *(= NAICS 31–33)* | 13 | direct |
| `40` | Trade, transportation, and utilities | `1021` | 13 | direct |
| `50` | Information | `1022` *(= NAICS 51)* | 13 | direct |
| `55` | Financial activities | `1023` | 13 | direct |
| `60` | Professional and business services | `1024` | 13 | direct |
| `65` | Education and health services | `1025` | 13 | direct |
| `70` | Leisure and hospitality | `1026` | 13 | direct |
| `80` | Other services | `1027` *(= NAICS 81)* | 13 | direct |

Only `10` is special. For every other supersector the QCEW supersector aggregate at agglvl 13 already
equals the sum of its private NAICS sectors, so a single pull suffices. (Equivalently, each could be summed
from the member sectors in 6.2 — identical result.)

### 6.2 Sectors (`own_code == '5'`)

The `id` column is the NAICS 2-digit identifier from the request / `industry_codes.csv`.

| id | Title | QCEW `industry_code` | agglvl | Method |
|---|---|---|---|---|
| `11` | Logging | `1133` | 16 | direct (4-digit) |
| `21` | Mining, quarrying, oil & gas extraction | `21` | 14 | direct |
| `22` | Utilities | `22` | 14 | direct |
| `23` | Construction | `23` | 14 | direct |
| `31` | Durable goods | *(see 6.3)* | 15 | **sum** |
| `32` | Nondurable goods | *(see 6.3)* | 15 | **sum** |
| `42` | Wholesale trade | `42` | 14 | direct |
| `44` | Retail trade | `44-45` | 14 | direct |
| `48` | Transportation and warehousing | `48-49` | 14 | direct |
| `51` | Information | `51` | 14 | direct |
| `52` | Finance and insurance | `52` | 14 | direct |
| `53` | Real estate and rental and leasing | `53` | 14 | direct |
| `54` | Professional, scientific, and technical services | `54` | 14 | direct |
| `55` | Management of companies and enterprises | `55` | 14 | direct |
| `56` | Administrative, support, and waste management | `56` | 14 | direct |
| `61` | Private educational services | `61` | 14 | direct |
| `62` | Health care and social assistance | `62` | 14 | direct |
| `71` | Arts, entertainment, and recreation | `71` | 14 | direct |
| `72` | Accommodation and food services | `72` | 14 | direct |
| `81` | Other services | `81` | 14 | direct |

`23`, `51`, and `81` are single-NAICS-sector supersectors, so the sector value equals the supersector value
(`1012` / `1022` / `1027`). Either pull is valid.

### 6.3 Manufacturing 3-digit split (NAICS 2022)

Pull each 3-digit code at agglvl 15, `own_code == '5'`, then sum within group.

| CES sector | NAICS 3-digit subsectors |
|---|---|
| `31` Durable goods | `321 327 331 332 333 334 335 336 337 339` |
| `32` Nondurable goods | `311 312 313 314 315 316 322 323 324 325 326` |

Check: the union is all 21 manufacturing subsectors = NAICS 31–33, so `31` + `32` = `30` (= `1013`).
Validated against the manufacturing subsectors present in `industry_codes.csv`.

---

## 7. Build order

1. **Leaf pulls** (`own_code == '5'`):
   - all 18 single-code sectors at agglvl 14;
   - `1133` at agglvl 16 (Logging);
   - the 21 manufacturing 3-digit codes at agglvl 15.
2. **Compose sectors:** `31` = Σ durable, `32` = Σ nondurable. (The other 18 are leaves.)
3. **Compose supersectors:** `10` = `21` + `1133`; `30` = `31` + `32` (or pull `1013`); `20 40 50 55 60 65 70 80`
   are single QCEW supersector pulls (agglvl 13).
4. **Compose top aggregates:** `06` = `10`+`20`+`30`; `08` = `40`+`50`+`55`+`60`+`65`+`70`+`80`; `05` = `06`+`08`.

Sum the QCEW measure columns (e.g. `month3_emplvl` for a CES-comparable employment level, or
`annual_avg_emplvl` for annual averages); never sum a rate.

---

## 8. Coverage caveats (CES vs QCEW universe)

These rules align the *classification*. Residual differences remain because CES is a sample survey
benchmarked annually to QCEW, not QCEW itself:

- **Religious organizations (NAICS 8131):** CES *Other services* (`81`) omits 8131; QCEW NAICS 81 includes
  whatever is UI-covered. Expect a small positive residual in `81` / `80` / `08` / `05`.
- **Private households (NAICS 814):** effectively excluded by both (UI-exempt); not a material gap.
- **Agriculture beyond Logging:** excluded by construction (Exception A).
- **Benchmark vs sample:** between March benchmarks CES is model/sample-based, so monthly QCEW-derived sums
  will not match CES month-to-month. Validate at the benchmark month or on annual averages and expect small
  residuals; do not expect exact equality.

---

## 9. Implementation sketch (Polars)

```python
QCEW_OWN_PRIVATE = '5'
QCEW_AREA_US = 'US000'

AGGLVL = {'domain': '12', 'supersector': '13', 'sector': '14', 'naics_3': '15', 'naics_4': '16'}

# Leaf-level QCEW pulls keyed by NAICS 2-digit id -> (industry_codes, agglvl)
SECTOR_PULLS = {
    '11': (['1133'], '16'),                                                        # Logging only
    '21': (['21'], '14'),   '22': (['22'], '14'),   '23': (['23'], '14'),
    '31': (['321', '327', '331', '332', '333', '334', '335', '336', '337', '339'], '15'),  # Durable
    '32': (['311', '312', '313', '314', '315', '316', '322', '323', '324', '325', '326'], '15'),  # Nondurable
    '42': (['42'], '14'),   '44': (['44-45'], '14'), '48': (['48-49'], '14'),
    '51': (['51'], '14'),   '52': (['52'], '14'),   '53': (['53'], '14'),
    '54': (['54'], '14'),   '55': (['55'], '14'),   '56': (['56'], '14'),
    '61': (['61'], '14'),   '62': (['62'], '14'),
    '71': (['71'], '14'),   '72': (['72'], '14'),   '81': (['81'], '14'),
}

# Supersector -> usable single QCEW aggregate (None forces a member sum) + member sector ids
SUPERSECTOR = {
    '10': {'qcew_code': None,   'sectors': ['11', '21']},          # 1011 over-includes agriculture
    '20': {'qcew_code': '1012', 'sectors': ['23']},
    '30': {'qcew_code': '1013', 'sectors': ['31', '32']},
    '40': {'qcew_code': '1021', 'sectors': ['42', '44', '48', '22']},
    '50': {'qcew_code': '1022', 'sectors': ['51']},
    '55': {'qcew_code': '1023', 'sectors': ['52', '53']},
    '60': {'qcew_code': '1024', 'sectors': ['54', '55', '56']},
    '65': {'qcew_code': '1025', 'sectors': ['61', '62']},
    '70': {'qcew_code': '1026', 'sectors': ['71', '72']},
    '80': {'qcew_code': '1027', 'sectors': ['81']},
}

DOMAIN = {
    '06': ['10', '20', '30'],
    '08': ['40', '50', '55', '60', '65', '70', '80'],
    '05': ['06', '08'],
}

def pull(df, industry_codes, agglvl):
    return df.filter(
        pl.col('own_code').eq(QCEW_OWN_PRIVATE)
        .and_(pl.col('area_fips').eq(QCEW_AREA_US))
        .and_(pl.col('agglvl_code').eq(agglvl))
        .and_(pl.col('industry_code').is_in(industry_codes))
    )
```

Roll-ups then sum the measure column over the pulled rows: `SECTOR_PULLS` → sectors,
`SUPERSECTOR` (aggregate code *or* member sectors) → supersectors, `DOMAIN` → top aggregates.
