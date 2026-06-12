# QCEW → CES Mapping — TODO

## Objective

Extract national employment from QCEW quarterly bulk files and map to CES
sector/supersector codes for use as a benchmark comparison series.

---

## 1. Government Employment (by ownership)

### Source

- **URL pattern**: `https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip`
- **Layout**: [NAICS-based quarterly layout](https://www.bls.gov/cew/about-data/downloadable-file-layouts/quarterly/naics-based-quarterly-layout.htm)
- **Note**: Singlefiles exclude title fields (fields 9–13); codes must be joined
  to reference tables for labels.

### Filters

| Field            | Value    | Meaning                                  |
|------------------|----------|------------------------------------------|
| `area_fips`      | `US000`  | National                                 |
| `own_code`       | `1`      | Federal Government                       |
| `own_code`       | `2`      | State Government                         |
| `own_code`       | `3`      | Local Government                         |
| `own_code`       | `8`      | Total Government (sum of 1 + 2 + 3)     |
| `industry_code`  | `10`     | Total, all industries                    |
| `agglvl_code`    | `11`     | National, Total — by ownership sector    |

### CES Mapping

| QCEW `own_code` | QCEW Ownership     | CES Code | CES Label            |
|------------------|--------------------|----------|----------------------|
| `1`              | Federal Government | 91       | Federal Government   |
| `2`              | State Government   | 92       | State Government     |
| `3`              | Local Government   | 93       | Local Government     |
| `8`              | Total Government   | 90       | Government (supersector) |

**Why ownership, not NAICS?** Government employment spans many NAICS industries
(education 61, healthcare 62, public administration 92, etc.). The QCEW
ownership code captures *all* workers employed by government entities regardless
of their NAICS classification, which aligns with how CES defines its government
supersector.

---

## 2. Manufacturing: Durable / Nondurable Split

### The Problem

CES publishes sector 31 (Durable goods) and sector 32 (Nondurable goods) as
distinct series under supersector 30 (Manufacturing). QCEW has no pre-built
durable/nondurable aggregation — its supersector code `1013` covers all of
manufacturing (NAICS 31-33) without the split.

### Approach

Pull NAICS 3-digit subsectors from QCEW and aggregate to match the CES
durable/nondurable grouping.

### Filters

| Field            | Value    | Meaning                                         |
|------------------|----------|-------------------------------------------------|
| `area_fips`      | `US000`  | National                                        |
| `own_code`       | `5`      | Private (CES manufacturing = private sector)    |
| `agglvl_code`    | `15`     | National, by NAICS 3-digit, by ownership sector |
| `industry_code`  | *(see mapping below)* | NAICS 3-digit subsectors           |

### CES Sector 32 — Nondurable Goods

| NAICS | Industry                                        |
|-------|-------------------------------------------------|
| 311   | Food manufacturing                              |
| 312   | Beverage and tobacco product manufacturing      |
| 313   | Textile mills                                   |
| 314   | Textile product mills                           |
| 315   | Apparel manufacturing                           |
| 316   | Leather and allied product manufacturing        |
| 322   | Paper manufacturing                             |
| 323   | Printing and related support activities          |
| 324   | Petroleum and coal products manufacturing       |
| 325   | Chemical manufacturing                          |
| 326   | Plastics and rubber products manufacturing      |

### CES Sector 31 — Durable Goods (for completeness / validation)

| NAICS | Industry                                                    |
|-------|-------------------------------------------------------------|
| 321   | Wood product manufacturing                                  |
| 327   | Nonmetallic mineral product manufacturing                   |
| 331   | Primary metal manufacturing                                 |
| 332   | Fabricated metal product manufacturing                      |
| 333   | Machinery manufacturing                                     |
| 334   | Computer and electronic product manufacturing               |
| 335   | Electrical equipment, appliance, and component manufacturing|
| 336   | Transportation equipment manufacturing                      |
| 337   | Furniture and related product manufacturing                 |
| 339   | Miscellaneous manufacturing                                 |

### Validation

- Durable + Nondurable should equal QCEW `industry_code='1013'` (supersector
  Manufacturing) at `agglvl_code='13'` for `own_code='5'`
- Cross-check against CES published sector 31 and 32 levels

---

## Key Employment Fields

- `month1_emplvl`, `month2_emplvl`, `month3_emplvl` — monthly employment levels
  within each quarter (reference period: pay period including the 12th of each month)

## Tasks

### Shared

- [ ] Download and cache quarterly singlefiles (2003–present)
- [ ] Handle disclosure codes (`disclosure_code = 'N'`)
- [ ] Reshape monthly employment levels into a time series
      (each quarter yields 3 monthly observations)
- [ ] Output a clean panel: `year`, `month`, `ces_code`, `employment`

### Government (Section 1)

- [ ] Filter to government rows (`own_code` in `['1','2','3','8']`,
      `industry_code='10'`, `agglvl_code='11'`)
- [ ] Map `own_code` → CES sector/supersector codes (91, 92, 93, 90)
- [ ] Validate: `own_code='8'` totals ≈ sum of `own_code` 1+2+3
- [ ] Cross-check against published CES government levels

### Manufacturing Durable/Nondurable (Section 2)

- [ ] Filter to private manufacturing subsectors (`own_code='5'`,
      `agglvl_code='15'`, NAICS 3-digit codes per mapping)
- [ ] Aggregate subsectors into CES sector 31 (durable) and 32 (nondurable)
- [ ] Validate: sector 31 + 32 = QCEW `industry_code='1013'` at `agglvl_code='13'`
- [ ] Cross-check against published CES sector 31 and 32 levels

## Notes

- Data available from 2003 onward (NAICS-based files)
- `own_code='8'` may include a small number of tribal government workers not
  separately broken out in own_codes 1/2/3
- QCEW employment is an establishment count (jobs, not persons); same concept as CES