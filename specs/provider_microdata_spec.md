# Payroll Provider Microdata Analysis Spec

**Alt-NFP Nowcasting System — Provider Representativeness Diagnostics**

Version: 1.0 \| Date: 2026-03-26 \| Author: Lowell Mason

------------------------------------------------------------------------

## 1. Overview

This spec defines a standalone diagnostics pipeline for analyzing payroll provider microdata. The pipeline ingests pay-period-level records, constructs monthly employment snapshots, and produces diagnostic outputs assessing data quality, sample representativeness, and labor market dynamics.

### 1.1 Relationship to Alt-NFP

The alt_nfp model consumes QCEW-weighted national composites of cell-level provider signals (region x supersector). This pipeline operates **upstream** in a separate private repo, providing the analytical foundation for understanding provider data properties. It does not directly feed the model but informs:

-   Whether cell-level compositing weights need adjustment beyond QCEW shares
-   Which cells have sufficient coverage for reliable signals
-   How sample composition drift affects aggregate trends
-   Data quality exclusion criteria

### 1.2 Phase Structure

| Phase | Scope | Priority |
|---------------------|---------------------|------------------------------|
| 1 | Data ingestion, monthification, representativeness & quality diagnostics | High |
| 2 | Full employment dynamics, earnings analysis, survival curves | Medium |

### 1.3 Cell Grid

The canonical analysis grid is **4 Census regions x 11 BLS supersectors x 9 QCEW size classes = 396 cells**. Not all analyses require the full grid; most roll up to coarser slices.

------------------------------------------------------------------------

## 2. Raw Data Schema

One row per employee per pay period.

| Column | Type | Description |
|---------------------|------------------|---------------------------------|
| `ref_date` | Date | Reference month (BLS convention: 12th of month) |
| `clt_id` | String | Client (establishment) identifier |
| `clt_state` | String | Client state (FIPS or postal abbreviation) |
| `clt_zip` | String | Client ZIP code |
| `naics_code` | String | Client NAICS code (variable length: 2-6 digits) |
| `ee_id` | String | Employee identifier |
| `check_date` | Date | Pay check date |
| `run_date` | Datetime | Payroll run timestamp |
| `payroll_start` | Date | Pay period start |
| `payroll_end` | Date | Pay period end |
| `payroll_freq` | String | Pay frequency (weekly, biweekly, semimonthly, monthly) |
| `ee_state` | String | Employee work state |
| `ee_zip` | String | Employee ZIP code |
| `ee_start_date` | Date | Employee hire/start date |
| `ee_exit_date` | Date | Employee termination date (null if active) |
| `soc_code` | String | Standard Occupational Classification code |
| `full_part_time` | String | Full-time / part-time indicator |
| `ee_seasonal` | Boolean | Seasonal employee flag |
| `ee_salary_annual` | Float64 | Annualized salary (salaried employees) |
| `ee_hourly_rate` | Float64 | Hourly rate (hourly employees) |
| `ee_unit_rate` | Float64 | Piece/unit rate (unit-pay employees) |
| `ee_gross_income` | Float64 | Gross pay for this pay period |
| `ee_gross_income_ttm` | Float64 | Trailing 12-month gross income |
| `ee_net_income` | Float64 | Net pay for this pay period |
| `payroll_expense_tax` | Float64 | Employer-side payroll tax expense |
| `is_normalized` | Boolean | Whether pay has been normalized to calendar month |

------------------------------------------------------------------------

## 3. Hierarchy Definitions

### 3.1 Geography

Standard Census hierarchy, mapped from `clt_state`:

| Level | Code | Count | Example |
|-----------------|-----------------|-----------------|--------------------|
| State | FIPS 2-digit or postal abbrev | 51 | CA, NY, TX |
| Division | Census division code | 9 | Pacific, Mountain, ... |
| Region | Census region code | 4 | West (4), South (3), Midwest (2), Northeast (1) |

Mapping source: BLS/Census geographic crosswalk (same as `nfp_lookups/geography.py`).

### 3.2 Industry

NAICS-based hierarchy, mapped from `naics_code`:

| Level | Digits | Count | Example |
|------------------|------------------|------------------|--------------------|
| Subsector | 3-digit NAICS | \~100 | 541 (Professional Services) |
| Sector | 2-digit NAICS → CES sector code | 18 | 54 → CES sector 54 |
| Supersector | BLS supersector | 11 | 60 (Professional & Business Services) |
| Domain | BLS domain | 5 | 08 (Private Service-Providing) |

Mapping source: `nfp_lookups/industry.py`. Special cases:

-   **Manufacturing split**: 3-digit NAICS → durable (CES 31) / nondurable (CES 32) via `NAICS3_TO_MFG_SECTOR`
-   **Government**: not applicable (provider data is private-sector only)
-   **Trade/Transport**: NAICS 42/44-45/48-49 → CES supersector 40 (TTU)
-   **Truncation**: if `naics_code` has \>3 digits, truncate to 3 for subsector, 2 for sector

### 3.3 QCEW Establishment Size Classes

Assigned per client-month based on employee headcount at that client in that month.

| Size class | Employment range | QCEW label   |
|------------|------------------|--------------|
| 1          | 1-4              | Size class 1 |
| 2          | 5-9              | Size class 2 |
| 3          | 10-19            | Size class 3 |
| 4          | 20-49            | Size class 4 |
| 5          | 50-99            | Size class 5 |
| 6          | 100-249          | Size class 6 |
| 7          | 250-499          | Size class 7 |
| 8          | 500-999          | Size class 8 |
| 9          | 1000+            | Size class 9 |

**Assignment rule**: For each `(clt_id, ref_date)`, count distinct `ee_id` with nonzero gross pay in the reference period. Assign to the size class bin containing that count. Size class is **time-varying** — a client can migrate between classes as it grows or shrinks.

------------------------------------------------------------------------

## 4. Phase 1: Core Representativeness & Quality Diagnostics

### 4.1 Step 0: Monthification

**Purpose**: Convert pay-period-level records to monthly employment snapshots.

#### 4.1.1 Monthly Employment Definition

An employee is **employed at a client in month M** if any of the following hold:

1.  A pay period overlaps with the BLS reference period (the pay period containing the 12th of the month), **and** `ee_gross_income > 0`
2.  `ee_start_date <= ref_date` and (`ee_exit_date` is null or `ee_exit_date >= ref_date`)

Rule (1) is primary; rule (2) is fallback when pay-period records are missing but administrative dates confirm employment.

#### 4.1.2 Monthly Snapshot Schema

One row per client-employee-month:

| Column | Type | Description |
|---------------------|------------------|---------------------------------|
| `ref_date` | Date | First of month |
| `clt_id` | String | Client ID |
| `ee_id` | String | Employee ID |
| `clt_state` | String | Client state |
| `clt_zip` | String | Client ZIP |
| `naics_code` | String | Client NAICS |
| `ee_state` | String | Employee work state |
| `soc_code` | String | SOC code |
| `full_part_time` | String | FT/PT |
| `ee_seasonal` | Boolean | Seasonal flag |
| `ee_start_date` | Date | Hire date |
| `ee_exit_date` | Date | Exit date |
| `monthly_gross_pay` | Float64 | Sum of `ee_gross_income` for pay periods overlapping this month |
| `is_normalized` | Boolean | Whether pay was calendar-normalized |

#### 4.1.3 Client-Month Summary Schema

One row per client-month (aggregated from snapshots):

| Column | Type | Description |
|---------------------|------------------|---------------------------------|
| `ref_date` | Date | First of month |
| `clt_id` | String | Client ID |
| `clt_state` | String | Client state |
| `clt_zip` | String | Client ZIP |
| `naics_code` | String | Client NAICS |
| `employment` | Int64 | Distinct employee count |
| `size_class` | Int8 | QCEW size class (1-9) |
| `region` | String | Census region code |
| `division` | String | Census division code |
| `supersector` | String | BLS supersector code |
| `sector` | String | CES sector code |
| `domain` | String | BLS domain code |
| `total_gross_pay` | Float64 | Sum of monthly_gross_pay |
| `avg_gross_pay` | Float64 | Mean monthly_gross_pay |
| `median_gross_pay` | Float64 | Median monthly_gross_pay |
| `n_hires` | Int64 | Employees in current month not in prior month (at same client) |
| `n_separations` | Int64 | Employees in current month not in next month (at same client) |
| `n_continuing` | Int64 | Employees in both current and prior month |

### 4.2 Client Tenure & Churn

#### 4.2.1 Client Tenure Table

One row per client. Computed from the client-month summary.

| Column | Type | Description |
|---------------------|------------------|---------------------------------|
| `clt_id` | String | Client ID |
| `first_observed` | Date | Earliest ref_date |
| `last_observed` | Date | Latest ref_date |
| `tenure_months` | Int32 | Calendar months between first and last |
| `months_observed` | Int32 | Count of months with data (may differ if gaps) |
| `initial_emp` | Int64 | Employment in first observed month |
| `final_emp` | Int64 | Employment in last observed month |
| `avg_emp` | Float64 | Mean employment over tenure |
| `initial_size_class` | Int8 | Size class in first month |
| `final_size_class` | Int8 | Size class in last month |
| `clt_state` | String | Client state |
| `naics_code` | String | Client NAICS |
| `supersector` | String | BLS supersector |
| `region` | String | Census region |

**Birth identification**: A client is flagged as a **likely birth** if `ee_start_date` for the majority of its initial employees falls within 90 days of `first_observed`. This requires the monthly snapshot to carry `ee_start_date`.

**Death identification**: A client is flagged as a **potential death** if `last_observed` is \>3 months before the end of the sample period. Clients still active at sample end are **censored**.

#### 4.2.2 Monthly Client Entry/Exit

One row per month:

| Column              | Type    | Description                                    |
|-----------------|---------------|----------------------------------------|
| `ref_date`          | Date    | Month                                          |
| `n_active`          | Int64   | Active clients                                 |
| `n_entries`         | Int64   | New clients (first_observed = this month)      |
| `n_exits`           | Int64   | Departing clients (last_observed = this month) |
| `entry_rate`        | Float64 | n_entries / n_active                           |
| `exit_rate`         | Float64 | n_exits / n_active                             |
| `churn_rate`        | Float64 | (n_entries + n_exits) / n_active               |
| `net_client_change` | Int64   | n_entries - n_exits                            |
| `entry_emp`         | Int64   | Total employment at entering clients           |
| `exit_emp`          | Int64   | Total employment at exiting clients            |

Stratified versions by: `region`, `supersector`, `size_class`.

### 4.3 Composition Analysis

#### 4.3.1 Composition Shares

Time series of employment distribution across each dimension. One row per `(ref_date, dimension, category)`:

| Column       | Type    | Description                                           |
|---------------------|------------------|---------------------------------|
| `ref_date`   | Date    | Month                                                 |
| `dimension`  | String  | `'region'`, `'supersector'`, `'size_class'`           |
| `category`   | String  | Category code within dimension                        |
| `employment` | Int64   | Total employment in this cell                         |
| `share`      | Float64 | Employment share (sums to 1.0 within dimension-month) |
| `n_clients`  | Int64   | Number of clients in this cell                        |

#### 4.3.2 Composition Shift Index (CSI)

One row per `(ref_date, dimension)`:

```         
CSI_t = 0.5 * Σ_c |share_c,t - share_c,t-1|
```

Bounded \[0, 1\]. Values \>0.02 warrant investigation; \>0.05 indicate significant composition instability.

#### 4.3.3 QCEW Benchmarking

Compare provider composition shares against QCEW universe shares at each level of the cell grid. For each `(ref_date, region, supersector, size_class)`:

| Column           | Type    | Description                                 |
|------------------|---------|---------------------------------------------|
| `ref_date`       | Date    | Month (QCEW quarterly, carried forward)     |
| `region`         | String  | Census region                               |
| `supersector`    | String  | BLS supersector                             |
| `size_class`     | Int8    | QCEW size class                             |
| `provider_emp`   | Int64   | Provider employment in cell                 |
| `provider_share` | Float64 | Provider share of total provider employment |
| `qcew_emp`       | Int64   | QCEW employment in cell                     |
| `qcew_share`     | Float64 | QCEW share of total QCEW employment         |
| `coverage_ratio` | Float64 | provider_emp / qcew_emp                     |
| `share_ratio`    | Float64 | provider_share / qcew_share                 |
| `n_clients`      | Int64   | Provider client count in cell               |

**QCEW size class data**: available only for Q1 each year from the BLS QCEW size data files (URL pattern: `https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip`, filtered by size-class aggregation levels). For months outside Q1, carry forward the most recent Q1 size distribution. This is a known limitation — QCEW publishes size-stratified data only annually.

**Over/under-representation score**: `log(share_ratio)`. Positive = over-represented, negative = under-represented. A heatmap of this score across the cell grid is the primary representativeness diagnostic.

### 4.4 Vintage Analysis

#### 4.4.1 Vintage Shares

Clients are assigned to a **vintage cohort** by year of `first_observed`. For each `(ref_date, vintage_year)`:

| Column         | Type    | Description                            |
|----------------|---------|----------------------------------------|
| `ref_date`     | Date    | Month                                  |
| `vintage_year` | Int32   | Year client first appeared             |
| `n_clients`    | Int64   | Clients from this vintage still active |
| `employment`   | Int64   | Employment at these clients            |
| `emp_share`    | Float64 | Share of total employment              |

**Contamination flag**: if vintages from the most recent 12 months account for \>25% of total employment in any month, flag that month as potentially contaminated by sample composition effects.

#### 4.4.2 Vintage-Controlled Growth

Compute employment growth separately for: - **Continuing clients only** (tenure \>= 13 months, present in both current and year-ago month): isolates true economic dynamics - **All clients**: includes composition effects

The gap between these two growth rates quantifies composition contamination.

### 4.5 Employment Change Decomposition

Monthly decomposition into intensive and extensive margins:

| Column | Type | Description |
|---------------------|------------------|---------------------------------|
| `ref_date` | Date | Month |
| `total_change` | Int64 | Total employment change |
| `within_client` | Int64 | Change at continuing clients (present in both months) |
| `entry_contribution` | Int64 | Employment from entering clients |
| `exit_contribution` | Int64 | Employment lost from exiting clients (negative) |
| `within_share` | Float64 | within_client / total_change |

Stratified versions by: `region`, `supersector`, `size_class`.

**Key diagnostic**: if `within_share` is consistently \<0.7, aggregate trends are dominated by sample composition rather than economic dynamics.

### 4.6 Data Quality Flags

#### 4.6.1 Record-Level Flags

Applied to the monthly snapshot before aggregation:

| Flag | Condition | Action |
|-------------------|-------------------------------|-----------------------|
| `extreme_emp_change` | Client MoM employment change \>50% (and employment \>10) | Flag; investigate |
| `zero_employment` | Client-month with 0 employees despite having records | Flag; exclude from employment counts |
| `multi_client_employee` | Same `ee_id` at \>1 `clt_id` in same month | Flag; attribute to primary client (highest gross pay) |
| `negative_gross_pay` | `ee_gross_income < 0` | Flag; likely reversal/adjustment — exclude from earnings analysis |
| `exit_before_start` | `ee_exit_date < ee_start_date` | Flag; data error |
| `orphan_pay_period` | Pay period has no overlap with any BLS reference period | Flag; allocation logic needed |

#### 4.6.2 Client-Level Flags

| Flag | Condition | Action |
|-------------------|-------------------------------|-----------------------|
| `volatile_size_class` | Size class changes \>2 classes in a single month | Flag; investigate |
| `naics_missing` | `naics_code` is null or invalid | Flag; exclude from industry-stratified analysis |
| `state_missing` | `clt_state` is null | Flag; exclude from geography-stratified analysis |
| `single_month_client` | Client observed for exactly 1 month | Flag; likely data issue or transient client |

#### 4.6.3 Quality Summary

Monthly aggregate quality metrics:

| Metric                | Description                               |
|-----------------------|-------------------------------------------|
| `pct_flagged_records` | Share of employee-months with any flag    |
| `pct_multi_client`    | Share of employees at \>1 client          |
| `pct_naics_missing`   | Share of employment with missing NAICS    |
| `pct_state_missing`   | Share of employment with missing state    |
| `n_extreme_changes`   | Count of extreme employment change events |

### 4.7 Deliverables (Phase 1)

| Output | Format | Description |
|--------------------|--------------------|--------------------------------|
| Client-month summary | Parquet | Core analytical table (S4.1.3) |
| Client tenure table | Parquet | One row per client (S4.2.1) |
| Monthly entry/exit | Parquet | Aggregate + stratified (S4.2.2) |
| Composition shares | Parquet | By region, supersector, size_class (S4.3.1) |
| QCEW benchmarking | Parquet | Cell-level representativeness (S4.3.3) |
| Vintage shares | Parquet | Composition contamination (S4.4.1) |
| Employment decomposition | Parquet | Within vs. entry/exit (S4.5) |
| Quality flags | Parquet | Record + client level (S4.6) |
| Diagnostic report | HTML/PDF | Summary visualizations (see S4.8) |

### 4.8 Diagnostic Visualizations (Phase 1)

1.  **Representativeness heatmap**: `log(share_ratio)` across region x supersector, with size-class marginals
2.  **Coverage ratio time series**: provider employment / QCEW employment by supersector
3.  **Composition shift index**: CSI time series by dimension (region, supersector, size_class)
4.  **Vintage stacked area**: employment share by vintage cohort over time
5.  **Employment decomposition**: stacked bar of within-client vs. entry/exit contributions
6.  **Client churn rates**: entry/exit rate time series, stratified by size_class
7.  **Quality flag summary**: monthly flagged-record share time series

------------------------------------------------------------------------

## 5. Phase 2: Employment Dynamics & Earnings

Phase 2 builds on the Phase 1 monthly snapshot and adds analyses that require the full employee-level detail.

### 5.1 Gross Job Flows

Computed from the monthly snapshot by tracking `ee_id` presence at each `clt_id` across consecutive months.

| Metric | Definition |
|-------------------------------|-----------------------------------------|
| Hires | `ee_id` in month M at `clt_id` but not in month M-1 at same `clt_id` |
| Separations | `ee_id` in month M at `clt_id` but not in month M+1 at same `clt_id` |
| Continuing | `ee_id` in both month M and M-1 at same `clt_id` |
| Hire rate | Hires / Employment |
| Separation rate | Separations / Employment |
| Churn rate | (Hires + Separations) / Employment |
| Net growth rate | (Hires - Separations) / Employment |

Stratified by: `region`, `division`, `state`, `supersector`, `sector`, `domain`, `size_class`, and all two-way crosses of `{region, supersector, size_class}`.

**Validation targets**: BLS JOLTS (national, by supersector) for hire/separation rates.

### 5.2 Employment Growth Rates

| Metric         | Definition                          |
|----------------|-------------------------------------|
| MoM growth     | `(emp_t - emp_{t-1}) / emp_{t-1}`   |
| YoY growth     | `(emp_t - emp_{t-12}) / emp_{t-12}` |
| Log-difference | `ln(emp_t) - ln(emp_{t-1})`         |

Computed at every level of each hierarchy dimension and for the full cell grid (region x supersector x size_class). YoY is the preferred trend metric (removes seasonality). MoM log-difference is comparable to the alt_nfp model's growth measure.

### 5.3 Earnings Distribution

Computed from the monthly snapshot using `monthly_gross_pay` for employees with `is_normalized = true` (or all, with a flag).

| Metric                   | Description                           |
|--------------------------|---------------------------------------|
| Mean monthly gross pay   | Arithmetic mean                       |
| Median monthly gross pay | P50                                   |
| P10, P25, P75, P90       | Distribution percentiles              |
| Std dev                  | Standard deviation                    |
| CV                       | Coefficient of variation (std / mean) |
| YoY growth (mean)        | Year-over-year change in mean         |
| YoY growth (median)      | Year-over-year change in median       |

Stratified by: `supersector`, `size_class`, `full_part_time`.

**Validation target**: QCEW average weekly wage (quarterly, converted to monthly).

### 5.4 Client Survival Analysis

Kaplan-Meier survival curves by entry cohort (year of `first_observed`).

| Milestone | Metric                   |
|-----------|--------------------------|
| 6 months  | P(survive \>= 6 months)  |
| 12 months | P(survive \>= 12 months) |
| 24 months | P(survive \>= 24 months) |
| 36 months | P(survive \>= 36 months) |

Stratified by: `size_class`, `supersector`, `region`.

Right-censored at sample end. Cohort comparison reveals changes in provider onboarding quality or economic conditions affecting business survival.

### 5.5 Deliverables (Phase 2)

| Output                | Format   | Description                       |
|-----------------------|----------|-----------------------------------|
| Job flows             | Parquet  | Aggregate + stratified (S5.1)     |
| Growth rates          | Parquet  | All hierarchy levels (S5.2)       |
| Earnings distribution | Parquet  | Stratified percentiles (S5.3)     |
| Survival curves       | Parquet  | By cohort x stratification (S5.4) |
| Diagnostic report     | HTML/PDF | Phase 2 visualizations            |

------------------------------------------------------------------------

## 6. Implementation Notes

### 6.1 Technology

-   **Language**: Python 3.12
-   **Data manipulation**: Polars (consistent with alt_nfp)
-   **Package manager**: uv
-   **Visualization**: Matplotlib / Plotly for HTML reports
-   **Output format**: Parquet for data, HTML for reports

### 6.2 Performance Considerations

Pay-period-level data at scale (millions of employees x monthly x 7 years) is large. Key optimizations:

-   **Monthification is the bottleneck**: do it once, write to Parquet, everything downstream reads from the monthly snapshot
-   **Lazy evaluation**: use Polars LazyFrames for all aggregations
-   **Partitioning**: partition client-month summary by `ref_date` for efficient time-range queries
-   **Size class assignment**: compute once during monthification, store in client-month summary

### 6.3 NAICS Handling

Provider `naics_code` may be 2-6 digits, or missing. Rules:

1.  Truncate to 2 digits for sector assignment, 3 digits for subsector
2.  Map 2-digit NAICS to CES sector codes via `nfp_lookups/industry.py` crosswalk
3.  Sector → supersector → domain via existing hierarchy
4.  Missing NAICS: include in totals, exclude from industry-stratified analyses, flag in quality report

### 6.4 Multi-Job Employees

When `ee_id` appears at multiple `clt_id` in the same month:

-   **Employment counts**: attribute to primary client (highest `monthly_gross_pay`). This matches BLS convention (CES counts jobs, not workers, but avoids double-counting at the same establishment)
-   **Earnings**: use total across all clients for earnings distribution; use primary-client earnings for client-level aggregations
-   **Job flows**: track hires/separations at each client independently (an employee leaving one client for another generates both a separation and a hire)

### 6.5 QCEW Size Class Data Ingestion

QCEW size-stratified data requires downloading the Q1 annual files from BLS. The download function should:

1.  Fetch `{year}_qtrly_singlefile.zip` for each year
2.  Filter to Q1 rows with size-class aggregation levels (agglvl_code 73-81 for national by size, or equivalent state-level codes)
3.  Extract employment by `(own_code, industry_code, size_code)` and map to the provider hierarchy
4.  Output: `(year, region, supersector, size_class, qcew_employment)` for benchmarking

This extends the existing QCEW download in `nfp-vintages/download/qcew.py` with additional agglvl_codes.

------------------------------------------------------------------------

## 7. Interface Contract with Alt-NFP

This pipeline does not feed the model directly, but its outputs inform decisions about the compositing pipeline in `nfp-ingest`. The key interface points:

### 7.1 Coverage Assessment

The QCEW benchmarking output (S4.3.3) identifies cells where provider coverage is too thin for reliable signals. This informs `MIN_PSEUDO_ESTABS_PER_CELL` in `nfp_lookups` and the weight redistribution logic in `nfp_ingest/compositing.py`.

### 7.2 Composition Drift Detection

The CSI and vintage analysis outputs (S4.3.2, S4.4) inform whether the static QCEW-weighted compositing approach is sufficient or whether time-varying composition adjustment is needed.

### 7.3 Stable Panel Definition

The client tenure analysis (S4.2.1) and quality flags (S4.6) inform criteria for constructing the "stable panel" subset of provider data — clients with sufficient tenure and clean data — that should be used for model signal construction.

### 7.4 Birth/Death Signal Quality

The employment decomposition (S4.5) and birth/death identification (S4.2.1) inform whether the provider's birth-rate data (`birth_file` in ProviderConfig) captures genuine establishment dynamics or is contaminated by sample composition effects.

------------------------------------------------------------------------

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|-------------------|----------------------|--------------------------------|
| NAICS missing for large share of clients | Industry-stratified analyses unreliable | Report missing rate; fall back to geography x size only |
| Size class assignment sensitive to pay-period timing | Monthly employee count may differ from BLS reference week | Use BLS reference period overlap (pay period containing the 12th) as primary |
| QCEW size data only available Q1 | Stale benchmarks for Q2-Q4 | Carry forward Q1; flag staleness in output |
| Multi-job employees inflate headcount | Provider employment overstated vs. QCEW | Primary-client attribution rule; report multi-job rate |
| Client churn conflates economic dynamics with provider sales | Misattribution of sample evolution as economic signal | Vintage analysis + within-client decomposition isolate the effect |
| Short `naics_code` (2-digit) prevents subsector analysis | Can't map to manufacturing durable/nondurable split | Fall back to sector level; flag in quality report |

------------------------------------------------------------------------

## 9. Open Questions

1.  **Pay period allocation for split months**: When a pay period spans two calendar months, should gross pay be pro-rated by calendar days, or attributed entirely to the month containing the check date? Recommend: pro-rate by overlap days with `is_normalized = true` flag.

2.  **Employee state vs. client state**: For geographic assignment, use `clt_state` (establishment location, consistent with QCEW) or `ee_state` (work location)? Recommend: `clt_state` for QCEW comparability, with `ee_state` as secondary for remote-work analysis.

3.  **SOC code utilization**: The raw data includes `soc_code`. Phase 2 could add occupational stratification, but this is out of scope for the current spec. Note for future work.

4.  **Payroll frequency normalization**: Different pay frequencies (weekly, biweekly, semimonthly, monthly) affect how many pay records map to a given month. The `is_normalized` flag suggests some normalization exists. Confirm: is normalization applied upstream, or does this pipeline need to handle it?