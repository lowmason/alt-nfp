# Concise Representativeness Analysis for Payroll Provider vs QCEW

## Objective

Build an agent-executable pipeline that replaces a brute-force table of `state × industry` comparisons with a concise representativeness diagnostic.

The output should answer three questions:

1.  **How different is the provider from QCEW overall?**
2.  **Is the mismatch mostly geography, mostly industry, or mostly their interaction?**
3.  **Which small set of cells are the truly important outliers after shrinkage?**

All geography is defined at the **state level**, with optional aggregation to:

-   **Census divisions (9)**
-   **Census regions (4)**

The system must gracefully handle cases where state information is incomplete or biased, and automatically aggregate to coarser geographic levels when necessary.

------------------------------------------------------------------------

## Non-goals

-   Do **not** run large numbers of independent significance tests.
-   Do **not** rely on a single omnibus chi-square test.
-   Do **not** report raw cell diagnostics before handling geographic bias.
-   Do **not** present fine-grained results when geography is not credible.

------------------------------------------------------------------------

## Deliverables

-   A reproducible pipeline that ingests provider data and QCEW benchmarks.

-   A concise diagnostics object with:

    -   `D_overall`: total variation (dissimilarity index)
    -   `D_state`: mismatch in state margins
    -   `D_industry`: mismatch in industry margins
    -   `decomposition`: geography vs industry vs interaction contributions
    -   `top_residual_cells`: top `K` shrunken outliers
    -   `grid_used`: one of `state_x_supersector`, `division_x_supersector`, `region_x_supersector`, or `national_x_supersector`
    -   `location_regime`: `state_observed`, `state_partial`, or `no_state`
    -   `warnings`: methodological caveats

-   A human-readable markdown or HTML report with an executive summary.

-   Unit tests for all core components.

------------------------------------------------------------------------

## Recommended Project Structure

``` text
src/
  representativeness/
    __init__.py
    config.py
    schemas.py
    qcew.py
    provider.py
    crosswalks.py
    cell_builder.py
    location_regime.py
    allocation.py
    metrics.py
    decomposition.py
    shrinkage.py
    diagnostics.py
    report.py
    cli.py

tests/
  test_metrics.py
  test_decomposition.py
  test_shrinkage.py
  test_allocation.py
  test_cell_builder.py

configs/
  default.yaml
  provider_<name>.yaml

outputs/
  tables/
  figures/
  reports/
```

------------------------------------------------------------------------

## Core Statistical Design

### 1) Treat the full table as one composition

For grid ( g \in G ) (e.g., `state × supersector`):

-   Provider counts: ( P_g )
-   QCEW counts: ( Q_g )
-   Provider shares: ( p_g = P_g / \sum\_g P_g )
-   QCEW shares: ( q_g = Q_g / \sum\_g Q_g )

------------------------------------------------------------------------

### 2) Headline mismatch

Use total variation distance:

``` text
D_overall = 0.5 * sum_g |p_g - q_g|
```

Interpretation: fraction of employment share that must be reallocated to match QCEW.

------------------------------------------------------------------------

### 3) Marginal mismatch

``` text
D_state    = 0.5 * sum_s |p_s. - q_s.|
D_industry = 0.5 * sum_i |p_.i - q_.i|
```

------------------------------------------------------------------------

### 4) Two-way decomposition

Fit a benchmark-relative Poisson log-linear model:

``` text
P_si ~ Poisson(mu_si)
log(mu_si) = log(sum_g P_g) + log(q_si) + alpha_s + beta_i + gamma_si
```

Where:

-   ( \alpha\_s ): state effects
-   ( \beta\_i ): industry effects
-   ( \gamma\_{si} ): interaction residuals

Output:

-   Share of deviance attributable to state
-   Share attributable to industry
-   Residual interaction share

------------------------------------------------------------------------

### 5) Hierarchical shrinkage

Do not rank raw residuals.

Initial approach:

-   Empirical Bayes shrinkage on residuals

Final approach:

-   Hierarchical Bayesian partial pooling with structure:

``` text
state → division → region
industry → supersector
```

Output:

-   Shrunken residuals
-   Uncertainty
-   Top ( K ) meaningful outliers

------------------------------------------------------------------------

### 6) Reporting rule

Each report must include:

-   One headline mismatch number
-   One decomposition summary
-   One short outlier list
-   One paragraph on geographic credibility

------------------------------------------------------------------------

## Methodological Guardrails

### A. Detect state availability regime

Classify input as:

-   `state_observed`: reliable state assignment
-   `state_partial`: biased (e.g., HQ concentration)
-   `no_state`: missing

------------------------------------------------------------------------

### B. Choose the analysis grid

Default:

| Regime | Grid |
|----------------|--------------------------------------------------------|
| state_observed | state × supersector |
| state_partial | state × supersector (post-adjustment) or division × supersector |
| no_state | national × supersector (optional model-based division) |

Always prefer **credible aggregation** over noisy detail.

------------------------------------------------------------------------

### C. Correct geography before diagnostics

If `state_partial`:

-   retain small clients when credible
-   redistribute large multi-state firms using QCEW shares

If `no_state`:

-   allocate using QCEW state-industry shares
-   mark all geography as model-based

------------------------------------------------------------------------

### D. Aggregate before over-interpreting

If state-level results are unstable:

-   step up to division or region

------------------------------------------------------------------------

### E. Compute diagnostics after correction

All metrics must use **post-adjustment data**.

------------------------------------------------------------------------

### F. Use size class if available

-   Include size class in modeling and shrinkage
-   Warn if missing

------------------------------------------------------------------------

## Implementation Tasks

### Phase 0: Scaffolding

-   Build repo structure
-   Add config system
-   CLI entrypoint

------------------------------------------------------------------------

### Phase 1: Data contracts

Provider:

-   period
-   employment_count
-   industry_code
-   state
-   size_class (optional)

QCEW:

-   period
-   state
-   industry_code
-   employment_count

Mappings:

-   NAICS → supersector
-   state → division (9)
-   state → region (4)

------------------------------------------------------------------------

### Phase 2: Cell table

-   Build `state × industry` grid
-   Support aggregation to division/region
-   Preserve totals and shares

------------------------------------------------------------------------

### Phase 3: Location regime

-   Rule-based classification
-   Emit label, confidence, explanation

------------------------------------------------------------------------

### Phase 4: State allocation

For `state_partial`:

-   detect HQ bias
-   redistribute large firms

For `no_state`:

-   allocate using QCEW shares

Track:

-   observed vs allocated
-   allocation diagnostics

------------------------------------------------------------------------

### Phase 5: Metrics

-   Implement `D_overall`, `D_state`, `D_industry`
-   Validate on toy cases

------------------------------------------------------------------------

### Phase 6: Decomposition

-   Fit Poisson log-linear model
-   Compute deviance shares
-   Generate narrative summary

------------------------------------------------------------------------

### Phase 7: Shrinkage

-   Empirical Bayes initial
-   Hierarchical Bayes later
-   Rank top ( K ) outliers

------------------------------------------------------------------------

### Phase 8: Reliability

Labels:

-   `reliable`
-   `marginal`
-   `insufficient`

Based on:

-   allocation share
-   cell size
-   shrinkage intensity

------------------------------------------------------------------------

### Phase 9: Reporting

Summary table:

-   grid used
-   regime
-   metrics
-   decomposition
-   top outliers

Charts:

-   state (or aggregated) margins
-   industry margins
-   residuals

------------------------------------------------------------------------

### Phase 10: Temporal extension (optional)

-   Track metrics over time
-   Stability diagnostics

------------------------------------------------------------------------

## Suggested Output Schema

``` json
{
  "provider": "string",
  "period": "YYYY-MM",
  "qcew_vintage": "string",
  "location_regime": "state_observed|state_partial|no_state",
  "grid_used": "state_x_supersector|division_x_supersector|region_x_supersector|national_x_supersector",
  "metrics": {
    "D_overall": 0.0,
    "D_state": 0.0,
    "D_industry": 0.0
  },
  "decomposition": {
    "state_share": 0.0,
    "industry_share": 0.0,
    "interaction_share": 0.0
  },
  "top_residual_cells": [],
  "warnings": []
}
```

------------------------------------------------------------------------

## Test Cases

1.  Perfect match → all metrics = 0
2.  Industry tilt → `D_industry` dominates
3.  Geography tilt → `D_state` dominates
4.  Sparse noise → shrinkage suppresses noise
5.  HQ bias → redistribution improves results

------------------------------------------------------------------------

## Open Design Choices

-   TVD vs Jensen-Shannon divergence
-   Deviance vs variance decomposition
-   EB vs full Bayesian shrinkage
-   Default aggregation level under partial geography
-   Inclusion of size class in v1

------------------------------------------------------------------------

## Final Acceptance Checklist

-   End-to-end pipeline works from config
-   Output is concise and interpretable
-   Clear decomposition of mismatch
-   Outliers reflect signal, not noise
-   Geographic limitations are explicit
-   Aggregation used when needed

------------------------------------------------------------------------

## Plain-English Definition of Done

The system is complete when a user can input provider data and QCEW benchmarks and receive a short report that explains:

-   how different the provider is from the benchmark,
-   whether that difference is driven by geography or industry,
-   which specific areas matter most after accounting for noise,
-   and whether the geographic detail shown is trustworthy.