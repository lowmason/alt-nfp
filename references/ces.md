# CES methodology: a complete technical reference for NFP nowcasting
 
The Current Employment Statistics survey produces the headline nonfarm payroll number through a **recursive weighted link relative estimator** applied to a stratified sample of ~631,000 worksites, supplemented by an ARIMA-based net birth/death model that adds **+700K to +1,400K jobs annually** but introduces systematic bias at business cycle turning points. Recent benchmark revisions of **−598,000 (March 2024)** and **−862,000 (March 2025)** — the largest on record — reveal that declining response rates (from >60% to <43%) and persistent birth/death forecast errors have degraded CES accuracy, making real-time nowcasting from alternative data sources increasingly valuable. This document extracts every methodologically relevant detail from five BLS source pages to inform the construction of a Bayesian nowcasting model that uses payroll provider microdata to forecast official employment statistics.
 
---
 
## The CES estimation pipeline: from sample to headline number
 
The CES estimate is built through a chain of multiplicative link relatives anchored to an annual benchmark. Understanding this pipeline is essential for modeling how CES numbers are constructed and where errors enter.
 
**The weighted link relative estimator** is the core formula. For each basic estimating cell (typically 3- to 6-digit NAICS industry):
 
```
AE_c = (AE_p − Σ_j ae*_p,j) × [Σ_i(w_i × ae_c,i) − Σ_j(w_j × ae*_c,j)] / [Σ_i(w_i × ae_p,i) − Σ_j(w_j × ae*_p,j)] + Σ_j ae*_c,j + b_c
```
 
Here *i* indexes matched sample units (reporting in both current and previous months with nonzero employment), *j* indexes atypical units (strikes, disasters) removed from the ratio but added back to levels, *w* is the sampling weight, and **b_c is the net birth/death forecast**. The critical property: this is a **recursive estimator** — it estimates month-over-month *change* via the weighted ratio and chains it to the prior month's published level. It does not independently estimate the employment *level* each month. This chaining means errors accumulate until the annual benchmark correction.
 
**Atypical handling** strips outlier microdata from the ratio calculation, preventing a single establishment's anomalous event from distorting the aggregate growth rate. The atypical unit's current-month employment is then added back to the level. Screening tests use X and K factors varying by industry and data type, with 1-month, 2-month, and 12-month change tests plus seasonal comparison tests at ±11, ±12, and ±13 months.
 
**For hours and earnings** (production employees, average weekly hours, average hourly earnings), CES uses a distinct **weighted-difference-link-and-taper estimator** with **α = 0.9** (weight on prior month) and **β = 0.1** (weight on sample average). This heavy persistence parameter means hours/earnings estimates are slow to reflect actual changes and drift toward the sample mean over time.
 
**Small domain model (SDM)** industries with inadequate samples use a composite estimator blending sample-based change with ARIMA projections from 10 years of QCEW history. Weights on the sample component are restricted to [0.2, 0.8] nationally; for tax preparation services, the sample weight is zero (purely model-driven).
 
---
 
## Sample design, stratification, and data collection
 
The CES sample is a **stratified simple random sample of worksites, clustered by UI account number**, drawn from the Longitudinal Database (LDB) built from the Quarterly Census of Employment and Wages. The LDB covers ~97% of nonfarm employment through Unemployment Insurance records and contains **~11.8 million establishments**.
 
**Stratification uses three dimensions**: state × industry (13 sectors) × employment size (8 classes: 0–9, 10–19, 20–49, 50–99, 100–249, 250–499, 500–999, 1,000+), yielding **104 allocation cells per state**. Within each cell, units are sorted by Metropolitan Statistical Area for implicit geographic stratification. Allocation follows **Neyman's optimum allocation** to minimize variance of over-the-month change for total state nonfarm employment.
 
**The universe is heavily skewed**: establishments with 1,000+ employees constitute only **0.2% of all UI accounts** but **28.9% of total employment**, while those with 0–9 employees represent **72.9% of accounts** but only **10.4% of employment**. This skew means certainty selections (all size-class-8 units, EDI reporters, and noncovered employment units) anchor the estimate's stability. The active sample covers approximately **26% of total nonfarm payroll employees** across ~121,000 businesses and ~631,000 worksites.
 
**Sample rotation** redraws the entire sample annually from first-quarter LDB data, with a supplemental birth update from the third quarter. Approximately **one-third of the sample rotates each year**, with ~66% overlapping between adjacent annual samples. Units serve a minimum of 2 years, with swap-out after 4+ consecutive years. **Permanent Random Numbers (PRN)** ensure consistent selection across draws.
 
**Quarterly enrollment by industry group** determines when new sample enters estimation:
- Q1 (April estimation): Mining/logging, Wholesale, Retail, Transportation, Utilities, Financial
- Q2 (July): Construction, Leisure/hospitality
- Q3 (October): Information, Professional/business services, Other services
- Q4 (January): Manufacturing, Education/health, all birth units
**Data collection** operates through four modes: CATI (computer-assisted telephone interviewing) for initial enrollment, web reporting for ongoing collection, EDI (electronic data interchange) for large multi-establishment firms submitting 50–1,000+ worksites per file, and legacy touchtone/electronic methods for a small residual. Federal participation is **voluntary** (no mandatory reporting authority), though it is mandatory by state law in New Mexico, Oregon, South Carolina, and Puerto Rico.
 
**Non-response handling** is implicit: non-responding units are automatically excluded from the matched sample. No explicit imputation is performed for standard non-response. Only select certainty units with consistent seasonal patterns receive imputation based on historical over-the-month change data. This means the matched-sample design inherently assumes non-respondents have similar growth rates to respondents — an assumption that recent benchmark analysis shows is **increasingly violated** as response rates decline.
 
**Coverage varies dramatically by industry**: government achieves **67%** coverage, retail trade **38%**, manufacturing **17%**, construction only **9%**, and other services just **6%**. These coverage differentials directly affect industry-level estimation precision, with standard errors ranging from **0.1%** relative SE for total nonfarm to **0.7%** for construction.
 
---
 
## The net birth/death model: methodology, performance, and known failures
 
This is the single most important component for a Bayesian nowcasting model to understand and potentially improve upon. The birth/death model addresses a fundamental sampling problem: new businesses (births) do not appear on the UI-based sampling frame for approximately one year after opening, and business closures (deaths) are indistinguishable from ordinary non-response.
 
**The model operates in two components.** Component 1 is an implicit offset: by excluding zero-employment reporters and non-respondents from the matched sample, the weighted link relative effectively imputes growth rates from continuing businesses to death establishments, which roughly offsets most birth employment. Component 2 is an **ARIMA time series forecast** of the residual birth/death employment not captured by this implicit offset.
 
**Deriving the historical birth/death residual from QCEW**: BLS classifies every QCEW establishment as continuing, birth, or death. A population link relative is computed from all continuing establishments (no weights). A "CIMP" (continuous-plus-imputed) series applies this population link relative to population employment from the March initialization month. The actual birth/death value equals the over-the-month change in the residual: **BD_t = (1 − B)(AE_t^LDB − CIMP_t)**, where B is the backshift operator. Two full years (24 months) of these residuals are calculated from each initialization and chained across multiple years to form the ARIMA input series.
 
**The ARIMA specification** historically followed a seasonal integrated moving average process: **BD_t = Θ(B^12) × a_t / (1 − B^12)**, with seasonal differencing capturing annual patterns. No covariates were included in the production model, and additive outliers could be specified but were unavailable for the forecast period. The model is re-estimated **quarterly** since 2011 (previously annually) using 5 years of historical birth/death residuals. Input data ends **6–9 months** before the forecast target due to QCEW publication lag.
 
**Starting January 2026, BLS permanently modified the model** to incorporate a regression component exploiting the linear relationship between **ln(WLR)** — the natural log of the sample-based weighted link relative — and the actual birth/death input values. Since the WLR is available concurrently with the survey (no lag), this provides real-time signal that was previously absent. Modeling is performed at the major industry sector level, then raked down to detailed industries proportional to their forecast error variances. BLS research found this modification could reduce forecast RMSE from **27.1K to ~10.7K** at the supersector level — roughly halving the error.
 
**Seasonal patterns in birth/death adjustments** are pronounced. For 2024, total nonfarm monthly B/D ranged from **−128K (September)** to **+368K (October)**, with the annual cumulative total of **+1,201K**. The largest positive months are consistently April (+363K) and October (+368K), coinciding with seasonal business formation; the most negative are January (−121K) and September (−128K). The B/D is added on a **not seasonally adjusted basis** — comparing it to seasonally adjusted employment changes overstates the model's apparent contribution.
 
### Business cycle turning point failures
 
The birth/death model's most consequential known limitation is its **systematic failure at turning points**. The ARIMA model projects historical seasonal patterns forward, producing mostly positive forecasts reflecting the normal excess of births over deaths. At recession onset, births decline sharply and deaths surge, but the model continues projecting positive contributions, creating systematic overestimation. During recoveries, the ARIMA inputs reflect depressed formation rates from the recession period, producing forecasts that are too low.
 
Historical evidence is stark. In **2009** (Great Recession), the benchmark revision was **−902,000** while the cumulative B/D contribution was **+717,000**; without the B/D model, the revision would have been only −185,000. The model effectively added 717,000 nonexistent jobs. In **2019**, the revision was **−489,000** against B/D of +1,068,000. In **2024**, the revision was **−598,000** against B/D of +1,365,000. BLS acknowledges that "persistent and relatively large birth-death forecast errors" have occurred since the 2020 benchmark.
 
In 17 of 20 years (2004–2024), the B/D model improved CES estimates. But in the 3 years it failed — all near turning points — the errors were enormous. For a Bayesian nowcasting model, this means **the B/D adjustment should be treated as a strong informative prior during stable growth periods but assigned substantially wider uncertainty bands when leading indicators suggest a regime change**.
 
A 2023 BLS research paper (Grieves, Mance, and Witt) simulated 44 prediction approaches and found the production ARIMA was the worst performer (RMSE 27.1K vs. 10.7K for the best individual approach). Simple forecast combination methods performed comparably to the best individual model, providing robustness to misspecification. The paper also found that birth/death and the CES sample link likely share a **cointegrating seasonal relationship**, suggesting seasonal differencing may cause model misspecification.
 
---
 
## Benchmarking: the wedge-back procedure and revision mechanics
 
Annual benchmarking replaces the sample-based March employment estimate with a near-complete count derived from QCEW (~97%) plus noncovered sources (~3%, from Railroad Retirement Board, County Business Patterns, Annual Survey of Public Employment and Payroll, and state agencies). This is CES's primary self-correction mechanism.
 
**The wedge-back procedure** distributes the March benchmark revision linearly across the preceding 11 months:
- February receives **11/12** of the March revision
- January receives **10/12**
- Continuing back to prior April, which receives **1/12**
This assumes estimation error accumulated at a steady rate since the prior benchmark — a simplification that may not hold if errors cluster around specific months (e.g., seasonal hiring periods where B/D model errors are largest). For the 9 months following the benchmark March (April through December), estimates are recalculated using the benchmark March as base, applying sample-based link relatives with **updated** birth/death forecasts.
 
**Scope of revision**: each benchmark affects **21 months of not seasonally adjusted data** (prior April through current December) and **5 years of seasonally adjusted data**. The timeline runs: March reference period → QCEW data available ~7–9 months later → preliminary benchmark published in August/September (~6 months after March) → final benchmark incorporated in February (~11 months after March), published with the January Employment Situation release.
 
**Recent benchmark magnitudes reveal a deteriorating trend**:
 
| Benchmark (March) | NSA Revision | Percent | Preliminary Estimate |
|---|---|---|---|
| 2025 | −862,000 | −0.5% | −911,000 (−0.6%) |
| 2024 | −598,000 | −0.4% | −818,000 (−0.5%) |
| 2023 | ~−105,000 | ~−0.07% | — |
| 2019 | −505,000 | ~−0.3% | — |
| 10-year average | — | ±0.1–0.2% | — |
 
The 2024 and 2025 revisions far exceed the historical 0.1–0.2% average. The preliminary-to-final gap is also informative: preliminary estimates of −818K and −911K were revised to −598K and −862K respectively, suggesting preliminary benchmarks systematically overstate the downward revision by **100K–220K**.
 
**Industry-level revision patterns show persistent biases**. In the March 2024 benchmark, the largest negative revisions hit professional and business services (−380K, −1.7%), information (−63K, −2.1%), leisure and hospitality (−116K, −0.7%), retail trade (−126K, −0.8%), and financial activities (−76K, −0.8%). Consistently positive revisions appeared in education and health services (+143K, +0.5%), transportation and warehousing (+65K, +1.0%), and other services (+52K, +0.9%). These industry-level biases are remarkably stable across benchmark years and provide **strong priors for industry-level nowcasting adjustments**.
 
**Declining survey quality metrics** compound the problem. CES response rates fell from over 60% (2016–2019) to **below 43%** recently, and initiation rates for new sample members fell from over 70% to **below 40%**. BLS identified two primary contributors to recent overestimation beyond the B/D model: **response error** (businesses reported less employment to QCEW than to CES) and **nonresponse error** (non-responding CES firms had lower QCEW employment than responding firms). Both suggest the matched-sample assumption of representative growth rates among respondents is breaking down.
 
---
 
## Reference period and pay period length effects
 
The CES reference period is the **pay period including the 12th of the month**. Employment counts all persons who received pay for any part of this period — including those on paid sick leave, vacation, or holiday, and those who worked part of the period even if unemployed or on strike for the rest. Persons are counted at **each worksite** (job count, not person count), so multiple jobholders are counted multiple times.
 
**Pay period length varies substantially by industry**, creating alignment complications:
 
| Industry | Weekly | Biweekly | Semimonthly | Monthly |
|---|---|---|---|---|
| Construction | **65.4%** | 22.3% | 6.5% | 5.9% |
| Manufacturing | 43.4% | 46.6% | 8.6% | 1.5% |
| Education & health | 9.9% | **63.6%** | 18.4% | 8.1% |
| Financial activities | 14.0% | 37.4% | **30.2%** | **18.4%** |
| Information | 4.4% | 47.2% | **37.5%** | 10.9% |
| All private | 27.0% | **43.0%** | 19.8% | 10.3% |
 
Construction's 65.4% weekly pay rate means its reference period captures approximately 7 days around the 12th, while financial activities' 18.4% monthly rate means those establishments report for the entire calendar month. **CES does not collect pay period start/end dates**, so the exact days covered are unknown. All payroll and hours are normalized to weekly equivalents based on reported length of pay period before estimation.
 
**Biweekly pay dominance increases with establishment size**: from 39.0% at 1–9 employees to 66.6% at 1,000+ employees. This matters for payroll-provider-based nowcasting because large firms' biweekly pay cycles create a more consistent reference period alignment than the heterogeneous mix at small firms.
 
**Two calendar effects distort CES data**. First, the **4-vs-5-week effect**: the interval between successive reference periods alternates between 4 and 5 weeks (~1 in 3 months has a 5-week gap), significantly affecting measured seasonal hiring magnitudes. Second, the **10/11-day effect** on hours and earnings: AWH and AHE show more growth in short months (20–21 weekdays) than long months (22–23 weekdays), particularly for salaried workers whose employers report fixed hours regardless of actual month length. Both effects are addressed through REGARIMA modeling during seasonal adjustment, but they introduce non-economic noise into not seasonally adjusted estimates that a nowcasting model must account for.
 
---
 
## Seasonal adjustment: X-13ARIMA-SEATS with concurrent factors
 
CES applies **concurrent seasonal adjustment** using **X-13ARIMA-SEATS** software with the **X-11 filtering procedure** for decomposition (not SEATS). Under concurrent methodology adopted in June 2003, new seasonal factors are calculated each month using all data through the current month, producing smaller revisions than the prior projected-factor approach.
 
**ARIMA model selection** uses the **TRAMO procedure** (since January 2017 data) during the annual review, offering a wider model selection space than X-12-ARIMA's automatic procedure. Model selection uses **AICc** (corrected Akaike Information Criterion). Models are series-specific — for example, Logging uses ARIMA(1,1,0)(1,0,0), while other industries have different specifications. All model specifications are held **fixed within the year** between annual benchmarks: ARIMA model, outliers, transformation (multiplicative or additive), and calendar treatments remain constant.
 
**Input data span** is **10 years** of not seasonally adjusted history. The most recent **5 years** of seasonally adjusted data are re-adjusted with each annual benchmark; data older than 5 years is effectively frozen.
 
**Outlier treatment** evolved significantly due to COVID-19. Pre-2022, only **additive outliers (AO)** were used. Since the 2021 benchmark, BLS allows **AO, temporary change (TC), and level shift (LS)** outliers, detected automatically with a critical value of **3.5**. A two-run process generates results with AO-only and with all types, and analysts manually select the better-fitting series. TC outliers better capture industries with large initial pandemic drops followed by gradual recovery; LS outliers capture permanent level shifts.
 
**Calendar effect adjustments** include:
- **4/5-week survey interval**: 11 separate REGARIMA dummy variables (one per month except March) estimate interval effects, producing filtered series before X-11 decomposition
- **10/11-day length-of-pay-period effect**: for AWH and AHE series in most service-providing industries and mining/logging
- **Good Friday/Easter and Labor Day**: for hours series where holiday timing affects the reference period
- **Special adjustments**: election poll workers for local government (November), December postal surge for USPS
**Direct vs. indirect adjustment**: CES generally adjusts published series **directly at the 3-digit NAICS level**, then sums components to form higher-level aggregates (indirect aggregation of directly adjusted components). A few series with heterogeneous seasonal patterns are indirectly adjusted at the 3-digit level by directly adjusting and summing more detailed components.
 
**Prior adjustments** remove quantifiable non-seasonal events (strikes, decennial census hiring) from original data before the X-13 run. After seasonal factors are computed, prior adjustments are reincorporated. The prior adjustment file updates annually for structure and monthly for strike data.
 
---
 
## Implications for building a Bayesian NFP nowcasting model
 
Several structural features of CES methodology create specific opportunities and constraints for a payroll-provider-based nowcasting model.
 
**The recursive estimator structure means CES carries forward past errors.** Because each month's estimate is the prior month's level multiplied by the sample growth rate plus the B/D adjustment, any error in level persists until the annual benchmark. A nowcasting model that independently estimates levels — as payroll microdata can support — should systematically outperform CES's chained estimates over multi-month horizons, while potentially showing larger single-month noise.
 
**The birth/death model is the dominant error source at turning points** and contributes +1,000K to +1,400K annually. For a nowcaster, the B/D adjustment values are published with each Employment Situation release and should be modeled as a known additive component with a strong seasonal prior and **fat-tailed uncertainty** that widens when coincident indicators suggest regime change. The January 2026 methodology change incorporating WLR regression will reduce but not eliminate turning-point bias.
 
**Industry-level biases are persistent and exploitable.** Professional and business services, information, and leisure and hospitality are systematically overestimated; education/health and transportation are underestimated. These biases likely reflect differential birth/death dynamics and non-response patterns by industry. A nowcasting model should incorporate industry-specific bias priors calibrated to the historical benchmark revision record.
 
**Pay period alignment is a first-order data engineering problem.** Since CES counts anyone paid during the pay period including the 12th, and 43% of establishments use biweekly pay, the exact mapping from payroll-provider transaction data to the CES reference period requires knowing each establishment's pay schedule. Construction (65.4% weekly) offers the tightest alignment; financial activities (18.4% monthly) the loosest.
 
**Revision patterns provide calibration targets.** The first preliminary estimate (released ~3 weeks after the reference period) uses less than the full sample. Second and third estimates incorporate additional responses. Over-the-month changes of approximately **122,000** are needed for statistical significance at the 90% confidence level. The standard error for total nonfarm is **82,554**. Monthly revisions between first and third estimates, plus the eventual benchmark revision, define the uncertainty envelope that a nowcasting model's posterior should encompass.
 
**Seasonal adjustment is concurrent but specification-fixed within years.** The seasonal factors recalculate monthly, but the underlying ARIMA models, outlier sets, and calendar treatments are locked annually. This means the SA process can be replicated given the published specification files, and any divergence between a nowcaster's seasonal model and BLS's can be explicitly modeled as a source of forecast error relative to the official number.
 
## Conclusion
 
The CES methodology is a sophisticated but increasingly strained system. Its core weighted link relative estimator performs well when the sample is representative and birth/death dynamics follow historical patterns, but three compounding problems — declining response rates below 43%, persistent birth/death model overestimation during slowdowns, and the inherent 6–9 month QCEW lag — have produced record benchmark revisions exceeding **−800,000 jobs** in consecutive years. BLS's January 2026 modification incorporating concurrent sample information into the B/D model represents the most significant methodological improvement in over a decade, but the fundamental architecture of chaining monthly growth rates from a voluntary survey sample remains vulnerable to non-response bias and regime changes. For a Bayesian nowcasting model using payroll provider microdata, the key advantages lie in three areas: **independent level estimation** that avoids CES's error accumulation, **real-time birth/death signal** from observing actual payroll transactions rather than forecasting from lagged QCEW, and **industry-specific bias correction** calibrated to the remarkably persistent patterns in historical benchmark revisions. The 4/5-week calendar effect, pay period alignment heterogeneity, and CES's specific seasonal adjustment specifications must all be modeled explicitly to produce forecasts comparable to the official seasonally adjusted headline number.