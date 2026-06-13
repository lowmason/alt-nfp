## Program origins, legal foundation, and the UI administrative backbone
 
The QCEW traces its lineage to the **Social Security Act of 1935**, which established the Federal-State unemployment insurance system, and the **Federal Unemployment Tax Act (FUTA) of 1938**, which created the federal tax mechanism compelling employer participation. The program was originally known as the **Employment Security Report 202 (ES-202)**, named after the form on which data were collected. BLS assumed publication responsibility from the Social Security Administration in **1972**, renamed the program to Covered Employment and Wages (CEW) in **1998**, and adopted its current name in **2002** when it began quarterly publication and transitioned from SIC to NAICS-based industry coding.
 
The program's statutory authority rests on several pillars. The FUTA establishes the employer reporting obligation. **29 U.S.C. § 2** authorizes the Annual Refiling Survey and Multiple Worksite Report. **Title XV of the Social Security Act** (1954) created the Unemployment Compensation for Federal Employees (UCFE) program, which brings federal civilian workers into scope. The **Confidential Information Protection and Statistical Efficiency Act of 2002 (CIPSEA)**, reauthorized under Title III of the Foundations for Evidence-Based Policymaking Act of 2018, governs data confidentiality.
 
The fundamental architecture is that QCEW data are a **byproduct of state UI tax administration** rather than a purpose-built statistical program. Employers must file Quarterly Contribution Reports (QCRs) with their State Workforce Agencies (SWAs), reporting monthly employment counts and quarterly total wages for all covered workers. BLS receives these administrative files from **53 reporting entities** — the 50 states, the District of Columbia, Puerto Rico, and the U.S. Virgin Islands — and transforms them into statistical products through extensive editing, validation, and supplementary data collection. This administrative foundation gives QCEW its near-universe coverage but also means data quality is **inherently sensitive to changes in state UI systems**, a vulnerability that has become increasingly salient as states modernize legacy IT infrastructure.
 
---
 
## What QCEW covers — and what falls through the gaps
 
QCEW coverage encompasses roughly **97.1% of civilian wage and salary employment** (2019 figure). Using December 2022 reference data, the program counted approximately **11.75 million establishments** employing **152.5 million workers**: 10.97 million private-sector establishments (130.7 million workers), 61,000 federal government establishments (4.6 million workers), 72,000 state government establishments (5.1 million workers), and 172,000 local government establishments (12.1 million workers). Covered workers received **$8.769 trillion in pay** in 2019, representing 40.9% of GDP.
 
Coverage scope has expanded dramatically since 1935, when the threshold was employers with **8 or more workers**. The employer-size threshold dropped to **4 or more** in 1956, and then to **1 or more workers** (in 20 weeks or $1,500 quarterly payroll) with the Employment Security Amendments of 1970, effective January 1, 1972. The **Unemployment Compensation Amendments of 1976** (effective January 1, 1978) represent the other watershed expansion, bringing in agricultural labor (employers with 10+ workers in 20 weeks or $20,000+ in quarterly cash wages), household workers (employers paying $1,000+ per quarter), state and local government employees, and employees of nonprofit elementary and secondary schools. Since 1978, UI and UCFE coverage has been basically comparable from state to state.
 
The remaining **~3% coverage gap** reflects specific statutory exclusions that matter for nowcasting applications:
 
- **Self-employed, proprietors, and unpaid family members** — the single largest excluded category, encompassing sole proprietors, partners in unincorporated businesses, and gig workers classified as independent contractors
- **Railroad workers** — covered by a separate federal unemployment insurance system under the Railroad Unemployment Insurance Act of 1938
- **Members of the armed forces** and the Commissioned Corps of NOAA
- **Certain agricultural and domestic workers** below statutory thresholds
- **Elected officials** in most states, members of legislative bodies and the judiciary
- **Student workers at their schools**, certain small nonprofit employees, and temporary emergency workers
- **Workers earning no wages during the entire applicable pay period** (those on unpaid leave, work stoppages, or unpaid vacation)
A critical recent development is **California's Assembly Bill No. 5** (effective January 1, 2020), which introduced the ABC test for independent contractor classification. This state-level change was expected to shift a significant number of workers from independent contractor to employee status, thereby expanding QCEW coverage in California. Such state-level variation in worker classification represents a persistent source of cross-state non-comparability.
 
---
 
## How data flow from employers through states to BLS
 
### The quarterly contribution report pipeline
 
The primary data pipeline begins when employers file QCRs with their SWAs on a quarterly basis. Private-sector employers normally submit **one consolidated report** covering all economic activities in a given state. These reports contain monthly employment counts (for each of the three months in the quarter) and total quarterly wages. For multi-establishment firms, the QCR reflects only statewide aggregated employment and wages — it does not disaggregate by worksite.
 
Federal government data follow a parallel path through the UCFE program. Federal agencies submit the **Report of Federal Employment and Wages (RFEW)**, a purely statistical report containing employment and wages data for each installation within each state. Unlike private-sector QCRs, these are not tax documents. UCFE employment data use the same reference period — the **payroll period including the 12th day of the month** — but differ from OPM data, which reference the last workday of the month.
 
### The Multiple Worksite Report — disaggregating multi-establishment firms
 
Because the QCR only provides statewide totals, BLS operates the **Multiple Worksite Report (MWR)** to obtain establishment-level breakouts from multi-location employers. The MWR is sent quarterly to employers operating multiple establishments and collects **prior-quarter** employment and wage data at the individual worksite level. As of Q4 2023, MWR statistics reveal the scale of this challenge: while only **4.7% of employers** file the MWR (those operating multiple establishments), these employers account for **17.2% of national worksites** and a remarkable **45% of national employment** (70.3 million workers across 2.1 million establishments). The remaining 82.8% of establishments are single-unit employers accounting for 55% of employment.
 
MWR collection is split between BLS and states. **BLS directly collects from 9% of establishments** (representing 23.7% of employment) through two channels: the Electronic Data Interchange (EDI) Center in Chicago (established 1995), which handles approximately 485,000 reports from ~100 large firms covering ~15 million employees; and BLS web collection (MWRWeb, operational since Q1 2007), which gathers 572,617 establishment reports covering 21.2 million employees. States collect the remaining 91% of establishments (76.3% of employment). **Thirty-four states** have laws mandating MWR completion, which materially affects response quality in those jurisdictions.
 
### The Annual Refiling Survey — maintaining industry and location accuracy
 
The **Annual Refiling Survey (ARS)** contacts approximately one-third of all private-sector businesses with more than 3 employees each year, completing a full cycle every three years. The ARS verifies industrial activity classification (NAICS code), geographic location, and physical versus mailing address. Establishments in industries with traditionally stable classifications (e.g., cemeteries) are placed on a **6-year cycle**. Unclassified establishments (NAICS 999999) are surveyed annually. States must attain either a **70% reporting rate** for surveyed establishments or an **80% reporting rate** measured by employees. Thirty-one states have laws mandating ARS completion.
 
### Editing, validation, and imputation
 
BLS processing transforms approximately **10.5 million records** per quarter through extensive review at state, regional, and national levels. The editing system ignores establishments showing no change or statistically insignificant change and flags records based on over-the-year changes in employment or wages relative to total county employment. The **basic monthly employment edit** applies a 6-step statistical test using multiple t-tests for month-to-month changes, over-the-year changes, and 12-month variation. The **wage edit** uses an interquartile test (Hoaglin, Iglewicz, and Tukey, 1986). Records failing edits are individually reviewed, and state staff contact respondents to validate significant movements.
 
As of June 2022, approximately **3% of establishments** failed to respond timely, requiring imputation. The current imputation method (effective November 2020) applies the **trend from reported data for similar employers** to estimate non-respondents' current employment and wages, replacing the previous year-ago-trend method. Records are imputed for a maximum of **two quarters** of non-reporting; after two consecutive quarters without a filed report, BLS drops the establishment from the universe — effectively treating it as a business death. Non-respondents typically account for **less than 5%** of QCEW employment.
 
---
 
## Key concepts and definitions that affect measurement
 
### Employment: the 12th-of-month pay period convention
 
QCEW monthly employment equals the number of covered workers who **worked during, or received pay for, the pay period including the 12th day of the month**. This measures **filled jobs** (not persons), counted by **place of work** (not residence). Workers holding multiple jobs are counted separately by each employer. Both full-time and part-time, temporary and permanent positions are included. Workers remain in the count even after their wages exceed the federal UI taxable wage base of **$7,000** per calendar year. Workers on paid sick leave, holiday, or vacation are counted; workers on unpaid leave for the entire pay period are excluded.
 
### Wages: a broad but not universal compensation measure
 
Total wages represent compensation paid during the calendar quarter regardless of when services were performed. The definition is **broader than base salary** but **narrower than total compensation**. Included are bonuses, stock options, severance pay, profit distributions, cash value of meals and lodging, tips and other gratuities, and in some states employer contributions to deferred compensation plans such as 401(k)s. Excluded are **employer contributions** to OASDI, health insurance, UI, workers' compensation, and private pension and welfare funds. Employee contributions to these same programs (deducted from gross pay) are reported as wages. A few states require wages to be reported for the period services are performed rather than the period compensation is paid, creating minor cross-state inconsistencies.
 
Average weekly wages are calculated as quarterly total wages divided by the average of three monthly employment levels, divided by 13 weeks. A known artifact is **calendar-driven wage fluctuation**: most federal employees receive biweekly pay, meaning some quarters contain **7 pay dates** versus the typical 6, creating pronounced over-the-year wage swings in government-dominated counties.
 
### NAICS coding: BLS-specific deviations from the standard
 
BLS implements several deviations from the official NAICS manual maintained by Census and OMB. The most significant involves **NAICS 238 (Specialty Trade Contractors)**, where BLS uses proprietary six-digit codes ending in "1" for residential construction and "2" for nonresidential construction, instead of the standard "0" suffix. This creates **38 BLS-only codes** (19 residential/nonresidential pairs) spanning from 238111/238112 (poured foundation contractors) through 238991/238992 (all other trade contractors). Data users must combine these pairs to reconstruct standard NAICS totals.
 
Two additional NAICS codes — **112130** (dual-purpose cattle ranching) and **541120** (offices of notaries) — are included in the NAICS 2022 manual but are not valid in the United States. One code, **517122** (agents of wireless telecommunications services), exists in the manual but is not used by BLS; these establishments are instead classified under **449210** (retail electronics). BLS also employs **12 supersector aggregation levels** and 2 domain levels (goods-producing and service-providing) above the NAICS hierarchy, providing comparability with CES and other programs. Establishments lacking sufficient information to assign an industry receive the placeholder code **999999** (unclassified).
 
### Geographic coding and the geocoding infrastructure
 
The primary geographic designation is the **county**, assigned based on physical location of the establishment. BLS converts physical addresses to geocodes (latitude/longitude coordinates) using Geographic Information Software maintained by the Office of Technology and Survey Processing. Township is a secondary designation used mainly in New England states and New Jersey. All geographic codes follow the Federal Information Processing Standards (FIPS) system.
 
---
 
## UI coverage expansion: from 1935 to the present
 
The historical timeline of federal UI coverage expansion reveals a program that has grown from a narrow base to near-universal coverage through a series of legislative expansions over nine decades.
 
The original Social Security Act of 1935 covered only employers with **8 or more workers** and excluded agricultural labor, domestic service, vessel crews, family employment, government employees, and nonprofit employees. Coverage first expanded meaningfully in **1954** (P.L. 83-767), which established the UCFE program for federal civilian employees and lowered the employer threshold to **4 or more workers in 20 weeks** (effective 1956). The 1960 Social Security Amendments brought **Puerto Rico** into the system and extended coverage to employees on American aircraft outside the U.S. and certain nonprofit organizations.
 
The two largest expansions came in **1970 and 1976**. The Employment Security Amendments of 1970 (P.L. 91-373, effective January 1, 1972) dropped the threshold to **1 or more employees** in 20 weeks or $1,500 quarterly payroll, and brought in nonprofit organizations with 4+ employees, state hospitals, institutions of higher education, outside salesmen, certain agricultural processing workers, and U.S. citizens working for American firms abroad. The Unemployment Compensation Amendments of 1976 (P.L. 94-566, effective January 1, 1978) extended coverage to **agricultural labor** (employers with 10+ workers in 20 weeks or $20,000+ cash wages per quarter), **household workers** (employers paying $1,000+ per quarter), **state and local government employees**, and employees of nonprofit elementary and secondary schools.
 
Subsequent legislation has been primarily subtractive — carving out specific exclusions rather than expanding coverage. Notable exclusions added include commission-based direct sellers and real estate agents (1982), summer camp counselors who are full-time students (1986), and persons committed to penal institutions (1997). The most recent federal coverage change of substance was the **Consolidated Appropriations Act of 2001**, which treated Indian tribes similarly to state and local governments, requiring coverage under state UI law while removing FUTA tax liability.
 
---
 
## UI modernization: a live threat to data quality
 
Numerous states are currently overhauling legacy UI tax systems, and because the UI tax extract is the **primary source of QCEW data**, these modernizations pose direct risks to data quality. BLS has developed extensive documentation — including a QCEW UI Modernization Guide, a 120-page UI Extract Guide, and a State UI Modernization Checklist — to manage transition risks.
 
The stakes are concrete. Errors in UI extracts can produce **incorrect employment and wages, missing accounts, lost administrative data, loss of activation/inactivation dates, and unidentified predecessor/successor transactions**. The UI Account Number combined with the Reporting Unit Number (RUN) serves as the primary key in QCEW systems; if states change UI numbers on existing accounts during modernization, BLS requires **12–18 months of lead time** to manage the transition. BLS strongly urges states not to go live with new extract systems that are not producing correct data, recommending **parallel testing for at least 2 quarters** (6 months) for QCEW and 3 quarters for other programs.
 
The **Colorado case study** illustrates the real-world impact. On November 20, 2024, BLS suspended publication of industry and substate QCEW data for Colorado due to data quality issues caused by UI system modernization. Publication resumed February 19, 2025, after quality concerns were sufficiently addressed. During the disruption, BLS replaced Colorado's sample-based CES estimates with QCEW-derived administrative data for multiple periods spanning 2022–2024, using wedge methodology to address breaks in the series. South Carolina experienced similar data quality disruptions coinciding with its 2019 UI modernization.
 
Three legacy extract formats exist — **EXPO, WIN, and QUEST** (the standardized format). BLS encourages all states to transition to the QUEST format, which is divided into five separate extract files (non-quarterly, quarterly, predecessor/successor, electronic contact information, and supplemental state use) and supports larger field sizes (up to 12 digits for wage fields versus 11 in legacy formats). A critical implementation detail: state UI agencies may impute unreported employment and wages using their own methods, but these imputations **do not follow BLS standards** and must not be included in QCEW extracts.
 
---
 
## Professional Employer Organizations distort industry and geography
 
### The scale of the problem
 
PEOs operate in a **co-employment relationship** with client businesses: the PEO becomes the employer of record for tax purposes, handling payroll, benefits, and employment-tax remittance, while the client retains day-to-day management of the workers. Classified under **NAICS 561330**, PEOs have grown explosively — employment surged **682% from 1992 to 2017** (341,884 to 2.7 million workers), while annual payroll grew from $5.7 billion to $144.6 billion. By 2018, County Business Patterns data showed approximately **3 million workers** on PEO payrolls, with the industry highly concentrated: **451 PEOs with over 1,000 employees** accounted for 88% of all PEO-reported employment. Florida alone hosted 520 PEOs with 1.3 million workers and $62.5 billion in annual payroll.
 
### Industry and geographic miscoding mechanisms
 
The core measurement problem is that leased workers risk being reported under the PEO's industry code (NAICS 561330) rather than the client firm's actual industry, and at the PEO's geographic location rather than the client's worksite. A comparison of 2018 data reveals the magnitude: Census Bureau programs (CBP) reported **~3 million PEO employees** nationally, while QCEW reported only **377,749** — a difference of approximately 2.6 million workers representing those that QCEW successfully reclassifies to client worksites under correct industry and geography codes. Without this reclassification, the Administrative and Support Services sector would be massively inflated while manufacturing, healthcare, technology, and hundreds of other industries would be understated.
 
Geographic distortion is equally severe. Florida's 1.3 million PEO "employees" in CBP data may not have performed any work in Florida — they appear there because PEOs pay employees from centralized payroll offices, and Census data follow the payroll processing location. QCEW addresses this through the MWR, which requires PEOs to break out employment by client worksite with correct industry and location coding. Several states, notably Florida, have enacted laws requiring **client-level reporting** by PEOs, vastly improving industry and county-level data quality. However, state laws vary, and completeness of client-level data remains uneven across jurisdictions.
 
### Cascading effects on downstream programs
 
Because QCEW serves as the benchmark and sampling frame for CES, JOLTS, OEWS, and other establishment surveys, and because BEA uses QCEW wages as the primary source for personal income estimation, any PEO-related distortions that survive the editing process cascade into these downstream products. The Annual Refiling Survey (contacting one-third of establishments annually) and the 6-step employment edit provide additional checkpoints, but the rapid growth of the PEO industry — representing roughly **1.7–2% of total U.S. employment** — makes this an ongoing data quality challenge.
 
---
 
## Publication timeline, revisions, and the asymmetric revision pattern
 
### Quarterly release schedule
 
QCEW data are published quarterly, approximately **5 to 6 months** after the end of the reference quarter. The typical schedule is: Q1 data (January–March) released in **late August/early September**; Q2 data (April–June) in **late November/early December**; Q3 data (July–September) in **late February/early March**; Q4 data (October–December) in **late May/early June**. Beginning with Q4 2024 data (released June 4, 2025), BLS publishes the County Employment and Wages news release and the full data update **simultaneously**, having previously used a staggered two-release format since 2018. Annual averages are first released with Q4 data and finalized with the following year's Q1 release.
 
### The revision cascade
 
QCEW data undergo a **declining number of revisions** depending on which quarter they represent: **Q1 data are published 5 times** (initial release, then 4 revisions at approximately quarterly intervals through the following September); **Q2 data 4 times; Q3 data 3 times; Q4 data 2 times**. All quarters of a given year remain open for updates until publication of Q1 data for the next year, at which point data become final and are never subsequently edited.
 
The magnitude of revisions is **relatively small** — typically less than **±0.10 to 0.15 percentage points** for employment levels. The **largest revision** consistently occurs from initial publication to the first revision, driven primarily by late-arriving employer reports, including reports confirming business deaths. This creates a characteristic pattern: initial publications tend to **slightly overstate** employment because the imputation method assumes non-reporters are still operating, while first revisions incorporate death confirmations that reduce counts. This asymmetric revision pattern is important for vintage-tracking in nowcasting applications, as it means early QCEW releases have a modest upward bias that diminishes with successive publications.
 
### News release adjustments versus website data
 
An important subtlety: the County Employment and Wages news release reports **adjusted** over-the-year growth rates that account for administrative corrections (multi-establishment reporting changes, state-verified classification corrections). These adjusted figures use modified prior-year base levels that are **never published** and do not match the unadjusted data available on the QCEW website. News release data are themselves **never updated** after publication. Website data, in contrast, are unadjusted and preliminary until finalized. Users performing time-series analysis should rely exclusively on the downloadable website data files rather than news release figures.
 
---
 
## QCEW as the benchmark and sampling universe for CES
 
### The annual benchmarking process
 
QCEW's most consequential downstream role is serving as the **annual benchmark for the Current Employment Statistics program**, which produces the monthly nonfarm payroll employment number. Each year, CES **re-anchors its March employment levels** to comprehensive QCEW population counts covering approximately 97% of CES-scope employment (the remaining 3% is benchmarked from sources outside QCEW). The preliminary benchmark estimate is released in **August/September** (e.g., September 2025 for March 2025), and the final benchmark revision is incorporated with the annual CES data revision released each **February** alongside the January Employment Situation report.
 
The benchmark revision quantifies the cumulative error in CES sample-based estimates relative to the QCEW near-census. Over **2002–2023, the average absolute benchmark revision was 255,000 jobs**, typically ±0.1–0.2% of total nonfarm employment. The largest revision in this period was **−902,000 for March 2009** during the Great Recession. The March 2024 benchmark (released February 2025) showed a revision of **−589,000 (−0.4%)** on a seasonally adjusted basis, with the not-seasonally-adjusted figure reaching −861,000 (−0.5%) after accounting for data reconstruction — among the largest outside pandemic periods.
 
### The birth-death model connection
 
A critical channel through which QCEW feeds CES is the **net birth-death model**, which estimates employment at new businesses not yet captured by the CES sample. The CES sample cannot capture new establishments on a timely basis because UI quarterly tax records only become available for updating the sampling frame **7 to 9 months after the reference month**, and additional time is needed for sampling and soliciting cooperation. In practice, BLS cannot sample new establishments until they are at least a year old.
 
The ARIMA time-series model underlying the birth-death adjustment is calibrated using **5 years of QCEW microdata** that capture the actual residual of births minus deaths. Monthly birth-death inputs are derived by comparing QCEW employment data including business openings and closings against the same data excluding births and imputing deaths. Since January 2011, BLS updates birth-death model inputs **quarterly** rather than annually. Effective January 2026, BLS modified the ARIMA component to incorporate **current sample information** (weighted-link-relative ratios) each month, exploiting the linear relationship between sample-based employment change and QCEW-based birth-death inputs. For the 2024 post-benchmark period, this modification reduced cumulative net birth-death forecasts by **228,000** relative to the unmodified approach.
 
The **asymmetric revision pattern** of benchmark revisions is fundamentally a birth-death model phenomenon. During economic slowdowns, the model continues adding jobs based on historical birth-death patterns even as actual business formation decelerates and closures accelerate, producing large negative benchmark revisions. During recoveries, the pattern reverses, with positive revisions as actual formation outpaces model expectations.
 
### QCEW as sampling frame
 
Beyond benchmarking, QCEW provides the operational sampling frame — the **Longitudinal Database (LDB)** — from which virtually all BLS establishment surveys draw their samples: CES (~631,000 worksites), JOLTS (~21,000 establishments), OEWS (~1.1 million), SOII (~230,000 annually), NCS/ECI (~7,000), and PPI. The LDB's longitudinal linkage using Fellegi-Sunter methodology (1969) enables tracking establishments over time, separating them into openings (births), continuous operations, and closings (deaths). This linkage is the foundation of the **Business Employment Dynamics (BED)** program, first published in 2003, which produces quarterly gross job gains and losses.
 
---
 
## How QCEW feeds GDP estimation through BEA
 
BEA uses QCEW data to develop the **wage and salary component of personal income** for national, state, and local-area accounts. Because personal income is a major component of both the National Income and Product Accounts (NIPAs) and Regional Economic Accounts, QCEW data directly influence GDP measurement on the income side (Gross Domestic Income). BLS provides QCEW microdata to BEA during each quarterly data-production cycle. For quarters where QCEW data are not yet available, BEA extrapolates from historical QCEW-based estimates using CES monthly data (employment × average weekly hours × average hourly earnings), replacing these projections with actual QCEW data as they become available — typically during the **annual update of the National Economic Accounts** each September.
 
The impact of QCEW data incorporation on GDP measurement can be substantial. For instance, the 2022 statistical discrepancy between GDP and GDI was revised from **−$75.6 billion to −$0.4 billion** after incorporating updated QCEW wage data. Since 2008, BLS has provided early Q1 macrodata to BEA specifically for estimating irregular finance-industry wage payments such as bonuses and stock options, which are concentrated in the first quarter and can significantly affect quarterly GDP estimates.
 
---
 
## Data quality limitations and their magnitudes
 
QCEW data are a **census, not a sample**, and are therefore not subject to sampling error. All error is **non-sampling error**: invalid county, industry, or ownership codes; data entry mistakes; over- or under-reporting of employment and wages; and imputation error for non-respondents.
 
**Coverage gaps** (~3% of civilian wage and salary employment) exclude self-employed workers, proprietors, certain agricultural and domestic workers, railroad workers, armed forces, elected officials, and national security agency employees. These exclusions are particularly relevant for comparing QCEW with CES, which covers some categories outside QCEW scope (railroad workers, religious organizations, certain nonprofits) while excluding some categories QCEW covers.
 
**Timing and reporting lags** represent the most significant limitation for real-time analysis. The ~5–6 month publication lag from end of quarter to initial release means that, for any given month, QCEW data become available **6 to 9 months later**. The 3% nonresponse rate requiring imputation, combined with the 2-quarter imputation window before establishments are dropped, means that initial releases contain some modeled data that will be revised.
 
**Industry miscoding** arises from the 3-year ARS cycle (establishments can operate under incorrect NAICS codes for up to 3 years between surveys), PEO-related classification errors (where client-level reporting is incomplete), and NAICS version transitions (2007, 2012, 2017, 2022). **Geographic miscoding** results from similar sources — establishments may report at headquarters rather than worksite locations, remote work arrangements can obscure actual work locations, and PEOs may centralize reporting at payroll office locations. QCEW assigns establishments with unidentified locations to an "unknown county" or "statewide" classification.
 
**Confidentiality suppression** protects employer identity at detailed industry-geography intersections, following Statistical Policy Working Paper 22. Suppressed cells are included in higher-level totals without being individually revealed. A notable change occurred in Q2 2022 when BLS began publishing previously suppressed state and local government data in states where such data are fully disclosable by law, releasing approximately **4.2 million in June 2022 employment** that had always been included in totals but not separately shown.
 
---
 
## Why QCEW is the essential ground truth for NFP nowcasting
 
### The near-census benchmark property
 
For any alternative nonfarm payroll nowcasting system — whether using Bayesian state-space models, payroll-provider signals, or other approaches — QCEW serves as the **definitive ground truth** against which all estimates must be evaluated. Both the BLS CES estimates and the ADP National Employment Report are explicitly benchmarked to QCEW annually. ADP's methodology (redesigned in 2022 with the Stanford Digital Economy Lab) applies matched-sample weekly employment growth from ADP payroll accounts to a **QCEW base-period level** of employment, then re-weights by industry, state, and employer size using QCEW distributions each February. ADP has stated that both the ADP and BLS estimates can be thought of as "a timely estimate or nowcast of the QCEW."
 
### Representativeness assessment of convenience samples
 
QCEW provides the universe distribution against which the representativeness of payroll-provider convenience samples can be quantified. ADP's own October 2025 representativeness analysis found that its sample of >26 million workers at >500,000 employers **overrepresents manufacturing** (9% of payroll accounts versus 3.5% of QCEW establishments) and **skews larger** than the national distribution, while being more representative of small/medium employers than BLS (employers with <50 workers account for 32.4% of ADP active employees versus 43.9% of QCEW employment versus <3% of BLS sample). BLS **overrepresents large employers** and trade, transportation, and utilities (32.9% of BLS sample versus 21.6% of QCEW). These distributional comparisons, possible only with QCEW as the reference universe, are essential for constructing appropriate re-weighting schemes in a Bayesian framework.
 
### The publication lag and its implications for state-space models
 
QCEW's **5–6 month publication lag** and quarterly frequency create a specific pattern of information arrival that state-space models must accommodate. For any given reference month, the information timeline is approximately: CES first estimate at T+5 weeks; CES second estimate at T+9 weeks; CES third estimate at T+13 weeks; QCEW initial release at T+6 to 9 months; QCEW first revision at T+9 to 12 months; final QCEW data at T+13 to 18 months (for Q1 data). A Bayesian state-space nowcasting model should treat each of these releases as successive signals about the latent true employment level, with QCEW carrying progressively higher precision (lower measurement error variance) as revisions accumulate.
 
The **Philadelphia Fed's Early Benchmark** approach (operational since November 2022) demonstrates the value of incorporating QCEW data as it arrives. By combining QCEW quarterly releases with CES monthly data, the Philadelphia Fed produces **quarterly early benchmark estimates** for all 50 states plus D.C. that closely predict the official BLS annual benchmark revisions. Their Q2 2022 estimate indicated only **10,500 net new jobs** versus 1,121,500 estimated by sum-of-states CES — a massive discrepancy later substantially confirmed by official benchmark revisions. This approach validates the strategy of using QCEW as a strong prior update within a state-space framework.
 
### Birth-death dynamics and model calibration
 
QCEW microdata, through the BED program, provide the **only direct measurement** of gross establishment births and deaths in the U.S. economy. Q4 2024 BED data showed gross job gains at opening establishments of **1.6 million** alongside 6.0 million jobs lost at contracting establishments and 6.1 million gained at expanding ones. These flows are far larger than net employment change, and their relative magnitudes shift systematically over the business cycle. A Bayesian nowcasting model that incorporates QCEW-derived birth-death flow data as informative priors on the pace of business formation — rather than relying solely on the CES birth-death model's ARIMA forecasts — should produce more accurate estimates during turning points, precisely when the standard model's errors are largest.
 
The **declining CES response rate** (from ~58% pre-2020 to **43% in 2024**) amplifies the importance of QCEW as a benchmark anchor. As CES first-release collection rates average only **60.4%**, the sample-based employment estimates carry increasing measurement error that only QCEW's near-census can resolve. A properly specified state-space model should assign time-varying measurement error variances to CES signals that reflect these declining response rates, while treating QCEW releases as high-precision observations when they become available.
 
---
 
## Conclusion
 
QCEW occupies a unique position in the U.S. statistical architecture — it is simultaneously an administrative byproduct of the UI tax system and the most comprehensive direct measurement of employment and wages in the economy. Its near-universe coverage of **12.2 million establishments** and **155.9 million workers** makes it the indispensable anchor for the entire constellation of BLS establishment surveys and BEA income accounts. For NFP nowcasting, three properties are paramount: its **census character** provides the ground truth that all sample-based and convenience-sample estimates approximate; its **asymmetric revision pattern** (with initial releases biased slightly upward due to delayed death confirmations) creates predictable vintage structure for state-space models; and its **quarterly publication cadence with a 5–6 month lag** defines the natural rhythm of information arrival against which higher-frequency payroll signals must be calibrated. The program's known vulnerabilities — PEO-driven industry and geographic distortion, UI modernization disruptions, the 3-year ARS reclassification cycle, and declining CES response rates that amplify benchmark revision magnitudes — are not merely limitations to acknowledge but parameters to model explicitly in a Bayesian framework that treats all employment signals as noisy observations of a latent true state.
