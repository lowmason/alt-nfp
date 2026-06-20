Vintage Store Coverage Integration Test

File

New test file: tests/test_store_coverage.py

Expected industry sets (derived from actual store contents)

The test needs three canonical industry sets:

CES (SA and NSA): 38 combos — national/00 + 4 domains (05,06,07,08) + 11 supersectors (10,20,30,40,50,55,60,65,70,80,90) + 22 sectors (21,22,23,31,32,42,44,48,51,52,53,54,55,56,61,62,71,72,81,91,92,93). CES sector codes are recoded to NAICS (41→42, 42→44, 43→48) and single-sector supersectors are duplicated as sectors (20→23, 50→51, 80→81).

QCEW (NSA only): 38 combos — national/00 + 4 domains + 11 supersectors + 22 sectors (same NAICS-based sector set as CES)

These will be constructed from src/alt_nfp/lookups/industry.py (INDUSTRY_MAP, \_CES_SECTOR_TO_NAICS, SINGLE_SECTOR_SUPERSECTORS).

What to check

For each unique vintage_date \>= 2022-01-01 in the store:

Presence: every expected (industry_type, industry_code) for each source tag (ces_sa, ces_nsa, qcew) has at least one observation row

CES revision coverage: for a given (vintage_date, industry_type, industry_code), the available ref_date values should each have the expected revisions (0, 1, 2) depending on the gap between ref_date and vintage_date — revision k should exist when vintage_date - ref_date \>= (k+1) months

QCEW revision coverage: only national/00 has revisions 0-4 (from the revisions CSV). All other QCEW industries have only revision 0 (from bulk). The test checks that national/00 has the expected revision set per ref_date quarter, and other industries have revision 0

Parameterization

The industry level scope will be parameterized so the caller can run for:

'all' — domain + supersector + sector (default)

'supersector' — supersector only

'sector' — sector only

'domain' — domain only

This is passed as a pytest parameter or fixture, using @pytest.mark.parametrize.

Test marker

No new marker needed. The test reads local files, not the network. It will use pytest.importorskip or a skipif on data/store/ existence so it passes gracefully when the store isn't built.

Key design decisions

Build the expected industry sets programmatically from INDUSTRY_MAP and SINGLE_SECTOR_SUPERSECTORS rather than hardcoding, so the test stays in sync with the lookup definitions

Group the store data by (source, seasonally_adjusted, vintage_date) and check coverage per group to avoid O(vintage_dates x industries) individual assertions — collect all gaps and report them in one failure message

Separate the presence test (does each industry appear at all for this source?) from the revision coverage test (are the right revisions present per ref_date?)

The presence check allows for a small tolerance: some vintage_dates near the boundary may legitimately lack data for the newest ref_dates. The test will check against ref_date values that are well within the publication window (e.g., ref_date \<= vintage_date - 2 months for CES)