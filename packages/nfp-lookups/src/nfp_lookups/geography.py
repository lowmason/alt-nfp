"""Census Region and Division lookups by state FIPS.

Provides :data:`STATES` (50 states + DC + Puerto Rico), :data:`FIPS_TO_REGION`,
and :data:`FIPS_TO_DIVISION` for aggregating state-level estimates to Census
Regions (1-4) and Divisions (01-09). Puerto Rico (FIPS ``'72'``) is assigned
to Region 3 (South) and Division 05 (South Atlantic), the same as Florida.

Optionally exposes :data:`GEOGRAPHY_HIERARCHY` as a Polars LazyFrame for
consistency with :data:`~alt_nfp.lookups.industry.INDUSTRY_HIERARCHY`.
"""

from __future__ import annotations

import polars as pl

# 50 states + DC + Puerto Rico (2-digit FIPS codes)
STATES: list[str] = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12',
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23',
    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
    '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55',
    '56', '72',
]

# Census Bureau: Region 1 Northeast, 2 Midwest, 3 South, 4 West
# Division 01-09 (2-digit string)
_FIPS_REGION_DIVISION: dict[str, tuple[str, str]] = {
    '01': ('3', '06'),   # AL  South, East South Central
    '02': ('4', '09'),   # AK  West, Pacific
    '04': ('4', '08'),   # AZ  West, Mountain
    '05': ('3', '07'),   # AR  South, West South Central
    '06': ('4', '09'),   # CA  West, Pacific
    '08': ('4', '08'),   # CO  West, Mountain
    '09': ('1', '01'),   # CT  Northeast, New England
    '10': ('3', '05'),   # DE  South, South Atlantic
    '11': ('3', '05'),   # DC  South, South Atlantic
    '12': ('3', '05'),   # FL  South, South Atlantic
    '13': ('3', '05'),   # GA  South, South Atlantic
    '15': ('4', '09'),   # HI  West, Pacific
    '16': ('4', '08'),   # ID  West, Mountain
    '17': ('2', '03'),   # IL  Midwest, East North Central
    '18': ('2', '03'),   # IN  Midwest, East North Central
    '19': ('2', '04'),   # IA  Midwest, West North Central
    '20': ('2', '04'),   # KS  Midwest, West North Central
    '21': ('3', '06'),   # KY  South, East South Central
    '22': ('3', '07'),   # LA  South, West South Central
    '23': ('1', '01'),   # ME  Northeast, New England
    '24': ('3', '05'),   # MD  South, South Atlantic
    '25': ('1', '01'),   # MA  Northeast, New England
    '26': ('2', '03'),   # MI  Midwest, East North Central
    '27': ('2', '04'),   # MN  Midwest, West North Central
    '28': ('3', '06'),   # MS  South, East South Central
    '29': ('2', '04'),   # MO  Midwest, West North Central
    '30': ('4', '08'),   # MT  West, Mountain
    '31': ('2', '04'),   # NE  Midwest, West North Central
    '32': ('4', '08'),   # NV  West, Mountain
    '33': ('1', '01'),   # NH  Northeast, New England
    '34': ('1', '02'),   # NJ  Northeast, Mid-Atlantic
    '35': ('4', '08'),   # NM  West, Mountain
    '36': ('1', '02'),   # NY  Northeast, Mid-Atlantic
    '37': ('3', '05'),   # NC  South, South Atlantic
    '38': ('2', '04'),   # ND  Midwest, West North Central
    '39': ('2', '03'),   # OH  Midwest, East North Central
    '40': ('3', '07'),   # OK  South, West South Central
    '41': ('4', '09'),   # OR  West, Pacific
    '42': ('1', '02'),   # PA  Northeast, Mid-Atlantic
    '44': ('1', '01'),   # RI  Northeast, New England
    '45': ('3', '05'),   # SC  South, South Atlantic
    '46': ('2', '04'),   # SD  Midwest, West North Central
    '47': ('3', '06'),   # TN  South, East South Central
    '48': ('3', '07'),   # TX  South, West South Central
    '49': ('4', '08'),   # UT  West, Mountain
    '50': ('1', '01'),   # VT  Northeast, New England
    '51': ('3', '05'),   # VA  South, South Atlantic
    '53': ('4', '09'),   # WA  West, Pacific
    '54': ('3', '05'),   # WV  South, South Atlantic
    '55': ('2', '03'),   # WI  Midwest, East North Central
    '56': ('4', '08'),   # WY  West, Mountain
    '72': ('3', '05'),   # PR  South, South Atlantic (same as FL)
}

FIPS_TO_REGION: dict[str, str] = {
    fips: r for fips, (r, _) in _FIPS_REGION_DIVISION.items()
}

FIPS_TO_DIVISION: dict[str, str] = {
    fips: d for fips, (_, d) in _FIPS_REGION_DIVISION.items()
}

# Region and division names for human-readable output
REGION_NAMES: dict[str, str] = {
    '1': 'Northeast',
    '2': 'Midwest',
    '3': 'South',
    '4': 'West',
}

DIVISION_NAMES: dict[str, str] = {
    '01': 'New England',
    '02': 'Mid-Atlantic',
    '03': 'East North Central',
    '04': 'West North Central',
    '05': 'South Atlantic',
    '06': 'East South Central',
    '07': 'West South Central',
    '08': 'Mountain',
    '09': 'Pacific',
}


GEOGRAPHY_HIERARCHY: pl.LazyFrame = pl.LazyFrame(
    {
        'state_fips': list(_FIPS_REGION_DIVISION.keys()),
        'region_code': [r for r, _ in _FIPS_REGION_DIVISION.values()],
        'region_name': [REGION_NAMES[r] for r, _ in _FIPS_REGION_DIVISION.values()],
        'division_code': [d for _, d in _FIPS_REGION_DIVISION.values()],
        'division_name': [DIVISION_NAMES[d] for _, d in _FIPS_REGION_DIVISION.values()],
    },
    schema={
        'state_fips': pl.Utf8,
        'region_code': pl.Utf8,
        'region_name': pl.Utf8,
        'division_code': pl.Utf8,
        'division_name': pl.Utf8,
    },
)
