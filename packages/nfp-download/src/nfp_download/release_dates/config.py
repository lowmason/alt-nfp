"""Config for the release-dates pipeline.

Paths and publication definitions for scraping BLS release pages and
building release_dates.parquet / vintage_dates.parquet.
"""

from dataclasses import dataclass
from pathlib import Path

from nfp_lookups.paths import DOWNLOADS_DIR, INTERMEDIATE_DIR

BASE_URL = 'https://www.bls.gov'
START_YEAR = 2003

# Output paths
RELEASES_DIR: Path = DOWNLOADS_DIR / 'releases'
RELEASE_DATES_PATH: Path = INTERMEDIATE_DIR / 'release_dates.parquet'
VINTAGE_DATES_PATH: Path = INTERMEDIATE_DIR / 'vintage_dates.parquet'


@dataclass(frozen=True)
class Publication:
    """BLS publication: name, series code, index URL, and frequency."""

    name: str
    series: str
    index_url: str
    frequency: str  # 'monthly' | 'quarterly'


PUBLICATIONS: list[Publication] = [
    Publication(
        name='ces',
        series='empsit',
        index_url=f'{BASE_URL}/bls/news-release/empsit.htm',
        frequency='monthly',
    ),
    # Publication(
    #     name='sae',
    #     series='laus',
    #     index_url=f'{BASE_URL}/bls/news-release/laus.htm',
    #     frequency='monthly',
    # ),
    Publication(
        name='qcew',
        series='cewqtr',
        index_url=f'{BASE_URL}/bls/news-release/cewqtr.htm',
        frequency='quarterly',
    ),
]
