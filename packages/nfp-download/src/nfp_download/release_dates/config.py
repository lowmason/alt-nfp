"""Config for the release-dates pipeline.

Paths and publication definitions for scraping BLS release pages and
building release_dates.parquet / vintage_dates.parquet.
"""

from dataclasses import dataclass

# Re-exported for back-compat; canonical home is nfp_lookups.paths
from nfp_lookups.paths import (  # noqa: F401
    RELEASE_DATES_PATH,
    RELEASES_DIR,
    VINTAGE_DATES_PATH,
)

BASE_URL = 'https://www.bls.gov'
START_YEAR = 2003


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
