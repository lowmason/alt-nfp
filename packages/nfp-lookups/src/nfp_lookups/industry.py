"""Canonical BLS NAICS -> supersector -> domain mapping for CES employment data.

Provides the industry hierarchy as a Polars LazyFrame, CES series ID mappings,
CES-to-QCEW industry cross-mapping (:class:`IndustryEntry`, :data:`INDUSTRY_MAP`),
EN (QCEW) series ID construction, and index-builder functions for the
model layer's hierarchical indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nfp_lookups.series_ids import build_series_id

# --- Industry hierarchy ---
# Each row maps a 2-digit NAICS sector to its BLS CES supersector and domain.
# Sector codes use simplified forms: '31' (not '31-33'), '44' (not '44-45'),
# '48' (not '48-49').

_HIERARCHY_ROWS = [
    # Goods-producing (G)
    ('21', 'Mining', '10', 'Mining and Logging', 'G', 'Goods-producing'),
    ('23', 'Construction', '20', 'Construction', 'G', 'Goods-producing'),
    ('31', 'Manufacturing', '30', 'Manufacturing', 'G', 'Goods-producing'),
    # Service-providing (S)
    ('42', 'Wholesale Trade', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('44', 'Retail Trade', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('48', 'Transportation and Warehousing', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('22', 'Utilities', '40', 'Trade, Transportation, and Utilities', 'S', 'Service-providing'),
    ('51', 'Information', '50', 'Information', 'S', 'Service-providing'),
    ('52', 'Finance and Insurance', '55', 'Financial Activities', 'S', 'Service-providing'),
    ('53', 'Real Estate', '55', 'Financial Activities', 'S', 'Service-providing'),
    ('54', 'Professional and Technical Services', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('55', 'Management of Companies', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('56', 'Administrative and Waste Services', '60', 'Professional and Business Services', 'S', 'Service-providing'),
    ('61', 'Educational Services', '65', 'Private Education and Health Services', 'S', 'Service-providing'),
    ('62', 'Health Care and Social Assistance', '65', 'Private Education and Health Services', 'S', 'Service-providing'),
    ('71', 'Arts, Entertainment, and Recreation', '70', 'Leisure and Hospitality', 'S', 'Service-providing'),
    ('72', 'Accommodation and Food Services', '70', 'Leisure and Hospitality', 'S', 'Service-providing'),
    ('81', 'Other Services', '80', 'Other Services', 'S', 'Service-providing'),
]

INDUSTRY_HIERARCHY: pl.LazyFrame = pl.LazyFrame(
    {
        'sector_code': [r[0] for r in _HIERARCHY_ROWS],
        'sector_title': [r[1] for r in _HIERARCHY_ROWS],
        'supersector_code': [r[2] for r in _HIERARCHY_ROWS],
        'supersector_title': [r[3] for r in _HIERARCHY_ROWS],
        'domain_code': [r[4] for r in _HIERARCHY_ROWS],
        'domain_title': [r[5] for r in _HIERARCHY_ROWS],
    },
    schema={
        'sector_code': pl.Utf8,
        'sector_title': pl.Utf8,
        'supersector_code': pl.Utf8,
        'supersector_title': pl.Utf8,
        'domain_code': pl.Utf8,
        'domain_title': pl.Utf8,
    },
)


def _build_ces_series_id(supersector_code: str, sa: bool) -> str:
    """Build a BLS CES series ID for all-employees employment level.

    Parameters
    ----------
    supersector_code : str
        Two-digit BLS CES supersector code (e.g., '30').
    sa : bool
        True for seasonally adjusted, False for not seasonally adjusted.

    Returns
    -------
    str
        BLS series ID (e.g., 'CES3000000001' for Manufacturing SA).
    """
    prefix = 'S' if sa else 'U'
    return f'CE{prefix}{supersector_code}00000001'


# CES_SERIES_MAP: maps (supersector_code, sa) -> BLS series ID
# Includes all 10 supersectors plus total private ('05').
_SUPERSECTOR_CODES = sorted(
    {r[2] for r in _HIERARCHY_ROWS}
)  # ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']

CES_SERIES_MAP: dict[tuple[str, bool], str] = {}
for _code in _SUPERSECTOR_CODES + ['05']:
    CES_SERIES_MAP[(_code, True)] = _build_ces_series_id(_code, sa=True)
    CES_SERIES_MAP[(_code, False)] = _build_ces_series_id(_code, sa=False)


def get_domain_codes() -> list[str]:
    """Return sorted unique domain codes.

    Returns
    -------
    list[str]
        ['G', 'S']
    """
    return sorted({r[4] for r in _HIERARCHY_ROWS})


def get_supersector_codes() -> list[str]:
    """Return sorted unique supersector codes.

    Returns
    -------
    list[str]
        ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']
    """
    return sorted({r[2] for r in _HIERARCHY_ROWS})


def get_sector_codes() -> list[str]:
    """Return sorted unique sector codes.

    Returns
    -------
    list[str]
        All 18 sector codes in sorted order.
    """
    return sorted({r[0] for r in _HIERARCHY_ROWS})


def supersector_to_domain_idx() -> np.ndarray:
    """Map each supersector to its parent domain index.

    Returns
    -------
    np.ndarray
        Integer array of length n_supersectors. Entry i is the domain index
        for the i-th supersector (sorted order).
    """
    domain_codes = get_domain_codes()
    domain_to_idx = {d: i for i, d in enumerate(domain_codes)}

    ss_codes = get_supersector_codes()
    # Build supersector -> domain lookup from hierarchy rows
    ss_to_domain: dict[str, str] = {}
    for r in _HIERARCHY_ROWS:
        ss_to_domain[r[2]] = r[4]

    return np.array([domain_to_idx[ss_to_domain[ss]] for ss in ss_codes], dtype=np.intp)


def sector_to_supersector_idx() -> np.ndarray:
    """Map each sector to its parent supersector index.

    Returns
    -------
    np.ndarray
        Integer array of length n_sectors. Entry i is the supersector index
        for the i-th sector (sorted order).
    """
    ss_codes = get_supersector_codes()
    ss_to_idx = {ss: i for i, ss in enumerate(ss_codes)}

    sec_codes = get_sector_codes()
    # Build sector -> supersector lookup from hierarchy rows
    sec_to_ss: dict[str, str] = {}
    for r in _HIERARCHY_ROWS:
        sec_to_ss[r[0]] = r[2]

    return np.array([ss_to_idx[sec_to_ss[sec]] for sec in sec_codes], dtype=np.intp)


def get_supersector_components() -> dict[str, list[str]]:
    """Map each supersector to its component NAICS-based sector codes.

    Derived from :data:`_HIERARCHY_ROWS` (private sectors) plus government
    sectors ``'91'``, ``'92'``, ``'93'`` under supersector ``'90'``.

    Returns
    -------
    dict[str, list[str]]
        Supersector code -> sorted list of component sector codes.
        Sector codes are NAICS-based (e.g. ``'42'`` for Wholesale,
        ``'44'`` for Retail).
    """
    result: dict[str, list[str]] = {}
    for sector_code, _, ss_code, _, _, _ in _HIERARCHY_ROWS:
        result.setdefault(ss_code, []).append(sector_code)
    result['90'] = ['91', '92', '93']
    return {k: sorted(v) for k, v in sorted(result.items())}


# Supersector -> domain membership for aggregation.
_GOODS_SUPERSECTORS = frozenset({'10', '20', '30'})
_ALL_PRIVATE_SUPERSECTORS = frozenset(
    get_supersector_codes()
)  # excludes '90' which is added separately

DOMAIN_DEFINITIONS: dict[str, dict] = {
    '00': {'name': 'Total Non-Farm', 'includes_govt': True, 'goods_only': False},
    '05': {'name': 'Total Private', 'includes_govt': False, 'goods_only': False},
    '06': {'name': 'Goods-Producing', 'includes_govt': False, 'goods_only': True},
    '07': {'name': 'Service-Providing', 'includes_govt': True, 'goods_only': False},
    '08': {'name': 'Private Service-Providing', 'includes_govt': False, 'goods_only': False},
}


def get_domain_supersectors(domain_code: str) -> list[str]:
    """Return the supersector codes that compose a given domain.

    Parameters
    ----------
    domain_code : str
        One of ``'00'``, ``'05'``, ``'06'``, ``'07'``, ``'08'``.

    Returns
    -------
    list[str]
        Sorted list of supersector codes belonging to this domain.
    """
    all_private = sorted(_ALL_PRIVATE_SUPERSECTORS)
    goods = sorted(_GOODS_SUPERSECTORS)
    services_private = sorted(_ALL_PRIVATE_SUPERSECTORS - _GOODS_SUPERSECTORS)

    if domain_code == '00':
        return all_private + ['90']
    elif domain_code == '05':
        return all_private
    elif domain_code == '06':
        return goods
    elif domain_code == '07':
        return services_private + ['90']
    elif domain_code == '08':
        return services_private
    else:
        raise ValueError(f'Unknown domain code: {domain_code!r}')


# ---------------------------------------------------------------------------
# CES-to-QCEW industry cross-mapping
# ---------------------------------------------------------------------------

# CES domain/supersector/sector tuples: (ces_6digit, industry_code_2digit, industry_name)
_CES_DOMAIN = [
    ('000000', '00', 'Total Non-Farm'),
    ('050000', '05', 'Total Private'),
    ('060000', '06', 'Goods-Producing Industries'),
    ('070000', '07', 'Service-Providing Industries'),
    ('080000', '08', 'Private Service-Providing'),
]

_CES_SUPERSECTOR = [
    ('100000', '10', 'Natural Resources and Mining'),
    ('200000', '20', 'Construction'),
    ('300000', '30', 'Manufacturing'),
    ('400000', '40', 'Trade, Transportation, and Utilities'),
    ('500000', '50', 'Information'),
    ('550000', '55', 'Financial Activities'),
    ('600000', '60', 'Professional and Business Services'),
    ('650000', '65', 'Education and Health Services'),
    ('700000', '70', 'Leisure and Hospitality'),
    ('800000', '80', 'Other Services'),
    ('900000', '90', 'Government'),
]

_CES_SECTOR = [
    ('102100', '21', 'Mining, quarrying, and oil and gas extraction'),
    ('310000', '31', 'Durable goods'),
    ('320000', '32', 'Nondurable goods'),
    ('414200', '41', 'Wholesale trade'),
    ('420000', '42', 'Retail trade'),
    ('430000', '43', 'Transportation and warehousing'),
    ('442200', '22', 'Utilities'),
    ('555200', '52', 'Finance and insurance'),
    ('555300', '53', 'Real estate and rental and leasing'),
    ('605400', '54', 'Professional, scientific, and technical services'),
    ('605500', '55', 'Management of companies and enterprises'),
    ('605600', '56', 'Administrative and support and waste management'),
    ('656100', '61', 'Private educational services'),
    ('656200', '62', 'Health care and social assistance'),
    ('707100', '71', 'Arts, entertainment, and recreation'),
    ('707200', '72', 'Accommodation and food services'),
    ('909100', '91', 'Federal'),
    ('909200', '92', 'State government'),
    ('909300', '93', 'Local government'),
]

# CES sector code → NAICS code.  Most are identity but CES uses its own
# codes for Wholesale/Retail/Transportation that differ from NAICS.
CES_SECTOR_TO_NAICS: dict[str, str] = {
    '21': '21',     # Mining
    '31': '31',     # Manufacturing (NAICS 31-33 mapped to simplified '31')
    '32': '32',     # Nondurable goods (sub-split of manufacturing)
    '41': '42',     # CES Wholesale trade → NAICS 42
    '42': '44',     # CES Retail trade → NAICS 44 (simplified from 44-45)
    '43': '48',     # CES Transportation → NAICS 48 (simplified from 48-49)
    '22': '22',     # Utilities
    '52': '52',     # Finance and insurance
    '53': '53',     # Real estate
    '54': '54',     # Professional, scientific, technical
    '55': '55',     # Management of companies
    '56': '56',     # Administrative and support
    '61': '61',     # Educational services
    '62': '62',     # Health care
    '71': '71',     # Arts, entertainment, recreation
    '72': '72',     # Accommodation and food services
    '91': '91',     # Federal government
    '92': '92',     # State government
    '93': '93',     # Local government
}

# Government ownership codes (QCEW) → CES sector codes.
GOVT_OWNERSHIP_TO_SECTOR: dict[str, str] = {
    '1': '91',  # Federal
    '2': '92',  # State
    '3': '93',  # Local
}

# NAICS 3-digit manufacturing subsectors → CES durable/nondurable sector code.
# CES sector 31 = Durable goods, sector 32 = Nondurable goods.
# Used to split QCEW total manufacturing into the CES durable/nondurable grouping.
NAICS3_TO_MFG_SECTOR: dict[str, str] = {
    # Nondurable goods (CES sector 32)
    '311': '32', '312': '32', '313': '32', '314': '32', '315': '32',
    '316': '32', '322': '32', '323': '32', '324': '32', '325': '32', '326': '32',
    # Durable goods (CES sector 31)
    '321': '31', '327': '31', '331': '31', '332': '31', '333': '31',
    '334': '31', '335': '31', '336': '31', '337': '31', '339': '31',
}

# Supersectors that contain exactly one NAICS sector.  These supersector rows
# can be duplicated as sector rows by remapping the industry_code.
# Used by QCEW (and eventually SAE) to fill sector-level gaps.
SINGLE_SECTOR_SUPERSECTORS: dict[str, str] = {
    '20': '23',  # Construction
    '50': '51',  # Information
    '80': '81',  # Other Services
}


@dataclass(frozen=True)
class IndustryEntry:
    """Single industry mapping for cross-program consistency.

    Attributes
    ----------
    industry_code : str
        Unified 2-digit code (e.g. ``'00'``, ``'10'``).
    industry_type : str
        One of ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_name : str
        Human-readable industry name.
    ces_code : str
        Six-digit CES industry code (e.g. ``'000000'``, ``'100000'``).
    qcew_naics : str
        QCEW NAICS code for the CSV slice API (e.g. ``'10'``, ``'1011'``).
    en_industry : str
        Six-digit industry code for EN series ID construction.
    """

    industry_code: str
    industry_type: str
    industry_name: str
    ces_code: str
    qcew_naics: str
    en_industry: str


def _build_industry_map() -> list[IndustryEntry]:
    """Build the canonical industry map across domain, supersector, and sector levels."""
    entries: list[IndustryEntry] = []

    # Domain: aggregated from supersectors, no direct QCEW download code.
    for ces_code, code, name in _CES_DOMAIN:
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='domain',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics='',
            en_industry=ces_code,
        ))

    # Supersector: aggregated from component sectors, no single QCEW code.
    for ces_code, code, name in _CES_SUPERSECTOR:
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='supersector',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics='',
            en_industry=ces_code,
        ))

    # Sector: qcew_naics is the NAICS code that appears in QCEW responses.
    for ces_code, code, name in _CES_SECTOR:
        qcew = CES_SECTOR_TO_NAICS.get(code, code)
        entries.append(IndustryEntry(
            industry_code=code,
            industry_type='sector',
            industry_name=name,
            ces_code=ces_code,
            qcew_naics=qcew,
            en_industry=ces_code,
        ))

    return entries


INDUSTRY_MAP: list[IndustryEntry] = _build_industry_map()
"""Complete industry mapping table spanning domain, supersector, and sector levels."""


def qcew_to_sector() -> dict[str, str]:
    """Return a mapping from QCEW codes to NAICS-based sector codes.

    Maps both QCEW CSV API codes (``'1012'``, ``'1023'``, ...) and raw
    NAICS codes (``'21'``, ``'42'``, ...) to the NAICS-based sector codes
    used in :data:`_HIERARCHY_ROWS` and :func:`get_sector_codes`.

    Returns
    -------
    dict[str, str]
        QCEW code -> NAICS sector code (e.g. ``'1012'`` -> ``'21'``,
        ``'42'`` -> ``'42'``).
    """
    # 4-digit QCEW CSV API codes → NAICS-based sector codes
    mapping: dict[str, str] = {
        '1012': '21',  # Mining (NAICS 21)
        '1013': '22',  # Utilities (NAICS 22)
        '1021': '23',  # Construction (NAICS 23)
        '1022': '31',  # Manufacturing (NAICS 31-33, simplified to '31')
        '1023': '42',  # Wholesale Trade (NAICS 42)
        '1024': '44',  # Retail Trade (NAICS 44-45, simplified to '44')
        '1025': '48',  # Transportation (NAICS 48-49, simplified to '48')
        '1026': '51',  # Information (NAICS 51)
        '1027': '52',  # Finance and Insurance (NAICS 52)
        '1028': '53',  # Real Estate (NAICS 53)
        '1029': '54',  # Professional Services (NAICS 54)
        '102A': '55',  # Management of Companies (NAICS 55)
        '102B': '56',  # Administrative Services (NAICS 56)
        '102C': '61',  # Educational Services (NAICS 61)
        '102D': '62',  # Health Care (NAICS 62)
        '102E': '71',  # Arts and Recreation (NAICS 71)
        '102F': '72',  # Accommodation and Food (NAICS 72)
        '102G': '81',  # Other Services (NAICS 81)
    }
    # Raw NAICS codes that appear in QCEW response data → identity mapping
    for sector_code, _, _, _, _, _ in _HIERARCHY_ROWS:
        mapping[sector_code] = sector_code
    return mapping


def en_series_id(
    industry_entry: IndustryEntry,
    area: str = 'US000',
    ownership: str = '0',
) -> str:
    """Build an EN (QCEW) series ID for the given industry and area.

    Parameters
    ----------
    industry_entry : IndustryEntry
        Industry mapping entry (provides en_industry).
    area : str
        Area code: ``'US000'`` for national, or state FIPS + ``'000'``
        (e.g. ``'26000'`` for Michigan).
    ownership : str
        ``'0'`` = all ownerships, ``'5'`` = private.

    Returns
    -------
    str
        Full EN series ID string (e.g. ``'ENUUS0001010000001'``).
    """
    return build_series_id(
        'EN',
        seasonal='N',
        area=area,
        data_type='1',
        size='0',
        ownership=ownership,
        industry=industry_entry.en_industry,
    )


def en_series_id_for_state(
    industry_entry: IndustryEntry,
    state_fips: str,
    ownership: str = '0',
) -> str:
    """Build an EN series ID for a state-level series.

    Parameters
    ----------
    industry_entry : IndustryEntry
        Industry mapping entry.
    state_fips : str
        Two-digit state FIPS code (e.g. ``'26'`` for Michigan).
    ownership : str
        ``'0'`` = all ownerships, ``'5'`` = private.

    Returns
    -------
    str
        Full EN series ID with area ``{state_fips}000``.
    """
    area = f'{state_fips}000'
    return en_series_id(industry_entry, area=area, ownership=ownership)
