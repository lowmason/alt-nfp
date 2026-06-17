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
# Vintage-store industry taxonomy (industry_type × ownership)  — store_rebuild
# ---------------------------------------------------------------------------
#
# The rebuilt vintage store splits the two axes the legacy single
# ``industry_code`` column conflated: ``industry_type`` (level within the
# industry partition) and ``ownership``. See ``specs/store_rebuild.md`` §3.
#
# ``industry_type`` retires the legacy ``'national'`` value (which collided with
# ``geographic_type='national'``); the ``'00'`` total-nonfarm series is now the
# ``(total, total)`` anchor and ``'05'`` total private the ``(total, private)``
# private-tree root.

INDUSTRY_TYPES: tuple[str, ...] = ('total', 'domain', 'supersector', 'sector')
"""Valid ``industry_type`` values in the rebuilt store (``'national'`` retired)."""

OWNERSHIPS: tuple[str, ...] = ('total', 'private', 'government')
"""Valid ``ownership`` values; ``'government'`` is reserved (deferred, §11)."""

_SUPERSECTOR_PRIVATE_CODES: tuple[str, ...] = (
    '10', '20', '30', '40', '50', '55', '60', '65', '70', '80',
)
_SECTOR_PRIVATE_CODES: tuple[str, ...] = (
    '11', '21', '22', '23', '31', '32', '42', '44', '48', '51',
    '52', '53', '54', '55', '56', '61', '62', '71', '72', '81',
)

# Canonical (industry_type, ownership, industry_code) taxonomy of the stored
# axis. ``ownership='government'`` and codes ``07``/``90``-``93`` are deferred
# (§11) and intentionally absent. Note code ``55`` appears at *both* the
# supersector (Financial Activities) and sector (Management of companies) level.
_TAXONOMY_ROWS: list[tuple[str, str, str]] = [
    ('total', 'total', '00'),      # total nonfarm — anchor, stored not modeled
    ('total', 'private', '05'),    # total private — private-tree root
    ('domain', 'private', '06'),   # goods-producing
    ('domain', 'private', '08'),   # private service-providing
    *[('supersector', 'private', c) for c in _SUPERSECTOR_PRIVATE_CODES],
    *[('sector', 'private', c) for c in _SECTOR_PRIVATE_CODES],
]

# (industry_type, industry_code) → ownership. Keyed on the *pair* so that code
# ``'55'`` survives the cross-level collision (supersector 55 vs sector 55)
# instead of being collapsed into one entry.
INDUSTRY_TAXONOMY: dict[tuple[str, str], str] = {
    (itype, code): own for itype, own, code in _TAXONOMY_ROWS
}


def ownership_for(industry_type: str, industry_code: str) -> str:
    """Return the ``ownership`` for a stored ``(industry_type, industry_code)``.

    Raises
    ------
    ValueError
        If the pair is not part of the stored taxonomy (e.g. a deferred
        government code, or a code at the wrong level).
    """
    try:
        return INDUSTRY_TAXONOMY[(industry_type, industry_code)]
    except KeyError:
        raise ValueError(
            f'No stored taxonomy entry for industry_type={industry_type!r}, '
            f'industry_code={industry_code!r}'
        ) from None


def codes_for(industry_type: str, ownership: str | None = None) -> list[str]:
    """Return the sorted stored ``industry_code``s at *industry_type*.

    Optionally restrict to a single *ownership*.
    """
    return sorted(
        code
        for (itype, code), own in INDUSTRY_TAXONOMY.items()
        if itype == industry_type and (ownership is None or own == ownership)
    )


def industry_types_for_code(industry_code: str) -> list[str]:
    """Return the sorted ``industry_type``s a code is stored at.

    Code ``'55'`` returns ``['sector', 'supersector']`` — the lone cross-level
    collision the store keys must keep distinct.
    """
    return sorted(
        {itype for (itype, code) in INDUSTRY_TAXONOMY if code == industry_code}
    )


def remap_industry_type(
    industry_type: str, industry_code: str
) -> tuple[str, str]:
    """Map a legacy ``(industry_type, industry_code)`` to ``(industry_type, ownership)``.

    Bridges the pre-rebuild store (no ``ownership`` axis; ``'national'`` for the
    ``00`` total) to the rebuilt taxonomy so the ≤2023 history-consistency gate
    (``specs/store_rebuild.md`` §10) keys on the same axes:

    - ``('national', '00')`` → ``('total', 'total')``
    - ``('domain', '05')``   → ``('total', 'private')``
    - ``('domain', '06'|'08')`` → ``('domain', 'private')``
    - supersectors / sectors → unchanged level, ``ownership='private'``

    Already-rebuilt inputs are idempotent. Raises ``ValueError`` for codes
    outside the stored taxonomy (deferred government, etc.).
    """
    if industry_type == 'national':
        new_type = 'total'
    elif industry_type == 'domain' and industry_code == '05':
        new_type = 'total'
    else:
        new_type = industry_type
    return new_type, ownership_for(new_type, industry_code)


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


# ---------------------------------------------------------------------------
# Rebuild QCEW→CES private crosswalk (ces_qcew_industry.md §9)  — store_rebuild
# ---------------------------------------------------------------------------
#
# Reconstructs the CES *private* published series from QCEW national, private
# (own_code=='5'), agglvl-coded cells. Distinct from the legacy ``qcew_to_sector``
# CSV-API scheme above: this is the agglvl 13/14/15/16 ``singlefile`` pull spec.

QCEW_OWN_PRIVATE: str = '5'
"""QCEW ``own_code`` for private establishments (ces_qcew_industry.md §3.1)."""

QCEW_OWN_TOTAL: str = '0'
"""QCEW ``own_code`` for total-covered (all ownerships) employment."""

# Single published total-covered area row (own_code=0, industry '10' all-industries,
# agglvl 10) → CES '00' total-nonfarm anchor. Verified 2026-06-17: own_code=0 returns
# exactly one area row at (agglvl 10, industry '10'). Total-covered ≠ CES nonfarm
# (incl. agriculture / UI-covered); the reconstruction gate bands the residual (T3/T4).
QCEW_TOTAL_PULL: dict[str, str] = {
    'industry_code': '10',
    'agglvl_code': '10',
}
"""Industry/agglvl coordinates of the QCEW total-covered area row → CES '00'
anchor. ``own_code`` is :data:`QCEW_OWN_TOTAL` (kept separate to mirror
:data:`QCEW_OWN_PRIVATE`, the private-track own_code filter)."""

QCEW_AREA_NATIONAL: str = 'US000'
"""QCEW ``area_fips`` for the national total."""

QCEW_AGGLVL: dict[str, str] = {
    'domain': '12',
    'supersector': '13',
    'sector': '14',
    'naics_3': '15',
    'naics_4': '16',
}
"""QCEW national-by-ownership aggregation-level codes (ces_qcew_industry.md §3.2)."""

# CES private sector id → (QCEW industry_codes, agglvl_code). Leaf pulls: most
# are direct agglvl-14 NAICS sectors; Logging is 4-digit 1133; Durable/Nondurable
# sum the 3-digit subsectors (ces_qcew_industry.md §6.2-6.3).
QCEW_SECTOR_PULLS: dict[str, tuple[tuple[str, ...], str]] = {
    '11': (('1133',), '16'),                          # Logging only (NAICS 1133)
    '21': (('21',), '14'),
    '22': (('22',), '14'),
    '23': (('23',), '14'),
    '31': (  # Durable goods
        ('321', '327', '331', '332', '333', '334', '335', '336', '337', '339'),
        '15',
    ),
    '32': (  # Nondurable goods
        ('311', '312', '313', '314', '315', '316', '322', '323', '324', '325', '326'),
        '15',
    ),
    '42': (('42',), '14'),
    '44': (('44-45',), '14'),
    '48': (('48-49',), '14'),
    '51': (('51',), '14'),
    '52': (('52',), '14'),
    '53': (('53',), '14'),
    '54': (('54',), '14'),
    '55': (('55',), '14'),
    '56': (('56',), '14'),
    '61': (('61',), '14'),
    '62': (('62',), '14'),
    '71': (('71',), '14'),
    '72': (('72',), '14'),
    '81': (('81',), '14'),
}

# CES supersector id → single agglvl-13 QCEW pull (``qcew_code``) plus its member
# CES sector ids. ``qcew_code=None`` forces a member-sector sum: only ``'10'``
# (the QCEW ``1011`` aggregate over-includes all agriculture, ces_qcew §4 Exc. A).
QCEW_SUPERSECTOR: dict[str, dict[str, object]] = {
    '10': {'qcew_code': None, 'sectors': ('11', '21')},
    '20': {'qcew_code': '1012', 'sectors': ('23',)},
    '30': {'qcew_code': '1013', 'sectors': ('31', '32')},
    '40': {'qcew_code': '1021', 'sectors': ('42', '44', '48', '22')},
    '50': {'qcew_code': '1022', 'sectors': ('51',)},
    '55': {'qcew_code': '1023', 'sectors': ('52', '53')},
    '60': {'qcew_code': '1024', 'sectors': ('54', '55', '56')},
    '65': {'qcew_code': '1025', 'sectors': ('61', '62')},
    '70': {'qcew_code': '1026', 'sectors': ('71', '72')},
    '80': {'qcew_code': '1027', 'sectors': ('81',)},
}

# CES top aggregate id → component ids (pure roll-ups, ces_qcew_industry.md §5).
# ``06``/``08`` are ``industry_type='domain'``; ``05`` is ``industry_type='total'``.
QCEW_DOMAIN: dict[str, tuple[str, ...]] = {
    '06': ('10', '20', '30'),
    '08': ('40', '50', '55', '60', '65', '70', '80'),
    '05': ('06', '08'),
}


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
