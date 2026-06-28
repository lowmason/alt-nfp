"""FRED/ALFRED real-time vintage access for national CES series.

Resolution tables map each vintage-store ``(industry_type, industry_code)``
key (NAICS-coded sectors) to its ALFRED series id. SA aggregates have NO
systematic ``CES…01`` archive, so they resolve to FRED friendly aliases
(PAYEMS, USPRIV, MANEMP, …); NSA resolves to systematic ``CEU…01`` except
``00`` (``PAYNSA``). All ids verified live 2026-06-27/28.

This module imports no ``nfp_*`` package (download-layer boundary).
"""
from __future__ import annotations

# (industry_type, industry_code NAICS) -> ALFRED SA series id.
CES_SERIES_SA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYEMS",
    ("total", "05"): "USPRIV",
    ("domain", "06"): "USGOOD",
    ("domain", "08"): "CES0800000001",
    ("supersector", "10"): "USMINE",
    ("supersector", "20"): "USCONS",
    ("supersector", "30"): "MANEMP",
    ("supersector", "40"): "USTPU",
    ("supersector", "50"): "USINFO",
    ("supersector", "55"): "USFIRE",
    ("supersector", "60"): "USPBS",
    ("supersector", "65"): "USEHS",
    ("supersector", "70"): "USLAH",
    ("supersector", "80"): "USSERV",
    ("sector", "21"): "CES1021000001",
    ("sector", "22"): "CES4422000001",
    ("sector", "31"): "DMANEMP",
    ("sector", "32"): "NDMANEMP",
    ("sector", "42"): "USWTRADE",
    ("sector", "44"): "USTRADE",
    ("sector", "48"): "CES4300000001",
    ("sector", "52"): "CES5552000001",
    ("sector", "53"): "CES5553000001",
    ("sector", "54"): "CES6054000001",
    ("sector", "55"): "CES6055000001",
    ("sector", "56"): "CES6056000001",
    ("sector", "61"): "CES6561000001",
    ("sector", "62"): "CES6562000001",
    ("sector", "71"): "CES7071000001",
    ("sector", "72"): "CES7072000001",
}

# NSA: 29/30 systematic CEU{8digit}01; only 00 -> PAYNSA.
CES_SERIES_NSA: dict[tuple[str, str], str] = {
    ("total", "00"): "PAYNSA",
    ("total", "05"): "CEU0500000001",
    ("domain", "06"): "CEU0600000001",
    ("domain", "08"): "CEU0800000001",
    ("supersector", "10"): "CEU1000000001",
    ("supersector", "20"): "CEU2000000001",
    ("supersector", "30"): "CEU3000000001",
    ("supersector", "40"): "CEU4000000001",
    ("supersector", "50"): "CEU5000000001",
    ("supersector", "55"): "CEU5500000001",
    ("supersector", "60"): "CEU6000000001",
    ("supersector", "65"): "CEU6500000001",
    ("supersector", "70"): "CEU7000000001",
    ("supersector", "80"): "CEU8000000001",
    ("sector", "21"): "CEU1021000001",
    ("sector", "22"): "CEU4422000001",
    ("sector", "31"): "CEU3100000001",
    ("sector", "32"): "CEU3200000001",
    ("sector", "42"): "CEU4142000001",
    ("sector", "44"): "CEU4200000001",
    ("sector", "48"): "CEU4300000001",
    ("sector", "52"): "CEU5552000001",
    ("sector", "53"): "CEU5553000001",
    ("sector", "54"): "CEU6054000001",
    ("sector", "55"): "CEU6055000001",
    ("sector", "56"): "CEU6056000001",
    ("sector", "61"): "CEU6561000001",
    ("sector", "62"): "CEU6562000001",
    ("sector", "71"): "CEU7071000001",
    ("sector", "72"): "CEU7072000001",
}


def resolve_series_id(industry_type: str, industry_code: str, *, sa: bool) -> str:
    """Return the ALFRED series id for a vintage-store industry key.

    Parameters
    ----------
    industry_type : str
        One of ``'total'``, ``'domain'``, ``'supersector'``, ``'sector'``.
    industry_code : str
        Store NAICS-coded industry code (e.g. ``'42'`` for wholesale).
    sa : bool
        ``True`` for seasonally adjusted (CES/alias), ``False`` for NSA (CEU/PAYNSA).

    Returns
    -------
    str
        The resolved ALFRED series id.

    Raises
    ------
    KeyError
        If ``(industry_type, industry_code)`` is not a stored CES key.
    """
    table = CES_SERIES_SA if sa else CES_SERIES_NSA
    return table[(industry_type, industry_code)]
