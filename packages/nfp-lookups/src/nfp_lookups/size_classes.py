"""QCEW establishment size-class schemes (``specs/size_classes.md``).

Four nested aggregation schemes over the nine native QCEW establishment size
classes (``size_code`` ``'1'``-``'9'``, assigned by March employment):

- ``'total'``  — 1 bucket  (all sizes; native code ``'0'``)
- ``'small'``  — 3 buckets
- ``'medium'`` — 5 buckets
- ``'large'``  — 9 buckets (the native QCEW size classes themselves)

Because the schemes are fully nested, a frame delivered at native ``size_code``
rolls up to any coarser scheme without re-binning from raw employment
(:func:`native_to_scheme`).
"""

from __future__ import annotations

SIZE_CLASS_TYPES: tuple[str, ...] = ("total", "small", "medium", "large")
"""Scheme names, coarse → fine (number of buckets)."""

NATIVE_SIZE_CODES: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
"""The nine native QCEW ``size_code`` values (the ``'large'`` scheme)."""

# {size_class_type: {code: (emp_lower, emp_upper | None)}} — inclusive ranges.
SIZE_CLASS_CODE: dict[str, dict[str, tuple[int, int | None]]] = {
    "total": {
        "0": (1, None),  # all sizes
    },
    "small": {
        "1": (1, 99),     # < 100
        "2": (100, 499),  # 100-499
        "3": (500, None),  # 500+
    },
    "medium": {
        "1": (1, 49),
        "2": (50, 99),
        "3": (100, 249),
        "4": (250, 499),
        "5": (500, None),
    },
    "large": {
        "1": (1, 4),
        "2": (5, 9),
        "3": (10, 19),
        "4": (20, 49),
        "5": (50, 99),
        "6": (100, 249),
        "7": (250, 499),
        "8": (500, 999),
        "9": (1000, None),
    },
}

# {scheme: {bucket_code: native size_codes}} for the aggregated schemes.
SIZE_CLASS_MEMBERS: dict[str, dict[str, tuple[str, ...]]] = {
    "small": {
        "1": ("1", "2", "3", "4", "5"),  # < 100   = medium 1 + 2
        "2": ("6", "7"),                  # 100-499 = medium 3 + 4
        "3": ("8", "9"),                  # 500+    = medium 5
    },
    "medium": {
        "1": ("1", "2", "3", "4"),  # 1-49
        "2": ("5",),                 # 50-99
        "3": ("6",),                 # 100-249
        "4": ("7",),                 # 250-499
        "5": ("8", "9"),             # 500+
    },
}


def native_to_scheme(scheme: str) -> dict[str, str]:
    """Map each native ``size_code`` (``'1'``-``'9'``) to its *scheme* bucket code.

    - ``'large'`` is the identity (native codes are the scheme codes).
    - ``'total'`` maps every native to the all-sizes code ``'0'``.
    - ``'small'`` / ``'medium'`` use :data:`SIZE_CLASS_MEMBERS`.

    Raises
    ------
    ValueError
        If *scheme* is not one of :data:`SIZE_CLASS_TYPES`.
    """
    if scheme == "large":
        return {c: c for c in NATIVE_SIZE_CODES}
    if scheme == "total":
        return dict.fromkeys(NATIVE_SIZE_CODES, "0")
    if scheme in SIZE_CLASS_MEMBERS:
        return {
            native: code
            for code, natives in SIZE_CLASS_MEMBERS[scheme].items()
            for native in natives
        }
    raise ValueError(f"Unknown size_class_type {scheme!r}; valid: {SIZE_CLASS_TYPES}")
