# QCEW Size-Class Scheme — Variable Spec

Defines the size-class aggregation schemes used to bucket establishment employment
from the QCEW Q1 size-class files (and provider microdata benchmarked against them).
Every scheme is an aggregation of the nine native QCEW establishment size classes.

> **Interpretation note.** The request enumerated `size_class_type` values as
> `'total', 'small', 'medium', 'large'`, but referred to `size_class_type == 'national'`
> for code `'0'`. There is no `'national'` value in the enumerated set, and QCEW
> `size_code` `'0'` is the *all-sizes total*, so `'national'` is read here as `'total'`.
> Swap the key if a separate geographic concept was actually intended.

## `size_class_type`

The four valid scheme names, ordered by increasing resolution (number of buckets).
The labels denote the **granularity of the partition**, not business size: `'large'`
is the finest scheme (most buckets), `'total'` the coarsest (one bucket).

| type       | levels | description                                                       |
|------------|--------|-------------------------------------------------------------------|
| `'total'`  | 1      | all sizes; native QCEW `size_code` `'0'`                          |
| `'small'`  | 3      | coarse 3-bucket aggregation                                       |
| `'medium'` | 5      | intermediate 5-bucket aggregation                                 |
| `'large'`  | 9      | the native QCEW establishment size classes (`size_code` `'1'`–`'9'`) |

```python
from typing import Literal

SizeClassType = Literal['total', 'small', 'medium', 'large']

size_class_type = ('total', 'small', 'medium', 'large')
```

## `size_class_code`

Nested mapping `{size_class_type: {code: (emp_lower, emp_upper)}}`. Each employment
range is **inclusive** on both ends; an upper bound of `None` means the class is
open-ended (no upper limit).

- `'large'` codes are the **native QCEW `size_code` values** — join directly to QCEW
  size-class files on these.
- `'total'` code `'0'` is the all-sizes total. As a *scheme* it is the sum over
  every native code; but in the **rebuilt store** the `'0'` headline is overridden
  to the published **area-levels** total, because the native-code sum drops
  suppressed (`disclosure_code='N'`) cells and would undercount the un-suppressed
  headline (see `store_rebuild.md` §8). The buckets need not sum to `'0'` under
  suppression.
- `'small'` / `'medium'` codes are **sequential aggregation indices**, not QCEW codes:
  the same code string means different ranges under different types, so always key by
  type first.

```python
size_class_code = {
    'total': {
        '0': (1, None),    # all sizes
    },
    'small': {
        '1': (1, 99),      # < 100
        '2': (100, 499),   # 100-499
        '3': (500, None),  # 500+
    },
    'medium': {
        '1': (1, 49),      # 1-49
        '2': (50, 99),     # 50-99
        '3': (100, 249),   # 100-249
        '4': (250, 499),   # 250-499
        '5': (500, None),  # 500+
    },
    'large': {
        '1': (1, 4),
        '2': (5, 9),
        '3': (10, 19),
        '4': (20, 49),
        '5': (50, 99),
        '6': (100, 249),
        '7': (250, 499),
        '8': (500, 999),
        '9': (1000, None),
    },
}
```

## `size_class_members` (aggregation map)

Because `'small'` and `'medium'` are aggregations of `'large'`, each of their buckets
maps to a tuple of native `size_code` values. The scheme is **fully nested**: every
`'small'` bucket is a union of `'medium'` buckets, and every `'medium'` bucket is a
union of `'large'` codes — so QCEW data delivered at native `size_code` can be rolled
up to either coarser scheme without re-binning from raw employment.

```python
size_class_members = {
    'small': {
        '1': ('1', '2', '3', '4', '5'),  # < 100   = medium 1 + 2
        '2': ('6', '7'),                 # 100-499 = medium 3 + 4
        '3': ('8', '9'),                 # 500+    = medium 5
    },
    'medium': {
        '1': ('1', '2', '3', '4'),       # 1-49
        '2': ('5',),                     # 50-99
        '3': ('6',),                     # 100-249
        '4': ('7',),                     # 250-499
        '5': ('8', '9'),                 # 500+
    },
}
```

Native-code coverage (sanity check):

| native `size_code` | range    | `medium` | `small` |
|--------------------|----------|----------|---------|
| `'1'`              | 1-4      | `'1'`    | `'1'`   |
| `'2'`              | 5-9      | `'1'`    | `'1'`   |
| `'3'`              | 10-19    | `'1'`    | `'1'`   |
| `'4'`              | 20-49    | `'1'`    | `'1'`   |
| `'5'`              | 50-99    | `'2'`    | `'1'`   |
| `'6'`              | 100-249  | `'3'`    | `'2'`   |
| `'7'`              | 250-499  | `'4'`    | `'2'`   |
| `'8'`              | 500-999  | `'5'`    | `'3'`   |
| `'9'`              | 1,000+   | `'5'`    | `'3'`   |

## Conventions

- Ranges are inclusive lower/upper integer employment counts; a `None` upper bound is
  open-ended.
- The `'small'` 3-way split (`<100 / 100-499 / 500+`) follows BEA's establishment-based
  small/medium/large business breakpoints, keeping it directly comparable to BEA's QCEW
  size-class work.
- Sizing is **point-in-time**: QCEW assigns each establishment to a class by its March
  employment. When benchmarking provider microdata, bin the provider's March
  (third-month) establishment counts to match.
- `'small'` / `'medium'` codes are scheme-local — do not join them to raw QCEW files;
  only `'large'` and `'total'` codes match the native `size_code`.

## Usage (polars)

Roll up a QCEW size-class frame (native `size_code`, `'1'`–`'9'`, plus an `emp`
measure) to a coarser scheme via a flat native→bucket lookup:

```python
import polars as pl


def native_to_scheme(scheme: str) -> dict[str, str]:
    return {
        native: code
        for code, natives in size_class_members[scheme].items()
        for native in natives
    }


scheme = 'small'
rolled = (
    df.with_columns(
        pl.col('size_code')
        .replace_strict(native_to_scheme(scheme), default=None)
        .alias(f'{scheme}_code')
    )
    .group_by(f'{scheme}_code')
    .agg(pl.col('emp').sum())
)
```