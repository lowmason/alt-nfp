# alt-nfp

Bayesian state-space nowcasting of US nonfarm payrolls (NFP) from real-time
data vintages: CES survey prints, QCEW administrative anchors, private payroll
provider microdata, and cyclical indicators, combined with strict as-of
censoring so every backtest sees only what was knowable on the day.

This is the v2 repo: the data layer is ported from a working reference
implementation, and the model layer is being rewritten in JAX (dynamax /
NumPyro) behind parity gates. See [docs/](docs/) and [specs/](specs/) for the
design record.

## Layout

A [uv](https://docs.astral.sh/uv/) workspace with a linear dependency chain:

```
nfp-lookups  →  nfp-download  →  nfp-ingest  →  nfp-vintages
```

| Package | Role |
|---|---|
| [`nfp-lookups`](packages/nfp-lookups/) | Static reference data: schemas, industry/geography hierarchies, revision schedules, series-ID grammar, canonical paths |
| [`nfp-download`](packages/nfp-download/) | HTTP clients and scrapers for BLS and FRED — fetching only, no transformation |
| [`nfp-ingest`](packages/nfp-ingest/) | Vintage store, as-of censoring, panel construction, provider ingestion, compositing |
| [`nfp-vintages`](packages/nfp-vintages/) | Historical vintage reconstruction pipeline (CES triangular revisions, QCEW bulk) and the `alt-nfp` CLI |

Each package has a `CLAUDE.md` with its internal map.

## Quickstart

```bash
uv sync                              # install workspace + dev tools
uv run pytest -m "not network"       # test suite (network-marked tests excluded)
uv run ruff check .                  # lint
uv run alt-nfp --help                # vintage pipeline CLI
```

## Data

`data/` holds proprietary provider files and pipeline artifacts; it is not in
the repository. Tests that need the vintage store self-skip when it is absent.
The directory layout is centralized in `nfp_lookups.paths`; set `NFP_BASE_DIR`
to point the layout at a different root (e.g. a snapshot directory).

## License

MIT
