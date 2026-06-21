# alt-nfp

Bayesian state-space nowcasting of US nonfarm payrolls (NFP) from real-time
data vintages: CES survey prints, QCEW administrative anchors, private payroll
provider microdata, and cyclical indicators, combined with strict as-of
censoring so every backtest sees only what was knowable on the day.

This is the v2 repo: the data layer is ported from a prior reference
implementation, and the model layer is rewritten in JAX/NumPyro. The port was
gated against the old repo for fidelity, but that repo is a work-in-progress,
not validated truth — correctness is validated against external ground truth
(published BLS / ALFRED real-time vintages). See [docs/](docs/) and
[specs/](specs/) for the design record.

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

The **vintage store** lives in S3-compatible object storage (a local MinIO by
default), configured via a gitignored `.env`:

```
NFP_STORE_URI=s3://alt-nfp/store
AWS_ACCESS_KEY_ID=…
AWS_SECRET_ACCESS_KEY=…
AWS_ENDPOINT_URL=http://127.0.0.1:9000
```

With `NFP_STORE_URI` unset the store falls back to the local `data/store/`
(this is how CI runs). Tests that need the store self-skip when it is
unavailable. `scripts/mirror_store.py` uploads a local store into the bucket.

The rest of `data/` (downloads, intermediate pipeline artifacts, proprietary
provider files) stays local and is not in the repository. The directory
layout is centralized in `nfp_lookups.paths`; set `NFP_BASE_DIR` to point the
layout at a different root (e.g. a snapshot directory).

## License

MIT
