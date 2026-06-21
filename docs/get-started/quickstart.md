# Quickstart

This page walks through the four production `alt-nfp` commands. Before you start, make sure
you have [installed the workspace](installation.md) and configured your `.env` with valid
store credentials.

## Explore the CLI

```bash
uv run alt-nfp --help
```

```
Usage: alt-nfp [OPTIONS] COMMAND [ARGS]...

  alt-nfp — Bayesian state-space NFP nowcasting CLI.

Options:
  --help  Show this message and exit.

Commands:
  status    Store coverage + uncaptured/corrected alarm.
  update    Capture knowable prints for --as-of, append to store.
  snapshot  Bake a hash-pinned ModelData handoff.
  watch     BLS-feed-driven trigger (designed for cron).
```

## Check store coverage

Before running an update, inspect what the store already contains:

```bash
uv run alt-nfp status
```

`status` is read-only. It reports:

- The resolved store URI and whether it is `REMOTE`, `LOCAL`, or `CANONICAL`
- Per-source coverage: latest/earliest `ref_date`, row count, last capture date
- **`UNCAPTURED` alarm** — BLS releases the CLI knows should be in the store but aren't yet

!!! tip "LOCAL FALLBACK warning"
    If `status` prints `LOCAL FALLBACK`, either `NFP_STORE_URI` is not set in your `.env`
    or your process didn't load `.env` before starting. See the [`.env` gotcha](installation.md#the-env-gotcha).

You can also pass an explicit store URI or a reference date:

```bash
uv run alt-nfp status --as-of 2026-03-12 --store s3://alt-nfp/store
```

## Capture a monthly update

`update` is the everyday production command. It advances the release calendar, fetches the
CES and QCEW prints that became knowable as of `--as-of`, and appends them to the store:

```bash
uv run alt-nfp update --as-of 2026-01-12
```

You can restrict which sources are captured:

```bash
uv run alt-nfp update --as-of 2026-01-12 --only ces
uv run alt-nfp update --as-of 2026-01-12 --only qcew
uv run alt-nfp update --as-of 2026-01-12 --only indicators
```

!!! note "API key requirement"
    `update` requires `BLS_API_KEY` in your `.env` to fetch CES data via the BLS JSON API.
    It also requires `FRED_API_KEY` for indicator refresh. Missing keys cause a loud failure
    at startup — not a silent empty capture.

`update` is idempotent: running it twice for the same `--as-of` is safe — the second run
appends zero rows and re-compacts any fragmented partitions.

## Bake a ModelData snapshot

After a successful update, bake the model-data handoff:

```bash
uv run alt-nfp snapshot --as-of 2026-01-12
```

`--as-of` must be the 12th of the month (enforced). The snapshot is written to
`NFP_SNAPSHOTS_URI` as a hash-pinned `.npz` archive ready for `nfp-model`.

## Feed-driven automation (cron)

`watch` polls the BLS RSS feeds and triggers `update` when a new CES or QCEW release appears:

```bash
uv run alt-nfp watch --source all
```

This is designed to be called by a **daily cron job**. It is a clean no-op on days when
nothing new has been published. Add `--snapshot` to also bake a ModelData snapshot
immediately after each successful capture.

```bash
uv run alt-nfp watch --source all --snapshot
```

## One-time historical bootstrap

The initial store is built with a separate script, not a CLI command:

```python
# scripts/bootstrap_store.py — run ONCE to reconstruct the full triangular store
uv run python scripts/bootstrap_store.py \
    --scratch s3://alt-nfp/store-rebuild \
    --canonical s3://alt-nfp/store
```

`bootstrap_store.py` is a **one-time script**, not an everyday command. It rebuilds the full
CES/QCEW history from public BLS bulk files and the CEW API, then promotes the result to the
canonical store via a safe copy-then-delete cutover. See the CLI Reference for the full
command surface.

## Run the test suite

```bash
uv run pytest -m "not network and not slow" --no-cov   # fast (~30 s), skips MCMC + network
uv run pytest -m "not network" --no-cov               # full local suite (~3 min, incl. MCMC smoke)
```

Tests that require the vintage store self-skip when the store is unavailable. Network tests
are marked `@pytest.mark.network` and excluded from CI.
