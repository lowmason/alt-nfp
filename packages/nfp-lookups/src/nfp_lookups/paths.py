"""Canonical data directory layout for the NFP project.

All path constants are derived from a single BASE_DIR (the repository root).
Other packages import these paths rather than constructing their own.

Set the ``NFP_BASE_DIR`` environment variable (before first import) to point
the layout at a different root, e.g. a snapshot or fixture directory.

The vintage store can live in S3-compatible object storage instead of the
local ``data/store/``. Set (before first import):

- ``NFP_STORE_URI`` — e.g. ``s3://alt-nfp/store``; unset means local
- ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` — credentials
- ``AWS_ENDPOINT_URL`` — e.g. ``http://127.0.0.1:9000`` for MinIO
- ``AWS_REGION`` — optional, defaults to ``us-east-1``

``VINTAGE_STORE_PATH`` is then a ``upath.UPath`` and pathlib-style operations
go through s3fs; hand Polars I/O the matching options via
:func:`storage_options_for`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _find_base_dir() -> Path:
    """Resolve the workspace root.

    Precedence: the ``NFP_BASE_DIR`` environment variable, then walking up
    from this file to the first directory containing both ``packages/`` and
    ``pyproject.toml`` (committed markers — ``data/`` is gitignored, so it
    cannot be relied on after a fresh clone), then a fixed-depth fallback
    for editable installs.
    """
    env = os.environ.get("NFP_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()

    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "packages").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    # Fallback: four levels above packages/nfp-lookups/src/nfp_lookups/
    return Path(__file__).resolve().parents[4]


BASE_DIR: Path = _find_base_dir()
DATA_DIR: Path = BASE_DIR / "data"
STORE_DIR: Path = DATA_DIR / "store"
DOWNLOADS_DIR: Path = DATA_DIR / "downloads"
INTERMEDIATE_DIR: Path = DATA_DIR / "intermediate"
OUTPUT_DIR: Path = BASE_DIR / "output"

# Release-dates pipeline artifacts (scraped BLS schedules → parquet)
RELEASES_DIR: Path = DOWNLOADS_DIR / "releases"


# ---------------------------------------------------------------------------
# Vintage store location (local dir or S3-compatible object storage)
# ---------------------------------------------------------------------------


def is_remote(path: Any) -> bool:
    """True if *path* points at object storage rather than the local fs."""
    return str(getattr(path, "protocol", "")) in ("s3", "s3a")


def is_canonical_store(path: Any) -> bool:
    """True if *path* is the canonical (production) vintage store.

    The canonical store (``s3://alt-nfp/store``) is the production store the
    model reads (rebuilt schema since the 2026-06-18 promotion; see ``plans/10``
    T8). It holds reconstructable public CES/QCEW data — replaceable, **not**
    append-only/irreplaceable (the older framing is retired) — but a from-scratch
    build or mirror straight to it would clobber the live store, so this predicate
    gates the write doors (``build_store``, ``mirror_store``) against accidental
    overwrite. Promotion to canonical is a deliberate, backup-first cutover, not a
    routine write.

    A value is treated as the canonical store when it is a *remote* path
    whose URI ends in ``/store``. Remoteness is detected for both ``UPath``
    objects (via :func:`is_remote`, duck-typed on ``.protocol``) and plain
    ``str`` URIs beginning with ``s3://`` / ``s3a://`` — the latter matters
    because callers like ``mirror_store`` build the destination as a string.

    A scratch rebuild prefix (e.g. ``s3://alt-nfp/store-rebuild``) and any
    local path return ``False``; rebuilds target a scratch prefix.
    """
    text = str(path)
    if not (is_remote(path) or text.startswith(("s3://", "s3a://"))):
        return False
    return text.rstrip("/").endswith("/store")


def _store_location() -> Any:
    """Resolve the vintage store root.

    ``NFP_STORE_URI`` (e.g. ``s3://alt-nfp/store``) selects object storage;
    unset selects the local ``STORE_DIR``. Returns ``Path`` or ``UPath``.
    """
    uri = os.environ.get("NFP_STORE_URI")
    if not uri:
        return STORE_DIR

    from upath import UPath  # deferred: s3fs only needed in remote mode

    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return UPath(
        uri,
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs=client_kwargs,
    )


def storage_options_for(path: Any) -> dict[str, str] | None:
    """Polars/object_store ``storage_options`` for *path*.

    Returns ``None`` for local paths (Polars default). For remote paths,
    builds options from the same ``AWS_*`` environment variables used by
    :func:`_store_location`; ``aws_allow_http`` is derived from the endpoint
    scheme (MinIO on localhost is plain http).
    """
    if not is_remote(path):
        return None

    options: dict[str, str] = {
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
    }
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if key:
        options["aws_access_key_id"] = key
    if secret:
        options["aws_secret_access_key"] = secret
    if endpoint:
        options["aws_endpoint_url"] = endpoint
        if endpoint.startswith("http://"):
            options["aws_allow_http"] = "true"
    return options


# Vintage store root. Local alias of STORE_DIR, or a UPath when
# NFP_STORE_URI is set (see module docstring).
VINTAGE_STORE_PATH = _store_location()


# ---------------------------------------------------------------------------
# Persistent non-store data artifacts (indicators, competitors, derived files)
# ---------------------------------------------------------------------------


def _upath(uri: str) -> Any:
    """Build a credentialed UPath from env (shared by all *_location helpers)."""
    from upath import UPath  # deferred: s3fs only needed in remote mode

    client_kwargs = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    return UPath(
        uri,
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        client_kwargs=client_kwargs,
    )


def data_location() -> Any:
    """Root for PERSISTENT non-store data artifacts (indicators, competitors,
    derived release/vintage schedules).

    ``NFP_DATA_URI`` (e.g. ``s3://alt-nfp``) selects object storage; unset selects
    the local ``DATA_DIR``. Returns ``Path`` or ``UPath`` — same env/credential
    contract as :func:`_store_location`. Bulky rebuild byproducts (downloads, the
    revisions intermediates) are NOT routed here; they use tempfile (plans/15 Tier C).
    Provider data lives on a SEPARATE store — see :func:`providers_location`.
    """
    uri = os.environ.get("NFP_DATA_URI")
    return _upath(uri) if uri else DATA_DIR


def providers_location() -> Any:
    """Root for provider parquets — a SEPARATE store from the alt-nfp data bucket.

    On Bloomberg the provider data lives on its own compute store (maintainer,
    2026-06-20), so it gets its own env var ``NFP_PROVIDERS_URI`` and is NOT seeded
    by this repo. Unset → local ``DATA_DIR`` (current dev behaviour). The relative
    ``ProviderConfig.file`` (e.g. ``providers/g/g_provider.parquet``) joins to this root.
    """
    uri = os.environ.get("NFP_PROVIDERS_URI")
    return _upath(uri) if uri else DATA_DIR


_DATA_ROOT = data_location()
INDICATORS_DIR = _DATA_ROOT / "indicators"
COMPETITORS_DIR = _DATA_ROOT / "competitors"
PROVIDERS_DIR = providers_location()  # the provider store root; cfg.file joins to it
RELEASE_DATES_PATH = _DATA_ROOT / "intermediate" / "release_dates.parquet"
VINTAGE_DATES_PATH = _DATA_ROOT / "intermediate" / "vintage_dates.parquet"
