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
INDICATORS_DIR: Path = DATA_DIR / "indicators"
OUTPUT_DIR: Path = BASE_DIR / "output"

# Release-dates pipeline artifacts (scraped BLS schedules → parquet)
RELEASES_DIR: Path = DOWNLOADS_DIR / "releases"
RELEASE_DATES_PATH: Path = INTERMEDIATE_DIR / "release_dates.parquet"
VINTAGE_DATES_PATH: Path = INTERMEDIATE_DIR / "vintage_dates.parquet"


# ---------------------------------------------------------------------------
# Vintage store location (local dir or S3-compatible object storage)
# ---------------------------------------------------------------------------


def is_remote(path: Any) -> bool:
    """True if *path* points at object storage rather than the local fs."""
    return str(getattr(path, "protocol", "")) in ("s3", "s3a")


def is_canonical_store(path: Any) -> bool:
    """True if *path* is the canonical, append-only vintage store.

    The canonical store (``s3://alt-nfp/store``) holds live-captured,
    release-day vintage rows that exist in no raw input and are therefore
    irreplaceable — it only ever takes appends and must never be rebuilt in
    place (see root ``CLAUDE.md``). This predicate gates the write doors that
    could clobber it (``build_store``, ``mirror_store``).

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
