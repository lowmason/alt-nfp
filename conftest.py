"""Workspace test bootstrap.

Loads ``.env`` (gitignored) before any ``nfp_*`` import so storage config
(``NFP_STORE_URI``, ``AWS_*``) is visible to ``nfp_lookups.paths``, which
reads the environment at import time. Without a ``.env`` the suite runs in
local mode and store-dependent tests skip.
"""

import pytest
from dotenv import load_dotenv

load_dotenv()


class _BlockedLiveStoreAccess(FileNotFoundError):
    """Raised when an unmarked test reaches the live store via s3fs network I/O.

    Subclasses ``FileNotFoundError`` so s3fs ``.exists()``/``.info()`` degrade
    to "not found" (the graceful ``False``/empty an unavailable store should
    yield, which is exactly what ``_store_available()``-style probes expect),
    while the distinctive class name keeps any *propagated* traceback
    self-explanatory rather than looking like a real missing object.
    """


def _sever_s3fs_object_methods(monkeypatch):
    """Sever the UPath-object-method path to the live store for unmarked tests.

    Blanking the environment (below) severs the *call-time* credential path:
    Polars I/O and every store *write* resolve creds via
    ``storage_options_for(path)`` at call time, so blanked env = no creds =
    fail closed. But ``VINTAGE_STORE_PATH`` is an import-time ``upath.UPath``
    whose s3fs filesystem instance has ``key=``/``secret=`` baked in; its
    object methods (``.exists``/``.glob``/``.iterdir``) keep reaching the live
    store regardless of a runtime ``delenv`` (clearing the fsspec *instance*
    cache does not help — the creds are bound into the live instance).

    Close that residual at a single class-level choke. Every s3fs S3 API call
    funnels through ``S3FileSystem._call_s3``; make it raise. ``set_session``
    is *not* a reliable choke — it returns early once a session exists (a
    collection-time availability probe establishes one with real creds), so a
    patch there would no-op and leak. Separately, ``.exists()``/``.info()``
    consult the per-instance dircache via ``_ls_from_cache`` *before* any
    network call, so a listing cached by an earlier ``real_store`` test (or a
    collection-time ``.glob`` probe) could serve a stale hit and bypass the
    ``_call_s3`` patch entirely; force that cache read to miss so the call
    always reaches the raising ``_call_s3``. Both patches use ``raising=True``
    so a future s3fs rename fails the suite loudly instead of silently
    reopening the hole.
    """
    try:
        import s3fs
    except ImportError:
        return  # no s3fs installed => no S3 access is possible; nothing to sever

    def _blocked_call_s3(self, *args, **kwargs):
        raise _BlockedLiveStoreAccess(
            "live store blocked: an unmarked test reached s3fs network I/O; "
            "opt in with @pytest.mark.real_store if the access is intentional"
        )

    def _force_cache_miss(self, *args, **kwargs):
        return None

    monkeypatch.setattr(s3fs.S3FileSystem, "_call_s3", _blocked_call_s3)
    monkeypatch.setattr(s3fs.S3FileSystem, "_ls_from_cache", _force_cache_miss)


@pytest.fixture(autouse=True)
def _block_live_store(request, monkeypatch):
    """Fail closed: tests cannot reach the live MinIO unless marked real_store.

    The store is a local MinIO whose creds are auto-loaded from .env by this
    conftest. A test that writes to it can destroy irreplaceable data (it has
    happened). Blank the creds/endpoint/URI for unmarked tests so any store
    I/O fails harmlessly instead of hitting production, and sever the residual
    UPath-object-method path that baked-in creds would otherwise keep open
    (see :func:`_sever_s3fs_object_methods`). Read-only tests that genuinely
    need the store opt in with @pytest.mark.real_store.
    """
    if request.node.get_closest_marker("real_store") is None:
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                    "AWS_ENDPOINT_URL", "AWS_SESSION_TOKEN", "NFP_STORE_URI"):
            monkeypatch.delenv(var, raising=False)
        _sever_s3fs_object_methods(monkeypatch)
