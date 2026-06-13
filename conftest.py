"""Workspace test bootstrap.

Loads ``.env`` (gitignored) before any ``nfp_*`` import so storage config
(``NFP_STORE_URI``, ``AWS_*``) is visible to ``nfp_lookups.paths``, which
reads the environment at import time. Without a ``.env`` the suite runs in
local mode and store-dependent tests skip.
"""

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def _block_live_store(request, monkeypatch):
    """Fail closed: tests cannot reach the live MinIO unless marked real_store.

    The store is a local MinIO whose creds are auto-loaded from .env by this
    conftest. A test that writes to it can destroy irreplaceable data (it has
    happened). Blank the creds/endpoint/URI for unmarked tests so any store
    I/O fails harmlessly instead of hitting production. Read-only tests that
    genuinely need the store opt in with @pytest.mark.real_store.
    """
    if request.node.get_closest_marker("real_store") is None:
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                    "AWS_ENDPOINT_URL", "AWS_SESSION_TOKEN", "NFP_STORE_URI"):
            monkeypatch.delenv(var, raising=False)
