"""Workspace test bootstrap.

Loads ``.env`` (gitignored) before any ``nfp_*`` import so storage config
(``NFP_STORE_URI``, ``AWS_*``) is visible to ``nfp_lookups.paths``, which
reads the environment at import time. Without a ``.env`` the suite runs in
local mode and store-dependent tests skip.
"""

from dotenv import load_dotenv

load_dotenv()
