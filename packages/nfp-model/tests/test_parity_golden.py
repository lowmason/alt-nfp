"""A3 posterior-parity spot check against the frozen-reference fixtures.

**FROZEN PRE-REBUILD (2026-06-18).** The A3 references are PyMC posteriors computed
on the OLD 2012+ vintage store. The store rebuild (``plans/10`` T8) moved the
canonical store to the 2017+ rebuilt schema, so parity must run against the
preserved old-store backup — ``NFP_STORE_URI=s3://alt-nfp/store-prev-20260618`` —
NOT the canonical store (against which the data diverges and some spot dates are the
Oct-2025 shutdown gap). The port is already validated (this gate passed 2026-06-12);
a forward JAX-on-canonical regression baseline is deferred. See ``plans/5``
"Post-rebuild status".

The full 14-fit parity run lives in ``scripts/run_a3_parity.py`` (minutes of MCMC
per fixture — not pytest material). This test re-runs ONE fixture and applies the
same criteria, as a guard that the gate stays reproducible. It requires explicit
opt-in via ``NFP_A3_PARITY=1`` on top of store/data availability.

Fixtures: ``s3://alt-nfp/golden/a3`` (override: ``NFP_GOLDEN_A3_URI``);
committed manifest: ``tests/golden/a3_manifest.json``.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

SPOT_STEM = "asof_2025-07-12_light"  # pre-shutdown; valid in both old + rebuilt stores

pytestmark = [pytest.mark.slow, pytest.mark.real_store]  # reads s3://.../golden/a3 + real store


def _golden_root():
    uri = os.environ.get("NFP_GOLDEN_A3_URI", "s3://alt-nfp/golden/a3")
    from upath import UPath

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


def _available() -> str | None:
    if os.environ.get("NFP_A3_PARITY") != "1":
        return "set NFP_A3_PARITY=1 to run the (minutes-long) parity spot check"
    store = os.environ.get("NFP_STORE_URI")
    if not (store and os.environ.get("AWS_ACCESS_KEY_ID")):
        return "vintage store env not configured"
    from nfp_lookups.paths import DATA_DIR, is_canonical_store

    if is_canonical_store(store):
        return (
            "A3 references are frozen pre-rebuild (2012+ old store); they diverge from "
            "the rebuilt canonical store (plans/10 T8). Set "
            "NFP_STORE_URI=s3://alt-nfp/store-prev-20260618 to run parity."
        )
    if not (Path(DATA_DIR) / "providers" / "g" / "g_provider.parquet").exists():
        return "local provider data unavailable"
    try:
        root = _golden_root()
        if not (root / "a3_manifest.json").exists():
            return "A3 fixtures not found in golden store"
    except Exception as e:  # noqa: BLE001
        return f"golden store unreachable: {e}"
    return None


def test_parity_spot_check():
    reason = _available()
    if reason:
        pytest.skip(reason)

    import io

    from nfp_ingest.model_data import build_model_data
    from nfp_model import fit_model
    from nfp_model.parity import compare_fixture

    root = _golden_root()
    manifest = json.loads((root / "a3_manifest.json").read_text())
    fx = manifest["fixtures"][SPOT_STEM]
    with (root / f"ref_{SPOT_STEM}.npz").open("rb") as f:
        ref = dict(np.load(io.BytesIO(f.read()), allow_pickle=False))

    as_of = date.fromisoformat(fx["as_of"])
    data = build_model_data(as_of, end_year=2026)
    fit = fit_model(data, settings=fx["preset"], seed=fx["seed"])

    report = compare_fixture(ref, fx, fit, manifest["provenance"])
    failures = [r for r in report.rows if not r.passed]
    assert not failures, "\n".join(
        f"{r.name}: {r.detail}" for r in failures
    )
