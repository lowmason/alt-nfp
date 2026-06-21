"""Smoke test for scripts/bootstrap_store.py (no network, no real store).

Phase 9 of specs/cli_production_workflow.md. The bootstrap orchestration is
exercised with every heavy/network step monkeypatched and both store prefixes
pinned to tmp_path — the real bootstrap is NEVER run against MinIO here.
"""

from __future__ import annotations

import importlib.util

import polars as pl
import pytest
from nfp_lookups.paths import BASE_DIR


def _load_bootstrap():
    """Import scripts/bootstrap_store.py by path (scripts/ is not on testpaths)."""
    path = BASE_DIR / "scripts" / "bootstrap_store.py"
    spec = importlib.util.spec_from_file_location("bootstrap_store", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_ces_panel() -> pl.DataFrame:
    """One CES row in VINTAGE_STORE_SCHEMA (already remapped to total taxonomy)."""
    from datetime import date

    from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "total",
        "industry_type": "total",
        "industry_code": "00",
        "ref_date": date(2024, 1, 12),
        "vintage_date": date(2024, 2, 2),
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 158000.0,
        "size_class_type": None,
        "size_class_code": None,
        "source": "ces",
        "seasonally_adjusted": True,
    }
    return pl.DataFrame([row], schema=dict(VINTAGE_STORE_SCHEMA))


def _tiny_qcew_levels_panel() -> pl.DataFrame:
    """One QCEW-levels row in VINTAGE_STORE_SCHEMA (post-build_qcew_panel shape)."""
    from datetime import date

    from nfp_lookups.schemas import VINTAGE_STORE_SCHEMA

    row = {
        "geographic_type": "national",
        "geographic_code": "00",
        "ownership": "private",
        "industry_type": "total",
        "industry_code": "05",
        "ref_date": date(2024, 3, 12),
        "vintage_date": date(2024, 9, 1),
        "revision": 0,
        "benchmark_revision": 0,
        "employment": 130000.0,
        "size_class_type": None,
        "size_class_code": None,
        "source": "qcew",
        "seasonally_adjusted": False,
    }
    return pl.DataFrame([row], schema=dict(VINTAGE_STORE_SCHEMA))


def _patch_seams(monkeypatch, boot):
    """Replace every heavy/network seam with a zero-network synthetic stub."""
    monkeypatch.setattr(boot, "download_ces", lambda *a, **k: None)
    monkeypatch.setattr(boot, "advance_release_calendar", lambda *a, **k: None)
    monkeypatch.setattr(boot, "build_ces_panel", lambda *a, **k: _tiny_ces_panel())
    monkeypatch.setattr(boot, "acquire_qcew_levels", lambda *a, **k: pl.DataFrame())
    monkeypatch.setattr(boot, "acquire_qcew_size_native", lambda *a, **k: pl.DataFrame())
    # build_qcew_panel/build_size_class_panel are the crosswalk consumers; with
    # empty raw frames they would error, so stub the *panel* builders too. The
    # QCEW-levels panel carries the qcew partition; size returns empty (no Q1
    # coverage in this synthetic fixture).
    monkeypatch.setattr(boot, "build_qcew_panel", lambda *a, **k: _tiny_qcew_levels_panel())
    monkeypatch.setattr(boot, "build_size_class_panel", lambda *a, **k: pl.DataFrame())


def test_bootstrap_builds_scratch_then_promotes_to_canonical(monkeypatch, tmp_path):
    boot = _load_bootstrap()
    _patch_seams(monkeypatch, boot)

    scratch = tmp_path / "store-rebuild"
    canonical = tmp_path / "store"

    boot.main(
        argv=[
            "--scratch", str(scratch),
            "--canonical", str(canonical),
            "--start-year", "2024",
            "--end-year", "2024",
        ]
    )

    # Promotion left the canonical prefix populated with the composed partitions.
    canon_files = sorted(canonical.glob("**/*.parquet"))
    assert canon_files, "canonical store has no parquet partitions after bootstrap"
    ces_part = canonical / "source=ces" / "seasonally_adjusted=true"
    assert ces_part.exists(), "expected source=ces/seasonally_adjusted=true partition"
    df = pl.read_parquet(canon_files[0])
    assert df.height >= 1


def test_bootstrap_refuses_canonical_uri_as_scratch(monkeypatch, tmp_path):
    """--scratch must not be the canonical store (is_canonical_store guard)."""
    boot = _load_bootstrap()
    _patch_seams(monkeypatch, boot)
    with pytest.raises(SystemExit):
        boot.main(
            argv=[
                "--scratch", "s3://alt-nfp/store",
                "--canonical", str(tmp_path / "store"),
                "--start-year", "2024",
                "--end-year", "2024",
            ]
        )


def test_promote_rejects_mixed_backends(tmp_path):
    """scratch and canonical must share a backend — a remote scratch + local
    canonical (or vice versa) would misroute the cutover, so it aborts loudly.

    Routes remote-scratch/local-canonical (which without the guard falls through
    to _promote_local, no network) and asserts the distinctive 'share a backend'
    message — proving the guard fired, not the generic empty-scratch exit.
    """
    boot = _load_bootstrap()
    with pytest.raises(SystemExit) as exc:
        boot._promote_scratch_to_canonical(
            "s3://alt-nfp/store-rebuild", str(tmp_path / "store")
        )
    assert "share a backend" in str(exc.value)


def test_promote_local_refuses_empty_scratch_file(tmp_path):
    """A zero-byte parquet in scratch aborts the promote before any orphan delete
    (a truncated/empty copy is the canonical-store-corruption class this guards)."""
    boot = _load_bootstrap()
    scratch = tmp_path / "store-rebuild"
    canonical = tmp_path / "store"
    part = scratch / "source=ces" / "seasonally_adjusted=true"
    part.mkdir(parents=True)
    (part / "v0.parquet").write_bytes(b"")  # zero-byte scratch file
    with pytest.raises(SystemExit) as exc:
        boot._promote_local(scratch, canonical)
    assert "empty" in str(exc.value)
