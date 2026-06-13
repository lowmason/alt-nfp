"""Tests for ModelData snapshots: content hash, round-trip, hash stability."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
from nfp_ingest.snapshots import (
    collect_snapshot,
    content_hash,
    load_snapshot,
    save_snapshot,
)


def _synthetic():
    arrays = {
        "g": np.array([0.1, np.nan, -0.2], dtype=float),
        "obs": np.array([0, 2], dtype=np.int64),
        "mask": np.array([True, False, True]),
    }
    meta = {"schema_version": 1, "scalars": {"T": 3}, "dates": ["2024-01-12"]}
    return arrays, meta


class TestContentHash:
    def test_deterministic(self):
        a1, m1 = _synthetic()
        a2, m2 = _synthetic()
        assert content_hash(a1, m1) == content_hash(a2, m2)

    def test_key_order_insensitive(self):
        arrays, meta = _synthetic()
        reordered = dict(reversed(list(arrays.items())))
        assert content_hash(arrays, meta) == content_hash(reordered, meta)

    def test_value_change_changes_hash(self):
        arrays, meta = _synthetic()
        h0 = content_hash(arrays, meta)
        arrays["g"] = arrays["g"].copy()
        arrays["g"][0] = 0.10000001
        assert content_hash(arrays, meta) != h0

    def test_meta_change_changes_hash(self):
        arrays, meta = _synthetic()
        h0 = content_hash(arrays, meta)
        assert content_hash(arrays, {**meta, "scalars": {"T": 4}}) != h0


class TestRoundTrip:
    def test_save_load_preserves_everything(self, tmp_path):
        arrays, meta = _synthetic()
        p = tmp_path / "snap.npz"
        digest = save_snapshot(arrays, meta, p)

        loaded, loaded_meta = load_snapshot(p)
        assert loaded_meta["content_hash"] == digest
        assert loaded_meta["scalars"] == meta["scalars"]
        assert sorted(loaded) == sorted(arrays)
        for k in arrays:
            assert np.array_equal(loaded[k], arrays[k], equal_nan=(
                np.issubdtype(arrays[k].dtype, np.floating)
            ))

    def test_corruption_detected(self, tmp_path):
        arrays, meta = _synthetic()
        p = tmp_path / "snap.npz"
        save_snapshot(arrays, meta, p)
        # re-save different content under embedded hash by tampering: simulate
        # by writing a snapshot whose arrays were mutated after hashing
        arrays2 = {**arrays, "g": np.array([9.9, 9.9, 9.9])}
        p2 = tmp_path / "snap2.npz"
        save_snapshot(arrays2, meta, p2)
        # swap files' names: loading p2's bytes with p's name is fine — hashes
        # are embedded, so corruption means embedded != recomputed. Forge it:
        import io
        import json as _json

        with p.open("rb") as f:
            npz = np.load(io.BytesIO(f.read()), allow_pickle=False)
            loaded = {k: npz[k] for k in npz.files}
        m = _json.loads(bytes(loaded.pop("__meta__")).decode())
        m["content_hash"] = "0" * 64
        buf = io.BytesIO()
        np.savez(
            buf,
            __meta__=np.frombuffer(_json.dumps(m).encode(), dtype=np.uint8),
            **loaded,
        )
        bad = tmp_path / "bad.npz"
        bad.write_bytes(buf.getvalue())
        with pytest.raises(ValueError, match="hash mismatch"):
            load_snapshot(bad)


class TestSchemaVersionCompat:
    """H-3: v3 snapshots lack birth_rate/bd_proxy/bd_qcew_lagged; v2 snapshots
    contain them.  Both must round-trip through load_snapshot without error.

    load_snapshot verifies only the content hash over whatever arrays ARE in
    the file, so a v2 snapshot (with the 3 extra arrays) loads fine and the
    extra keys are simply present in the returned dict (the model ignores them).
    A v3 snapshot (without the 3) also loads fine — no schema_version check is
    needed; the hash contract is sufficient.
    """

    def test_v3_snapshot_round_trips_without_dropped_arrays(self, tmp_path):
        """A snapshot saved without birth_rate/bd_proxy/bd_qcew_lagged loads."""
        # v3: GLOBAL_ARRAY_KEYS no longer includes the 3 dropped names
        arrays = {
            "g_ces_sa": np.array([0.1, np.nan, -0.2], dtype=float),
            "qcew_obs": np.array([0, 1], dtype=np.int64),
        }
        meta = {"schema_version": 3, "scalars": {"T": 3}, "dates": ["2024-01-12"]}
        p = tmp_path / "v3_snap.npz"
        digest = save_snapshot(arrays, meta, p)

        loaded, loaded_meta = load_snapshot(p)
        assert loaded_meta["content_hash"] == digest
        assert "birth_rate" not in loaded
        assert "bd_proxy" not in loaded
        assert "bd_qcew_lagged" not in loaded

    def test_v2_snapshot_with_dropped_arrays_still_loads(self, tmp_path):
        """A v2 snapshot containing the 3 dead arrays loads without error.

        load_snapshot hashes whatever arrays are in the file, so including the
        extra keys is transparent — the hash matches and the arrays come back.
        The model layer (from_snapshot / model_inputs) ignores any extra keys.
        """
        arrays = {
            "g_ces_sa": np.array([0.1, np.nan, -0.2], dtype=float),
            "qcew_obs": np.array([0, 1], dtype=np.int64),
            # v2 dead arrays — present in old snapshots, never read by the model
            "birth_rate": np.full(3, np.nan),
            "bd_proxy": np.full(3, np.nan),
            "bd_qcew_lagged": np.full(3, np.nan),
        }
        meta = {"schema_version": 2, "scalars": {"T": 3}, "dates": ["2024-01-12"]}
        p = tmp_path / "v2_snap.npz"
        digest = save_snapshot(arrays, meta, p)

        loaded, loaded_meta = load_snapshot(p)
        assert loaded_meta["content_hash"] == digest
        # Extra keys survive the round-trip (load_snapshot does not strip them)
        assert "birth_rate" in loaded
        assert "bd_proxy" in loaded
        assert "bd_qcew_lagged" in loaded


def _store_available() -> bool:
    try:
        from nfp_lookups.paths import DATA_DIR, VINTAGE_STORE_PATH

        return (
            VINTAGE_STORE_PATH.exists()
            and (DATA_DIR / "providers/g/g_provider.parquet").exists()
            and (DATA_DIR / "indicators/claims.parquet").exists()
        )
    except Exception:
        return False


@pytest.mark.real_store  # build_model_data reads the real vintage store; exempt from cred blanking
@pytest.mark.skipif(not _store_available(), reason="store/providers/indicators unavailable")
class TestHashStability:
    AS_OF = date(2023, 7, 12)

    def test_build_twice_same_hash(self):
        from nfp_ingest.model_data import build_model_data

        d1 = build_model_data(self.AS_OF, start_year=2012, end_year=2026)
        d2 = build_model_data(self.AS_OF, start_year=2012, end_year=2026)
        a1, m1 = collect_snapshot(d1)
        a2, m2 = collect_snapshot(d2)
        assert content_hash(a1, m1) == content_hash(a2, m2)

    def test_snapshot_write_and_reload(self, tmp_path):
        from nfp_ingest.snapshots import snapshot_model_data

        path, digest = snapshot_model_data(
            self.AS_OF, out_root=Path(tmp_path), start_year=2012, end_year=2026
        )
        arrays, meta = load_snapshot(path)
        assert meta["content_hash"] == digest
        assert meta["as_of"] == self.AS_OF.isoformat()
        assert meta["scalars"]["T"] == 137  # pinned by the A2 golden master
