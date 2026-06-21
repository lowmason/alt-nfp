"""ModelData intake: normalization and the snapshot round trip."""

import numpy as np
from nfp_model.data import from_snapshot, model_inputs, provider_field
from synthetic_data import FakeProviderConfig, make_synthetic_data


class TestProviderField:
    def test_attribute_access(self):
        cfg = FakeProviderConfig(error_model="ar1")
        assert provider_field(cfg, "error_model") == "ar1"
        assert provider_field(cfg, "name") == "G"

    def test_mapping_access(self):
        cfg = {"name": "G", "error_model": "ar1"}
        assert provider_field(cfg, "error_model") == "ar1"

    def test_default(self):
        assert provider_field({}, "error_model", "iid") == "iid"
        assert provider_field(FakeProviderConfig(), "missing", 42) == 42


class TestModelInputs:
    def test_strips_to_model_keys(self):
        data = make_synthetic_data()
        data["panel"] = object()  # frames must never reach the model
        data["levels"] = object()
        inputs = model_inputs(data)
        assert "panel" not in inputs and "levels" not in inputs
        assert "dates" not in inputs
        assert inputs["T"] == data["T"]
        np.testing.assert_array_equal(inputs["g_qcew"], data["g_qcew"])

    def test_normalizes_provider_entries(self):
        inputs = model_inputs(make_synthetic_data(error_model="ar1"))
        (pp,) = inputs["pp_data"]
        assert set(pp) == {"name", "error_model", "g_pp", "pp_obs"}
        assert pp["name"] == "G"
        assert pp["error_model"] == "ar1"
        assert pp["pp_obs"].dtype.kind == "i"

    def test_none_cyclical_preserved(self):
        inputs = model_inputs(make_synthetic_data(with_jolts=False))
        assert inputs["jolts_c"] is None
        assert inputs["claims_c"] is not None


class TestSnapshotRoundTrip:
    """from_snapshot must invert nfp_ingest.snapshots.collect_snapshot."""

    def test_round_trip_matches_model_inputs(self):
        from nfp_ingest.snapshots import collect_snapshot

        data = make_synthetic_data(error_model="ar1")
        arrays, meta = collect_snapshot(data)
        assert meta["providers"][0]["error_model"] == "ar1"  # schema v2 field

        rebuilt = from_snapshot(arrays, meta)
        a = model_inputs(data)
        b = model_inputs(rebuilt)
        assert a.keys() == b.keys()
        for k in a:
            if k == "pp_data":
                continue
            if a[k] is None or np.isscalar(a[k]):
                assert a[k] == b[k], k
            else:
                np.testing.assert_array_equal(a[k], np.asarray(b[k]), err_msg=k)
        (pa,), (pb,) = a["pp_data"], b["pp_data"]
        assert pa["name"] == pb["name"]
        assert pa["error_model"] == pb["error_model"]
        np.testing.assert_array_equal(pa["g_pp"], pb["g_pp"])
        np.testing.assert_array_equal(pa["pp_obs"], pb["pp_obs"])

    def test_schema_v1_defaults_to_iid(self):
        from nfp_ingest.snapshots import collect_snapshot

        data = make_synthetic_data()
        arrays, meta = collect_snapshot(data)
        for pm in meta["providers"]:
            del pm["error_model"]  # simulate a v1 snapshot
        rebuilt = from_snapshot(arrays, meta)
        assert rebuilt["pp_data"][0]["config"]["error_model"] == "iid"

    def test_dates_and_vintage_map_restored(self):
        from nfp_ingest.snapshots import collect_snapshot

        data = make_synthetic_data()
        arrays, meta = collect_snapshot(data)
        rebuilt = from_snapshot(arrays, meta)
        assert rebuilt["dates"] == data["dates"]
        assert rebuilt["ces_vintage_map"] == data["ces_vintage_map"]
