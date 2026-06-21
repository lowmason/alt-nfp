"""Tests for provider and cyclical indicator configuration dataclasses."""

from __future__ import annotations

from nfp_lookups.provider_config import (
    CYCLICAL_INDICATORS_DEFAULT,
    MIN_PSEUDO_ESTABS_PER_CELL,
    CyclicalIndicator,
    ProviderConfig,
)


class TestProviderConfig:
    """ProviderConfig dataclass construction and defaults."""

    def test_minimal_construction(self):
        p = ProviderConfig(name="X", file="providers/x/x.parquet")
        assert p.name == "X"
        assert p.file == "providers/x/x.parquet"

    def test_defaults(self):
        p = ProviderConfig(name="X", file="x.parquet")
        assert p.error_model == "iid"
        assert p.birth_file is None
        assert p.industry_type == "national"
        assert p.industry_code == "00"
        assert p.geography_type == "national"
        assert p.geography_code == "00"

    def test_cell_level_provider(self):
        p = ProviderConfig(
            name="G",
            file="providers/g/g_provider.parquet",
            birth_file="providers/g/g_births.parquet",
            geography_type="region",
        )
        assert p.geography_type == "region"
        assert p.birth_file is not None

    def test_ar1_error_model(self):
        p = ProviderConfig(name="Y", file="y.parquet", error_model="ar1")
        assert p.error_model == "ar1"

    def test_mutable(self):
        """ProviderConfig is mutable (not frozen)."""
        p = ProviderConfig(name="X", file="x.parquet")
        p.name = "Z"
        assert p.name == "Z"


class TestCyclicalIndicator:
    """CyclicalIndicator dataclass construction and defaults."""

    def test_construction(self):
        ci = CyclicalIndicator(name="test", fred_id="TEST123", freq="monthly")
        assert ci.name == "test"
        assert ci.fred_id == "TEST123"
        assert ci.freq == "monthly"
        assert ci.pub_lag == 1  # default

    def test_custom_pub_lag(self):
        ci = CyclicalIndicator(name="test", fred_id="TEST", freq="weekly", pub_lag=3)
        assert ci.pub_lag == 3

    def test_frozen(self):
        """CyclicalIndicator is frozen (immutable)."""
        ci = CyclicalIndicator(name="test", fred_id="TEST", freq="monthly")
        try:
            ci.name = "other"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestDefaultIndicators:
    """Default CYCLICAL_INDICATORS_DEFAULT has expected structure."""

    def test_count(self):
        assert len(CYCLICAL_INDICATORS_DEFAULT) == 2

    def test_names(self):
        names = {ci.name for ci in CYCLICAL_INDICATORS_DEFAULT}
        assert names == {"claims", "jolts"}

    def test_claims_config(self):
        claims = next(ci for ci in CYCLICAL_INDICATORS_DEFAULT if ci.name == "claims")
        assert claims.fred_id == "ICNSA"
        assert claims.freq == "weekly"
        assert claims.pub_lag == 1

    def test_jolts_config(self):
        jolts = next(ci for ci in CYCLICAL_INDICATORS_DEFAULT if ci.name == "jolts")
        assert jolts.fred_id == "JTSJOL"
        assert jolts.freq == "monthly"
        assert jolts.pub_lag == 2


class TestMinPseudoEstabs:
    def test_value(self):
        assert MIN_PSEUDO_ESTABS_PER_CELL == 5
        assert isinstance(MIN_PSEUDO_ESTABS_PER_CELL, int)
