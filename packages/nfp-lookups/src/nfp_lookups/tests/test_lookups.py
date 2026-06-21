"""Tests for nfp_lookups — industry hierarchy, revision schedules, and geography."""

import numpy as np
import pytest
from nfp_lookups import (
    CES_REVISIONS,
    CES_SERIES_MAP,
    DIVISION_NAMES,
    DOMAIN_DEFINITIONS,
    FIPS_TO_DIVISION,
    FIPS_TO_REGION,
    GEOGRAPHY_HIERARCHY,
    GOVT_OWNERSHIP_TO_SECTOR,
    INDUSTRY_HIERARCHY,
    INDUSTRY_MAP,
    QCEW_REVISIONS,
    REGION_NAMES,
    STATES,
    en_series_id,
    en_series_id_for_state,
    get_domain_codes,
    get_domain_supersectors,
    get_sector_codes,
    get_supersector_codes,
    get_supersector_components,
    qcew_to_sector,
    sector_to_supersector_idx,
    supersector_to_domain_idx,
)


class TestIndustryHierarchy:
    """Tests for the BLS industry hierarchy."""

    def test_industry_hierarchy_completeness(self):
        """All 10 BLS supersectors present, all sectors map to a valid
        supersector, all supersectors map to a valid domain."""
        df = INDUSTRY_HIERARCHY.collect()

        # 10 unique supersectors
        ss_codes = df['supersector_code'].unique().sort().to_list()
        assert ss_codes == ['10', '20', '30', '40', '50', '55', '60', '65', '70', '80']

        # 2 domains
        domains = df['domain_code'].unique().sort().to_list()
        assert domains == ['G', 'S']

        # 18 sectors
        sectors = df['sector_code'].unique().sort().to_list()
        assert len(sectors) == 18

    def test_industry_hierarchy_no_orphans(self):
        """No sector codes without a supersector parent."""
        df = INDUSTRY_HIERARCHY.collect()

        # Every sector has a non-null supersector_code
        assert df['supersector_code'].null_count() == 0

        # Every supersector has a non-null domain_code
        assert df['domain_code'].null_count() == 0

    def test_sector_codes_simplified(self):
        """Sector codes use simplified forms: '31' not '31-33', etc."""
        df = INDUSTRY_HIERARCHY.collect()
        sector_codes = df['sector_code'].to_list()

        # Should not contain range notation
        for code in sector_codes:
            assert '-' not in code, f'Sector code {code!r} uses range notation'

        # Specific checks
        assert '31' in sector_codes  # not '31-33'
        assert '44' in sector_codes  # not '44-45'
        assert '48' in sector_codes  # not '48-49'
        assert '31-33' not in sector_codes
        assert '44-45' not in sector_codes
        assert '48-49' not in sector_codes

    def test_ces_series_id_generation(self):
        """CES series IDs have correct format."""
        # Manufacturing SA
        assert CES_SERIES_MAP[('30', True)] == 'CES3000000001'
        # Manufacturing NSA
        assert CES_SERIES_MAP[('30', False)] == 'CEU3000000001'
        # Total private SA
        assert CES_SERIES_MAP[('05', True)] == 'CES0500000001'
        # Total private NSA
        assert CES_SERIES_MAP[('05', False)] == 'CEU0500000001'

        # All series IDs should match pattern
        for (_code, sa), sid in CES_SERIES_MAP.items():
            prefix = 'CES' if sa else 'CEU'
            assert sid.startswith(prefix), f'{sid} should start with {prefix}'
            assert len(sid) == 13, f'{sid} should be 13 chars'
            assert sid.endswith('00000001'), f'{sid} should end with 00000001'

    def test_index_builders_shapes(self):
        """Index builder arrays have correct shapes and value ranges."""
        domains = get_domain_codes()
        ss_codes = get_supersector_codes()
        sec_codes = get_sector_codes()

        n_domains = len(domains)
        n_ss = len(ss_codes)
        n_sec = len(sec_codes)

        # supersector_to_domain_idx
        ss_to_dom = supersector_to_domain_idx()
        assert ss_to_dom.shape == (n_ss,)
        assert ss_to_dom.dtype == np.intp
        assert np.all(ss_to_dom >= 0)
        assert np.all(ss_to_dom < n_domains)

        # sector_to_supersector_idx
        sec_to_ss = sector_to_supersector_idx()
        assert sec_to_ss.shape == (n_sec,)
        assert sec_to_ss.dtype == np.intp
        assert np.all(sec_to_ss >= 0)
        assert np.all(sec_to_ss < n_ss)


class TestRevisionSchedules:
    """Tests for QCEW and CES revision schedules."""

    def test_revision_schedule_counts(self):
        """QCEW: Q1=5, Q2=4, Q3=3, Q4=2 vintages; CES=4 entries."""
        assert len(QCEW_REVISIONS['Q1']) == 5
        assert len(QCEW_REVISIONS['Q2']) == 4
        assert len(QCEW_REVISIONS['Q3']) == 3
        assert len(QCEW_REVISIONS['Q4']) == 2
        assert len(CES_REVISIONS) == 4

    def test_revision_schedule_monotonic_noise(self):
        """Noise multipliers decrease (or stay flat) with revision number."""
        for q_label, specs in QCEW_REVISIONS.items():
            sorted_specs = sorted(specs, key=lambda s: s.revision_number)
            for i in range(1, len(sorted_specs)):
                assert sorted_specs[i].noise_multiplier <= sorted_specs[i - 1].noise_multiplier, (
                    f'QCEW {q_label}: noise should decrease from rev {sorted_specs[i-1].revision_number} '
                    f'to rev {sorted_specs[i].revision_number}'
                )

        # CES: sort by revision_number (excluding benchmark at -1)
        ces_regular = sorted(
            [s for s in CES_REVISIONS if s.revision_number >= 0],
            key=lambda s: s.revision_number,
        )
        for i in range(1, len(ces_regular)):
            assert ces_regular[i].noise_multiplier <= ces_regular[i - 1].noise_multiplier

    def test_revision_schedule_monotonic_lag(self):
        """Lag_months increases with revision number within each schedule."""
        for q_label, specs in QCEW_REVISIONS.items():
            sorted_specs = sorted(specs, key=lambda s: s.revision_number)
            for i in range(1, len(sorted_specs)):
                assert sorted_specs[i].lag_months > sorted_specs[i - 1].lag_months, (
                    f'QCEW {q_label}: lag should increase from rev {sorted_specs[i-1].revision_number} '
                    f'to rev {sorted_specs[i].revision_number}'
                )

        # CES: sort by revision_number (excluding benchmark at -1)
        ces_regular = sorted(
            [s for s in CES_REVISIONS if s.revision_number >= 0],
            key=lambda s: s.revision_number,
        )
        for i in range(1, len(ces_regular)):
            assert ces_regular[i].lag_months > ces_regular[i - 1].lag_months


class TestGeography:
    """Tests for Census Region and Division lookups."""

    def test_states_count(self):
        """52 entries: 50 states + DC + Puerto Rico."""
        assert len(STATES) == 52

    def test_all_fips_mapped_to_region(self):
        """Every FIPS in STATES has a valid region code (1-4)."""
        valid_regions = {'1', '2', '3', '4'}
        for fips in STATES:
            assert fips in FIPS_TO_REGION, f'FIPS {fips} missing from FIPS_TO_REGION'
            assert FIPS_TO_REGION[fips] in valid_regions, (
                f'FIPS {fips} has invalid region {FIPS_TO_REGION[fips]}'
            )

    def test_all_fips_mapped_to_division(self):
        """Every FIPS in STATES has a valid division code (01-09)."""
        valid_divisions = {'01', '02', '03', '04', '05', '06', '07', '08', '09'}
        for fips in STATES:
            assert fips in FIPS_TO_DIVISION, f'FIPS {fips} missing from FIPS_TO_DIVISION'
            assert FIPS_TO_DIVISION[fips] in valid_divisions, (
                f'FIPS {fips} has invalid division {FIPS_TO_DIVISION[fips]}'
            )

    def test_puerto_rico_assignment(self):
        """Puerto Rico (72) → Region 3 (South), Division 05 (South Atlantic)."""
        assert '72' in STATES
        assert FIPS_TO_REGION['72'] == '3'
        assert FIPS_TO_DIVISION['72'] == '05'

    def test_region_names_complete(self):
        """All 4 regions have names."""
        assert len(REGION_NAMES) == 4
        assert set(REGION_NAMES.keys()) == {'1', '2', '3', '4'}

    def test_division_names_complete(self):
        """All 9 divisions have names."""
        assert len(DIVISION_NAMES) == 9
        assert set(DIVISION_NAMES.keys()) == {
            '01', '02', '03', '04', '05', '06', '07', '08', '09'
        }

    def test_geography_hierarchy_lazyframe(self):
        """GEOGRAPHY_HIERARCHY LazyFrame collects with expected shape."""
        df = GEOGRAPHY_HIERARCHY.collect()
        assert df.height == 52
        assert set(df.columns) == {
            'state_fips', 'region_code', 'region_name',
            'division_code', 'division_name',
        }
        assert df['state_fips'].null_count() == 0
        assert df['region_code'].null_count() == 0
        assert df['division_code'].null_count() == 0


class TestIndustryMap:
    """Tests for the CES-QCEW industry cross-mapping."""

    def test_industry_map_has_all_levels(self):
        """INDUSTRY_MAP contains domain, supersector, and sector entries."""
        types = {e.industry_type for e in INDUSTRY_MAP}
        assert types == {'domain', 'supersector', 'sector'}

    def test_industry_map_domain_count(self):
        """5 domain-level entries."""
        domains = [e for e in INDUSTRY_MAP if e.industry_type == 'domain']
        assert len(domains) == 5

    def test_industry_map_supersector_count(self):
        """11 supersector-level entries (10 private + government)."""
        supersectors = [e for e in INDUSTRY_MAP if e.industry_type == 'supersector']
        assert len(supersectors) == 11

    def test_industry_map_sector_count(self):
        """19 sector-level entries."""
        sectors = [e for e in INDUSTRY_MAP if e.industry_type == 'sector']
        assert len(sectors) == 19

    def test_industry_entry_is_frozen(self):
        """IndustryEntry instances are immutable."""
        entry = INDUSTRY_MAP[0]
        with pytest.raises(AttributeError):
            entry.industry_code = 'XX'

    def test_sector_entries_have_qcew_naics(self):
        """Sector-level INDUSTRY_MAP entries have non-empty qcew_naics."""
        for entry in INDUSTRY_MAP:
            if entry.industry_type == 'sector':
                assert entry.qcew_naics, f'{entry.industry_code} has empty qcew_naics'

    def test_aggregate_entries_have_empty_qcew_naics(self):
        """Supersector and domain entries have empty qcew_naics (aggregated)."""
        for entry in INDUSTRY_MAP:
            if entry.industry_type in ('supersector', 'domain'):
                assert entry.qcew_naics == '', (
                    f'{entry.industry_code} ({entry.industry_type}) should have '
                    f'empty qcew_naics, got {entry.qcew_naics!r}'
                )

    def test_every_entry_has_ces_code(self):
        """Every INDUSTRY_MAP entry has a 6-digit ces_code."""
        for entry in INDUSTRY_MAP:
            assert len(entry.ces_code) == 6, (
                f'{entry.industry_code} ces_code {entry.ces_code!r} not 6 digits'
            )

    def test_qcew_to_sector_mapping(self):
        """qcew_to_sector() returns expected mappings for known codes."""
        mapping = qcew_to_sector()
        # QCEW API codes -> NAICS sector codes
        assert mapping['1012'] == '21'   # Mining
        assert mapping['1022'] == '31'   # Manufacturing
        assert mapping['1023'] == '42'   # Wholesale (NAICS 42)
        assert mapping['1024'] == '44'   # Retail (NAICS 44-45)
        assert mapping['1025'] == '48'   # Transport (NAICS 48-49)
        assert mapping['102F'] == '72'   # Accommodation
        # Raw NAICS codes -> identity (NAICS-based hierarchy)
        assert mapping['21'] == '21'
        assert mapping['42'] == '42'    # Wholesale
        assert mapping['44'] == '44'    # Retail
        assert mapping['48'] == '48'    # Transport
        assert mapping['72'] == '72'

    def test_sector_qcew_naics_correctness(self):
        """CES sectors 41/42/43 have NAICS (not CES) codes in qcew_naics."""
        ces_41 = next(e for e in INDUSTRY_MAP if e.industry_code == '41')
        ces_42 = next(e for e in INDUSTRY_MAP if e.industry_code == '42')
        ces_43 = next(e for e in INDUSTRY_MAP if e.industry_code == '43')
        assert ces_41.qcew_naics == '42', 'CES Wholesale → NAICS 42'
        assert ces_42.qcew_naics == '44', 'CES Retail → NAICS 44'
        assert ces_43.qcew_naics == '48', 'CES Transport → NAICS 48'

    def test_en_series_id_format(self):
        """EN series IDs have expected prefix and length."""
        # Find the Total Non-Farm domain entry
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid = en_series_id(total_nf)
        assert sid.startswith('EN'), f'EN series ID should start with EN: {sid}'
        assert len(sid) == 17, f'EN series ID should be 17 chars: {sid}'

    def test_en_series_id_for_state(self):
        """State-level EN series ID embeds the state FIPS."""
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid = en_series_id_for_state(total_nf, '26')  # Michigan
        assert '26000' in sid, f'State FIPS 26 should appear in EN series ID: {sid}'

    def test_en_series_id_ownership(self):
        """Private ownership parameter changes EN series ID."""
        total_nf = next(e for e in INDUSTRY_MAP if e.industry_code == '00')
        sid_all = en_series_id(total_nf, ownership='0')
        sid_prv = en_series_id(total_nf, ownership='5')
        assert sid_all != sid_prv, 'Different ownership should produce different IDs'


class TestSupersectorComponents:
    """Tests for supersector component mappings and domain definitions."""

    def test_all_supersectors_present(self):
        """get_supersector_components covers all 10 private + government."""
        sc = get_supersector_components()
        expected = {'10', '20', '30', '40', '50', '55', '60', '65', '70', '80', '90'}
        assert set(sc.keys()) == expected

    def test_ttu_components(self):
        """Supersector 40 (TTU) = Wholesale + Retail + Transport + Utilities."""
        sc = get_supersector_components()
        assert sorted(sc['40']) == ['22', '42', '44', '48']

    def test_financial_components(self):
        """Supersector 55 (Financial) = Finance + Real Estate."""
        sc = get_supersector_components()
        assert sorted(sc['55']) == ['52', '53']

    def test_government_components(self):
        """Supersector 90 (Government) = Federal + State + Local."""
        sc = get_supersector_components()
        assert sorted(sc['90']) == ['91', '92', '93']

    def test_component_sectors_in_hierarchy(self):
        """All component sector codes exist in the sector list or are govt."""
        sc = get_supersector_components()
        hierarchy_sectors = set(get_sector_codes())
        govt_sectors = {'91', '92', '93'}
        for ss, sectors in sc.items():
            for s in sectors:
                assert s in hierarchy_sectors or s in govt_sectors, (
                    f'Sector {s} in supersector {ss} not in hierarchy or govt'
                )

    def test_domain_definitions(self):
        """Domain definitions exist for all 5 codes."""
        assert set(DOMAIN_DEFINITIONS.keys()) == {'00', '05', '06', '07', '08'}

    def test_domain_00_includes_government(self):
        """Domain 00 (Total Non-Farm) includes supersector 90."""
        ss = get_domain_supersectors('00')
        assert '90' in ss

    def test_domain_05_excludes_government(self):
        """Domain 05 (Total Private) excludes supersector 90."""
        ss = get_domain_supersectors('05')
        assert '90' not in ss

    def test_domain_06_goods_only(self):
        """Domain 06 (Goods-Producing) includes only goods supersectors."""
        ss = get_domain_supersectors('06')
        assert set(ss) == {'10', '20', '30'}

    def test_domain_07_service_with_govt(self):
        """Domain 07 (Service-Providing) includes services + government."""
        ss = get_domain_supersectors('07')
        assert '90' in ss
        assert '10' not in ss  # goods excluded

    def test_domain_08_private_service(self):
        """Domain 08 (Private Service-Providing) excludes goods and govt."""
        ss = get_domain_supersectors('08')
        assert '90' not in ss
        assert '10' not in ss

    def test_govt_ownership_mapping(self):
        """Government ownership codes map to correct CES sectors."""
        assert GOVT_OWNERSHIP_TO_SECTOR == {'1': '91', '2': '92', '3': '93'}


class TestAggregation:
    """Tests for QCEW aggregation logic."""

    def test_aggregate_to_hierarchy_basic(self):
        """aggregate_to_hierarchy produces sector, supersector, domain rows."""
        from datetime import date

        import polars as pl
        from nfp_ingest.qcew import aggregate_to_hierarchy

        rows = []
        for month in range(1, 4):
            for sector in ['52', '53']:
                rows.append({
                    'industry_code': sector,
                    'ref_date': date(2023, month, 1),
                    'employment': 100000,
                    'qtr': 1,
                    'geographic_type': 'national',
                    'geographic_code': 'US',
                })

        df = pl.DataFrame(rows)
        result = aggregate_to_hierarchy(df)

        levels = set(result['industry_level'].unique().to_list())
        assert levels == {'sector', 'supersector', 'domain'}

    def test_supersector_sums_components(self):
        """Supersector employment equals sum of component sector employment."""
        from datetime import date

        import polars as pl
        from nfp_ingest.qcew import aggregate_to_hierarchy

        d = date(2023, 1, 1)
        rows = [
            {'industry_code': '52', 'ref_date': d, 'employment': 100,
             'qtr': 1, 'geographic_type': 'national', 'geographic_code': 'US'},
            {'industry_code': '53', 'ref_date': d, 'employment': 50,
             'qtr': 1, 'geographic_type': 'national', 'geographic_code': 'US'},
        ]
        result = aggregate_to_hierarchy(pl.DataFrame(rows))

        ss55 = result.filter(
            (pl.col('industry_code') == '55')
            & (pl.col('industry_level') == 'supersector')
        )
        assert len(ss55) == 1
        assert ss55['employment'][0] == 150

    def test_domain_private_excludes_govt(self):
        """Domain 05 (Total Private) excludes government employment."""
        from datetime import date

        import polars as pl
        from nfp_ingest.qcew import aggregate_to_hierarchy

        d = date(2023, 1, 1)
        rows = [
            {'industry_code': '51', 'ref_date': d, 'employment': 300,
             'qtr': 1, 'geographic_type': 'national', 'geographic_code': 'US'},
            {'industry_code': '91', 'ref_date': d, 'employment': 100,
             'qtr': 1, 'geographic_type': 'national', 'geographic_code': 'US'},
        ]
        result = aggregate_to_hierarchy(pl.DataFrame(rows))

        d05 = result.filter(
            (pl.col('industry_code') == '05')
            & (pl.col('industry_level') == 'domain')
        )
        d00 = result.filter(
            (pl.col('industry_code') == '00')
            & (pl.col('industry_level') == 'domain')
        )
        assert d05['employment'][0] == 300
        assert d00['employment'][0] == 400

    def test_build_growth_panel_schema(self):
        """_build_growth_panel produces PANEL_SCHEMA-conforming output."""
        from datetime import date

        import polars as pl
        from nfp_ingest.qcew import _build_growth_panel, aggregate_to_hierarchy
        from nfp_lookups.schemas import PANEL_SCHEMA

        rows = []
        for month in range(1, 7):
            rows.append({
                'industry_code': '51', 'ref_date': date(2023, month, 1),
                'employment': 100000 + month * 1000,
                'qtr': (month - 1) // 3 + 1,
                'geographic_type': 'national', 'geographic_code': 'US',
            })

        all_levels = aggregate_to_hierarchy(pl.DataFrame(rows))
        panel = _build_growth_panel(all_levels)

        assert set(panel.columns) == set(PANEL_SCHEMA.keys())
        for col, dtype in PANEL_SCHEMA.items():
            assert panel.schema[col] == dtype, f'{col}: {panel.schema[col]} != {dtype}'

    def test_government_extraction(self):
        """_extract_government_employment maps ownership to CES sectors."""
        import polars as pl
        from nfp_ingest.qcew import _extract_government_employment

        raw = pl.DataFrame({
            'own_code': ['1', '2', '3', '5'],
            'industry_code': ['10', '10', '10', '10'],
            'year': [2023, 2023, 2023, 2023],
            'qtr': [1, 1, 1, 1],
            'month1_emplvl': [100, 200, 400, 9999],
            'month2_emplvl': [101, 201, 401, 9999],
            'month3_emplvl': [102, 202, 402, 9999],
            'geographic_type': ['national'] * 4,
            'geographic_code': ['US'] * 4,
        })
        result = _extract_government_employment(raw)

        codes = sorted(result['industry_code'].unique().to_list())
        assert codes == ['91', '92', '93']
        assert len(result) == 9  # 3 ownership × 3 months
