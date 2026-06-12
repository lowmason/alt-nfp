"""Observation panel builder: combines vintage store and provider data.

Reads from the Hive-partitioned vintage store and payroll provider parquet
files, producing a unified panel conforming to PANEL_SCHEMA.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime
from pathlib import Path

import polars as pl
from nfp_lookups.paths import VINTAGE_STORE_PATH
from nfp_lookups.provider_config import ProviderConfig
from nfp_lookups.schemas import PANEL_SCHEMA, validate_panel

from nfp_ingest.payroll import ingest_provider
from nfp_ingest.vintage_store import read_vintage_store, transform_to_panel

logger = logging.getLogger(__name__)


def build_panel(
    store_path: Path | None = None,
    providers: list[ProviderConfig] | None = None,
    start_year: int = 2012,
    end_year: int | None = None,
    as_of_ref: date | None = None,
) -> pl.DataFrame:
    """Build a unified observation panel from vintage store and providers.

    Parameters
    ----------
    store_path : Path, optional
        Root of the Hive-partitioned vintage store.
        Defaults to ``VINTAGE_STORE_PATH``.
    providers : list[ProviderConfig], optional
        Provider list. Defaults to PROVIDERS from config.
    start_year : int
        First year for data (default 2012).
    end_year : int, optional
        Last year for data. Defaults to current year.
    as_of_ref : date, optional
        When set, apply rank-based censoring so the panel contains only
        the data available at horizon *as_of_ref* (``YYYY-MM-12``).
        CES and QCEW rows are selected via the triangular diagonal rules
        and ``vintage_date <= as_of_ref``.

    Returns
    -------
    pl.DataFrame
        Validated observation panel conforming to PANEL_SCHEMA.
    """
    if providers is None:
        providers = []
    if end_year is None:
        end_year = date.today().year
    if store_path is None:
        store_path = VINTAGE_STORE_PATH

    parts: list[pl.DataFrame] = []
    start_ref = date(start_year, 1, 12)
    end_ref = date(end_year, 12, 12)

    if store_path.exists():
        lf = read_vintage_store(
            store_path=store_path,
            ref_date_range=(start_ref, end_ref),
        )
        vintage_df = transform_to_panel(
            lf, geographic_scope='national', as_of_ref=as_of_ref
        )
        if len(vintage_df) > 0:
            parts.append(vintage_df)
            logger.info('Panel built from vintage store (%d rows)', len(vintage_df))

    for cfg in providers:
        try:
            pp_df = ingest_provider(cfg)
            if len(pp_df) > 0:
                parts.append(pp_df)
        except (FileNotFoundError, ValueError, KeyError) as e:
            logger.warning('Provider %s ingestion failed: %s', cfg.name, e)

    if not parts:
        logger.warning('No data ingested from any source')
        return pl.DataFrame(schema=PANEL_SCHEMA)

    panel = pl.concat(parts).sort('period', 'source', 'industry_code', 'revision_number')

    return validate_panel(panel)


def save_panel(panel: pl.DataFrame, output_dir: Path) -> None:
    """Save observation panel to parquet with a manifest file.

    Parameters
    ----------
    panel : pl.DataFrame
        Validated observation panel.
    output_dir : Path
        Directory to write observation_panel.parquet and panel_manifest.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    panel.write_parquet(output_dir / 'observation_panel.parquet')

    source_counts = (
        panel.group_by('source')
        .len()
        .sort('source')
        .to_dicts()
    )

    git_hash = None
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    manifest = {
        'build_timestamp': datetime.now().isoformat(),
        'row_count': len(panel),
        'source_counts': {d['source']: d['len'] for d in source_counts},
        'date_range': {
            'min': str(panel['period'].min()),
            'max': str(panel['period'].max()),
        },
        'git_hash': git_hash,
    }

    manifest_path = output_dir / 'panel_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f'Panel saved: {len(panel)} rows to {output_dir / "observation_panel.parquet"}'
    )


def load_panel(panel_dir: Path) -> pl.DataFrame:
    """Load a previously saved observation panel from parquet.

    Parameters
    ----------
    panel_dir : Path
        Directory containing observation_panel.parquet.

    Returns
    -------
    pl.DataFrame
        Observation panel.
    """
    parquet_path = panel_dir / 'observation_panel.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(f'Panel file not found: {parquet_path}')

    return pl.read_parquet(parquet_path)
