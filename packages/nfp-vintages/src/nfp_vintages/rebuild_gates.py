"""The four §10 store-rebuild acceptance gates.

Replaces the retired frozen-reference parity gate (``specs/store_rebuild.md``
§10, ``plans/10-store_rebuild.md`` T6).  Each gate is a **pure gap-collector**:
it takes Polars frame(s) and returns a ``list[str]`` of human-readable gap
descriptions.  An empty list means the gate PASSES; a non-empty list means it
FAILS, and each entry names the specific problem.  This mirrors the
``_check_*``-style functions in
``packages/nfp-ingest/tests/test_store_coverage.py`` — gates are tests, not
prose.

All four gates key on ``industry_type + industry_code + ownership +
(revision, benchmark_revision) + values``.  ``industry_type`` MUST stay in the
key: code ``55`` is the lone cross-level collision (supersector ``55``
Financial Activities vs sector ``55`` Management of companies) that
``industry_code + ownership`` alone cannot disambiguate.

Two cross-cutting invariants every gate respects:

* **All-sizes selection.** The rebuilt store carries the QCEW Q1 size
  cross-product (``size_class_*`` non-null bucket rows plus a ``total``/``'0'``
  all-sizes row).  Before *any* aggregation, join, or duplicate check, frames
  must be reduced to the headline (all-sizes) level via
  ``size_class_type IS NULL OR size_class_code == '0'``
  (``nfp_ingest.size_class.all_sizes_predicate``); otherwise buckets +
  total double-count and every sum / residual / dup check is wrong.
* **Soft vs hard.** Some checks are launch-blocking ("hard"); others are
  "reconstruct-and-validate" diagnostics the caller weighs ("soft", per §10).
  Because the signature only returns gaps (empty = pass), soft findings are
  prefixed ``"SOFT: "`` and a missing sector-month is silently skipped (never a
  gap).  A good frame yields ``[]``.  A soft-only finding yields a *non-empty*
  list whose entries are all ``"SOFT: "``-prefixed; callers MUST filter out
  ``"SOFT: "`` entries before deciding pass/fail
  (``hard = [g for g in gaps if not g.startswith("SOFT:")]``), so soft findings
  never block the hard gate.

Gate functions are **pure** (frames in, gaps out) — no I/O.  Only the
``@pytest.mark.real_store`` test wrappers touch the store, and they self-skip
when it is unavailable.  No gate or test writes any store.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from nfp_ingest.size_class import all_sizes_predicate
from nfp_lookups.industry import (
    QCEW_SUPERSECTOR,
    get_domain_supersectors,
    remap_industry_type,
)

# ---------------------------------------------------------------------------
# Shared keys / helpers
# ---------------------------------------------------------------------------

# The disambiguating industry key (``industry_type`` keeps code 55 unambiguous).
_INDUSTRY_KEY = ["industry_type", "industry_code", "ownership"]

# Full per-series key (geography + industry).  National is the only geography
# in this rebuild, but we never assume it.
_SERIES_KEY = ["geographic_type", "geographic_code", *_INDUSTRY_KEY]

# The four legitimate (revision, benchmark_revision) cohorts.
_REQUIRED_REV_BMR: set[tuple[int, int]] = {(0, 0), (1, 0), (2, 0), (2, 1)}

# Stored private supersector codes (industry_type='supersector').
_SUPERSECTOR_CODES: tuple[str, ...] = (
    "10", "20", "30", "40", "50", "55", "60", "65", "70", "80",
)

# The 10 supersectors plus 05 (total private) — the hard-gate frontier set.
_FRONTIER_INDUSTRIES: list[tuple[str, str, str]] = [
    ("total", "05", "private"),
    *[("supersector", c, "private") for c in _SUPERSECTOR_CODES],
]


def _all_sizes(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce to the headline (all-sizes) level (store_rebuild §7).

    Drops the QCEW Q1 size-bucket rows so sums / joins / dup checks see exactly
    one employment value per (series, ref_date, rev, bmr).
    """
    if "size_class_type" not in df.columns:
        return df
    return df.filter(all_sizes_predicate())


def _best_available(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (series, ref_date): the most-revised, most-recent print.

    Selection order: highest ``revision``, then highest ``benchmark_revision``,
    then latest ``vintage_date`` — the published/benchmarked value the
    reconstruction gate compares.  Callers must reduce to the all-sizes level
    first so the single retained row is a headline level, not a size bucket.
    """
    return (
        df.sort(
            ["revision", "benchmark_revision", "vintage_date"], descending=True
        )
        .unique(subset=[*_SERIES_KEY, "ref_date"], keep="first")
    )


# ---------------------------------------------------------------------------
# Gate 1 — History consistency
# ---------------------------------------------------------------------------


def gate_history_consistency(
    rebuilt_ces: pl.DataFrame,
    existing_store: pl.DataFrame,
    *,
    cutoff: date = date(2024, 1, 1),
    abs_tol: float = 0.5,
    rel_tol: float = 1e-6,
) -> list[str]:
    """Rebuilt CES prints reproduce the current store on the ≤2023 known-good core.

    For ``ref_date < cutoff``:

    * Apply :func:`remap_industry_type` to the *existing* store's
      ``(industry_type, industry_code)`` so it keys on the rebuilt axes
      (``national/00 → (total, total)``, ``domain/05 → (total, private)``,
      supersectors/sectors unchanged with ``ownership='private'``).  The
      *rebuilt* store already carries those axes — leave it alone.
    * For (series, ref_date, revision, benchmark_revision) rows present in
      BOTH stores, ``employment`` is compared within ``abs_tol + rel_tol * |old|``
      (``abs_tol=0.5`` covers thousands-precision rounding; ``rel_tol=1e-6`` is a
      near-zero relative rail).  The verdict is **split by cohort**:

      - **HARD** — the benchmark-free prints ``(0,0)``/``(1,0)`` must reproduce the
        legacy store to that tolerance.  These carry no annual-benchmark
        ambiguity (verified 2026-06-16: 0/2520 diverge on the real stores), so a
        drift is a genuine reconstruction bug.
      - **SOFT** — the benchmark-bearing cohorts ``(2,0)``/``(2,1)`` are reported
        ``"SOFT: "`` only.  The legacy benchmark splice deviates from
        BLS-published cesvinall (its ``(2,1)`` mis-stamped the *latest* benchmark
        value under the *earliest* ``vintage_date`` — a lookahead bug); the
        rebuilt store reproduces the literal cesvinall cells to the unit, so it
        *correctly* diverges from legacy here.  The HARD accuracy rail on these
        cohorts is :func:`gate_ces_fidelity` (rebuilt vs cesvinall), not this
        legacy-store comparison.
    * The four ``(revision, benchmark_revision)`` cohorts
      ``{(0,0),(1,0),(2,0),(2,1)}`` must all appear in the rebuilt CES for the
      private hierarchy **and** the ``00`` anchor.

    The rebuilt store legitimately carries EXTRA per-benchmark ``(2,1)`` rows
    the old store lacks.  Because each annual benchmark revises prior history,
    two ``(2,1)`` rows for the same ref_date but different benchmark cohorts
    (``vintage_date``) hold legitimately different levels — comparing
    "latest-on-each-side" would align mismatched cohorts and spuriously fail.
    So the ``(2,1)`` comparison is keyed on ``vintage_date`` as well: a rebuilt
    ``(2,1)`` row is checked only against the existing store's ``(2,1)`` row for
    the *same* benchmark cohort, and rebuilt ``(2,1)`` cohorts the old store
    never held are not flagged.  The ``(0,0)/(1,0)/(2,0)`` cohorts carry one
    vintage per series-month and join on the plain key.

    Parameters
    ----------
    rebuilt_ces, existing_store : pl.DataFrame
        ``VINTAGE_STORE_SCHEMA``-conformant CES frames (rebuilt = new axes;
        existing = legacy axes with null ``ownership``).
    cutoff : date
        Exclusive upper bound on ``ref_date`` for the history join.
    abs_tol, rel_tol : float
        Employment match tolerance (absolute + relative).  Defaults
        (``abs_tol=0.5``, ``rel_tol=1e-6``) admit only rounding slack; the gate
        is the parity replacement, so a realistic reconstruction error of a few
        thousand jobs must fail it.
    """
    gaps: list[str] = []

    rebuilt = _all_sizes(rebuilt_ces).filter(pl.col("ref_date") < cutoff)
    existing = _all_sizes(existing_store).filter(pl.col("ref_date") < cutoff)

    # --- (rev, bmr) cohort population on the rebuilt CES -------------------
    # Private hierarchy + the 00 anchor; each must carry all four cohorts.
    # The existence check is intentionally scoped to the private root (05) +
    # the 00 anchor — a defensible reading of §10's "private hierarchy AND the
    # 00 anchor"; the value-match join below keys on the full _SERIES_KEY and so
    # already compares every overlapping series across the whole hierarchy.
    for itype, icode, own in [("total", "00", "total"), ("total", "05", "private")]:
        sub = rebuilt.filter(
            (pl.col("industry_type") == itype)
            & (pl.col("industry_code") == icode)
            & (pl.col("ownership") == own)
        )
        if sub.is_empty():
            gaps.append(
                f"history: rebuilt CES missing the {itype}/{icode} "
                f"(ownership={own}) series entirely"
            )
            continue
        present = set(
            sub.select("revision", "benchmark_revision").unique().iter_rows()
        )
        missing = _REQUIRED_REV_BMR - present
        if missing:
            gaps.append(
                f"history: rebuilt CES {itype}/{icode} (ownership={own}) "
                f"missing (rev,bmr) cohorts {sorted(missing)}"
            )

    if existing.is_empty():
        gaps.append(f"history: existing store has no CES rows before {cutoff}")
        return gaps

    # --- remap the existing store onto the rebuilt axes -------------------
    # Compute the remap over the distinct legacy pairs, then join back (avoids
    # a row-wise map over the whole frame).
    legacy_pairs = (
        existing.select("industry_type", "industry_code").unique()
    )
    remap_rows: list[dict[str, str]] = []
    for itype, icode in legacy_pairs.iter_rows():
        try:
            new_type, ownership = remap_industry_type(itype, icode)
        except ValueError:
            # Deferred government / out-of-taxonomy codes: skip — never our gate.
            continue
        remap_rows.append(
            {
                "industry_type": itype,
                "industry_code": icode,
                "_new_industry_type": new_type,
                "_new_ownership": ownership,
            }
        )
    if not remap_rows:
        gaps.append("history: no existing-store industries remapped to rebuilt axes")
        return gaps

    remap_df = pl.DataFrame(remap_rows)
    # The remap only rewrites ``industry_type`` and ``ownership``; ``industry_code``
    # is carried through unchanged, so drop only the two columns being replaced.
    existing_remapped = (
        existing.join(remap_df, on=["industry_type", "industry_code"], how="inner")
        .drop("industry_type", "ownership")
        .rename(
            {
                "_new_industry_type": "industry_type",
                "_new_ownership": "ownership",
            }
        )
    )

    # --- value-match join, cohort-aligned -------------------------------
    # The (0,0)/(1,0)/(2,0) cohorts carry one vintage per series-month, so they
    # join on the plain (series, ref_date, rev, bmr) key.  The (2,1) cohort fans
    # out across annual benchmarks (each revising prior history), so it joins on
    # ``vintage_date`` too — a rebuilt (2,1) is compared only against the
    # existing store's same-benchmark (2,1), never a different cohort.
    matched = _cohort_aligned_match(rebuilt, existing_remapped)
    if matched.is_empty():
        gaps.append(
            "history: no overlapping (series, ref_date, rev, bmr) rows between "
            "rebuilt and existing CES on the ≤2023 core"
        )
        return gaps

    mismatched = matched.filter(
        (pl.col("employment") - pl.col("_employment_old")).abs()
        > (abs_tol + rel_tol * pl.col("_employment_old").abs())
    )

    # Split the divergence by cohort.  The benchmark-FREE prints (0,0)/(1,0)
    # carry no annual-benchmark ambiguity, so the rebuilt store must reproduce
    # the legacy store's first/second prints to rounding — a drift here is a real
    # reconstruction bug (HARD).  The benchmark-BEARING cohorts (2,0)/(2,1) are a
    # different matter: verified against the real stores (2026-06-16), the legacy
    # benchmark splice deviates from BLS-published cesvinall — its (2,1) even
    # mis-stamped the latest benchmark value under the earliest vintage_date (a
    # lookahead bug) — while the rebuilt store reproduces the literal cesvinall
    # cells to the unit.  So a (2,0)/(2,1)-vs-legacy divergence is a *documented
    # expected* divergence (the rebuild diverges toward ground truth), reported
    # SOFT; the HARD accuracy rail on those cohorts is :func:`gate_ces_fidelity`
    # (rebuilt vs cesvinall), not this legacy-store comparison.
    _benchmark_free = (pl.col("benchmark_revision") == 0) & (pl.col("revision") < 2)
    prebench = mismatched.filter(_benchmark_free)
    bench = mismatched.filter(~_benchmark_free)

    def _examples(frame: pl.DataFrame) -> str:
        return "\n".join(
            f"  {r['industry_type']}/{r['industry_code']} "
            f"(own={r['ownership']}) ref={r['ref_date']} "
            f"(rev,bmr)=({r['revision']},{r['benchmark_revision']}): "
            f"rebuilt={r['employment']} old={r['_employment_old']}"
            for r in frame.head(5).iter_rows(named=True)
        )

    if not prebench.is_empty():
        gaps.append(
            f"history: {prebench.height} CES benchmark-free (rev 0/1) employment "
            f"values diverge from the existing store on the ≤2023 core:\n"
            + _examples(prebench)
        )
    if not bench.is_empty():
        gaps.append(
            f"SOFT: history {bench.height} CES benchmark-cohort (2,0)/(2,1) values "
            f"diverge from the existing store — expected: the legacy benchmark "
            f"splice deviates from BLS-published cesvinall, which gate_ces_fidelity "
            f"checks the rebuild reproduces:\n" + _examples(bench)
        )

    return gaps


def _cohort_aligned_match(
    rebuilt: pl.DataFrame, existing: pl.DataFrame
) -> pl.DataFrame:
    """Inner-join rebuilt vs existing CES, cohort-aligning the (2,1) fan-out.

    Returns the rebuilt rows that overlap the existing store, carrying the
    existing employment as ``_employment_old`` for the divergence check.

    The ``(0,0)/(1,0)/(2,0)`` cohorts carry a single vintage per series-month,
    so they join on the plain ``(series, ref_date, rev, bmr)`` key.  The
    ``(2,1)`` cohort fans out across annual benchmarks (each revising prior
    history); aligning "latest-on-each-side" would compare a newer rebuilt
    benchmark against the older existing one and report a spurious divergence.
    So ``(2,1)`` adds ``vintage_date`` to the join key — a rebuilt ``(2,1)`` is
    matched only to the existing store's ``(2,1)`` for the *same* benchmark
    cohort, and rebuilt ``(2,1)`` cohorts the old store never held drop out of
    the inner join (not flagged).
    """
    is_2_1 = (pl.col("revision") == 2) & (pl.col("benchmark_revision") == 1)

    base_key = [*_SERIES_KEY, "ref_date", "revision", "benchmark_revision"]

    def _join(left: pl.DataFrame, right: pl.DataFrame, key: list[str]) -> pl.DataFrame:
        return left.join(
            right.select([*key, "employment"]).rename(
                {"employment": "_employment_old"}
            ),
            on=key,
            how="inner",
        )

    # ``vintage_date`` is already a column on both sides (store schema), so
    # adding it to the (2,1) join key introduces no new column — both result
    # frames share the same schema and concat cleanly.
    non21 = _join(
        rebuilt.filter(~is_2_1), existing.filter(~is_2_1), base_key
    )
    only21 = _join(
        rebuilt.filter(is_2_1), existing.filter(is_2_1), [*base_key, "vintage_date"]
    )
    return pl.concat([non21, only21])


# ---------------------------------------------------------------------------
# Gate 1b — CES fidelity (rebuilt CES vs cesvinall reference)
# ---------------------------------------------------------------------------


def gate_ces_fidelity(
    rebuilt_ces: pl.DataFrame,
    reference_ces: pl.DataFrame,
    *,
    abs_tol: float = 0.5,
    rel_tol: float = 0.0,
) -> list[str]:
    """Rebuilt CES reproduces the cesvinall-derived reference to the unit.

    The TRUE CES reconstruction-accuracy rail (the CES analogue of
    :func:`gate_qcew_fidelity`).  Where :func:`gate_history_consistency`
    compares against the *legacy store* — whose benchmark splice deviates from
    BLS-published cesvinall on the ``(2,0)``/``(2,1)`` cohorts — this gate
    compares against the cesvinall triangle itself (the reference is a fresh
    ``build_ces_panel`` of the raw ``cesvinall`` CSVs).  It is therefore the
    HARD accuracy check on the benchmark-bearing cohorts the history gate now
    treats SOFT: a real benchmark-walk regression (off-by-one, wrong
    ``vintage_date`` stamp) that the legacy comparison can no longer catch must
    fail HERE.

    Both frames are reduced to the all-sizes level, then compared on the full
    per-vintage key ``(geographic_type, geographic_code, ownership,
    industry_type, industry_code, ref_date, revision, benchmark_revision,
    vintage_date)`` — ``vintage_date`` is in the key so the ``(2,1)``
    per-benchmark fan-out aligns cohort-for-cohort:

    * **Hard** (failing entry): for rows present in BOTH, an absolute mismatch
      ``|rebuilt - reference| > abs_tol + rel_tol * |reference|``.
    * **Soft** (``"SOFT: "``): rows present on only one side (a coverage /
      ``as_of`` frontier difference, not a value corruption).

    This is a SAME-SOURCE, to-the-unit reproduction (rebuilt CES == the
    cesvinall triangle), so the tolerance is a flat absolute rounding rail:
    ``rel_tol=0`` and ``abs_tol=0.5`` (employment in thousands, stored rounded
    to integer thousands).  A magnitude-scaling ``rel_tol`` is rejected for the
    same reason as :func:`gate_qcew_fidelity` — it would wave through a
    hundreds-to-thousands-of-jobs store-write corruption at the total level.

    Pure: frames in, gaps out.  No I/O.
    """
    gaps: list[str] = []
    key = [*_SERIES_KEY, "ref_date", "revision", "benchmark_revision", "vintage_date"]
    rebuilt = _all_sizes(rebuilt_ces)
    reference = _all_sizes(reference_ces)
    if rebuilt.is_empty() or reference.is_empty():
        return ["fidelity: rebuilt or reference CES frame is empty"]

    left = rebuilt.select([*key, "employment"]).rename({"employment": "_rebuilt"})
    right = reference.select([*key, "employment"]).rename({"employment": "_reference"})

    matched = left.join(right, on=key, how="inner")
    bad = matched.filter(
        (pl.col("_rebuilt") - pl.col("_reference")).abs()
        > (abs_tol + rel_tol * pl.col("_reference").abs())
    )
    if not bad.is_empty():
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']} "
            f"(rev,bmr)=({r['revision']},{r['benchmark_revision']}): "
            f"rebuilt={r['_rebuilt']:.3f} reference={r['_reference']:.3f}"
            for r in bad.head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"fidelity: {bad.height} CES values differ from the cesvinall "
            f"reference beyond {abs_tol}+{rel_tol:.0e}*|ref|:\n" + "\n".join(ex)
        )

    only_rebuilt = left.join(right, on=key, how="anti")
    only_reference = right.join(left, on=key, how="anti")
    for frame, where in ((only_rebuilt, "rebuilt"), (only_reference, "reference")):
        if frame.is_empty():
            continue
        missing_side = "reference" if where == "rebuilt" else "rebuilt"
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']} "
            f"(rev,bmr)=({r['revision']},{r['benchmark_revision']})"
            for r in frame.sort(key).head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"SOFT: fidelity {frame.height} CES rows present in {where} but "
            f"missing in {missing_side}:\n" + "\n".join(ex)
        )

    return gaps


# ---------------------------------------------------------------------------
# Gate 2 — Gap fill
# ---------------------------------------------------------------------------


def gate_gap_fill(
    rebuilt: pl.DataFrame,
    *,
    frontier_ref_date: date,
    dec_cohort_years: tuple[int, ...] = (2024, 2025),
    nesting_tol: float = 1.0,
) -> list[str]:
    """Gap-fill gate (priority-ordered, store_rebuild §10).

    **Hard** (failing entries):

    * ``total/05`` (total private) and the 10 supersectors must each have a
      row current to ``frontier_ref_date`` (the level the nowcast + supersector
      narrative read today).
    * The December ``(2,1)`` cohorts for ``dec_cohort_years`` must be complete
      for that same frontier set (one ``Dec-YYYY`` (2,1) row per industry).

    **Soft** (``"SOFT: "``-prefixed, caller decides; a missing component row is
    never a gap): the three §10 additive-nesting identities where all needed
    rows exist — ``05 == 06 + 08``, each domain summing its supersectors, and
    each supersector summing its stored sectors — within ``nesting_tol``
    (thousands of jobs).  Checked per ``(geography, ownership, ref_date,
    revision, benchmark_revision)`` group (the full series context) after
    collapsing the ``(2,1)`` per-benchmark fan-out to one row per cohort, and
    skipped where any component row is absent.
    """
    gaps: list[str] = []
    df = _all_sizes(rebuilt)
    if df.is_empty():
        return ["gap_fill: rebuilt frame is empty"]

    # --- HARD: frontier currency -----------------------------------------
    frontier = df.filter(pl.col("ref_date") == frontier_ref_date)
    present_frontier = set(
        frontier.select(_INDUSTRY_KEY).unique().iter_rows()
    )
    missing_frontier = [
        (it, ic, ow) for (it, ic, ow) in _FRONTIER_INDUSTRIES
        if (it, ic, ow) not in present_frontier
    ]
    if missing_frontier:
        gaps.append(
            f"gap_fill: {len(missing_frontier)} hard-gate industries missing at "
            f"frontier ref_date {frontier_ref_date}: {missing_frontier}"
        )

    # --- HARD: December (2,1) cohorts complete ----------------------------
    dec_2_1 = df.filter(
        (pl.col("revision") == 2)
        & (pl.col("benchmark_revision") == 1)
        & (pl.col("ref_date").dt.month() == 12)
    )
    for year in dec_cohort_years:
        year_rows = dec_2_1.filter(pl.col("ref_date").dt.year() == year)
        present_dec = set(year_rows.select(_INDUSTRY_KEY).unique().iter_rows())
        missing_dec = [
            (it, ic, ow) for (it, ic, ow) in _FRONTIER_INDUSTRIES
            if (it, ic, ow) not in present_dec
        ]
        if missing_dec:
            gaps.append(
                f"gap_fill: December {year} (2,1) cohort incomplete — "
                f"{len(missing_dec)} industries missing: {missing_dec}"
            )

    # --- SOFT: additive nesting where present -----------------------------
    gaps.extend(_check_additive_nesting(df, nesting_tol))

    return gaps


# Nesting group key: the FULL series context.  Geography + ownership are
# constant on today's national/private rebuild, but keying on them keeps the
# identity correct if a second geography or the deferred government ownership
# axis (§11) ever lands — a government supersector must not sum into a private
# parent.
_NESTING_GROUP_KEY = [
    "geographic_type", "geographic_code", "ownership",
    "ref_date", "revision", "benchmark_revision",
]


def _check_additive_nesting(df: pl.DataFrame, tol: float) -> list[str]:
    """Soft nesting validation of the three §10 additive identities.

    Validates, per ``_NESTING_GROUP_KEY`` group (full series context):

    * ``05 == 06 + 08``;
    * each domain (``06``/``08``) == sum of its stored supersectors;
    * each supersector == sum of its stored sectors (``QCEW_SUPERSECTOR``).

    The rebuilt store fans the ``(2,1)`` cohort out across annual benchmarks
    (multiple ``vintage_date`` rows per series-month-cohort); those are
    collapsed to one row per cohort (latest ``vintage_date``) BEFORE summing, so
    a parent and its components are never summed across mismatched benchmark
    depth (which would spuriously break the identity on exactly the benchmarked
    cohort the gate most wants to validate).

    Each identity is *skipped* (no gap) for any group where a component row is
    absent — a missing sector/supersector-month does not block promotion (§10).
    """
    gaps: list[str] = []

    # Collapse the (2,1) fan-out: one row per (series, ref_date, rev, bmr).
    df = (
        df.sort("vintage_date", descending=True)
        .unique(
            subset=[*_SERIES_KEY, "ref_date", "revision", "benchmark_revision"],
            keep="first",
        )
    )
    group_key = _NESTING_GROUP_KEY

    def _level(industry_type: str, industry_code: str) -> pl.DataFrame:
        return (
            df.filter(
                (pl.col("industry_type") == industry_type)
                & (pl.col("industry_code") == industry_code)
            )
            .group_by(group_key)
            .agg(pl.col("employment").sum().alias("_emp"))
        )

    def _parent_vs_components(
        parent_type: str,
        parent_code: str,
        comp_type: str,
        comp_codes: list[str],
        label: str,
    ) -> None:
        """Append a SOFT gap if parent != sum(components) where all present."""
        parent = _level(parent_type, parent_code).rename({"_emp": "_parent"})
        comp = (
            df.filter(
                (pl.col("industry_type") == comp_type)
                & (pl.col("industry_code").is_in(comp_codes))
            )
            .group_by(group_key)
            .agg(
                pl.col("employment").sum().alias("_comp_sum"),
                pl.col("industry_code").n_unique().alias("_n_comp"),
            )
            # Only groups where every stored component is present.
            .filter(pl.col("_n_comp") == len(comp_codes))
        )
        bad = (
            parent.join(comp, on=group_key, how="inner")
            .filter((pl.col("_parent") - pl.col("_comp_sum")).abs() > tol)
        )
        if not bad.is_empty():
            ex = bad.head(3).select(group_key + ["_parent", "_comp_sum"]).to_dicts()
            gaps.append(
                f"SOFT: gap_fill nesting {label} in {bad.height} groups: {ex}"
            )

    # (1) 05 == 06 + 08
    tot = _level("total", "05").rename({"_emp": "_tot"})
    goods = _level("domain", "06").rename({"_emp": "_goods"})
    svc = _level("domain", "08").rename({"_emp": "_svc"})
    joined = (
        tot.join(goods, on=group_key, how="inner")
        .join(svc, on=group_key, how="inner")
        .with_columns((pl.col("_goods") + pl.col("_svc")).alias("_sum"))
        .filter((pl.col("_tot") - pl.col("_sum")).abs() > tol)
    )
    if not joined.is_empty():
        ex = joined.head(3).select(group_key + ["_tot", "_sum"]).to_dicts()
        gaps.append(
            f"SOFT: gap_fill nesting 05 != 06+08 in {joined.height} groups: {ex}"
        )

    # (2) 06 == sum(goods supersectors); 08 == sum(service supersectors)
    for domain_code in ("06", "08"):
        components = [
            c for c in get_domain_supersectors(domain_code) if c in _SUPERSECTOR_CODES
        ]
        _parent_vs_components(
            "domain", domain_code, "supersector", components,
            f"{domain_code} != sum(supersectors)",
        )

    # (3) each supersector == sum of its stored sectors.  QCEW_SUPERSECTOR is the
    # authoritative stored-taxonomy parent map (e.g. 30 -> {31, 32} durable+
    # nondurable, 10 -> {11, 21}); get_supersector_components() is the CES
    # modeling hierarchy and is MISALIGNED with the stored sectors, so it must
    # NOT be used here.
    for ss_code, info in QCEW_SUPERSECTOR.items():
        sector_codes = [str(s) for s in info["sectors"]]
        _parent_vs_components(
            "supersector", ss_code, "sector", sector_codes,
            f"supersector {ss_code} != sum(sectors)",
        )

    return gaps


# ---------------------------------------------------------------------------
# Gate 3 — Reconstruction accuracy + Q1 continuity
# ---------------------------------------------------------------------------

# Verified per-series QCEW-minus-CES DEFINITIONAL residuals as a fraction
# (qcew/ces - 1), settled non-COVID, headline level.  QCEW counts UI-covered
# employment; CES estimates ALL nonfarm payroll incl. UI-exempt orgs (religious/
# membership NAICS 813, private households 814, etc.), so QCEW sits BELOW CES and
# the residual is NEGATIVE.  The Other Services gap (80/81 ~ -22.5%) is driven by
# UI-exempt religious/membership organizations in NAICS 813.
#
# Source: maintainer verification 2026-06-16, against published QCEW (rebuilt 80
# == published agglvl-13 '1027' == 4739.7 @ 2024-06).  RE-SEED these if the
# QCEW->CES crosswalk or the QCEW UI-coverage universe changes — they are
# empirical, not structural.
_EXPECTED_QCEW_CES_RESIDUAL: dict[str, float] = {
    "05": -0.025,
    "08": -0.029,
    "80": -0.225,
    "81": -0.225,
}

# Per-series acceptance band (half-width, fraction) around the expected residual.
# A single uniform band straddling a -22.5% series and a -2.9% series rubber-stamps
# the named adversarial case: an 8pp band on 08 (expected -0.029) admits 0% — a
# coverage bug that pulls CES-universe data or includes UI-exempt orgs (erasing the
# definitional gap) would slip through.  So the shallow 05/08 series get a tight
# ~2pp band (well above the verified <1pp p10-p90 spread, but tight enough to catch
# the 0% regression), while the deep 80/81 series keep the generous 8pp headroom
# their -22.5% magnitude warrants.  Tighten as confidence grows.  Every code in
# _EXPECTED_QCEW_CES_RESIDUAL MUST appear here.
_QCEW_CES_RESIDUAL_BAND: dict[str, float] = {
    "05": 0.02,
    "08": 0.02,
    "80": 0.08,
    "81": 0.08,
}

# A residual more negative than ``expected - _IMPLAUSIBLE_COLLAPSE_MARGIN`` is an
# implausible coverage collapse, not the definitional gap.  This per-MONTH HARD
# floor is independent of the frontier exclusion (which is now date-scoped to the
# incomplete window): a settled month whose QCEW falls far below CES cannot be
# waved through as a "frontier" exclusion and then hidden by the median.
_IMPLAUSIBLE_COLLAPSE_MARGIN: float = 0.15

# The frontier exclusion only applies to ref_dates in this incomplete window.  The
# data-relative value rule (QCEW << prior-year) is a NECESSARY but not sufficient
# condition: a settled-history month that collapses is a reconstruction error, not
# an incomplete frontier, so it must stay in the hard band and trip the floor.  The
# CES carries Other Services ONLY as supersector/80 (verified 0 CES sector/81
# rows), so QCEW sector/81 is compared against CES supersector/80.  Every other
# residual series maps to its own (industry_type, industry_code) on the CES side.
_CES_RESIDUAL_TARGET: dict[tuple[str, str], tuple[str, str]] = {
    ("total", "05"): ("total", "05"),
    ("domain", "08"): ("domain", "08"),
    ("supersector", "80"): ("supersector", "80"),
    ("sector", "81"): ("supersector", "80"),
}

# COVID years dropped from the median (employment levels were anomalous).
_COVID_YEARS: frozenset[int] = frozenset({2020, 2021})


def gate_reconstruction_accuracy(
    rebuilt_qcew: pl.DataFrame,
    published_ces: pl.DataFrame,
    *,
    band: dict[str, float] | None = None,
    pos_margin: float = 0.01,
    frontier_window_start: date | None = None,
    implausible_collapse_margin: float = _IMPLAUSIBLE_COLLAPSE_MARGIN,
) -> list[str]:
    """Rebuilt-QCEW vs published-CES DEFINITIONAL-residual band gate (§10).

    The rebuilt QCEW is faithful to published QCEW (UI-covered employment); CES
    estimates ALL nonfarm payroll including UI-exempt employers.  So ``QCEW < CES``
    is the EXPECTED direction and the per-series residual is a stable, negative,
    *definitional* gap — not a reconstruction error.  (Reconstruction fidelity is
    :func:`gate_qcew_fidelity`, a QCEW-vs-QCEW check.)  This gate therefore checks
    each series' MEDIAN residual against :data:`_EXPECTED_QCEW_CES_RESIDUAL`.

    For each series in :data:`_CES_RESIDUAL_TARGET`, the per-ref_date residual is
    ``qcew.employment / ces.employment - 1`` over shared ref_dates, then:

    * **Excluded from the median** — COVID years (2020, 2021) and
      incomplete-frontier months.  Frontier rule (collapse-signal AND date-scoped):
      a month qualifies as incomplete-frontier ONLY when it both (a) falls on/after
      ``frontier_window_start`` (the unsettled window; defaults to the latest
      calendar year in the data, where the QCEW size / sector-detail tables lag the
      area tables — verified 2025-Q1: domain 06 -88%, total 05 -20%) AND (b) shows
      an implausible collapse (residual more negative than
      ``expected - implausible_collapse_margin``).  The date gate is load-bearing:
      the SAME collapse signal in SETTLED history is a reconstruction error, so it
      stays in the hard band and trips the floor below — it is never downgraded to
      SOFT.  Normal-magnitude months in the frontier window stay in the band (only
      the implausibly-collapsed ones are excused as lag).

      Both classes of exclusion — COVID years and incomplete-frontier months —
      are SOFT-reported (one line each) so the maintainer sees every month that
      dropped out of the hard band check, not just the frontier ones.

    * **Hard fail (implausible collapse, per-month floor)** — for every clean
      (non-COVID, non-frontier) month, a residual more negative than
      ``expected - implausible_collapse_margin`` (default 0.15 below expected)
      hard-fails as an implausible coverage collapse.  This floor catches a
      settled-month collapse the median would otherwise absorb, and is independent
      of the frontier exclusion (which the date gate now confines to the incomplete
      window).

    * **Hard fail (median band)** if the median of the *remaining* (clean)
      residuals is outside ``expected ± band[code]`` (the per-series band in
      :data:`_QCEW_CES_RESIDUAL_BAND` — ~2pp for the shallow 05/08, ~8pp for the
      deep 80/81).  A single uniform band would straddle a -22.5% series and a
      -2.9% series and admit a 0% residual on 08, rubber-stamping a coverage bug
      that erased the definitional gap; the per-series band closes that.  A median
      more positive than ``pos_margin`` (default +0.01) ALSO hard-fails as an
      anomalous direction (QCEW > CES).

    If, after exclusions, a series has NO clean months to band-check, that is a
    HARD failure (not a silent SOFT pass): an all-excluded series would otherwise
    let a systematic collapse — which the value rule excludes month-by-month —
    slip through.  The date-scoped frontier rule makes this unreachable for settled
    history, but the hard path closes the residual hole.

    The 81->80 mapping is explicit (:data:`_CES_RESIDUAL_TARGET`): QCEW sector/81
    joins CES supersector/80 but the median is grouped on the ORIGINAL QCEW
    ``(industry_type, industry_code)`` so 80 and 81 stay distinct series and 81 is
    never silently dropped.

    Both sides are reduced to the all-sizes level and to a single best-available
    cohort per (series, ref_date) before the residual is taken — NEVER summed
    across cohorts (which would inflate the level and make residuals meaningless).
    """
    gaps: list[str] = []
    band = band if band is not None else _QCEW_CES_RESIDUAL_BAND
    codes = {c for (_it, c) in _CES_RESIDUAL_TARGET}
    qcew = _all_sizes(rebuilt_qcew).filter(pl.col("industry_code").is_in(codes))
    ces = _all_sizes(published_ces)
    if qcew.is_empty() or ces.is_empty():
        return ["reconstruction: no rows for residual codes {05,08,80,81}"]

    # Best-available single cohort per (series, ref_date) on each side.
    qcew_ba = _best_available(qcew)
    ces_ba = _best_available(ces)

    # Map each QCEW residual series onto its CES target series, join on the CES
    # axes + ref_date, then group/median on the ORIGINAL QCEW series so 80 and 81
    # remain distinct (and 81 is never dropped by the 80 join).
    ces_lookup = (
        ces_ba.select(
            ["geographic_type", "geographic_code", "ownership",
             "industry_type", "industry_code", "ref_date", "employment"]
        )
        .rename(
            {
                "industry_type": "_ces_itype",
                "industry_code": "_ces_icode",
                "employment": "_ces",
            }
        )
    )

    rows: list[pl.DataFrame] = []
    for (q_it, q_ic), (c_it, c_ic) in _CES_RESIDUAL_TARGET.items():
        side = qcew_ba.filter(
            (pl.col("industry_type") == q_it) & (pl.col("industry_code") == q_ic)
        ).select(
            ["geographic_type", "geographic_code", "ownership",
             "industry_type", "industry_code", "ref_date", "employment"]
        ).rename({"employment": "_qcew"})
        if side.is_empty():
            continue
        merged = side.join(
            ces_lookup.filter(
                (pl.col("_ces_itype") == c_it) & (pl.col("_ces_icode") == c_ic)
            ),
            left_on=["geographic_type", "geographic_code", "ownership", "ref_date"],
            right_on=["geographic_type", "geographic_code", "ownership", "ref_date"],
            how="inner",
        )
        if not merged.is_empty():
            rows.append(
                merged.with_columns(
                    (pl.col("_qcew") / pl.col("_ces") - 1.0).alias("_resid")
                )
            )

    if not rows:
        return ["reconstruction: no shared (industry, ref_date) rows to compare"]
    merged = pl.concat(rows, how="vertical")

    # Expected per-series definitional residual + the implausible-collapse signal
    # (a residual more negative than ``expected - implausible_collapse_margin``).
    merged = merged.with_columns(
        pl.col("industry_code")
        .replace_strict(_EXPECTED_QCEW_CES_RESIDUAL, default=None)
        .alias("_expected")
    ).with_columns(
        (pl.col("_resid") < pl.col("_expected") - implausible_collapse_margin)
        .alias("_implausible")
    )

    # Incomplete-frontier detection: an implausible collapse INSIDE the unsettled
    # frontier window — the latest, still-settling calendar year, where the QCEW
    # size / sector-detail tables lag the area tables (verified 2025-Q1: domain 06
    # -88%, total 05 -20%).  An implausible collapse in SETTLED history is NOT
    # frontier — it stays in the hard band and trips the floor below (a settled
    # collapse is a reconstruction error, not lag).  The window auto-detects from
    # the data (max ref_date year) unless pinned via ``frontier_window_start``; the
    # date gate is load-bearing so a settled-month break is never downgraded to SOFT.
    if frontier_window_start is None:
        max_year = merged["ref_date"].dt.year().max()
        frontier_window_start = date(int(max_year), 1, 1)

    year = pl.col("ref_date").dt.year()
    is_covid = year.is_in(list(_COVID_YEARS))
    is_frontier = (pl.col("ref_date") >= frontier_window_start) & pl.col("_implausible")

    # SOFT-report the excluded frontier months.  ``& ~is_covid`` so a COVID-year
    # month is reported once (under COVID), partitioning the two SOFT lines.
    excluded = merged.filter(is_frontier & ~is_covid)
    if not excluded.is_empty():
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']}: "
            f"resid={r['_resid']:+.3f} (expected {r['_expected']:+.3f}; incomplete "
            f"frontier >= {frontier_window_start} — QCEW size/detail tables lag)"
            for r in excluded.sort("ref_date").head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"SOFT: reconstruction excluded {excluded.height} incomplete-frontier "
            f"months from the band check:\n" + "\n".join(ex)
        )

    # SOFT-report the excluded COVID months too (both classes stay visible).
    covid_excluded = merged.filter(is_covid)
    if not covid_excluded.is_empty():
        cx = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']}: "
            f"resid={r['_resid']:.3f} (COVID year)"
            for r in covid_excluded.sort("ref_date").head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"SOFT: reconstruction excluded {covid_excluded.height} COVID "
            f"months from the band check:\n" + "\n".join(cx)
        )

    clean = merged.filter(~is_covid & ~is_frontier)
    if clean.is_empty():
        # HARD (not SOFT): an all-excluded series would otherwise let a systematic
        # collapse slip through.
        gaps.append(
            "reconstruction: no clean (non-COVID, non-frontier) months to "
            "band-check — every comparable month was excluded"
        )
        return gaps

    # HARD per-month floor: a SETTLED clean month with an implausible collapse.
    # In-window implausible collapses are SOFT frontier (above) and excluded from
    # ``clean``; what remains implausible in ``clean`` is therefore settled history
    # — a real reconstruction error the median would otherwise absorb.
    collapsed = clean.filter(pl.col("_implausible"))
    if not collapsed.is_empty():
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']}: "
            f"resid={r['_resid']:+.3f} (expected {r['_expected']:+.3f}; "
            f">{implausible_collapse_margin:.0%} below)"
            for r in collapsed.sort("ref_date").head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"reconstruction: {collapsed.height} settled months show an implausible "
            f"collapse (residual >{implausible_collapse_margin:.0%} below "
            f"expected — not the definitional gap):\n" + "\n".join(ex)
        )

    # Per-series median residual, keyed on the ORIGINAL QCEW series.
    medians = (
        clean.group_by(["industry_type", "industry_code"])
        .agg(pl.col("_resid").median().alias("_med"), pl.len().alias("_n"))
    )
    for r in medians.sort("industry_type", "industry_code").iter_rows(named=True):
        code = r["industry_code"]
        med = r["_med"]
        expected = _EXPECTED_QCEW_CES_RESIDUAL[code]
        band_tol = band[code]
        if med > pos_margin:
            gaps.append(
                f"reconstruction: {r['industry_type']}/{code} median residual "
                f"{med:+.3f} is anomalously positive (QCEW > CES by "
                f">{pos_margin:.0%}; expected ~{expected:+.3f}) over {r['_n']} "
                f"clean months"
            )
        elif abs(med - expected) > band_tol:
            gaps.append(
                f"reconstruction: {r['industry_type']}/{code} median residual "
                f"{med:+.3f} out-of-band (expected {expected:+.3f} ± "
                f"{band_tol:.0%}) over {r['_n']} clean months"
            )

    return gaps


def gate_qcew_fidelity(
    rebuilt_qcew: pl.DataFrame,
    reference_qcew: pl.DataFrame,
    *,
    abs_tol: float = 0.05,
    rel_tol: float = 0.0,
) -> list[str]:
    """Rebuilt-QCEW vs reference-QCEW near-exact fidelity gate (store_rebuild §10).

    The TRUE reconstruction-accuracy check.  Unlike
    :func:`gate_reconstruction_accuracy` (which measures the *definitional* gap
    between QCEW and CES), this compares the rebuilt QCEW against the published
    QCEW it is meant to reproduce — catching store-write corruption or a
    stale/``!=published`` build.

    Both frames are reduced to the all-sizes level (the stored rebuild carries Q1
    size buckets; a ``build_qcew_panel`` reference does not), then compared on the
    full key ``(geographic_type, geographic_code, ownership, industry_type,
    industry_code, ref_date, revision, benchmark_revision)``:

    * **Hard** (failing entry): for rows present in BOTH, an absolute mismatch
      ``|rebuilt - reference| > abs_tol + rel_tol * |reference|`` (near-exact).
    * **Soft** (``"SOFT: "``): rows present on only one side are reported (a
      coverage/vintage difference, not a value corruption).

    This is a SAME-SOURCE, to-the-unit reproduction (rebuilt QCEW == published
    QCEW), so the tolerance is a flat absolute rounding rail: ``rel_tol=0`` and
    ``abs_tol=0.05`` (= 50 jobs, employment in thousands).  A magnitude-scaling
    ``rel_tol`` is deliberately rejected — at ``rel_tol=1e-4`` the admitted slack
    is ``1e-4 * |ref|``, i.e. ~13,050 jobs at a 130,000 total-private level and
    ~474 jobs at the 80 level, which would wave through a real store-write
    corruption of hundreds-to-thousands of jobs (the same reason
    :func:`gate_history_consistency` uses ``rel_tol=1e-6``, not ``1e-4``).  With a
    flat ``abs_tol`` the rail is the same 50 jobs at every level.

    Pure: frames in, gaps out.  No I/O.
    """
    gaps: list[str] = []
    key = [
        "geographic_type", "geographic_code", "ownership",
        "industry_type", "industry_code", "ref_date",
        "revision", "benchmark_revision",
    ]
    rebuilt = _all_sizes(rebuilt_qcew)
    reference = _all_sizes(reference_qcew)
    if rebuilt.is_empty() or reference.is_empty():
        return ["fidelity: rebuilt or reference QCEW frame is empty"]

    left = rebuilt.select([*key, "employment"]).rename({"employment": "_rebuilt"})
    right = reference.select([*key, "employment"]).rename({"employment": "_reference"})

    matched = left.join(right, on=key, how="inner")
    bad = matched.filter(
        (pl.col("_rebuilt") - pl.col("_reference")).abs()
        > (abs_tol + rel_tol * pl.col("_reference").abs())
    )
    if not bad.is_empty():
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']} "
            f"(rev,bmr)=({r['revision']},{r['benchmark_revision']}): "
            f"rebuilt={r['_rebuilt']:.3f} reference={r['_reference']:.3f}"
            for r in bad.head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"fidelity: {bad.height} QCEW values differ from the reference "
            f"beyond {abs_tol}+{rel_tol:.0e}*|ref|:\n" + "\n".join(ex)
        )

    # Rows present on only one side (coverage / vintage difference) — SOFT.
    only_rebuilt = left.join(right, on=key, how="anti")
    only_reference = right.join(left, on=key, how="anti")
    for frame, where in ((only_rebuilt, "rebuilt"), (only_reference, "reference")):
        if frame.is_empty():
            continue
        missing_side = "reference" if where == "rebuilt" else "rebuilt"
        ex = [
            f"  {r['industry_type']}/{r['industry_code']} ref={r['ref_date']} "
            f"(rev,bmr)=({r['revision']},{r['benchmark_revision']})"
            for r in frame.sort(key).head(5).iter_rows(named=True)
        ]
        gaps.append(
            f"SOFT: fidelity {frame.height} rows present in {where} but missing "
            f"in {missing_side}:\n" + "\n".join(ex)
        )

    return gaps


def gate_q1_continuity(
    rebuilt: pl.DataFrame,
    *,
    clean_supersectors: tuple[str, ...] = _SUPERSECTOR_CODES,
    tol: float = 0.05,
) -> list[str]:
    """Q1 headline continuity for suppression-free supersectors (T5 carry-over).

    **Diagnostic-only / never hard.** This gate emits only ``"SOFT: "``-prefixed
    findings — it can never block promotion.  The promotion caller MUST
    partition findings on the prefix
    (``hard = [g for g in gaps if not g.startswith("SOFT:")]``) before deciding
    pass/fail, or every real run with a genuine Q1 discontinuity would
    spuriously fail.

    **Divergence from the literal T6 instruction (plans/10:148).** T6 asks to
    "compare the Q1 ``total``/``'0'`` against the area-endpoint all-sizes level
    within tolerance".  That same-row diff is **moot** on the composed store: the
    §7 fix overrides the Q1 ``total``/``'0'`` headline *to* the area-levels total
    (``compose_rebuild_panel``), so the headline now **is** the area level —
    ``gate_qcew_fidelity`` checks that to the unit across all four quarters.  This
    gate therefore keeps a deliberately weaker **temporal** proxy — each Q1 month
    vs the interpolation of its immediate non-Q1 neighbours — as a light
    month-over-month continuity check; it remains diagnostic-only.

    For each clean supersector, take the all-sizes level
    (``size_class_type IS NULL OR size_class_code == '0'``) per ref_date, and
    require each Q1 month's level to be within ``tol`` (fractional) of the
    interpolation of its immediate, **calendar-adjacent** non-Q1 neighbours.  A
    Q1 month with no surrounding non-Q1 months (frontier / store edge), or whose
    nearest non-Q1 neighbour is not the adjacent calendar month (a gap in the
    series), is skipped — as is February (both neighbours are Q1).

    ``clean_supersectors`` defaults to all 10 stored supersectors.  Per
    plans/10:136 suppression is contained to sectors ``31``/``32``/``11``
    (3-/4-digit NAICS); the supersector level (size agglvl 23 = area agglvl 13)
    is exact for **all** 10, so all 10 are certified clean and validated here.
    These are **stored** supersector codes (``'20'``, ``'30'``, …) — not QCEW
    aggregate codes (``'1012'``).
    """
    gaps: list[str] = []
    df = _all_sizes(rebuilt).filter(
        (pl.col("industry_type") == "supersector")
        & (pl.col("industry_code").is_in(list(clean_supersectors)))
    )
    if df.is_empty():
        return []

    # Use the latest available level per (series, ref_date) so we compare
    # apples-to-apples across the size/area boundary.
    latest = (
        df.sort("vintage_date", descending=True)
        .unique(subset=[*_SERIES_KEY, "ref_date"], keep="first")
    )

    for key_vals, series in latest.group_by(_SERIES_KEY):
        s = series.sort("ref_date")
        ref = s["ref_date"]
        emp = s["employment"]
        is_q1 = ref.dt.month().is_in([1, 2, 3])
        month_idx = (ref.dt.year() * 12 + ref.dt.month()).to_list()
        emp_list = emp.to_list()
        q1_flags = is_q1.to_list()

        for i, q1 in enumerate(q1_flags):
            if not q1:
                continue
            # Only use a neighbour that is non-Q1 AND the adjacent calendar month
            # (a month gap in the series makes the positional neighbour 2+ months
            # away, so its level is not a valid interpolation target).
            prev_i = (
                i - 1
                if i - 1 >= 0
                and not q1_flags[i - 1]
                and month_idx[i] - month_idx[i - 1] == 1
                else None
            )
            next_i = (
                i + 1
                if i + 1 < len(q1_flags)
                and not q1_flags[i + 1]
                and month_idx[i + 1] - month_idx[i] == 1
                else None
            )
            if prev_i is None and next_i is None:
                continue  # isolated Q1 (store edge / gap) — nothing to compare
            neighbours = [emp_list[j] for j in (prev_i, next_i) if j is not None]
            expected = sum(neighbours) / len(neighbours)
            if expected <= 0:
                continue
            actual = emp_list[i]
            if abs(actual - expected) > tol * expected:
                key = key_vals if isinstance(key_vals, tuple) else (key_vals,)
                month = month_idx[i]
                gaps.append(
                    f"SOFT: q1_continuity {key[2]}/{key[3]} "
                    f"ref={ref[i]}: Q1 all-sizes level {actual:.1f} deviates "
                    f">{tol:.0%} from neighbour mean {expected:.1f} "
                    f"(month_idx={month})"
                )

    return gaps


# ---------------------------------------------------------------------------
# Gate 4 — Vintage integrity (as-of slice)
# ---------------------------------------------------------------------------


def gate_vintage_integrity(as_of_slice: pl.DataFrame) -> list[str]:
    """``_validate_censored_selection``-style checks on an as-of-censored slice.

    Mirrors the *style* of
    ``nfp_ingest.vintage_store._validate_censored_selection`` (vintage_store.py)
    — but collects gaps into a ``list[str]`` instead of raising, and is a pure
    function (the private fn is **not** imported).

    Checks (all hard):

    1. **No duplicate (series, ref_date).** One row per series-month after
       censoring.
    2. **No cross-vintage sums.** Exactly one ``vintage_date`` per (series,
       ref_date) — a censored slice must not mix vintages within a series-month.
    3. **No null/zero/NaN employment.**

    The per-series key includes the size dimension so QCEW Q1 size-bucket rows
    are not mistaken for duplicate series-months.
    """
    gaps: list[str] = []
    if as_of_slice.is_empty():
        return ["vintage_integrity: as-of slice is empty"]

    # Size dims are part of the series identity (Q1 buckets are distinct series).
    size_cols = [
        c for c in ("size_class_type", "size_class_code")
        if c in as_of_slice.columns
    ]
    series_month = [*_SERIES_KEY, *size_cols, "ref_date"]

    # 1. No duplicate (series, ref_date).
    dup = (
        as_of_slice.group_by(series_month)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not dup.is_empty():
        ex = dup.head(3).to_dicts()
        gaps.append(
            f"vintage_integrity: {dup.height} duplicate (series, ref_date) "
            f"keys: {ex}"
        )

    # 2. No cross-vintage sums — one vintage_date per (series, ref_date).
    multi_vintage = (
        as_of_slice.group_by(series_month)
        .agg(pl.col("vintage_date").n_unique().alias("_n_vintage"))
        .filter(pl.col("_n_vintage") > 1)
    )
    if not multi_vintage.is_empty():
        ex = multi_vintage.head(3).to_dicts()
        gaps.append(
            f"vintage_integrity: {multi_vintage.height} (series, ref_date) keys "
            f"mix multiple vintage_dates (cross-vintage sum risk): {ex}"
        )

    # 3. No null/zero/NaN employment.
    bad_emp = as_of_slice.filter(
        pl.col("employment").is_null()
        | pl.col("employment").is_nan()
        | (pl.col("employment") <= 0)
    )
    if not bad_emp.is_empty():
        ex = bad_emp.head(3).select(
            [*_INDUSTRY_KEY, "ref_date", "employment"]
        ).to_dicts()
        gaps.append(
            f"vintage_integrity: {bad_emp.height} rows with null/zero/NaN "
            f"employment: {ex}"
        )

    return gaps
