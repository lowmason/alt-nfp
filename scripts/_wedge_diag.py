"""Read-only diagnostic: characterize the 00-05 government wedge.

Throwaway orientation script for Track-B brainstorming. Not committed.
"""

from dotenv import load_dotenv

load_dotenv(".env")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from nfp_ingest.first_print import first_print_changes  # noqa: E402
from nfp_ingest.vintage_store import read_vintage_store  # noqa: E402

# --- First-print wedge change: the actual forecast target ---
fp00 = first_print_changes(industry_type="total", industry_code="00").rename(
    {"first_print_change_k": "chg00"}
).select("ref_date", "chg00")
fp05 = first_print_changes(industry_type="total", industry_code="05").rename(
    {"first_print_change_k": "chg05"}
).select("ref_date", "chg05")

w = fp00.join(fp05, on="ref_date", how="inner").with_columns(
    (pl.col("chg00") - pl.col("chg05")).alias("wedge_chg")
).sort("ref_date").drop_nulls()

print(f"=== First-print WEDGE change (govt = 00 - 05), n={w.height} ===")
print(f"date range: {w['ref_date'].min()} .. {w['ref_date'].max()}")
arr = w["wedge_chg"].to_numpy()
print(f"wedge_chg  mean={arr.mean():7.1f}k  std={arr.std():7.1f}k  "
      f"min={arr.min():7.1f}k  max={arr.max():7.1f}k")
c00 = w["chg00"].to_numpy()
c05 = w["chg05"].to_numpy()
print(f"chg00      mean={c00.mean():7.1f}k  std={c00.std():7.1f}k")
print(f"chg05      mean={c05.mean():7.1f}k  std={c05.std():7.1f}k")

# Seasonality of the wedge change by calendar month
wm = w.with_columns(pl.col("ref_date").dt.month().alias("mo"))
print("\n=== Wedge first-print change by calendar month (SA series) ===")
by = wm.group_by("mo").agg(
    pl.col("wedge_chg").mean().alias("mean"),
    pl.col("wedge_chg").std().alias("std"),
    pl.len().alias("n"),
).sort("mo")
for r in by.iter_rows(named=True):
    print(f"  month {r['mo']:2d}: mean={r['mean']:7.1f}k  std={r['std'] or 0:6.1f}k  n={r['n']}")

# --- Naive forecast skill on the wedge: how well do simple models do? ---
# Drop COVID (2020-2021) for the error budget
mask = (w["ref_date"].dt.year() < 2020) | (w["ref_date"].dt.year() > 2021)
wc = w.filter(mask)
y = wc["wedge_chg"].to_numpy()
months = wc["ref_date"].dt.month().to_numpy()

# Naive 1: random walk (predict last month's wedge change)
rw_err = y[1:] - y[:-1]
# Naive 2: zero (wedge change ~ 0)
zero_err = y
# Naive 3: calendar-month mean (leave-one-out would be ideal; use in-sample mean as floor)
mo_mean = {m: y[months == m].mean() for m in range(1, 13)}
seas_err = np.array([y[i] - mo_mean[months[i]] for i in range(len(y))])

print(f"\n=== Naive wedge-forecast error (COVID-excluded, n={len(y)}) ===")
print(f"  random-walk      MAE={np.abs(rw_err).mean():6.1f}k  RMSE={np.sqrt((rw_err**2).mean()):6.1f}k")
print(f"  predict-zero     MAE={np.abs(zero_err).mean():6.1f}k  RMSE={np.sqrt((zero_err**2).mean()):6.1f}k")
print(f"  cal-month-mean   MAE={np.abs(seas_err).mean():6.1f}k  RMSE={np.sqrt((seas_err**2).mean()):6.1f}k (in-sample, optimistic)")
print("\n  For reference: consensus Total-NFP ceiling ~48k MAE / ~60-65k RMSE")
print(f"  Private first-print change std (the dominant term) ~ {c05.std():.0f}k")

# === Settled-level wedge: underlying govt change, free of first-print noise ===
def settled_levels(code):
    lf = read_vintage_store(source="ces", seasonally_adjusted=True,
                            geographic_type="national", industry_type="total",
                            industry_code=code)
    df = (lf.select("ref_date", "vintage_date", "revision", "benchmark_revision", "employment")
            .filter(pl.col("employment") > 0).collect())
    # latest (most-revised) value per ref_date
    settled = (df.sort(["benchmark_revision", "revision", "vintage_date"])
                 .group_by("ref_date").last()
                 .select("ref_date", pl.col("employment").alias(f"L{code}")))
    return settled

s = settled_levels("00").join(settled_levels("05"), on="ref_date").sort("ref_date")
s = s.with_columns((pl.col("L00") - pl.col("L05")).alias("wedge_lvl"))
s = s.with_columns((pl.col("wedge_lvl") - pl.col("wedge_lvl").shift(1)).alias("wedge_chg_settled")).drop_nulls()

sc = s.filter((s["ref_date"].dt.year() < 2020) | (s["ref_date"].dt.year() > 2021))
ys = sc["wedge_chg_settled"].to_numpy()
print(f"\n=== SETTLED-level wedge change (COVID-excl, n={len(ys)}) ===")
print(f"  mean={ys.mean():6.1f}k  std={ys.std():6.1f}k  min={ys.min():6.1f}k  max={ys.max():6.1f}k")
mos = sc["ref_date"].dt.month().to_numpy()
mo_mean_s = {m: ys[mos == m].mean() for m in range(1, 13)}
seas_err_s = np.array([ys[i] - mo_mean_s[mos[i]] for i in range(len(ys))])
rw_s = ys[1:] - ys[:-1]
print(f"  random-walk    MAE={np.abs(rw_s).mean():5.1f}k")
print(f"  cal-month-mean MAE={np.abs(seas_err_s).mean():5.1f}k (in-sample)")
print("  by month (settled): " + "  ".join(f"{m}:{mo_mean_s[m]:.0f}" for m in range(1,13)))

# First-print vs settled gap = the first-print/SA-additivity noise we'd inherit
fp_wedge = w.filter((w["ref_date"].dt.year()<2020)|(w["ref_date"].dt.year()>2021))
print(f"\n  first-print wedge-change std = {fp_wedge['wedge_chg'].std():.1f}k")
print(f"  settled     wedge-change std = {ys.std():.1f}k")
print(f"  => excess first-print noise (variance diff) ~ "
      f"{np.sqrt(max(fp_wedge['wedge_chg'].std()**2 - ys.std()**2,0)):.1f}k")
