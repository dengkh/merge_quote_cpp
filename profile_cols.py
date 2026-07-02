#!/usr/bin/env python3
"""Column-level profiling for one day's three outputs, to inform sanity checks."""
import sys
import polars as pl

BASE = "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple"
day = sys.argv[1] if len(sys.argv) > 1 else "20210104"
ym, ymd = f"{day[:4]}.{day[4:6]}", f"{day[:4]}.{day[4:6]}.{day[6:8]}"


def path(q):
    return f"{BASE}/{ym}/{ymd}/default/{q}/all.parquet"


def profile(name, q, cols_small):
    print("=" * 70)
    print(f"{name}  ({path(q)})")
    print("=" * 70)
    df = pl.read_parquet(path(q))
    print("rows:", df.height)
    for c in df.columns:
        dt = df[c].dtype
        if dt == pl.List or str(dt).startswith("Array") or str(dt).startswith("List"):
            print(f"  {c:20s} {str(dt):18s} [array-skip scalar stats]")
            continue
        nulls = df[c].null_count()
        try:
            mn, mx = df[c].min(), df[c].max()
        except Exception:
            mn = mx = "?"
        extra = ""
        if c in cols_small:
            u = df[c].unique().sort().to_list()
            extra = f"  uniq({len(u)})={u[:15]}"
        print(f"  {c:20s} {str(dt):10s} null={nulls:<9} min={mn} max={mx}{extra}")
    return df


t = profile("TICK", "1",
            {"mi_type", "action_day", "trading_day", "status", "prefix"})
# tick cross-column
print("  -- checks --")
print("   change == last-preclose ?",
      t.filter((pl.col("last_px") > 0) &
               (pl.col("change") != (pl.col("last_px").cast(pl.Int64) -
                                     pl.col("pre_close_px").cast(pl.Int64)))).height, "mismatch")
print("   wind_code startswith ticker ?",
      t.filter(~pl.col("wind_code").str.starts_with(pl.col("ticker"))).height
      if t["wind_code"].null_count() == 0 else "has-null")
del t

o = profile("ORDER", "3",
            {"mi_type", "market", "order_type", "bsflag", "channel"})
print("  -- checks --")
print("   order_volume<=0:", o.filter(pl.col("order_volume") <= 0).height)
print("   order_price<0:", o.filter(pl.col("order_price") < 0).height)
del o

tr = profile("TRANS", "2",
             {"mi_type", "market", "bsflag", "trade_type", "channel"})
print("  -- checks --")
print("   trade_volume<=0:", tr.filter(pl.col("trade_volume") <= 0).height)
print("   trade_price<0:", tr.filter(pl.col("trade_price") < 0).height)
print("   sell_id<0 or buy_id<0:",
      tr.filter((pl.col("sell_id") < 0) | (pl.col("buy_id") < 0)).height)
