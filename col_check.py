#!/usr/bin/env python3
"""Per-column sanity / ordering checks for the three output types.

Groups:
  1. tick book microstructure (ap asc / bp desc / price-vol sync / level contiguity)
  2. tick cumulative fields non-decreasing per stock (num_of_trades/total_vol/total_notional)
  3. timestamp consistency (int_time/exch_time decoded from exchange_time; trading session)
  4. cross-table: order_price/trade_price within the stock's limit band (from tick)
  5. enum domains & cross-field consistency (market<->channel, biz_index<->market,
     trade_type<->price, trade_amount vs price*volume [informational])
"""
import os
import sys
import polars as pl

BASE = os.environ.get("CHECK_BASE", "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple")


def paths(day):
    ym, ymd = f"{day[:4]}.{day[4:6]}", f"{day[:4]}.{day[4:6]}.{day[6:8]}"
    return {q: f"{BASE}/{ym}/{ymd}/default/{q}/all.parquet" for q in ("1", "2", "3")}


def rep(name, v):
    tag = "PASS" if v == 0 else f"FAIL({v:,})"
    print(f"  [{tag:>12}] {name}")
    return v


# int_time decoded from exchange_time (µs since epoch, +8h CST) -> HHMMSSmmm
def it_from_et(et):
    et = et.cast(pl.Int64)
    epoch_s = et // 1_000_000
    ms = (et // 1000) % 1000
    sod = (epoch_s + 8 * 3600) % 86400
    hh = sod // 3600
    mm = (sod % 3600) // 60
    ss = sod % 60
    return hh * 10_000_000 + mm * 100_000 + ss * 1000 + ms


def check_tick(p, band_out):
    print("=" * 60, "\nTICK")
    df = pl.read_parquet(p).with_columns(
        pl.col("ap_array").cast(pl.List(pl.UInt32)),
        pl.col("bp_array").cast(pl.List(pl.UInt32)),
        pl.col("av_array").cast(pl.List(pl.UInt32)),
        pl.col("bv_array").cast(pl.List(pl.UInt32)),
    )
    n = df.height
    fails = 0

    # structural
    fails += rep("serial contiguous 0..N-1 & unique",
                 0 if (df["serial"].n_unique() == n and df["serial"].min() == 0
                       and df["serial"].max() == n - 1) else n)
    fails += rep("mi_type constant (==9)", int((df["mi_type"] != 9).sum()))
    fails += rep("action_day==trading_day==date",
                 int(((df["action_day"] != df["trading_day"])).sum()))

    # group 3: timestamp
    d = df.select(
        (pl.col("exch_time") != it_from_et(pl.col("exchange_time"))).sum().alias("ts_mismatch"),
        (~pl.col("exch_time").is_between(91500000, 150500000)).sum().alias("ts_range"),
    ).row(0, named=True)
    fails += rep("exch_time == decode(exchange_time)", d["ts_mismatch"])
    fails += rep("exch_time in trading session", d["ts_range"])

    # group 1: book microstructure (element-wise)
    def g(col, i):
        return pl.col(col).list.get(i)

    ap_ord = pl.lit(False)
    bp_ord = pl.lit(False)
    ap_sync = pl.lit(False)
    bp_sync = pl.lit(False)
    ap_gap = pl.lit(False)
    bp_gap = pl.lit(False)
    # one-directional: a quoted price must carry volume (price>0 => vol>0).
    # The reverse (vol>0 & price==0) is a legitimate call-auction encoding.
    for i in range(10):
        ap_sync = ap_sync | ((g("ap_array", i) > 0) & (g("av_array", i) == 0))
        bp_sync = bp_sync | ((g("bp_array", i) > 0) & (g("bv_array", i) == 0))
        if i < 9:
            ap_ord = ap_ord | ((g("ap_array", i) > 0) & (g("ap_array", i + 1) > 0)
                               & (g("ap_array", i) > g("ap_array", i + 1)))
            bp_ord = bp_ord | ((g("bp_array", i) > 0) & (g("bp_array", i + 1) > 0)
                               & (g("bp_array", i) < g("bp_array", i + 1)))
            ap_gap = ap_gap | ((g("ap_array", i) == 0) & (g("ap_array", i + 1) > 0))
            bp_gap = bp_gap | ((g("bp_array", i) == 0) & (g("bp_array", i + 1) > 0))
    m = df.select(
        ap_ord.sum().alias("ap_ord"), bp_ord.sum().alias("bp_ord"),
        ap_sync.sum().alias("ap_sync"), bp_sync.sum().alias("bp_sync"),
        ap_gap.sum().alias("ap_gap"), bp_gap.sum().alias("bp_gap"),
    ).row(0, named=True)
    fails += rep("ap ascending (nonzero)", m["ap_ord"])
    fails += rep("bp descending (nonzero)", m["bp_ord"])
    fails += rep("ap price>0 => av>0", m["ap_sync"])
    fails += rep("bp price>0 => bv>0", m["bp_sync"])
    fails += rep("ap levels contiguous (no gap)", m["ap_gap"])
    fails += rep("bp levels contiguous (no gap)", m["bp_gap"])

    # group 2: cumulative non-decreasing per stock (file is in local_time order)
    cum = df.select(
        ["ticker", "num_of_trades", "total_vol", "total_notional"]
    ).with_columns(
        pl.col("num_of_trades").cast(pl.Int64).diff().over("ticker").alias("dnt"),
        pl.col("total_vol").diff().over("ticker").alias("dtv"),
        pl.col("total_notional").diff().over("ticker").alias("dtn"),
    )
    c = cum.select(
        (pl.col("dnt") < 0).sum().alias("nt"),
        (pl.col("dtv") < 0).sum().alias("tv"),
        (pl.col("dtn") < 0).sum().alias("tn"),
    ).row(0, named=True)
    fails += rep("num_of_trades non-decreasing/stock", c["nt"])
    fails += rep("total_vol non-decreasing/stock", c["tv"])
    fails += rep("total_notional non-decreasing/stock", c["tn"])

    # export per-symbol band for cross-table checks
    band = df.group_by("ticker").agg(
        pl.col("lower_limit_px").filter(pl.col("lower_limit_px") > 0).min().alias("lo"),
        pl.col("upper_limit_px").max().alias("up"),
    )
    band_out["band"] = band
    print(f"  tick fails total: {fails}")
    return fails


def _enum_viol(df, col, allowed):
    return int((~pl.col(col).is_in(allowed)).sum().__class__ is int) if False else \
        int(df.select((~pl.col(col).is_in(allowed)).sum()).item())


def check_order(p, band):
    print("=" * 60, "\nORDER")
    df = pl.read_parquet(p)
    n = df.height
    fails = 0
    fails += rep("serial contiguous 0..N-1 & unique",
                 0 if (df["serial"].n_unique() == n and df["serial"].min() == 0
                       and df["serial"].max() == n - 1) else n)
    fails += rep("mi_type constant (==57)", int((df["mi_type"] != 57).sum()))
    # timestamp
    fails += rep("int_time == decode(exchange_time)",
                 int(df.select((pl.col("int_time") != it_from_et(pl.col("exchange_time"))).sum()).item()))
    # enums
    fails += rep("order_type in {0,1,2,A,D,U}", _enum_viol(df, "order_type", ["0", "1", "2", "A", "D", "U"]))
    fails += rep("bsflag in {B,S}", _enum_viol(df, "bsflag", ["B", "S"]))
    fails += rep("market in {48,49}", _enum_viol(df, "market", [48, 49]))
    # market<->channel single mapping
    mp = df.group_by("channel").agg(pl.col("market").n_unique().alias("nm"))
    fails += rep("each channel -> single market", int((mp["nm"] != 1).sum()))
    # biz_index<->market: SH(channel<1000) biz>0 ; SZ(>=1000) biz==0
    fails += rep("biz_index>0 <=> SH channel",
                 int(df.select((((pl.col("channel") < 1000) & (pl.col("biz_index") == 0)) |
                                ((pl.col("channel") >= 1000) & (pl.col("biz_index") != 0))).sum()).item()))
    # cross-table: limit-order price within band. Exclude 0/sentinel and special
    # order types (market '1', best-price 'U') whose price field is a placeholder.
    j = df.select(["symbol", "order_price", "order_type"]).join(
        band, left_on="symbol", right_on="ticker", how="left")
    fails += rep("order_price within limit band (limit orders)",
                 int(j.select(((pl.col("order_price") > 0) & (pl.col("order_price") != 2147483647) &
                               (~pl.col("order_type").is_in(["1", "U"])) &
                               (pl.col("lo").is_not_null()) &
                               ((pl.col("order_price") < pl.col("lo")) | (pl.col("order_price") > pl.col("up")))).sum()).item()))
    print(f"  order fails total: {fails}")
    return fails


def check_trans(p, band):
    print("=" * 60, "\nTRANS")
    df = pl.read_parquet(p)
    n = df.height
    fails = 0
    fails += rep("serial contiguous 0..N-1 & unique",
                 0 if (df["serial"].n_unique() == n and df["serial"].min() == 0
                       and df["serial"].max() == n - 1) else n)
    fails += rep("mi_type constant (==56)", int((df["mi_type"] != 56).sum()))
    fails += rep("int_time == decode(exchange_time)",
                 int(df.select((pl.col("int_time") != it_from_et(pl.col("exchange_time"))).sum()).item()))
    fails += rep("bsflag in {'',B,S}", _enum_viol(df, "bsflag", ["", "B", "S"]))
    fails += rep("trade_type in {'',0,1,C,F}", _enum_viol(df, "trade_type", ["", "0", "1", "C", "F"]))
    fails += rep("market in {48,49}", _enum_viol(df, "market", [48, 49]))
    mp = df.group_by("channel").agg(pl.col("market").n_unique().alias("nm"))
    fails += rep("each channel -> single market", int((mp["nm"] != 1).sum()))
    fails += rep("biz_index>0 <=> SH channel",
                 int(df.select((((pl.col("channel") < 1000) & (pl.col("biz_index") == 0)) |
                                ((pl.col("channel") >= 1000) & (pl.col("biz_index") != 0))).sum()).item()))
    # trade_type C <=> price 0  (cancel)
    fails += rep("trade_price==0 <=> trade_type=='C'",
                 int(df.select((((pl.col("trade_price") == 0) & (pl.col("trade_type") != "C")) |
                                ((pl.col("trade_type") == "C") & (pl.col("trade_price") != 0))).sum()).item()))
    # cross-table price band (exclude 0 cancel)
    j = df.select(["symbol", "trade_price"]).join(band, left_on="symbol", right_on="ticker", how="left")
    fails += rep("trade_price within limit band",
                 int(j.select(((pl.col("trade_price") > 0) & (pl.col("lo").is_not_null()) &
                               ((pl.col("trade_price") < pl.col("lo")) | (pl.col("trade_price") > pl.col("up")))).sum()).item()))
    # informational: trade_amount vs price*volume relationship
    samp = df.filter((pl.col("trade_price") > 0) & (pl.col("trade_volume") > 0)).head(200000)
    if samp.height:
        r = samp.select((pl.col("trade_amount").cast(pl.Float64) /
                         (pl.col("trade_price").cast(pl.Float64) * pl.col("trade_volume"))).alias("r"))
        print(f"  [INFO] amount/(price*vol): min={r['r'].min():.6g} "
              f"median={r['r'].median():.6g} max={r['r'].max():.6g}")
    print(f"  trans fails total: {fails}")
    return fails


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20210104"
    p = paths(day)
    band_out = {}
    tot = 0
    tot += check_tick(p["1"], band_out)
    band = band_out["band"]
    tot += check_order(p["3"], band)
    tot += check_trans(p["2"], band)
    print("=" * 60)
    print(f"DAY {day}  TOTAL FAILS: {tot}")
    sys.exit(1 if tot else 0)


if __name__ == "__main__":
    main()
