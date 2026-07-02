#!/usr/bin/env python3
"""Price sanity checks for tick output (OHLC + level prices).

Rules (violations counted only on rows where the relevant values are > 0,
so that pre-open / empty snapshots and empty book levels are ignored):

  A. OHLC internal consistency:  low <= open/last/high  and  high >= open/last/low
  B. limit sanity:               upper_limit >= lower_limit
  C. OHLC within limit band:     lower_limit <= {open,high,low,last} <= upper_limit
  D. low/high within pre_close +/-50%:  0.5*pc <= low <= high <= 1.5*pc
  E. ask/bid levels within limit band: every non-zero ap/bp in [lower_limit, upper_limit]
  F. best ask >= best bid (ap[0] >= bp[0] when both > 0)
"""
import argparse
import os
import polars as pl

BASE = os.environ.get("CHECK_BASE", "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple")


def tick_path(day: str) -> str:
    ym = f"{day[:4]}.{day[4:6]}"
    ymd = f"{day[:4]}.{day[4:6]}.{day[6:8]}"
    return f"{BASE}/{ym}/{ymd}/default/1/all.parquet"


def check_day(day: str) -> dict:
    cols = ["pre_close_px", "open_px", "high_px", "low_px", "last_px",
            "upper_limit_px", "lower_limit_px", "ap_array", "bp_array"]
    lf = pl.scan_parquet(tick_path(day)).select(cols).with_columns(
        pl.col("ap_array").cast(pl.List(pl.UInt32)),
        pl.col("bp_array").cast(pl.List(pl.UInt32)),
    )

    pc = pl.col("pre_close_px")
    o, h, l, c = pl.col("open_px"), pl.col("high_px"), pl.col("low_px"), pl.col("last_px")
    up, lo = pl.col("upper_limit_px"), pl.col("lower_limit_px")

    lim_ok = (up > 0) & (lo > 0)
    pos = lambda x: x > 0

    # E: level prices — max and min-of-nonzero per row
    ap_max = pl.col("ap_array").list.max()
    bp_max = pl.col("bp_array").list.max()
    ap_min = pl.col("ap_array").list.eval(
        pl.element().filter(pl.element() > 0)).list.min()
    bp_min = pl.col("bp_array").list.eval(
        pl.element().filter(pl.element() > 0)).list.min()
    ap0 = pl.col("ap_array").list.get(0)
    bp0 = pl.col("bp_array").list.get(0)

    lf = lf.with_columns(
        # A. OHLC internal consistency
        (pos(l) & pos(o) & (l > o)).alias("A_low_gt_open"),
        (pos(l) & pos(c) & (l > c)).alias("A_low_gt_last"),
        (pos(l) & pos(h) & (l > h)).alias("A_low_gt_high"),
        (pos(h) & pos(o) & (h < o)).alias("A_high_lt_open"),
        (pos(h) & pos(c) & (h < c)).alias("A_high_lt_last"),
        # B. limit sanity
        (lim_ok & (up < lo)).alias("B_up_lt_lo"),
        # C. OHLC within band
        (lim_ok & pos(o) & ((o > up) | (o < lo))).alias("C_open_oob"),
        (lim_ok & pos(h) & ((h > up) | (h < lo))).alias("C_high_oob"),
        (lim_ok & pos(l) & ((l > up) | (l < lo))).alias("C_low_oob"),
        (lim_ok & pos(c) & ((c > up) | (c < lo))).alias("C_last_oob"),
        # D. low/high within pre_close +/-50%
        (pos(pc) & pos(l) & (l.cast(pl.Float64) < 0.5 * pc)).alias("D_low_below"),
        (pos(pc) & pos(h) & (h.cast(pl.Float64) > 1.5 * pc)).alias("D_high_above"),
        # E. level prices within band
        (lim_ok & (ap_max > up)).alias("E_ap_above"),
        (lim_ok & ap_min.is_not_null() & (ap_min < lo)).alias("E_ap_below"),
        (lim_ok & (bp_max > up)).alias("E_bp_above"),
        (lim_ok & bp_min.is_not_null() & (bp_min < lo)).alias("E_bp_below"),
        # F. best ask >= best bid
        (pos(ap0) & pos(bp0) & (ap0 < bp0)).alias("F_cross_book"),
    )

    flag_cols = [x for x in lf.columns if x[:2] in
                 ("A_", "B_", "C_", "D_", "E_", "F_")]
    agg = lf.select(
        [pl.len().alias("rows")] +
        [pl.col(x).sum().alias(x) for x in flag_cols]
    ).collect(streaming=True)
    return agg.row(0, named=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+", help="YYYYMMDD ...")
    args = ap.parse_args()
    grand = {}
    for day in args.days:
        r = check_day(day)
        rows = r.pop("rows")
        total_viol = sum(r.values())
        tag = "PASS" if total_viol == 0 else f"FAIL({total_viol})"
        print(f"{day}  rows={rows:>11,}  {tag}")
        if total_viol:
            for k, v in r.items():
                if v:
                    print(f"    {k:16s} {v:,}")
        for k, v in r.items():
            grand[k] = grand.get(k, 0) + v
    if len(args.days) > 1:
        tv = sum(grand.values())
        print("-" * 50)
        print(f"TOTAL violations: {tv:,}")
        for k, v in grand.items():
            if v:
                print(f"    {k:16s} {v:,}")


if __name__ == "__main__":
    main()
