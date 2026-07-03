#!/usr/bin/env python3
"""Directly verify the global merge order & local_time assignment rule.

Reconstruct the global merge stream by sorting all three types by local_time
(local_time is strictly increasing in merge order, so this recovers it exactly).

Then verify:
  (a) local_time formula:  lt[0]==et[0];  lt[j]==max(et[j], lt[j-1]+1)
      (i.e. local_time = exchange_time, +1 only on ties, in merge order)
  (b) merge key primary:  exchange_time is non-decreasing ACROSS sub-sequences
      (tick=-1, else channel); any strict decrease must be intra-channel
      (biz_index-preserved), never cross sub-sequence.
  (c) tie order: within equal exchange_time, cross sub-seq (type_priority,
      stock_code) strictly increasing.
"""
import os
import sys
import polars as pl

BASE = os.environ.get("CHECK_BASE", "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple")


def paths(day):
    ym, ymd = f"{day[:4]}.{day[4:6]}", f"{day[:4]}.{day[4:6]}.{day[6:8]}"
    return {q: f"{BASE}/{ym}/{ymd}/default/{q}/all.parquet" for q in ("1", "2", "3")}


def load(day):
    p = paths(day)
    parts = []
    for dt, q, tp, sc in [("tick", "1", 0, "ticker"),
                          ("order", "3", 1, "symbol"),
                          ("trans", "2", 2, "symbol")]:
        cols = ["local_time", "exchange_time", sc] + ([] if dt == "tick" else ["channel"])
        df = pl.read_parquet(p[q], columns=cols)
        ss = pl.lit(-1, dtype=pl.Int64) if dt == "tick" else pl.col("channel").cast(pl.Int64)
        parts.append(df.select(
            pl.col("local_time").cast(pl.Int64).alias("lt"),
            pl.col("exchange_time").cast(pl.Int64).alias("et"),
            pl.lit(tp, dtype=pl.Int64).alias("tp"),
            pl.col(sc).cast(pl.Int64).alias("sc"),
            ss.alias("ss"),
        ))
    return pl.concat(parts).sort("lt")


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else "20210104"
    g = load(day)
    n = g.height
    print(f"DAY {day}  global rows={n:,}")

    # (a) local_time formula: expected lt = cummax over max(et, prev+1)
    #     equivalently lt[j] == max(et[j], lt[j-1]+1).  Build expected via prev lt.
    # local_time is exchange_time in NANOSECONDS (et is µs); auto-detect the
    # scale (1 for legacy µs output, 1000 for ns output) so both are supported.
    lt = g["lt"]
    et_raw = g["et"]
    med_lt = lt.filter(lt > 0).median()
    med_et = et_raw.filter(et_raw > 0).median()
    scale = 1000 if (med_et and med_lt and round(med_lt / med_et) == 1000) else 1
    print(f"  lt/et scale = {scale} ({'nanoseconds' if scale == 1000 else 'microseconds'})")
    et = et_raw * scale
    prev_lt = lt.shift(1)
    # expected[j] = max(et[j], prev_lt[j] + 1) ; expected[0] = et[0]
    expected = pl.select(
        pl.when(prev_lt.is_null())
        .then(et)
        .otherwise(pl.max_horizontal(et, prev_lt + 1))
    ).to_series()
    a_viol = int((lt != expected).sum())
    a_ge = int((lt < et).sum())              # lt must be >= et always
    a_mono = int((lt.diff().drop_nulls() <= 0).sum())  # strictly increasing
    print(f"  (a) local_time == max(et, prev+1) : violations={a_viol}")
    print(f"      local_time >= exchange_time    : violations={a_ge}")
    print(f"      local_time strictly increasing : violations={a_mono}")

    # (b) cross sub-sequence exchange_time must be non-decreasing
    same_ss = g["ss"] == g["ss"].shift(1)
    et_dec = et.diff() < 0                    # et[j] < et[j-1]
    cross_dec = int((et_dec & ~same_ss).fill_null(False).sum())
    intra_dec = int((et_dec & same_ss).fill_null(False).sum())
    print(f"  (b) exch_time decrease CROSS sub-seq (must=0): {cross_dec}")
    print(f"      exch_time decrease intra-channel (ok)    : {intra_dec:,}")

    # (c) within equal exchange_time, cross sub-seq (tp, sc) strictly increasing
    same_et = et == et.shift(1)
    same_tp = g["tp"] == g["tp"].shift(1)
    tp_bad = int((same_et & ~same_ss & (g["tp"] < g["tp"].shift(1))).fill_null(False).sum())
    sc_bad = int((same_et & ~same_ss & same_tp & (g["sc"] <= g["sc"].shift(1))).fill_null(False).sum())
    print(f"  (c) same-et cross-subseq tp not increasing: {tp_bad}")
    print(f"      same-et cross-subseq sc not increasing: {sc_bad}")

    ok = (a_viol == 0 and a_ge == 0 and a_mono == 0 and cross_dec == 0
          and tp_bad == 0 and sc_bad == 0)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
