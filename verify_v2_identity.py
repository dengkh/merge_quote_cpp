#!/usr/bin/env python3
"""Read-only: confirm ParquetDataV2 (final, per-stock split) integrity & identity
to the already fully-verified CppGenSimple output.

For a day, for each type (tick/trans/order):
  1. split integrity:  sum(rows of SH*/SZ* per-stock files) == all.parquet rows
  2. identity to CppGenSimple all.parquet (row-aligned by serial):
       same row count, local_time 0 mismatch, exchange_time 0 mismatch
"""
import sys
import glob
import polars as pl
import pyarrow.parquet as pq

V2 = "/mnt/data/data/ParquetDataV2"
CS = "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple"


def dpath(base, day, q):
    ym, ymd = f"{day[:4]}.{day[4:6]}", f"{day[:4]}.{day[4:6]}.{day[6:8]}"
    return f"{base}/{ym}/{ymd}/default/{q}"


def main():
    day = sys.argv[1]
    fails = []
    for t, name in [("1", "tick"), ("2", "trans"), ("3", "order")]:
        vdir = dpath(V2, day, t)
        vall = f"{vdir}/all.parquet"
        # split integrity
        perstock = sum(pq.read_metadata(f).num_rows
                       for f in glob.glob(f"{vdir}/S[HZ]*.parquet"))
        vrows = pq.read_metadata(vall).num_rows
        if perstock != vrows:
            fails.append(f"{name}: split sum {perstock} != all {vrows}")
        # identity to CppGenSimple
        call = f"{dpath(CS, day, t)}/all.parquet"
        v = pl.read_parquet(vall, columns=["serial", "local_time", "exchange_time"]).sort("serial")
        c = pl.read_parquet(call, columns=["serial", "local_time", "exchange_time"]).sort("serial")
        if v.height != c.height:
            fails.append(f"{name}: rows v2={v.height} cs={c.height}")
            continue
        ltm = int((v["local_time"] != c["local_time"]).sum())
        etm = int((v["exchange_time"] != c["exchange_time"]).sum())
        if ltm or etm:
            fails.append(f"{name}: lt_mismatch={ltm} et_mismatch={etm}")
    if fails:
        print(f"{day}  FAIL  " + "; ".join(fails))
        sys.exit(1)
    print(f"{day}  PASS")


if __name__ == "__main__":
    main()
