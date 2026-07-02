#!/usr/bin/env python3
from __future__ import annotations
"""
Verification for the *simple* merge output (no NPY, no 09:40 split).

The simple pipeline produces, for each type, a per-type parquet sorted globally
by (exchange_time, type_priority, stock_code) with per-channel order+trans merged
by biz_index/applseq, and local_time = exchange_time with +1 on ties (global,
strictly increasing across the 3-type merge stream).

Checks (all applied to the whole day uniformly — there is no front/back split):
  1. local_time strictly increasing (+ no nulls)
  2. output row count == raw input row count
  3. channel-internal index (order_index / trade_index) non-decreasing
  4a. SH biz_index interleaving (order+trans merged by local_time) non-decreasing
  4b. SZ applseq  interleaving (order+trans merged by local_time) non-decreasing
  5.  cross-channel sub-sequence ordering (3-type merge, grouped by exchange_time)

Usage:
  python3 verify_simple.py 20210104
  python3 verify_simple.py 20210104 --output-base /path/to/out --raw-base /path/to/raw
"""

import argparse
import gc
import glob
import os
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from loguru import logger

OUTPUT_BASE_DEFAULT = "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple"
RAW_BASE_DEFAULT = "/mnt/beegfs_ssd/public/quant002/hds_work/CppGenRaw"

QUOTE_SPECS = {
    "tick": {"quote_num": "1", "channel_col": None, "index_col": None},
    "trans": {"quote_num": "2", "channel_col": "channel", "index_col": "trade_index"},
    "order": {"quote_num": "3", "channel_col": "channel", "index_col": "order_index"},
}


def get_output_path(date_str: str, quote_num: str, output_base: str) -> str:
    ym = f"{date_str[:4]}.{date_str[4:6]}"
    ymd = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
    return f"{output_base}/{ym}/{ymd}/default/{quote_num}/all.parquet"


def get_raw_dir(date_str: str, quote_num: str, raw_base: str) -> str:
    ym = f"{date_str[:4]}.{date_str[4:6]}"
    ymd = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
    return f"{raw_base}/{ym}/{ymd}/default/{quote_num}"


# ── Check 1: local_time strictly increasing ──────────────────────────


def check_strictly_increasing(out_path: str) -> dict:
    lt = pl.scan_parquet(out_path).select("local_time").collect()["local_time"]
    total = lt.len()
    null_count = lt.null_count()
    diffs = lt.diff().drop_nulls()
    violations = int((diffs <= 0).sum())
    return {
        "total_rows": total,
        "null_lt": null_count,
        "violations": violations,
        "pass": violations == 0 and null_count == 0,
    }


# ── Check 2: output row count == raw input row count ─────────────────


def check_row_count(date_str: str, quote_num: str, out_path: str, raw_base: str) -> dict:
    out_rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    raw_dir = get_raw_dir(date_str, quote_num, raw_base)
    if not Path(raw_dir).is_dir():
        return {"raw_rows": -1, "out_rows": out_rows, "diff": 0, "pass": True, "skipped": True}
    files = [
        f for f in glob.glob(f"{raw_dir}/*.parquet")
        if os.path.basename(f) != "all.parquet"
    ]
    raw_rows = sum(pq.read_metadata(f).num_rows for f in files)
    diff = out_rows - raw_rows
    return {
        "raw_rows": raw_rows,
        "out_rows": out_rows,
        "diff": diff,
        "pass": diff == 0,
        "skipped": False,
    }


# ── Check 3: channel-internal index non-decreasing ───────────────────


def _count_idx_violations(df: pl.DataFrame, ch_col: str, idx_col: str) -> int:
    return (
        df.with_columns(pl.col(idx_col).diff().over(ch_col).alias("_d"))
        .filter(pl.col("_d").is_not_null())
        .filter(pl.col("_d") < 0)
        .height
    )


def check_channel_index_ordering(data_type: str, out_path: str) -> dict:
    spec = QUOTE_SPECS[data_type]
    ch_col, idx_col = spec["channel_col"], spec["index_col"]
    if ch_col is None or idx_col is None:
        return {"applicable": False, "pass": True}
    df = (
        pl.scan_parquet(out_path)
        .select([ch_col, pl.col(idx_col).cast(pl.Int64)])
        .collect()
    )
    viol = _count_idx_violations(df, ch_col, idx_col)
    n_channels = df[ch_col].n_unique()
    del df
    gc.collect()
    return {
        "applicable": True,
        "channels": n_channels,
        "index_col": idx_col,
        "violations": viol,
        "pass": viol == 0,
    }


# ── Check 4: SH biz_index / SZ applseq interleaving ──────────────────


def _check_interleaving(
    out_paths: dict[str, str], ch_filter: pl.Expr, idx_col_order: str,
    idx_col_trans: str, key_name: str,
) -> dict:
    trans_path, order_path = out_paths.get("trans"), out_paths.get("order")
    if not trans_path or not order_path:
        return {"applicable": False, "pass": True}
    if not Path(trans_path).exists() or not Path(order_path).exists():
        return {"applicable": False, "pass": True}

    trans_df = (
        pl.scan_parquet(trans_path)
        .select(["local_time", "channel", pl.col(idx_col_trans).cast(pl.Int64).alias(key_name)])
        .filter(ch_filter)
        .collect()
    )
    order_df = (
        pl.scan_parquet(order_path)
        .select(["local_time", "channel", pl.col(idx_col_order).cast(pl.Int64).alias(key_name)])
        .filter(ch_filter)
        .collect()
    )
    if trans_df.height == 0 and order_df.height == 0:
        return {"applicable": False, "pass": True}

    merged = pl.concat([trans_df, order_df]).sort("local_time")
    del trans_df, order_df
    gc.collect()
    viol = _count_idx_violations(merged, "channel", key_name)
    n_channels = merged["channel"].n_unique()
    total = merged.height
    del merged
    gc.collect()
    return {
        "applicable": True,
        "channels": n_channels,
        "total_rows": total,
        "violations": viol,
        "pass": viol == 0,
    }


def check_sh_biz_interleaving(out_paths: dict[str, str]) -> dict:
    return _check_interleaving(
        out_paths, pl.col("channel") < 1000, "biz_index", "biz_index", "biz_index"
    )


def check_sz_applseq_interleaving(out_paths: dict[str, str]) -> dict:
    return _check_interleaving(
        out_paths, pl.col("channel") >= 1000, "order_index", "trade_index", "applseq"
    )


# ── Check 5: cross-channel sub-sequence ordering ─────────────────────


def _analyze_cross_subseq(df: pl.DataFrame, group_expr: pl.Expr) -> dict:
    zeros = {"same_tp": 0, "cross_tp": 0, "same_sc": 0, "cross_sc": 0}
    if df.height <= 1:
        return zeros
    same_group = group_expr == group_expr.shift(1)
    same_ss = pl.col("_ss") == pl.col("_ss").shift(1)
    same_tp = pl.col("_tp") == pl.col("_tp").shift(1)
    tp_viol = same_group & (pl.col("_tp") < pl.col("_tp").shift(1))
    sc_decreased = same_group & same_tp & (pl.col("_sc") < pl.col("_sc").shift(1))
    sc_not_increased = same_group & same_tp & (pl.col("_sc") <= pl.col("_sc").shift(1))
    return df.select(
        (tp_viol & same_ss).fill_null(False).sum().alias("same_tp"),
        (tp_viol & ~same_ss).fill_null(False).sum().alias("cross_tp"),
        (sc_decreased & same_ss).fill_null(False).sum().alias("same_sc"),
        (sc_not_increased & ~same_ss).fill_null(False).sum().alias("cross_sc"),
    ).row(0, named=True)


def check_cross_channel_ordering(out_paths: dict[str, str]) -> dict:
    """Merge tick+order+trans by local_time, group by exchange_time, and check
    cross-sub-sequence (tp, sc) strict ordering. order+trans of the same channel
    share one sub-sequence (channel); tick is its own (-1)."""
    parts: list[pl.DataFrame] = []
    for dt, tp_val in [("tick", 0), ("order", 1), ("trans", 2)]:
        p = out_paths.get(dt)
        if p is None or not Path(p).exists():
            continue
        sc_col = "ticker" if dt == "tick" else "symbol"
        ch_expr = (
            pl.lit(-1, dtype=pl.Int32) if dt == "tick"
            else pl.col("channel").cast(pl.Int32)
        )
        read_cols = ["local_time", "exchange_time"] + (
            [] if dt == "tick" else ["channel"]
        ) + [sc_col]
        try:
            part = (
                pl.scan_parquet(p).select(read_cols)
                .with_columns(
                    pl.lit(tp_val, dtype=pl.Int8).alias("_tp"),
                    ch_expr.alias("_ch"),
                    pl.col(sc_col).cast(pl.Int64).alias("_sc"),
                )
                .select(["local_time", "exchange_time", "_tp", "_ch", "_sc"])
                .collect()
            )
        except pl.exceptions.ComputeError:
            part = (
                pl.read_parquet(p, columns=read_cols)
                .with_columns(
                    pl.lit(tp_val, dtype=pl.Int8).alias("_tp"),
                    ch_expr.alias("_ch"),
                    pl.col(sc_col).cast(pl.Int64).alias("_sc"),
                )
                .select(["local_time", "exchange_time", "_tp", "_ch", "_sc"])
            )
        parts.append(part)

    if len(parts) < 2:
        return {"applicable": False, "pass": True}

    merged = pl.concat(parts).sort("local_time")
    del parts
    gc.collect()

    subseq = (
        pl.when(pl.col("_ch") == -1)
        .then(pl.lit(-1, dtype=pl.Int64))
        .otherwise(pl.col("_ch").cast(pl.Int64))
    )
    merged = merged.with_columns(subseq.alias("_ss"))

    lt_viol = int((merged["local_time"].diff().drop_nulls() <= 0).sum())
    res = _analyze_cross_subseq(merged, pl.col("exchange_time"))
    total = merged.height
    del merged
    gc.collect()

    cross = res["cross_tp"] + res["cross_sc"]
    return {
        "applicable": True,
        "total_rows": total,
        "lt_viol": lt_viol,
        "same_tp": res["same_tp"],
        "same_sc": res["same_sc"],
        "cross_tp": res["cross_tp"],
        "cross_sc": res["cross_sc"],
        "cross": cross,
        "pass": cross == 0 and lt_viol == 0,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify simple merge output.")
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--output-base", default=OUTPUT_BASE_DEFAULT)
    parser.add_argument("--raw-base", default=RAW_BASE_DEFAULT)
    parser.add_argument(
        "--types", nargs="*", default=["tick", "trans", "order"],
        choices=["tick", "trans", "order"],
    )
    args = parser.parse_args()
    date_str = args.date.strip()

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.info("=" * 65)
    logger.info("Verify(simple): date={} output={}", date_str, args.output_base)
    logger.info("=" * 65)

    all_pass = True
    out_paths: dict[str, str] = {}

    for dt in args.types:
        spec = QUOTE_SPECS[dt]
        out_path = get_output_path(date_str, spec["quote_num"], args.output_base)
        out_paths[dt] = out_path
        logger.info("")
        logger.info("── {} (quote={}) ──", dt, spec["quote_num"])

        if not Path(out_path).exists():
            logger.error("  Output not found: {}", out_path)
            all_pass = False
            continue

        r1 = check_strictly_increasing(out_path)
        tag = "PASS" if r1["pass"] else "FAIL"
        logger.info(
            "  [{}] 1. strictly increasing: rows={:,} null={} violations={}",
            tag, r1["total_rows"], r1["null_lt"], r1["violations"],
        )
        all_pass &= r1["pass"]

        r2 = check_row_count(date_str, spec["quote_num"], out_path, args.raw_base)
        if r2.get("skipped"):
            logger.info("  [SKIP] 2. row count vs raw: out={:,} (no raw dir)", r2["out_rows"])
        else:
            tag = "PASS" if r2["pass"] else "FAIL"
            logger.info(
                "  [{}] 2. row count vs raw: raw={:,} out={:,} diff={:+,}",
                tag, r2["raw_rows"], r2["out_rows"], r2["diff"],
            )
            all_pass &= r2["pass"]

        r3 = check_channel_index_ordering(dt, out_path)
        if r3["applicable"]:
            tag = "PASS" if r3["pass"] else "FAIL"
            logger.info(
                "  [{}] 3. channel {} non-decreasing: violations={} (channels={})",
                tag, r3["index_col"], r3["violations"], r3["channels"],
            )
            all_pass &= r3["pass"]
        else:
            logger.info("  [N/A] 3. channel index order: not applicable for {}", dt)

    # 4a. SH biz_index interleaving
    r4 = check_sh_biz_interleaving(out_paths)
    logger.info("")
    if r4["applicable"]:
        tag = "PASS" if r4["pass"] else "FAIL"
        logger.info(
            "  [{}] 4a. SH biz_index interleave: violations={} (rows={:,} channels={})",
            tag, r4["violations"], r4["total_rows"], r4["channels"],
        )
        all_pass &= r4["pass"]
    else:
        logger.info("  [N/A] 4a. SH biz_index interleave")

    # 4b. SZ applseq interleaving
    r4b = check_sz_applseq_interleaving(out_paths)
    if r4b["applicable"]:
        tag = "PASS" if r4b["pass"] else "FAIL"
        logger.info(
            "  [{}] 4b. SZ applseq interleave: violations={} (rows={:,} channels={})",
            tag, r4b["violations"], r4b["total_rows"], r4b["channels"],
        )
        all_pass &= r4b["pass"]
    else:
        logger.info("  [N/A] 4b. SZ applseq interleave")

    # 5. cross-channel ordering
    r5 = check_cross_channel_ordering(out_paths)
    if r5["applicable"]:
        tag = "PASS" if r5["pass"] else "FAIL"
        logger.info("")
        logger.info("── Merged 3-type analysis ──")
        logger.info(
            "  [{}] 5a. global local_time strictly increasing: violations={}",
            "PASS" if r5["lt_viol"] == 0 else "FAIL", r5["lt_viol"],
        )
        logger.info("  [{}] 5b. cross-channel sub-seq: rows={:,}", tag, r5["total_rows"])
        logger.info("       same-seq(ok): tp={:,} sc={:,}", r5["same_tp"], r5["same_sc"])
        logger.info("       cross-seq(BUG!): tp={:,} sc={:,}", r5["cross_tp"], r5["cross_sc"])
        all_pass &= r5["pass"]
    else:
        logger.info("  [N/A] 5. cross-channel ordering")

    logger.info("")
    logger.info("=" * 65)
    if all_pass:
        logger.info("  RESULT: ALL PASSED")
    else:
        logger.error("  RESULT: SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
