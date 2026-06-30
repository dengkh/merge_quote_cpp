#!/usr/bin/env python3
from __future__ import annotations
"""
Comprehensive verification for sorted_npy_merge output.

Per-type checks:
  1. local_time strictly increasing       (global)
  2. NPY local_time correctly applied     (truncate to μs, must match)
  3. Row count completeness               (vs sorted parquet)
  4. Business column integrity            (sample comparison)
  5. Channel-internal index non-decreasing (order_index / trade_index)

Cross-type checks:
  6a. SH BizIndex interleaving            (order + trans merged by local_time)
  6b. SZ ApplSeq interleaving             (order + trans merged by local_time)
  7.  Cross-channel sub-sequence ordering  (stg2 merged: same-seq vs cross-seq)

Usage:
  python scripts/verify_npy_output.py 20250221
  python scripts/verify_npy_output.py 20250221 --output-base /path/to/stg2
  python scripts/verify_npy_output.py 20250221 --sorted-dir /path/to/stg1/sorted_parquet
"""

import argparse
import gc
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltmerge.stage_paths import get_stage1_sorted_dir, get_stage2_output_root

NPY_BASE_CLASSIC = "/mnt/data/public/quant001/local_times/stocks/91500000-94000000"
NPY_BASE_DELTA = "/mnt/data/public/quant001/stocks"
OUTPUT_BASE_DEFAULT = get_stage2_output_root()
SORTED_DIR_DEFAULT = get_stage1_sorted_dir()
CST = timezone(timedelta(hours=8))

QUOTE_SPECS = {
    "tick": {
        "quote_num": "1",
        "struct_col": "tick_data",
        "join_keys": ["ticker", "exch_time"],
        "use_group_idx": True,
        "npy_loader": "snapshot",
        "channel_col": None,
        "index_col": None,
    },
    "trans": {
        "quote_num": "2",
        "struct_col": "trans_data",
        "join_keys": ["channel", "trade_index"],
        "use_group_idx": False,
        "npy_loader": "trans",
        "channel_col": "channel",
        "index_col": "trade_index",
    },
    "order": {
        "quote_num": "3",
        "struct_col": "order_data",
        "join_keys": ["channel", "order_index"],
        "use_group_idx": False,
        "npy_loader": "order",
        "channel_col": "channel",
        "index_col": "order_index",
    },
}


# ── NPY loading ──────────────────────────────────────────────────────


def load_npy(
    date_str: str, data_type: str, *, npy_base: str, npy_format: str
) -> pl.DataFrame:
    if npy_format == "delta":
        npy_dir = _resolve_delta_npy_dir(npy_base, date_str)
        if data_type == "snapshot":
            df = _load_single_npy(f"{npy_dir}/snapshot.npy")
        elif data_type in ("trans", "order"):
            df = _load_delta_channel_npy(npy_dir, data_type)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    elif npy_format == "classic":
        if data_type == "snapshot":
            df = _load_single_npy(f"{npy_base}/{date_str}/snapshot.npy")
        elif data_type == "trans":
            df = _load_classic_channel_dir(
                f"{npy_base}/{date_str}/trans_localtime_by_channel"
            )
        elif data_type == "order":
            df = _load_classic_channel_dir(
                f"{npy_base}/{date_str}/order_localtime_by_channel"
            )
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    else:
        raise ValueError(f"Unknown npy_format: {npy_format}")

    binary_cols = [n for n, d in df.schema.items() if d == pl.Binary]
    if binary_cols:
        df = df.with_columns([pl.col(c).cast(pl.String) for c in binary_cols])
    return df.with_columns(
        (pl.col("local_time") * 1000).cast(pl.Int64).alias("npy_lt"),
    ).drop("local_time")


def _load_single_npy(path: str) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPY file not found: {path}")
    return pl.from_numpy(np.load(str(p)))


def _load_classic_channel_dir(channel_dir: str) -> pl.DataFrame:
    p = Path(channel_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Channel dir not found: {channel_dir}")
    frames: list[pl.DataFrame] = []
    for f in sorted(p.iterdir()):
        if f.suffix != ".npy" or not f.stem.startswith("channel-"):
            continue
        try:
            ch = int(f.stem.split("-", 1)[1])
        except ValueError:
            continue
        data = np.load(str(f))
        if data.size == 0:
            continue
        ch_df = pl.from_numpy(data)
        if "channel" not in ch_df.columns:
            ch_df = ch_df.with_columns(pl.lit(ch).alias("channel"))
        frames.append(ch_df)
    if not frames:
        raise FileNotFoundError(f"No channel NPY files in {channel_dir}")
    return pl.concat(frames, how="vertical")


def _resolve_delta_npy_dir(npy_base: str, date_str: str) -> str:
    year = date_str[:4]
    d = Path(npy_base) / year / f"type1_{date_str}_94000000"
    if not d.is_dir():
        raise FileNotFoundError(f"Delta NPY dir not found: {d}")
    return str(d)


def _load_delta_channel_npy(npy_dir: str, data_type: str) -> pl.DataFrame:
    p = Path(npy_dir)
    prefix = "order_" if data_type == "order" else "trans_"
    idx_det_col = "order_index_det" if data_type == "order" else "trade_index_det"
    idx_col = "order_index" if data_type == "order" else "trade_index"

    frames: list[pl.DataFrame] = []
    for f in sorted(p.iterdir()):
        if f.suffix != ".npy" or not f.name.startswith(prefix):
            continue
        parts = f.stem.split("_")
        if len(parts) != 4:
            continue
        try:
            ch = int(parts[1])
            base_idx = int(parts[2])
            base_ts = int(parts[3])
        except ValueError:
            continue

        data = np.load(str(f))
        if data.size == 0:
            continue

        abs_idx = base_idx + np.cumsum(data[idx_det_col].astype(np.int64))
        abs_lt = base_ts + np.cumsum(data["local_time_det"].astype(np.int64))

        ch_df = pl.DataFrame(
            {
                "channel": np.full(len(data), ch, dtype=np.int16),
                idx_col: abs_idx,
                "local_time": abs_lt,
            }
        )
        frames.append(ch_df)

    if not frames:
        raise FileNotFoundError(f"No {prefix}*.npy files in {npy_dir}")
    return pl.concat(frames, how="vertical")


def get_output_path(date_str: str, quote_num: str, output_base: str) -> str:
    ym = f"{date_str[:4]}.{date_str[4:6]}"
    ymd = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
    return f"{output_base}/{ym}/{ymd}/default/{quote_num}/all.parquet"


# ── Check 1: local_time strictly increasing ──────────────────────────


def check_strictly_increasing(out_path: str) -> dict:
    lt = pl.scan_parquet(out_path).select("local_time").collect()["local_time"]
    total = lt.len()
    null_count = lt.null_count()
    diffs = lt.diff().drop_nulls()
    violations = int((diffs <= 0).sum())
    min_diff = diffs.min()
    return {
        "total_rows": total,
        "null_lt": null_count,
        "violations": violations,
        "min_diff": min_diff,
        "pass": violations == 0 and null_count == 0,
    }


# ── Check 2: NPY local_time matching ────────────────────────────────


def check_npy_match(
    date_str: str,
    data_type: str,
    out_path: str,
    *,
    npy_base: str,
    npy_format: str,
) -> dict:
    spec = QUOTE_SPECS[data_type]
    join_keys = spec["join_keys"]

    npy_df = load_npy(
        date_str, spec["npy_loader"], npy_base=npy_base, npy_format=npy_format
    )
    out_df = pl.scan_parquet(out_path).select(join_keys + ["local_time"]).collect()

    for k in join_keys:
        if out_df[k].dtype != npy_df[k].dtype:
            npy_df = npy_df.with_columns(pl.col(k).cast(out_df[k].dtype))

    if data_type == "tick":
        out_tickers = out_df[join_keys[0]].unique()
        npy_df = npy_df.filter(pl.col(join_keys[0]).is_in(out_tickers))

    if spec["use_group_idx"]:
        # Match by (join_keys, lt_us, _gidx_within_lt_us).
        # This handles both right-join and left-join tick data:
        # fallback rows (different lt_us) don't shift _gidx of NPY-matched rows.
        out_df = (
            out_df 
            .with_columns((pl.col("local_time").cast(pl.Int64) // 1000).alias("_lt_us"))
            .sort(join_keys + ["_lt_us", "local_time"])
            .with_columns(
                pl.int_range(pl.len()).over(join_keys + ["_lt_us"]).alias("_gidx")
            )
        )
        npy_df = (
            npy_df
            .with_columns((pl.col("npy_lt") // 1000).alias("_lt_us"))
            .sort(join_keys + ["_lt_us", "npy_lt"])
            .with_columns(
                pl.int_range(pl.len()).over(join_keys + ["_lt_us"]).alias("_gidx")
            )
        )
        actual_keys = join_keys + ["_lt_us", "_gidx"]
    else:
        actual_keys = join_keys

    merged = npy_df.select(actual_keys + ["npy_lt"]).join(
        out_df.select(actual_keys + ["local_time"]),
        on=actual_keys,
        how="left",
    )

    matched = merged.filter(pl.col("local_time").is_not_null())
    unmatched = merged.height - matched.height

    if matched.height == 0:
        return {
            "npy_rows": npy_df.height,
            "matched": 0,
            "us_mismatch": 0,
            "pass": False,
        }

    us_mismatch = int(
        matched.filter(
            (pl.col("local_time").cast(pl.Int64) // 1000)
            != (pl.col("npy_lt") // 1000)
        ).height
    )
    npy_total = npy_df.height
    matched_count = matched.height

    del out_df, npy_df, merged, matched
    gc.collect()

    return {
        "npy_rows": npy_total,
        "matched": matched_count,
        "unmatched": unmatched,
        "us_mismatch": us_mismatch,
        "pass": us_mismatch == 0,
    }


# ── Check 3: Row count completeness ─────────────────────────────────


def check_row_count(data_type: str, out_path: str, sorted_path: str) -> dict:
    out_rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    if not Path(sorted_path).exists():
        return {
            "sorted_rows": -1,
            "output_rows": out_rows,
            "diff": 0,
            "pass": True,
            "skipped": True,
        }
    sorted_rows = (
        pl.scan_parquet(sorted_path)
        .filter(pl.col("data_type") == data_type)
        .select(pl.len())
        .collect()
        .item()
    )
    diff = out_rows - sorted_rows
    ok = diff == 0 if data_type != "tick" else diff <= 0
    return {
        "sorted_rows": sorted_rows,
        "output_rows": out_rows,
        "diff": diff,
        "pass": ok,
        "skipped": False,
    }


# ── Check 4: Business column integrity ──────────────────────────────


def check_column_integrity(
    data_type: str, date_str: str, out_path: str, sorted_path: str
) -> dict:
    spec = QUOTE_SPECS[data_type]
    struct_col = spec["struct_col"]
    cutoff_ns = _cutoff_ns(date_str)

    sorted_schema = pl.scan_parquet(sorted_path).collect_schema()
    struct_fields = sorted_schema[struct_col].fields
    struct_field_names = [f.name for f in struct_fields]

    out_schema = pl.scan_parquet(out_path).collect_schema()
    skip_cols = {"local_time", "serial"}
    out_cols = [c for c in out_schema.names() if c not in skip_cols]
    common = [c for c in out_cols if c in struct_field_names]

    src_data = (
        pl.scan_parquet(sorted_path)
        .filter((pl.col("sort_time") > cutoff_ns) & (pl.col("data_type") == data_type))
        .head(1000)
        .select([pl.col(struct_col).struct.field(c).alias(c) for c in common])
        .collect()
    )

    front_count = (
        pl.scan_parquet(sorted_path)
        .filter((pl.col("sort_time") <= cutoff_ns) & (pl.col("data_type") == data_type))
        .select(pl.len())
        .collect()
        .item()
    )

    out_data = (
        pl.scan_parquet(out_path).select(common).slice(front_count, 1000).collect()
    )

    mismatched_cols: list[str] = []
    if src_data.shape == out_data.shape:
        for col in common:
            eq = src_data[col] == out_data[col]
            both_null = src_data[col].is_null() & out_data[col].is_null()
            if not (eq | both_null).all():
                mismatched_cols.append(col)

    return {
        "columns_checked": len(common),
        "mismatched": mismatched_cols,
        "pass": len(mismatched_cols) == 0,
    }


def _cutoff_ns(date_str: str) -> int:
    """9:40:00 CST as Unix nanoseconds for the given date."""
    dt = datetime(
        int(date_str[:4]),
        int(date_str[4:6]),
        int(date_str[6:8]),
        9,
        40,
        0,
        tzinfo=CST,
    )
    return int(dt.timestamp() * 1_000_000_000)


# ── Check 5: Channel-internal index non-decreasing ──────────────────


def _count_idx_violations(df: pl.DataFrame, ch_col: str, idx_col: str) -> int:
    return (
        df.with_columns(pl.col(idx_col).diff().over(ch_col).alias("_d"))
        .filter(pl.col("_d").is_not_null())
        .filter(pl.col("_d") < 0)
        .height
    )


def check_channel_index_ordering(data_type: str, date_str: str, out_path: str) -> dict:
    """Verify index_col non-decreasing within each channel, split by front/back."""
    spec = QUOTE_SPECS[data_type]
    ch_col = spec["channel_col"]
    idx_col = spec["index_col"]
    if ch_col is None or idx_col is None:
        return {"applicable": False, "pass": True}

    cutoff_us = _cutoff_ns(date_str) // 1000
    df = pl.scan_parquet(out_path).select([ch_col, idx_col, "exchange_time"]).collect()

    front = df.filter(pl.col("exchange_time") <= cutoff_us)
    back = df.filter(pl.col("exchange_time") > cutoff_us)

    front_viol = _count_idx_violations(front, ch_col, idx_col)
    back_viol = _count_idx_violations(back, ch_col, idx_col)
    n_channels = df[ch_col].n_unique()

    del df, front, back
    gc.collect()

    return {
        "applicable": True,
        "channels": n_channels,
        "index_col": idx_col,
        "front_violations": front_viol,
        "back_violations": back_viol,
        "pass": back_viol == 0,
    }


# ── Check 6: SH merged BizIndex interleaving ────────────────────────


def check_sh_biz_index_interleaving(date_str: str, out_paths: dict[str, str]) -> dict:
    """For SH channels (< 1000), verify that when order and trans rows are
    merged by local_time, their biz_index remains non-decreasing within each channel.

    Reports front (NPY section) and back (sort_time section) separately.
    """
    trans_path = out_paths.get("trans")
    order_path = out_paths.get("order")
    if trans_path is None or order_path is None:
        return {"applicable": False, "pass": True}
    if not Path(trans_path).exists() or not Path(order_path).exists():
        return {"applicable": False, "pass": True}

    cutoff_us = _cutoff_ns(date_str) // 1000

    trans_df = (
        pl.scan_parquet(trans_path)
        .select(["local_time", "channel", "biz_index", "exchange_time"])
        .filter(pl.col("channel") < 1000)
        .with_columns(pl.col("biz_index").cast(pl.Int64))
        .collect()
    )
    order_df = (
        pl.scan_parquet(order_path)
        .select(["local_time", "channel", "biz_index", "exchange_time"])
        .filter(pl.col("channel") < 1000)
        .with_columns(pl.col("biz_index").cast(pl.Int64))
        .collect()
    )

    if trans_df.height == 0 and order_df.height == 0:
        return {"applicable": False, "pass": True}

    merged = pl.concat([trans_df, order_df]).sort("local_time")

    del trans_df, order_df
    gc.collect()

    front = merged.filter(pl.col("exchange_time") <= cutoff_us)
    back = merged.filter(pl.col("exchange_time") > cutoff_us)

    front_viol = _count_idx_violations(front, "channel", "biz_index")
    back_viol = _count_idx_violations(back, "channel", "biz_index")
    n_channels = merged["channel"].n_unique()
    total_rows = merged.height

    del merged, front, back
    gc.collect()

    return {
        "applicable": True,
        "channels": n_channels,
        "total_rows": total_rows,
        "front_violations": front_viol,
        "back_violations": back_viol,
        "pass": back_viol == 0,
    }


# ── Check 6b: SZ merged ApplSeq interleaving ─────────────────────────


def check_sz_applseq_interleaving(date_str: str, out_paths: dict[str, str]) -> dict:
    """For SZ channels (>= 1000), verify that when order and trans rows are
    merged by local_time, their applseq (order_index / trade_index) remains
    non-decreasing within each channel.
    """
    trans_path = out_paths.get("trans")
    order_path = out_paths.get("order")
    if trans_path is None or order_path is None:
        return {"applicable": False, "pass": True}
    if not Path(trans_path).exists() or not Path(order_path).exists():
        return {"applicable": False, "pass": True}

    cutoff_us = _cutoff_ns(date_str) // 1000

    trans_df = (
        pl.scan_parquet(trans_path)
        .select(["local_time", "channel", "trade_index", "exchange_time"])
        .filter(pl.col("channel") >= 1000)
        .with_columns(pl.col("trade_index").cast(pl.Int64).alias("applseq"))
        .drop("trade_index")
        .collect()
    )
    order_df = (
        pl.scan_parquet(order_path)
        .select(["local_time", "channel", "order_index", "exchange_time"])
        .filter(pl.col("channel") >= 1000)
        .with_columns(pl.col("order_index").cast(pl.Int64).alias("applseq"))
        .drop("order_index")
        .collect()
    )

    if trans_df.height == 0 and order_df.height == 0:
        return {"applicable": False, "pass": True}

    merged = pl.concat([trans_df, order_df]).sort("local_time")

    del trans_df, order_df
    gc.collect()

    front = merged.filter(pl.col("exchange_time") <= cutoff_us)
    back = merged.filter(pl.col("exchange_time") > cutoff_us)

    front_viol = _count_idx_violations(front, "channel", "applseq")
    back_viol = _count_idx_violations(back, "channel", "applseq")
    n_channels = merged["channel"].n_unique()
    total_rows = merged.height

    del merged, front, back
    gc.collect()

    return {
        "applicable": True,
        "channels": n_channels,
        "total_rows": total_rows,
        "front_violations": front_viol,
        "back_violations": back_viol,
        "pass": back_viol == 0,
    }


# ── Check 7: Cross-channel ordering (stg2 sub-sequence analysis) ─────

SH_CH_THRESHOLD = 1000


def _analyze_cross_subseq(df: pl.DataFrame, group_expr: pl.Expr) -> dict:
    """Consecutive-row sub-sequence analysis within time groups.

    same-subseq:  (tp, sc) non-decreasing — allow ties (same stock in same channel)
    cross-subseq: (tp, sc) strictly increasing — each stock belongs to exactly
                  one channel, so same (tp, sc) across sub-sequences is impossible.
    """
    zeros = {"same_tp": 0, "cross_tp": 0, "same_sc": 0, "cross_sc": 0}
    if df.height <= 1:
        return zeros

    same_group = group_expr == group_expr.shift(1)
    same_ss = pl.col("_ss") == pl.col("_ss").shift(1)
    same_tp = pl.col("_tp") == pl.col("_tp").shift(1)

    tp_viol = same_group & (pl.col("_tp") < pl.col("_tp").shift(1))

    # same-subseq: sc decreased (non-decreasing check)
    sc_decreased = same_group & same_tp & (pl.col("_sc") < pl.col("_sc").shift(1))
    # cross-subseq: sc decreased OR equal (strictly increasing check)
    sc_not_increased = same_group & same_tp & (pl.col("_sc") <= pl.col("_sc").shift(1))

    return df.select(
        (tp_viol & same_ss).fill_null(False).sum().alias("same_tp"),
        (tp_viol & ~same_ss).fill_null(False).sum().alias("cross_tp"),
        (sc_decreased & same_ss).fill_null(False).sum().alias("same_sc"),
        (sc_not_increased & ~same_ss).fill_null(False).sum().alias("cross_sc"),
    ).row(0, named=True)


def _read_stg2_section(
    out_paths: dict[str, str],
    filt_expr: pl.Expr,
    separate_types: bool = False,
) -> pl.DataFrame | None:
    """Read tick+order+trans from stg2, filter by *filt_expr*, merge by local_time.

    separate_types: if True, order and trans from the same channel get distinct
        sub-sequence IDs (ch*10+1 for order, ch*10+2 for trans).
        If False, same channel = same sub-sequence (back-half behaviour).
    """
    parts: list[pl.DataFrame] = []

    for dt, tp_val in [("tick", 0), ("order", 1), ("trans", 2)]:
        p = out_paths.get(dt)
        if p is None or not Path(p).exists():
            continue
        sc_col = "ticker" if dt == "tick" else "symbol"
        ch_expr = (
            pl.lit(-1, dtype=pl.Int32)
            if dt == "tick"
            else pl.col("channel").cast(pl.Int32)
        )
        read_cols = ["local_time", "exchange_time"]
        if dt != "tick":
            read_cols.append("channel")
        read_cols.append(sc_col)

        try:
            part = (
                pl.scan_parquet(p)
                .select(read_cols)
                .filter(filt_expr)
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
                .filter(filt_expr)
                .with_columns(
                    pl.lit(tp_val, dtype=pl.Int8).alias("_tp"),
                    ch_expr.alias("_ch"),
                    pl.col(sc_col).cast(pl.Int64).alias("_sc"),
                )
                .select(["local_time", "exchange_time", "_tp", "_ch", "_sc"])
            )
        parts.append(part)
        del part

    if len(parts) < 2:
        return None

    merged = pl.concat(parts).sort("local_time")
    del parts
    gc.collect()

    is_tick = pl.col("_ch") == -1

    if separate_types:
        subseq = (
            pl.when(is_tick)
            .then(pl.lit(-1, dtype=pl.Int64))
            .when(pl.col("_tp") == 1)
            .then(pl.col("_ch").cast(pl.Int64) * 10 + 1)
            .otherwise(pl.col("_ch").cast(pl.Int64) * 10 + 2)
        )
    else:
        subseq = (
            pl.when(is_tick)
            .then(pl.lit(-1, dtype=pl.Int64))
            .otherwise(pl.col("_ch").cast(pl.Int64))
        )
    return merged.with_columns(subseq.alias("_ss"))


def check_cross_channel_ordering(date_str: str, out_paths: dict[str, str]) -> dict:
    """Cross-channel ordering on stg2 per-type outputs (tick+order+trans merged).

    Merges all types by local_time, assigns sub-sequence IDs, and checks
    within-time-group cross-subseq ordering violations.

    Front (9:15~9:40): group by local_time // 1000 (µs, pre-dedup precision)
    Back  (after 9:40): group by exchange_time

    Stage 1 already deduplicates local_time globally (sort_time ties get +1 ns),
    so the per-type stg2 outputs interleave perfectly by local_time with no ties.

    Sub-sequence IDs:
      Front (separate_types=True):
        tick:     -1
        order:    channel * 10 + 1
        trans:    channel * 10 + 2
      Back (separate_types=False):
        tick:     -1
        SH/SZ:   channel  (order+trans merged)
    """
    cutoff_us = _cutoff_ns(date_str) // 1000
    total_rows = 0

    front_lt_viol = 0
    back_lt_viol = 0

    # ── Front (order/trans per channel are separate sub-sequences) ──
    logger.debug("  check7: reading front …")
    front = _read_stg2_section(
        out_paths, pl.col("exchange_time") <= cutoff_us, separate_types=True
    )
    if front is not None:
        total_rows += front.height
        front_lt_viol = int((front["local_time"].diff().drop_nulls() <= 0).sum())
        fr = _analyze_cross_subseq(front, pl.col("local_time") // 1000)
        del front
        gc.collect()
    else:
        fr = {"same_tp": 0, "cross_tp": 0, "same_sc": 0, "cross_sc": 0}

    # ── Back (order+trans per channel share one sub-sequence) ──
    logger.debug("  check7: reading back …")
    back = _read_stg2_section(
        out_paths, pl.col("exchange_time") > cutoff_us, separate_types=False
    )
    if back is not None:
        total_rows += back.height
        back_lt_viol = int((back["local_time"].diff().drop_nulls() <= 0).sum())
        bk = _analyze_cross_subseq(back, pl.col("exchange_time"))
        del back
        gc.collect()
    else:
        bk = {"same_tp": 0, "cross_tp": 0, "same_sc": 0, "cross_sc": 0}

    # Front sorts by (local_time, stock_code) — tp not in sort key, so only check sc.
    # Back sorts by (exchange_time, type_priority, stock_code) — check both tp and sc.
    front_cross = fr["cross_sc"]
    back_cross = bk["cross_tp"] + bk["cross_sc"]
    lt_pass = front_lt_viol == 0 and back_lt_viol == 0

    return {
        "applicable": True,
        "total_rows": total_rows,
        "front_lt_viol": front_lt_viol,
        "back_lt_viol": back_lt_viol,
        "lt_pass": lt_pass,
        "front": fr,
        "back": bk,
        "front_cross": front_cross,
        "back_cross": back_cross,
        "front_same": fr["same_sc"],
        "back_same": bk["same_tp"] + bk["same_sc"],
        "pass": front_cross == 0 and back_cross == 0 and lt_pass,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive verification for sorted_npy_merge output."
    )
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--output-base", default=OUTPUT_BASE_DEFAULT)
    parser.add_argument("--sorted-dir", default=SORTED_DIR_DEFAULT)
    parser.add_argument(
        "--types",
        nargs="*",
        default=["tick", "trans", "order"],
        choices=["tick", "trans", "order"],
    )
    parser.add_argument(
        "--npy-format",
        default="classic",
        choices=["classic", "delta"],
        help="NPY layout: classic=channel-dir, delta=flat delta-encoded (default: classic)",
    )
    parser.add_argument(
        "--npy-base",
        default=None,
        help="NPY base directory (auto-set per format if omitted)",
    )
    args = parser.parse_args()
    date_str = args.date.strip()
    sorted_path = f"{args.sorted_dir}/sorted_{date_str}.parquet"
    npy_base = args.npy_base or (
        NPY_BASE_DELTA if args.npy_format == "delta" else NPY_BASE_CLASSIC
    )
    npy_format = args.npy_format

    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}")
    logger.info("=" * 65)
    logger.info("Verification: date={} output={}", date_str, args.output_base)
    logger.info("=" * 65)

    all_pass = True
    summary: list[tuple[str, str, str, bool]] = []
    out_paths: dict[str, str] = {}

    for dt in args.types:
        spec = QUOTE_SPECS[dt]
        out_path = get_output_path(date_str, spec["quote_num"], args.output_base)
        logger.info("")
        logger.info("── {} (quote={}) ──", dt, spec["quote_num"])

        out_paths[dt] = out_path

        if not Path(out_path).exists():
            logger.error("  Output not found: {}", out_path)
            summary.append((dt, "ALL", "MISSING", False))
            all_pass = False
            continue

        # ── 1. local_time strictly increasing ────────────────────
        r1 = check_strictly_increasing(out_path)
        tag = "PASS" if r1["pass"] else "FAIL"
        logger.info(
            "  [{}] 1. strictly increasing: rows={:,} null={} violations={}",
            tag,
            r1["total_rows"],
            r1["null_lt"],
            r1["violations"],
        )
        summary.append((dt, "1.strictly_incr", tag, r1["pass"]))
        if not r1["pass"]:
            all_pass = False

        # ── 2. NPY local_time matching ───────────────────────────
        r2 = check_npy_match(
            date_str, dt, out_path, npy_base=npy_base, npy_format=npy_format
        )
        tag = "PASS" if r2["pass"] else "FAIL"
        logger.info(
            "  [{}] 2. NPY match (μs): matched={:,} unmatched={} us_mismatch={}",
            tag,
            r2["matched"],
            r2.get("unmatched", 0),
            r2["us_mismatch"],
        )
        summary.append((dt, "2.npy_match", tag, r2["pass"]))
        if not r2["pass"]:
            all_pass = False

        # ── 3. Row count completeness ────────────────────────────
        r3 = check_row_count(dt, out_path, sorted_path)
        if r3.get("skipped"):
            tag = "SKIP"
            logger.info(
                "  [SKIP] 3. row count: no sorted intermediate, output={:,}",
                r3["output_rows"],
            )
        else:
            tag = "PASS" if r3["pass"] else "FAIL"
            logger.info(
                "  [{}] 3. row count: sorted={:,} output={:,} diff={:+,}",
                tag,
                r3["sorted_rows"],
                r3["output_rows"],
                r3["diff"],
            )
        summary.append((dt, "3.row_count", tag, r3["pass"]))
        if not r3["pass"]:
            all_pass = False

        # ── 4. Business column integrity ─────────────────────────
        if dt == "tick":
            tag = "SKIP"
            logger.info(
                "  [SKIP] 4. columns: tick uses right-join, positional compare invalid"
            )
            summary.append((dt, "4.col_integrity", tag, True))
        elif not Path(sorted_path).exists():
            tag = "SKIP"
            logger.info("  [SKIP] 4. columns: no sorted intermediate")
            summary.append((dt, "4.col_integrity", tag, True))
        else:
            r4 = check_column_integrity(dt, date_str, out_path, sorted_path)
            tag = "PASS" if r4["pass"] else "FAIL"
            logger.info(
                "  [{}] 4. columns: {} checked, mismatched={}",
                tag,
                r4["columns_checked"],
                r4["mismatched"] or "none",
            )
            summary.append((dt, "4.col_integrity", tag, r4["pass"]))
            if not r4["pass"]:
                all_pass = False

        # ── 5. Channel-internal index non-decreasing ──────────────
        r5 = check_channel_index_ordering(dt, date_str, out_path)
        if r5["applicable"]:
            tag = "PASS" if r5["pass"] else "FAIL"
            warn = "" if r5["front_violations"] == 0 else " (WARN)"
            logger.info(
                "  [{}] 5. channel {} non-decreasing: back={} front={:,}{}",
                tag,
                r5["index_col"],
                r5["back_violations"],
                r5["front_violations"],
                warn,
            )
        else:
            tag = "N/A"
            logger.info("  [N/A] 5. channel index order: not applicable for {}", dt)
        summary.append((dt, "5.channel_idx_order", tag, r5["pass"]))
        if not r5["pass"]:
            all_pass = False

    # ── 6. SH BizIndex interleaving (cross-type) ────────────────
    r6 = check_sh_biz_index_interleaving(date_str, out_paths)
    if r6["applicable"]:
        tag = "PASS" if r6["pass"] else "FAIL"
        warn = "" if r6["front_violations"] == 0 else " (WARN)"
        logger.info("")
        logger.info("── SH BizIndex interleaving (order + trans) ──")
        logger.info(
            "  [{}] 6. SH biz_index non-decreasing: back={:,} front={:,}{} (rows={:,})",
            tag,
            r6["back_violations"],
            r6["front_violations"],
            warn,
            r6["total_rows"],
        )
    else:
        tag = "N/A"
    summary.append(("ALL", "6a.sh_biz_interleave", tag, r6.get("pass", True)))
    if not r6.get("pass", True):
        all_pass = False

    # ── 6b. SZ ApplSeq interleaving (cross-type) ────────────────
    r6b = check_sz_applseq_interleaving(date_str, out_paths)
    if r6b["applicable"]:
        tag = "PASS" if r6b["pass"] else "FAIL"
        warn = "" if r6b["front_violations"] == 0 else " (WARN)"
        logger.info("")
        logger.info("── SZ ApplSeq interleaving (order + trans) ──")
        logger.info(
            "  [{}] 6b. SZ applseq non-decreasing: back={:,} front={:,}{} (rows={:,})",
            tag,
            r6b["back_violations"],
            r6b["front_violations"],
            warn,
            r6b["total_rows"],
        )
    else:
        tag = "N/A"
    summary.append(("ALL", "6b.sz_applseq_interleave", tag, r6b.get("pass", True)))
    if not r6b.get("pass", True):
        all_pass = False

    # ── 7. Cross-channel ordering (stg2 sub-seq analysis) ───────
    r7 = check_cross_channel_ordering(date_str, out_paths)
    if r7["applicable"]:
        tag = "PASS" if r7["pass"] else "FAIL"
        fr, bk = r7["front"], r7["back"]
        lt_tag = "PASS" if r7["lt_pass"] else "FAIL"
        logger.info("")
        logger.info("── Merged 3-type analysis (stg2) ──")
        logger.info(
            "  [{}] 7a. global local_time strictly increasing: "
            "front={:,} back={:,} violations",
            lt_tag,
            r7["front_lt_viol"],
            r7["back_lt_viol"],
        )
        logger.info(
            "  [{}] 7b. cross-channel sub-seq: rows={:,}",
            tag,
            r7["total_rows"],
        )
        logger.info(
            "    front  same-seq(expected): tp={:,} sc={:,}",
            fr["same_tp"],
            fr["same_sc"],
        )
        logger.info(
            "    front  cross-seq(BUG!):    tp={:,} sc={:,}",
            fr["cross_tp"],
            fr["cross_sc"],
        )
        logger.info(
            "    back   same-seq(expected): tp={:,} sc={:,}",
            bk["same_tp"],
            bk["same_sc"],
        )
        logger.info(
            "    back   cross-seq(BUG!):    tp={:,} sc={:,}",
            bk["cross_tp"],
            bk["cross_sc"],
        )
        if r7["front_cross"] > 0:
            logger.error(
                "    FRONT cross-seq violations: {:,} — sorting bug!",
                r7["front_cross"],
            )
        if r7["back_cross"] > 0:
            logger.error(
                "    BACK cross-seq violations: {:,} — sorting bug!",
                r7["back_cross"],
            )
    else:
        tag = "N/A"
    summary.append(("ALL", "7.cross_ch_order", tag, r7.get("pass", True)))
    if not r7.get("pass", True):
        all_pass = False

    # ── Summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("SUMMARY")
    logger.info("=" * 65)
    for dt, check, tag, _ in summary:
        logger.info("  {:>5s} | {:<18s} | {}", dt, check, tag)
    logger.info("-" * 65)

    if all_pass:
        logger.info("  RESULT: ALL PASSED ✓")
    else:
        logger.error("  RESULT: SOME CHECKS FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
