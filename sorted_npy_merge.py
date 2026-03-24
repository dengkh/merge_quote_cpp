#!/usr/bin/env python3
"""
Merge sorted parquet with NPY localtime data.

Input:  data/stg1/sorted_parquet/sorted_{date}.parquet
Output: data/stg2/{date}/default/{1,2,3}/all.parquet  (one per quote type)

Pipeline:
  1. Split at 9:40 cutoff (sort_time)
  2. Front half: per-type join with NPY (tick: right, trans/order: left)
     -> build per-channel sequences -> K-way merge by (local_time, stock_code)
     Channel-internal order is preserved: each channel forms an independent
     sequence and the heap only pops from sequence heads.
  3. Compute offset = last_front_lt + 1 - first_back_lt
  4. Per type: concat(front_type, back_type + offset) -> unnest struct -> write

Usage:
  cd merge_localtime
  python scripts/sorted_npy_merge.py 20250221
  python scripts/sorted_npy_merge.py 20250221 --sorted-dir /path/to/sorted
  python scripts/sorted_npy_merge.py 20250221 --post-process        # merge + fix serial + split
  python scripts/sorted_npy_merge.py 20250221 --post-process-only   # only fix serial + split
"""

import argparse
import gc
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltmerge.config import QuoteType
from ltmerge.kway import kway_merge_numba, two_way_merge_by_biz
from ltmerge.stage_paths import get_stage1_sorted_dir, get_stage2_output_root
from ltmerge.validation.schema import validate_schema

# ── Constants ────────────────────────────────────────────────────────

NPY_BASE = "/mnt/data/public/quant001/local_times/stocks/91500000-94000000"
TRADE_CHANNEL_DIR = "trans_localtime_by_channel"
ORDER_CHANNEL_DIR = "order_localtime_by_channel"
STAGE2_OUTPUT_BASE = get_stage2_output_root()
CST = timezone(timedelta(hours=8))
SH_CHANNEL_THRESHOLD = 1000

DATA_TYPE_TO_QUOTE = {
    "tick": QuoteType.SNAPSHOT,
    "trans": QuoteType.TRADE,
    "order": QuoteType.ORDER,
}

TYPE_CONFIG = {
    "tick": {
        "struct_col": "tick_data",
        "join_keys": ["ticker", "exch_time"],
        "need_group_idx": True,
        "join_how": "right",
        "quote_num": "1",
        "other_structs": ["order_data", "trans_data"],
    },
    "trans": {
        "struct_col": "trans_data",
        "join_keys": ["channel", "trade_index"],
        "need_group_idx": False,
        "join_how": "left",
        "quote_num": "2",
        "other_structs": ["tick_data", "order_data"],
    },
    "order": {
        "struct_col": "order_data",
        "join_keys": ["channel", "order_index"],
        "need_group_idx": False,
        "join_how": "left",
        "quote_num": "3",
        "other_structs": ["tick_data", "trans_data"],
    },
}

META_COLS = ["sort_time", "stock_code", "type_priority", "data_type"]


# ── 1. Cutoff ────────────────────────────────────────────────────────


def compute_cutoff_ns(date_str: str) -> int:
    """9:40:00 CST on the given date, as Unix nanoseconds."""
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


# ── 2. NPY loading ──────────────────────────────────────────────────


def load_npy(date_str: str, data_type: str) -> pl.DataFrame:
    """Load NPY localtime, return DataFrame with join keys + _npy_local_time (ns)."""
    if data_type == "tick":
        df = _load_single_npy(f"{NPY_BASE}/{date_str}/snapshot.npy")
    elif data_type == "trans":
        df = _load_channel_npy(f"{NPY_BASE}/{date_str}/{TRADE_CHANNEL_DIR}")
    elif data_type == "order":
        df = _load_channel_npy(f"{NPY_BASE}/{date_str}/{ORDER_CHANNEL_DIR}")
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    df = _cast_binary_cols(df)

    _validate_npy_monotonic(df, data_type)

    df = df.with_columns(
        (pl.col("local_time") * 1000).cast(pl.Int64).alias("_npy_local_time"),
    ).drop("local_time")

    logger.info("NPY {}: {:,} rows loaded", data_type, df.height)
    return df


def _load_single_npy(path: str) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPY file not found: {path}")
    data = np.load(str(p))
    logger.debug("  npy shape={} dtype={}", data.shape, data.dtype)
    return pl.from_numpy(data)


def _load_channel_npy(channel_dir: str) -> pl.DataFrame:
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


def _cast_binary_cols(df: pl.DataFrame) -> pl.DataFrame:
    binary_cols = [n for n, d in df.schema.items() if d == pl.Binary]
    if binary_cols:
        df = df.with_columns([pl.col(c).cast(pl.String) for c in binary_cols])
        logger.debug("  Binary -> String: {}", binary_cols)
    return df


def _validate_npy_monotonic(df: pl.DataFrame, data_type: str) -> None:
    """Assert that channel-internal index is monotonically non-decreasing in NPY data.

    trans: trade_index non-decreasing within each channel
    order: order_index non-decreasing within each channel
    tick:  exch_time non-decreasing within each ticker
    """
    if data_type == "trans":
        group_col, index_col = "channel", "trade_index"
    elif data_type == "order":
        group_col, index_col = "channel", "order_index"
    elif data_type == "tick":
        group_col, index_col = "ticker", "exch_time"
    else:
        return

    violations = df.with_columns(
        pl.col(index_col).diff().over(group_col).alias("_idx_diff")
    ).filter(pl.col("_idx_diff") < 0)

    if violations.height > 0:
        bad_groups = violations[group_col].unique().sort().to_list()
        raise ValueError(
            f"NPY {data_type}: {index_col} is NOT monotonically non-decreasing "
            f"within {group_col}. {violations.height:,} violations in "
            f"{group_col}s: {bad_groups[:10]}{'...' if len(bad_groups) > 10 else ''}"
        )
    logger.info(
        "  NPY {}: {} monotonic within {} ✓ ({:,} rows)",
        data_type,
        index_col,
        group_col,
        df.height,
    )


# ── 3. Front half join ───────────────────────────────────────────────


def join_front_type(
    sorted_path: str,
    cutoff_ns: int,
    data_type: str,
    npy_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join front-half rows of one type with NPY localtime.

    tick  → right join (NPY primary): drops frozen 926~930 snapshots
    trans → left join  (parquet primary): keeps all parquet rows
    order → left join  (parquet primary): keeps all parquet rows
    """
    cfg = TYPE_CONFIG[data_type]
    struct_col = cfg["struct_col"]
    join_keys = cfg["join_keys"]
    need_group_idx = cfg["need_group_idx"]
    join_how = cfg["join_how"]

    t0 = _time.time()

    # ── read & filter ────────────────────────────────────────────────
    front = (
        pl.scan_parquet(sorted_path)
        .filter((pl.col("sort_time") <= cutoff_ns) & (pl.col("data_type") == data_type))
        .collect()
    )
    logger.info(
        "  front {}: {:,} rows ({:.1f}s)", data_type, front.height, _time.time() - t0
    )

    # ── extract join keys from struct & align types with npy ────────
    key_exprs = [pl.col(struct_col).struct.field(k).alias(k) for k in join_keys]
    front = front.with_columns(key_exprs)

    npy_select = npy_df.select(join_keys + ["_npy_local_time"])

    for k in join_keys:
        if front[k].dtype != npy_select[k].dtype:
            target = npy_select[k].dtype
            logger.debug("  cast front.{}: {} -> {}", k, front[k].dtype, target)
            front = front.with_columns(pl.col(k).cast(target))

    # ── tick only: filter npy to ticker intersection ─────────────────
    if data_type == "tick":
        ticker_key = join_keys[0]
        front_tickers = front[ticker_key].unique()
        npy_tickers = npy_select[ticker_key].unique()
        excluded = npy_tickers.filter(~npy_tickers.is_in(front_tickers.implode()))
        if excluded.len() > 0:
            logger.warning(
                "  tick: {:,} tickers in NPY but absent from parquet ({}), filtering out",
                excluded.len(),
                excluded.sort().to_list(),
            )
            npy_select = npy_select.filter(
                pl.col(ticker_key).is_in(front_tickers.implode())
            )
            logger.info(
                "  tick: NPY after ticker filter: {:,} rows",
                npy_select.height,
            )

    # ── group_idx for snapshot duplicates ────────────────────────────
    if need_group_idx:
        front = front.with_columns(
            pl.int_range(pl.len()).over(join_keys).alias("_group_idx")
        )
        npy_select = npy_select.with_columns(
            pl.int_range(pl.len()).over(join_keys).alias("_group_idx")
        )
        actual_keys = join_keys + ["_group_idx"]
    else:
        actual_keys = join_keys

    # ── join ─────────────────────────────────────────────────────────
    npy_for_join = npy_select.select(actual_keys + ["_npy_local_time"])
    join_kwargs: dict = dict(on=actual_keys, how=join_how, coalesce=True)
    if join_how == "left":
        join_kwargs["maintain_order"] = "left"
    merged = front.join(npy_for_join, **join_kwargs)
    logger.info("  {} join ({}): {:,} rows", data_type, join_how, merged.height)

    # ── post-join validation ─────────────────────────────────────────
    if join_how == "right":
        # tick right join: after ticker intersection, all npy must match
        unmatched = merged.filter(pl.col(struct_col).is_null()).height
        if unmatched > 0:
            raise ValueError(
                f"tick: {unmatched:,} NPY rows unmatched after ticker "
                f"intersection filter — join logic error"
            )
        dropped = front.height - merged.height
        if dropped > 0:
            logger.info(
                "  tick: {:,} frozen parquet rows dropped (926~930 etc.)",
                dropped,
            )
    # left join: all parquet rows kept; unmatched just keep original local_time

    # ── replace local_time: npy value preferred, fallback to original ─
    merged = merged.with_columns(
        pl.coalesce(["_npy_local_time", "local_time"]).alias("local_time")
    )

    # ── keep _channel / _ch_seq for K-way merge, drop the rest ──────
    drop_cols = ["_npy_local_time"]
    if need_group_idx:
        drop_cols.append("_group_idx")

    if data_type in ("trans", "order"):
        index_key = join_keys[1]  # trade_index or order_index
        merged = merged.rename({"channel": "_channel", index_key: "_ch_seq"})
    else:
        drop_cols.extend(join_keys)  # drop ticker, exch_time
        merged = merged.with_columns(
            pl.lit(-1).cast(pl.Int32).alias("_channel"),
            pl.lit(0).cast(pl.Int64).alias("_ch_seq"),
        )
    merged = merged.drop(drop_cols)

    logger.info(
        "  joined {}: {:,} rows ({:.1f}s)",
        data_type,
        merged.height,
        _time.time() - t0,
    )
    return merged


# ── 3b. K-way merge for front half ───────────────────────────────────


def kway_merge_front(front: pl.DataFrame) -> pl.DataFrame:
    """K-way merge the concatenated front DataFrame by (local_time, type_priority, stock_code).

    Builds per-channel sequences (SH: merged order+trans by biz_index,
    SZ: separate order/trans per channel, tick: single sequence), then
    runs the Numba heap merge.  Channel-internal order is never violated
    because each sequence is consumed strictly head-to-tail.
    """
    t0 = _time.time()

    channels = front["_channel"].to_numpy()
    ch_seqs = front["_ch_seq"].to_numpy()
    local_times = front["local_time"].to_numpy().astype(np.int64)
    type_pris = front["type_priority"].to_numpy().astype(np.int8)
    stock_codes = front["stock_code"].to_numpy().astype(np.int64)
    data_types = front["data_type"].to_list()

    n = front.height
    row_indices = np.arange(n, dtype=np.uint32)

    seq_times: list[np.ndarray] = []
    seq_types: list[np.ndarray] = []
    seq_codes: list[np.ndarray] = []
    seq_locs: list[np.ndarray] = []

    tick_mask = channels == -1
    order_mask = np.array([d == "order" for d in data_types], dtype=bool)
    trans_mask = np.array([d == "trans" for d in data_types], dtype=bool)

    # ── tick: single sequence ────────────────────────────────────────
    tick_idx = np.where(tick_mask)[0]
    if len(tick_idx) > 0:
        seq_times.append(local_times[tick_idx])
        seq_types.append(type_pris[tick_idx])
        seq_codes.append(stock_codes[tick_idx])
        seq_locs.append(row_indices[tick_idx])

    # ── SH channels (< threshold): merge order+trans by biz_index ────
    sh_mask = (~tick_mask) & (channels < SH_CHANNEL_THRESHOLD)
    sh_channels_arr = np.unique(channels[sh_mask])
    n_sh_merged = 0

    for ch in sh_channels_arr:
        ch_order_mask = sh_mask & order_mask & (channels == ch)
        ch_trans_mask = sh_mask & trans_mask & (channels == ch)
        o_idx = np.where(ch_order_mask)[0]
        t_idx = np.where(ch_trans_mask)[0]

        if len(o_idx) == 0 and len(t_idx) == 0:
            continue

        # Sort within channel by _ch_seq (biz_index)
        if len(o_idx) > 0:
            o_idx = o_idx[np.argsort(ch_seqs[o_idx])]
        if len(t_idx) > 0:
            t_idx = t_idx[np.argsort(ch_seqs[t_idx])]

        if len(o_idx) == 0:
            seq_times.append(local_times[t_idx])
            seq_types.append(type_pris[t_idx])
            seq_codes.append(stock_codes[t_idx])
            seq_locs.append(row_indices[t_idx])
            n_sh_merged += len(t_idx)
            continue

        if len(t_idx) == 0:
            seq_times.append(local_times[o_idx])
            seq_types.append(type_pris[o_idx])
            seq_codes.append(stock_codes[o_idx])
            seq_locs.append(row_indices[o_idx])
            n_sh_merged += len(o_idx)
            continue

        merge_order = two_way_merge_by_biz(ch_seqs[o_idx], ch_seqs[t_idx])
        no = len(o_idx)
        merged_len = len(merge_order)
        is_ord = merge_order < no

        combined_idx = np.empty(merged_len, dtype=np.uint32)
        combined_idx[is_ord] = o_idx[merge_order[is_ord]]
        combined_idx[~is_ord] = t_idx[merge_order[~is_ord] - no]

        seq_times.append(local_times[combined_idx])
        seq_types.append(type_pris[combined_idx])
        seq_codes.append(stock_codes[combined_idx])
        seq_locs.append(row_indices[combined_idx])
        n_sh_merged += merged_len

    # ── SZ channels (>= threshold): separate order and trans ─────────
    sz_mask = (~tick_mask) & (channels >= SH_CHANNEL_THRESHOLD)
    for is_order_type, mask_type in [(True, order_mask), (False, trans_mask)]:
        combined_mask = sz_mask & mask_type
        ch_vals = np.unique(channels[combined_mask])
        for ch in ch_vals:
            idx = np.where(combined_mask & (channels == ch))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.argsort(ch_seqs[idx])]
            seq_times.append(local_times[idx])
            seq_types.append(type_pris[idx])
            seq_codes.append(stock_codes[idx])
            seq_locs.append(row_indices[idx])

    n_seqs = len(seq_times)
    logger.info(
        "  K-way merge: {} sequences ({} SH merged, {:,} SH rows), {:,} total rows",
        n_seqs,
        len(sh_channels_arr),
        n_sh_merged,
        n,
    )

    # ── Pack and run Numba heap merge ────────────────────────────────
    all_times = np.concatenate(seq_times)
    all_types = np.concatenate(seq_types)
    all_codes = np.concatenate(seq_codes)
    all_locs = np.concatenate(seq_locs)

    offsets = np.zeros(n_seqs, dtype=np.int64)
    lengths = np.array([len(s) for s in seq_times], dtype=np.int64)
    cum = 0
    for i in range(n_seqs):
        offsets[i] = cum
        cum += lengths[i]

    out_src = np.empty(n, dtype=np.uint8)
    out_loc = np.empty(n, dtype=np.uint32)

    kway_merge_numba(
        all_times,
        all_types,
        all_codes,
        all_locs,
        offsets,
        lengths,
        out_src,
        out_loc,
    )

    # out_loc contains the original row indices in merged order
    idx_series = pl.Series("_idx", out_loc.astype(np.uint32))
    result = front.select(pl.all().gather(idx_series)).drop(["_channel", "_ch_seq"])

    logger.info("  K-way merge done ({:.1f}s)", _time.time() - t0)
    return result


# ── 4. Unnest + clean ────────────────────────────────────────────────


def unnest_and_clean(df: pl.DataFrame, data_type: str) -> pl.DataFrame:
    """Unnest the target struct, drop metadata and other structs,
    replace internal local_time with the merged outer one."""
    cfg = TYPE_CONFIG[data_type]
    struct_col = cfg["struct_col"]
    other_structs = cfg["other_structs"]

    df = df.rename({"local_time": "_merged_lt"})
    df = df.drop(META_COLS + other_structs)
    df = df.unnest(struct_col)
    df = df.drop("local_time")
    df = df.rename({"_merged_lt": "local_time"})
    return df


# ── 5. Output path ───────────────────────────────────────────────────


def get_output_path(date_str: str, data_type: str, output_base: str) -> str:
    quote_num = TYPE_CONFIG[data_type]["quote_num"]
    ym = f"{date_str[:4]}.{date_str[4:6]}"
    ymd = f"{date_str[:4]}.{date_str[4:6]}.{date_str[6:8]}"
    return f"{output_base}/{ym}/{ymd}/default/{quote_num}/all.parquet"


# ── 6. Main ──────────────────────────────────────────────────────────


class _Args:
    date_str: str
    sorted_path: str
    output_base: str
    post_process: bool
    post_process_only: bool


def parse_args() -> _Args:
    parser = argparse.ArgumentParser(
        description="Merge sorted parquet with NPY localtime.",
    )
    parser.add_argument("date", help="Date in YYYYMMDD format, e.g. 20250221")
    parser.add_argument(
        "--sorted-dir",
        default=get_stage1_sorted_dir(),
        help="Directory containing sorted_{date}.parquet",
    )
    parser.add_argument(
        "--output-base",
        default=STAGE2_OUTPUT_BASE,
        help="Output base directory for stage2 outputs",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="After merge, fix serial and split into per-stock parquet files",
    )
    parser.add_argument(
        "--post-process-only",
        action="store_true",
        help="Skip merge, only fix serial and split existing all.parquet files",
    )
    raw = parser.parse_args()

    a = _Args()
    a.date_str = raw.date.strip()
    if len(a.date_str) != 8 or not a.date_str.isdigit():
        raise ValueError(f"Invalid date '{a.date_str}', expected YYYYMMDD.")

    a.output_base = raw.output_base
    a.post_process = raw.post_process
    a.post_process_only = raw.post_process_only

    if a.post_process_only:
        a.sorted_path = ""
        for dt in ["tick", "trans", "order"]:
            p = get_output_path(a.date_str, dt, a.output_base)
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"--post-process-only: {p} not found, run merge first"
                )
    else:
        a.sorted_path = f"{raw.sorted_dir}/sorted_{a.date_str}.parquet"
        if not Path(a.sorted_path).exists():
            raise FileNotFoundError(f"Sorted parquet not found: {a.sorted_path}")

    return a


def _wind_code_to_filename(wind_code: str | bytes) -> str:
    """'501001.SH' -> 'SH501001.parquet'."""
    if isinstance(wind_code, bytes):
        wind_code = wind_code.decode("utf-8")
    wind_code = str(wind_code)
    if "." not in wind_code:
        return f"{wind_code}.parquet"
    code, exchange = wind_code.split(".")
    return f"{exchange}{code}.parquet"


def _market_symbol_to_filename(market: int | str, symbol: str | bytes) -> str:
    """(49, '600000') -> 'SH600000.parquet'."""
    if isinstance(symbol, bytes):
        symbol = symbol.decode("utf-8")
    symbol = str(symbol)
    market = int(market)
    exchange = {48: "SZ", 49: "SH"}.get(market, f"M{market}")
    return f"{exchange}{symbol}.parquet"


def fix_serial_and_split(output_path: str, data_type: str) -> None:
    """Fix serial column to 0-based incremental, then split into per-stock files."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fpath = Path(output_path)
    output_dir = fpath.parent

    # ── fix serial ────────────────────────────────────────────────────
    logger.info("  [post] fix serial: {}", fpath.name)
    df = pl.read_parquet(fpath)
    df = df.with_columns(pl.int_range(pl.len(), dtype=pl.Int32).alias("serial"))
    df.write_parquet(fpath, compression="zstd")
    logger.info(
        "  [post] serial reset: {:,} rows, max={}", df.height, df["serial"].max()
    )

    # ── split into per-stock files ────────────────────────────────────
    if data_type == "tick":
        partition_field = "wind_code"
    else:
        partition_field = ["symbol", "market"]

    if isinstance(partition_field, str):
        partition_fields = [partition_field]
    else:
        partition_fields = partition_field

    partitions = df.partition_by(partition_fields, maintain_order=True)
    logger.info("  [post] splitting {} into {} stocks ...", data_type, len(partitions))

    def _save_one(part_df: pl.DataFrame) -> str | None:
        if part_df.height == 0:
            return None
        if "serial" in part_df.columns:
            part_df = part_df.with_columns(
                pl.int_range(pl.len(), dtype=pl.Int32).alias("serial")
            )
        if data_type == "tick":
            wc = part_df["wind_code"].item(0)
            fname = _wind_code_to_filename(wc)
        else:
            sym = part_df["symbol"].item(0)
            mkt = part_df["market"].item(0)
            fname = _market_symbol_to_filename(mkt, sym)
        part_df.write_parquet(output_dir / fname, compression="zstd")
        return fname

    ok = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = {pool.submit(_save_one, p): i for i, p in enumerate(partitions)}
        for fut in as_completed(futs):
            if fut.result() is not None:
                ok += 1

    logger.info("  [post] {} split done: {} stock files written", data_type, ok)
    del df, partitions
    gc.collect()


def main() -> None:
    ts = _time.time()
    a = parse_args()
    date_str, sorted_path, output_base = a.date_str, a.sorted_path, a.output_base

    if a.post_process_only:
        logger.info("date={}, mode=post-process-only", date_str)
        logger.info("=" * 60)
        logger.info("Post-process only (fix serial + split stocks)")
        for dt in ["tick", "trans", "order"]:
            t0 = _time.time()
            out_path = get_output_path(date_str, dt, output_base)
            fix_serial_and_split(out_path, dt)
            logger.info("  {} post-process done ({:.1f}s)", dt, _time.time() - t0)
        total_sec = _time.time() - ts
        logger.info("=" * 60)
        logger.info("All done in {:.1f}s ({:.1f} min)", total_sec, total_sec / 60)
        return

    logger.info("date={}, sorted={}", date_str, sorted_path)

    cutoff_ns = compute_cutoff_ns(date_str)
    logger.info("940 cutoff (ns): {}", cutoff_ns)

    # ── Step 1: front half — per-type join with NPY ─────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Front half join with NPY (tick=right, trans/order=left)")

    front_parts: list[pl.DataFrame] = []
    for dt in ["tick", "trans", "order"]:
        npy_df = load_npy(date_str, dt)
        part = join_front_type(sorted_path, cutoff_ns, dt, npy_df)
        front_parts.append(part)
        del npy_df
        gc.collect()

    front = pl.concat(front_parts, how="vertical_relaxed")
    del front_parts
    gc.collect()

    front = kway_merge_front(front)

    # ── dedup: same local_time → 2nd row +1, 3rd +2, … ──────────
    # Data is sorted, so identical local_time values are contiguous.
    # Use row_index + cum_max trick to compute within-group offset in O(n)
    # without .over() which chokes on pl.lit() scalars.
    dup_before = front["local_time"].is_duplicated().sum()
    if dup_before > 0:
        front = front.with_row_index("_ridx")
        is_boundary = (pl.col("local_time") != pl.col("local_time").shift(1)).fill_null(
            True
        )
        group_start = (
            pl.col("_ridx").cast(pl.Int64) * is_boundary.cast(pl.Int64)
        ).cum_max()
        front = front.with_columns(
            (pl.col("local_time") + pl.col("_ridx").cast(pl.Int64) - group_start).alias(
                "local_time"
            )
        ).drop("_ridx")
        dup_after = front["local_time"].is_duplicated().sum()
        logger.info(
            "  local_time dedup: {} duplicated before -> {} after",
            dup_before,
            dup_after,
        )

    last_front_lt = int(front["local_time"][-1])
    logger.info(
        "Front merged: {:,} rows, last_lt={}",
        front.height,
        last_front_lt,
    )

    # ── Step 2: offset from back half ────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Compute offset")

    first_back_lt = int(
        pl.scan_parquet(sorted_path)
        .filter(pl.col("sort_time") > cutoff_ns)
        .select("local_time")
        .head(1)
        .collect()
        .item()
    )
    offset = last_front_lt + 1 - first_back_lt
    logger.info("first_back_lt={}, offset={:+,}", first_back_lt, offset)

    # ── Step 3: per-type concat + unnest + write ─────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Per-type output")

    for dt in ["tick", "trans", "order"]:
        t0 = _time.time()

        front_type = front.filter(pl.col("data_type") == dt)
        logger.info("  {} front: {:,} rows", dt, front_type.height)

        back_type = (
            pl.scan_parquet(sorted_path)
            .filter((pl.col("sort_time") > cutoff_ns) & (pl.col("data_type") == dt))
            .with_columns(pl.col("local_time") + offset)
            .collect()
        )
        logger.info("  {} back:  {:,} rows", dt, back_type.height)

        combined = pl.concat([front_type, back_type], how="vertical_relaxed")
        combined_rows = combined.height
        del front_type, back_type
        gc.collect()

        cleaned = unnest_and_clean(combined, dt)
        del combined
        gc.collect()

        cleaned = validate_schema(
            cleaned, DATA_TYPE_TO_QUOTE[dt], schema_type="parquet"
        )

        # sanity check: local_time monotonic
        lt = cleaned["local_time"]
        if lt.is_null().any():
            logger.warning("  {} has null local_time!", dt)
        elif lt.len() > 1:
            diffs = lt.diff().drop_nulls()
            non_increasing = (diffs <= 0).sum()
            if non_increasing > 0:
                logger.warning(
                    "  {} local_time NOT strictly increasing: {:,} violations",
                    dt,
                    non_increasing,
                )
            else:
                logger.info("  {} local_time strictly increasing ✓", dt)

        output_path = get_output_path(date_str, dt, output_base)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cleaned.write_parquet(output_path, compression="zstd")
        logger.info(
            "  {} written: {:,} rows -> {} ({:.1f}s)",
            dt,
            cleaned.height,
            output_path,
            _time.time() - t0,
        )

        assert cleaned.height == combined_rows, (
            f"{dt}: row count mismatch after unnest "
            f"({cleaned.height} != {combined_rows})"
        )

        del cleaned
        gc.collect()

    del front
    gc.collect()

    # ── Step 4 (optional): fix serial + split per-stock ───────────────
    if a.post_process:
        logger.info("=" * 60)
        logger.info("Step 4: Post-process (fix serial + split stocks)")
        for dt in ["tick", "trans", "order"]:
            t0 = _time.time()
            out_path = get_output_path(date_str, dt, output_base)
            fix_serial_and_split(out_path, dt)
            logger.info("  {} post-process done ({:.1f}s)", dt, _time.time() - t0)

    total_sec = _time.time() - ts
    logger.info("=" * 60)
    logger.info("All done in {:.1f}s ({:.1f} min)", total_sec, total_sec / 60)


if __name__ == "__main__":
    main()
