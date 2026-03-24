#!/usr/bin/env python3
"""
K-way merge sort for tick/order/trans parquet quote data.

Modified: SH channels (< 1000) merge order+trans into single per-channel
sequences sorted by biz_index, preserving channel-internal index ordering.

Output schema (C1 struct layout):
  Metadata : sort_time, type_priority, stock_code, data_type, local_time
  Payloads : tick_data (struct|null), order_data (struct|null), trans_data (struct|null)

Sorting rules:
  1. time ascending  (exchange_time in nanoseconds)
  2. type priority   (tick=0 < order=1 < trans=2)
  3. stock code asc  (ticker / symbol, as integer)

  For SH channels: within each channel, biz_index (shared order_index /
  trade_index counter) ordering takes priority over type_priority and
  stock_code.  This is achieved by pre-merging order+trans per-channel
  into a single sequence sorted by biz_index.

sort_time: exchange_time (µs Unix) × 1000 → nanoseconds.
local_time: strictly increasing based on sort_time (ties get +1 ns).

Memory-optimised: no diagonal concat, no full-table reorder.
Three source tables stay separate; K-way merge streams chunks
directly to parquet via PyArrow ParquetWriter.
"""

import argparse
import gc
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ltmerge.kway import kway_merge_numba, two_way_merge_by_biz
from ltmerge.stage_paths import get_source_parquet_default_dir, get_stage1_sorted_dir

try:
    from write_mt import ParquetWriterMT as _MTWriter

    _USE_MT_WRITER = True
except ImportError:
    _MTWriter = None
    _USE_MT_WRITER = False

CHUNK_SIZE = 20_000_000
SH_CHANNEL_THRESHOLD = 1000


# ── helpers ──────────────────────────────────────────────────────────


def read_data(data_dir: str, sub_dir: str, label: str) -> pl.DataFrame:
    import glob

    folder = os.path.join(data_dir, sub_dir)
    files = sorted(
        f
        for f in glob.glob(os.path.join(folder, "*.parquet"))
        if os.path.basename(f) != "all.parquet"
    )
    t0 = _time.time()
    df = pl.read_parquet(files)
    logger.info(
        "[{}] {:>12,} rows from {} files ({:.1f}s)",
        label,
        df.shape[0],
        len(files),
        _time.time() - t0,
    )
    return df


def fix_array_cols(df: pl.DataFrame) -> pl.DataFrame:
    """FixedSizeList (Array) → List for parquet struct compatibility."""
    fsl = [n for n, d in df.schema.items() if d.base_type() == pl.Array]
    if fsl:
        df = df.with_columns([pl.col(c).cast(pl.List(pl.UInt32)) for c in fsl])
    return df


def build_meta_with_struct(
    df: pl.DataFrame,
    time_col: str,
    code_col: str,
    type_pri: int,
    dtype_str: str,
    struct_name: str,
) -> pl.DataFrame:
    """Create metadata columns + one struct payload from a source DataFrame."""
    meta = pl.DataFrame(
        {
            "sort_time": df[time_col] * 1000,
            "stock_code": df[code_col].cast(pl.Int64),
        }
    ).with_columns(
        pl.lit(type_pri, dtype=pl.Int8).alias("type_priority"),
        pl.lit(dtype_str).alias("data_type"),
    )
    payload = df.select(pl.struct(pl.all()).alias(struct_name))
    return pl.concat([meta, payload], how="horizontal")


def build_unified_schema(
    tick_schema: pa.Schema,
    order_schema: pa.Schema,
    trans_schema: pa.Schema,
) -> pa.Schema:
    """Build the full output schema with all struct columns + local_time."""
    fields = []
    for name in ["sort_time", "type_priority", "stock_code", "data_type"]:
        fields.append(tick_schema.field(name))
    fields.append(pa.field("local_time", pa.int64()))
    fields.append(tick_schema.field("tick_data"))
    fields.append(order_schema.field("order_data"))
    fields.append(trans_schema.field("trans_data"))
    return pa.schema(fields)


_DATA_TYPE_LABELS = np.array([b"tick", b"order", b"trans"])


def build_sorted_batch(
    chunk_src: np.ndarray,
    sources: list[pa.Table],
    cursors: list[int],
    struct_names: list[str],
    schema: pa.Schema,
) -> pa.Table:
    """Build output table by column-wise numpy scatter (avoids pc.take)."""
    chunk_len = len(chunk_src)
    src_masks = [chunk_src == i for i in range(3)]
    src_counts = [int(m.sum()) for m in src_masks]
    src_positions = [np.where(m)[0] for m in src_masks]

    slices: list[pa.Table] = []
    for sid in range(3):
        n = src_counts[sid]
        slices.append(sources[sid].slice(cursors[sid], n))
        cursors[sid] += n

    sort_time = np.empty(chunk_len, dtype=np.int64)
    type_pri = np.empty(chunk_len, dtype=np.int8)
    stock_code = np.empty(chunk_len, dtype=np.int64)
    for sid in range(3):
        if src_counts[sid] == 0:
            continue
        p = src_positions[sid]
        sort_time[p] = slices[sid].column("sort_time").to_numpy()
        type_pri[p] = slices[sid].column("type_priority").to_numpy()
        stock_code[p] = slices[sid].column("stock_code").to_numpy()

    struct_cols: dict[str, pa.Array] = {}
    for sid in range(3):
        col_name = struct_names[sid]
        n = src_counts[sid]
        if n == 0:
            struct_cols[col_name] = pa.nulls(chunk_len, type=schema.field(col_name).type)
            continue

        src_struct = slices[sid].column(col_name).combine_chunks()
        positions = src_positions[sid]
        st = src_struct.type

        child_arrays = []
        for fi in range(st.num_fields):
            child_np = src_struct.field(fi).to_numpy(zero_copy_only=False)
            buf = np.zeros(chunk_len, dtype=child_np.dtype)
            buf[positions] = child_np
            child_arrays.append(pa.array(buf, type=st.field(fi).type))

        validity = np.zeros(chunk_len, dtype=bool)
        validity[positions] = True
        struct_cols[col_name] = pa.StructArray.from_arrays(
            child_arrays,
            fields=list(st),
            mask=pa.array(~validity, type=pa.bool_()),
        )

    return pa.table(
        {
            "sort_time": pa.array(sort_time),
            "type_priority": pa.array(type_pri),
            "stock_code": pa.array(stock_code),
            "data_type": pa.array(_DATA_TYPE_LABELS[type_pri], type=pa.utf8()),
            "local_time": pa.nulls(chunk_len, type=pa.int64()),
            **struct_cols,
        },
        schema=schema,
    )


def compute_local_time(sort_time: np.ndarray, prev_last: int | None) -> np.ndarray:
    """Vectorised strictly-increasing local_time with cross-chunk carry-over."""
    n = len(sort_time)
    idx = np.arange(n, dtype=np.int64)
    adj = sort_time.astype(np.int64) - idx
    if prev_last is not None and n > 0:
        adj[0] = max(adj[0], prev_last + 1)
    np.maximum.accumulate(adj, out=adj)
    return adj + idx


def parse_date_arg() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="Sort parquet quote data by a YYYYMMDD date."
    )
    parser.add_argument("date", help="Date in YYYYMMDD format, e.g. 20250221")
    args = parser.parse_args()

    date_str = args.date.strip()
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"Invalid date '{date_str}', expected YYYYMMDD format.")

    data_dir = get_source_parquet_default_dir(date_str)
    return date_str, data_dir


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    ts = _time.time()
    date_str, data_dir = parse_date_arg()

    # ── 1. Read ──────────────────────────────────────────────────────
    logger.info("Reading parquet data ...")
    tick = read_data(data_dir, "1", "tick ")
    trans = read_data(data_dir, "2", "trans")
    order = read_data(data_dir, "3", "order")

    # ── 2. Pre-sort ─────────────────────────────────────────────────
    logger.info("Pre-sorting ...")
    t0 = _time.time()
    tick = tick.sort(["exchange_time", "ticker"], maintain_order=True)
    order = order.sort(["channel", "order_index"], maintain_order=True)
    trans = trans.sort(["channel", "trade_index"], maintain_order=True)
    logger.info("  done ({:.1f}s)", _time.time() - t0)

    n_tick = tick.shape[0]
    n_order = order.shape[0]
    n_trans = trans.shape[0]
    total = n_tick + n_order + n_trans

    # ── 3. Extract numpy arrays for K-way merge keys ─────────────────
    logger.info("Building sub-sequences ...")
    t0 = _time.time()

    tick_times = tick["exchange_time"].to_numpy() * 1000
    tick_codes = tick["ticker"].cast(pl.Int64).to_numpy()

    order_ch = order["channel"].to_numpy()
    order_times = order["exchange_time"].to_numpy() * 1000
    order_codes = order["symbol"].cast(pl.Int64).to_numpy()
    order_biz = order["order_index"].to_numpy()

    trans_ch = trans["channel"].to_numpy()
    trans_times = trans["exchange_time"].to_numpy() * 1000
    trans_codes = trans["symbol"].cast(pl.Int64).to_numpy()
    trans_biz = trans["trade_index"].to_numpy()

    # Sequence format:
    #   times    - sort_time (ns) for heap comparison
    #   types    - per-element type_priority (int or array)
    #   codes    - per-element stock_code for heap comparison
    #   locs     - per-element index into source Arrow table
    # For SH merged sequences, types/codes/locs vary per element.

    # s_times[i], s_types[i], s_codes[i], s_locs[i]
    # s_types can be int (uniform) or np.ndarray (per-element for SH merged)
    seq_times: list[np.ndarray] = []
    seq_types: list[int | np.ndarray] = []
    seq_codes: list[np.ndarray] = []
    seq_locs: list[np.ndarray] = []

    # tick: single sequence
    seq_times.append(tick_times)
    seq_types.append(0)
    seq_codes.append(tick_codes)
    seq_locs.append(np.arange(n_tick, dtype=np.uint32))

    # SH channels: merge order+trans per channel by (time, biz_index)
    sh_order_channels = set(
        int(c) for c in np.unique(order_ch) if c < SH_CHANNEL_THRESHOLD
    )
    sh_trans_channels = set(
        int(c) for c in np.unique(trans_ch) if c < SH_CHANNEL_THRESHOLD
    )
    sh_channels = sorted(sh_order_channels | sh_trans_channels)
    n_sh_merged = 0

    for ch in sh_channels:
        o_mask = order_ch == ch
        t_mask = trans_ch == ch
        o_idx = np.where(o_mask)[0]
        t_idx = np.where(t_mask)[0]

        if len(o_idx) == 0 and len(t_idx) == 0:
            continue

        if len(o_idx) == 0:
            seq_times.append(trans_times[t_idx])
            seq_types.append(2)
            seq_codes.append(trans_codes[t_idx])
            seq_locs.append(t_idx.astype(np.uint32))
            n_sh_merged += len(t_idx)
            continue

        if len(t_idx) == 0:
            seq_times.append(order_times[o_idx])
            seq_types.append(1)
            seq_codes.append(order_codes[o_idx])
            seq_locs.append(o_idx.astype(np.uint32))
            n_sh_merged += len(o_idx)
            continue

        merge_order = two_way_merge_by_biz(
            order_biz[o_idx],
            trans_biz[t_idx],
        )

        no = len(o_idx)
        merged_len = len(merge_order)
        is_order = merge_order < no

        m_times = np.empty(merged_len, dtype=np.int64)
        m_types = np.empty(merged_len, dtype=np.int8)
        m_codes = np.empty(merged_len, dtype=np.int64)
        m_locs = np.empty(merged_len, dtype=np.uint32)

        o_sel = merge_order[is_order]
        t_sel = merge_order[~is_order] - no

        m_times[is_order] = order_times[o_idx[o_sel]]
        m_times[~is_order] = trans_times[t_idx[t_sel]]

        m_types[is_order] = 1
        m_types[~is_order] = 2

        m_codes[is_order] = order_codes[o_idx[o_sel]]
        m_codes[~is_order] = trans_codes[t_idx[t_sel]]

        m_locs[is_order] = o_idx[o_sel].astype(np.uint32)
        m_locs[~is_order] = t_idx[t_sel].astype(np.uint32)

        seq_times.append(m_times)
        seq_types.append(m_types)
        seq_codes.append(m_codes)
        seq_locs.append(m_locs)
        n_sh_merged += merged_len

    # SZ channels: separate order and trans sequences (unchanged)
    for ch in np.unique(order_ch):
        if ch < SH_CHANNEL_THRESHOLD:
            continue
        idx = np.where(order_ch == ch)[0]
        seq_times.append(order_times[idx])
        seq_types.append(1)
        seq_codes.append(order_codes[idx])
        seq_locs.append(idx.astype(np.uint32))

    for ch in np.unique(trans_ch):
        if ch < SH_CHANNEL_THRESHOLD:
            continue
        idx = np.where(trans_ch == ch)[0]
        seq_times.append(trans_times[idx])
        seq_types.append(2)
        seq_codes.append(trans_codes[idx])
        seq_locs.append(idx.astype(np.uint32))

    n_seqs = len(seq_times)
    logger.info(
        "  {} sequences ({} SH merged channels, {:,} SH rows)  ({:.1f}s)",
        n_seqs,
        len(sh_channels),
        n_sh_merged,
        _time.time() - t0,
    )
    for i in range(n_seqs):
        tp = seq_types[i]
        if isinstance(tp, np.ndarray):
            label = "sh_mg"
        else:
            label = {0: "tick", 1: "order", 2: "trans"}[tp]
        logger.info("    [{:>2d}] {:>5s}  {:>12,} rows", i, label, len(seq_times[i]))

    del order_ch, trans_ch, tick_times, tick_codes
    del order_times, order_codes, trans_times, trans_codes
    del order_biz, trans_biz

    # ── 4. Pack struct payloads (per source, NO diagonal concat) ─────
    logger.info("Packing structs → Arrow tables ...")
    t0 = _time.time()

    tick = fix_array_cols(tick)
    order = fix_array_cols(order)
    trans = fix_array_cols(trans)

    tick_df = build_meta_with_struct(
        tick, "exchange_time", "ticker", 0, "tick", "tick_data"
    )
    del tick
    order_df = build_meta_with_struct(
        order, "exchange_time", "symbol", 1, "order", "order_data"
    )
    del order
    trans_df = build_meta_with_struct(
        trans, "exchange_time", "symbol", 2, "trans", "trans_data"
    )
    del trans

    tick_arrow = tick_df.to_arrow()
    del tick_df
    order_arrow = order_df.to_arrow()
    del order_df
    trans_arrow = trans_df.to_arrow()
    del trans_df
    gc.collect()

    sources = [tick_arrow, order_arrow, trans_arrow]
    struct_names = ["tick_data", "order_data", "trans_data"]
    schema = build_unified_schema(
        tick_arrow.schema, order_arrow.schema, trans_arrow.schema
    )

    logger.info("  Arrow tables ready ({:.1f}s)", _time.time() - t0)
    for lbl, tbl in zip(["tick", "order", "trans"], sources):
        logger.info(
            "  {:>5s} : {:>12,} rows × {} cols", lbl, tbl.num_rows, tbl.num_columns
        )

    # ── 5. K-way merge (Numba JIT min-heap) ────────────────────────
    logger.info("K-way merge ({:,} rows, heap size {}) ...", total, n_seqs)
    t0 = _time.time()

    s_lens = [len(seq_times[i]) for i in range(n_seqs)]

    # Expand seq_types: uniform int → per-element arrays
    s_types_arr: list[np.ndarray] = []
    for i in range(n_seqs):
        tp = seq_types[i]
        if isinstance(tp, np.ndarray):
            s_types_arr.append(tp.astype(np.int8))
        else:
            s_types_arr.append(np.full(s_lens[i], tp, dtype=np.int8))

    # Concatenate all sequences into contiguous arrays for Numba
    all_times = np.concatenate([seq_times[i] for i in range(n_seqs)])
    all_types = np.concatenate(s_types_arr)
    all_codes = np.concatenate([seq_codes[i] for i in range(n_seqs)])
    all_locs = np.concatenate([seq_locs[i] for i in range(n_seqs)])
    offsets = np.zeros(n_seqs, dtype=np.int64)
    lengths = np.array(s_lens, dtype=np.int64)
    cum = 0
    for i in range(n_seqs):
        offsets[i] = cum
        cum += s_lens[i]

    logger.info(
        "  arrays: times {:.1f} MB, types {:.1f} MB, codes {:.1f} MB, total {:,} elements",
        all_times.nbytes / 1e6,
        all_types.nbytes / 1e6,
        all_codes.nbytes / 1e6,
        len(all_times),
    )

    # JIT warmup on first call (cached for subsequent runs)
    merge_src = np.empty(total, dtype=np.uint8)
    merge_loc = np.empty(total, dtype=np.uint32)

    logger.info("  running Numba K-way merge ...")
    kway_merge_numba(
        all_times,
        all_types,
        all_codes,
        all_locs,
        offsets,
        lengths,
        merge_src,
        merge_loc,
    )

    merge_sec = _time.time() - t0
    logger.info(
        "  merge done in {:.1f}s  ({:,.0f} rows/s)", merge_sec, total / merge_sec
    )
    logger.info(
        "  index arrays: merge_src {:.1f} MB, merge_loc {:.1f} MB",
        merge_src.nbytes / 1e6,
        merge_loc.nbytes / 1e6,
    )

    del seq_times, seq_types, seq_codes, seq_locs, s_types_arr
    del all_times, all_types, all_codes, all_locs, offsets, lengths
    gc.collect()

    # ── 6a. Pre-reorder source tables + pre-compute sort_times ─────
    logger.info("Pre-reordering source tables by merge order ...")
    t0 = _time.time()

    sort_times_all = np.empty(total, dtype=np.int64)
    for i in range(3):
        mask = merge_src == i
        locs = merge_loc[mask]
        idx_arr = pa.array(locs, type=pa.uint32())
        sources[i] = pc.take(sources[i], idx_arr)
        sort_times_all[np.where(mask)[0]] = sources[i].column("sort_time").to_numpy()

    del merge_loc
    gc.collect()

    reorder_sec = _time.time() - t0
    logger.info(
        "  pre-reorder done in {:.1f}s  (sort_times_all {:.1f} MB)",
        reorder_sec,
        sort_times_all.nbytes / 1e6,
    )

    # ── 6b. Chunked gather + write ────────────────────────────────────
    out_dir = get_stage1_sorted_dir()
    final_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sorted_{date_str}.parquet")

    logger.info(
        "Chunked write ({:,} rows, chunk {}) → {} ...", total, CHUNK_SIZE, out_path
    )
    t0 = _time.time()

    prev_last_local: int | None = None
    written = 0
    n_chunks = 0
    cursors = [0, 0, 0]

    _writer_cls = _MTWriter if _USE_MT_WRITER else pq.ParquetWriter
    logger.info("  writer: {}", "multi-threaded C++" if _USE_MT_WRITER else "pyarrow default")

    with _writer_cls(
        out_path,
        schema,
        compression="zstd",
    ) as writer:
        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            chunk_len = end - start

            batch_table = build_sorted_batch(
                merge_src[start:end],
                sources,
                cursors,
                struct_names,
                schema,
            )

            local_times = compute_local_time(sort_times_all[start:end], prev_last_local)
            prev_last_local = int(local_times[-1])

            batch_table = batch_table.set_column(
                schema.get_field_index("local_time"),
                "local_time",
                pa.array(local_times, type=pa.int64()),
            )

            writer.write_table(batch_table)
            written += chunk_len
            n_chunks += 1

            if n_chunks % 5 == 0:
                el = _time.time() - t0
                eta = el / written * (total - written)
                logger.info(
                    "    {:>12,} / {:,} written  ({:.0f}s elapsed, ~{:.0f}s left)",
                    written,
                    total,
                    el,
                    eta,
                )

    write_sec = _time.time() - t0
    logger.info(
        "  write done in {:.1f}s  ({:,.0f} rows/s, {} chunks)",
        write_sec,
        total / write_sec,
        n_chunks,
    )

    del merge_src, sort_times_all
    del sources, tick_arrow, order_arrow, trans_arrow
    gc.collect()

    # ── 7. Move to final location (NFS) ──────────────────────────────
    import shutil

    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, f"sorted_{date_str}.parquet")
    if out_path != final_path:
        logger.info("Moving {} → {} ...", out_path, final_path)
        t0 = _time.time()
        shutil.move(out_path, final_path)
        logger.info("  move done in {:.1f}s", _time.time() - t0)
        out_path = final_path

    # ── Summary ──────────────────────────────────────────────────────
    total_sec = _time.time() - ts
    logger.info("=" * 60)
    logger.info("Total rows written : {:,}", written)
    logger.info("Output             : {}", out_path)
    logger.info("Elapsed            : {:.1f}s ({:.1f} min)", total_sec, total_sec / 60)

    logger.info("Verifying output ...")
    pf = pq.ParquetFile(out_path)
    logger.info("  row groups : {}", pf.metadata.num_row_groups)
    logger.info("  total rows : {:,}", pf.metadata.num_rows)
    logger.info("  schema     : {}", [f.name for f in pf.schema_arrow])

    meta_cols = [
        "sort_time",
        "type_priority",
        "stock_code",
        "data_type",
        "local_time",
    ]
    first_rg = pf.read_row_group(0, columns=meta_cols)
    logger.info("First 10 rows (metadata):\n{}", pl.from_arrow(first_rg).head(10))

    last_rg = pf.read_row_group(pf.metadata.num_row_groups - 1, columns=meta_cols)
    logger.info("Last 10 rows (metadata):\n{}", pl.from_arrow(last_rg).tail(10))


if __name__ == "__main__":
    main()
