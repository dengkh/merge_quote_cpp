#!/usr/bin/env python3
"""
Batch runner: iterate trading days in a date range and invoke parquet_sort for each.

Usage:
  python run_parquet.py <start_date> <end_date> <data_base> <npy_dir> <output_base> [extra_flags...]

Example:
  python run_parquet.py 2023-01-01 2023-01-31 \
      /mnt/data/data/ParquetDataV2 \
      /mnt/data/public/quant001/local_times/stocks/91500000-94000000 \
      ./output/stg2 \
      --no-sh-merge --post-process

  data_base:   root containing {YYYY.MM}/{YYYY.MM.DD}/default/  sub-dirs
  npy_dir:     root containing {YYYYMMDD}/  sub-dirs
  output_base: passed as --output-base to parquet_sort
  extra_flags: forwarded verbatim (e.g. --no-sh-merge, --post-process, --write-intermediate)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import exchange_calendars as ec


def _find_pyarrow_lib_dir() -> str:
    """Locate pyarrow's shared library directory for LD_LIBRARY_PATH."""
    try:
        import pyarrow
        return os.path.dirname(pyarrow.__file__)
    except ImportError:
        pass
    candidate = os.path.expandvars(
        "$HOME/.local/lib/python3.8/site-packages/pyarrow")
    if os.path.isdir(candidate):
        return candidate
    return ""


def main():
    if len(sys.argv) < 6:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    data_base = sys.argv[3]
    npy_dir = sys.argv[4]
    output_base = sys.argv[5]
    extra_flags = sys.argv[6:]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    binary = os.path.join(script_dir, "build", "parquet_sort")
    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    pyarrow_lib = _find_pyarrow_lib_dir()
    env = os.environ.copy()
    if pyarrow_lib:
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{pyarrow_lib}:{ld}" if ld else pyarrow_lib
        print(f"LD_LIBRARY_PATH prepended: {pyarrow_lib}")

    cal = ec.get_calendar("XSHG")
    sessions = cal.sessions_in_range(start_date, end_date)
    trading_days = [d.strftime("%Y%m%d") for d in sessions]

    print(f"Trading days in [{start_date}, {end_date}]: {len(trading_days)}")

    total_t0 = time.time()
    ok, fail, skip = 0, 0, 0

    for i, date_str in enumerate(trading_days, 1):
        ym = f"{date_str[:4]}.{date_str[4:6]}"
        ymd = f"{ym}.{date_str[6:8]}"
        data_dir = os.path.join(data_base, ym, ymd, "default")

        if not os.path.isdir(data_dir):
            print(f"[{i}/{len(trading_days)}] {date_str}  SKIP (dir not found: {data_dir})")
            skip += 1
            continue

        cmd = [
            binary, date_str, data_dir,
            "--npy-dir", npy_dir,
            "--output-base", output_base,
            *extra_flags,
        ]

        print(f"[{i}/{len(trading_days)}] {date_str}  START")
        sys.stdout.flush()
        t0 = time.time()

        ret = subprocess.call(cmd, env=env, cwd=script_dir)

        elapsed = time.time() - t0
        if ret == 0:
            ok += 1
            print(f"[{i}/{len(trading_days)}] {date_str}  OK     {elapsed:.1f}s")
        else:
            fail += 1
            print(f"[{i}/{len(trading_days)}] {date_str}  FAIL   exit={ret}  {elapsed:.1f}s")
        sys.stdout.flush()

    total_elapsed = time.time() - total_t0
    print(f"\nDone: {ok} ok, {fail} fail, {skip} skip  ({total_elapsed:.1f}s total)")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
