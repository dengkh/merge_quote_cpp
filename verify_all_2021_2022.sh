#!/usr/bin/env bash
set -euo pipefail

CPPGEN_BASE="/mnt/beegfs/quant002/hds_work/CppGen"
NPY_BASE="/mnt/data/public/quant001/local_times/stocks/91500000-94000000"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY="$SCRIPT_DIR/verify_npy_output.py"
LOG="$SCRIPT_DIR/verify_all_2021_2022.log"

ok=0; fail=0; skip=0
fail_dates=""

: > "$LOG"

for year in 2021 2022; do
    for month_dir in "$CPPGEN_BASE"/$year.*/; do
        [ -d "$month_dir" ] || continue
        for day_dir in "$month_dir"$year.*/; do
            [ -d "$day_dir" ] || continue
            basename_dir=$(basename "$day_dir")
            date_str="${basename_dir//./}"

            if [[ ! -f "$day_dir/default/1/all.parquet" ]]; then
                echo "[$(printf '%3d' $((ok+fail+skip+1)))] $date_str  SKIP (no output)" | tee -a "$LOG"
                ((skip++)) || true
                continue
            fi

            echo -n "[$(printf '%3d' $((ok+fail+skip+1)))] $date_str  " | tee -a "$LOG"

            t0=$SECONDS
            if python3 "$VERIFY" "$date_str" --output-base "$CPPGEN_BASE" --npy-base "$NPY_BASE" >> "$LOG" 2>&1; then
                elapsed=$((SECONDS - t0))
                echo "OK  (${elapsed}s)" | tee -a "$LOG"
                ((ok++)) || true
            else
                elapsed=$((SECONDS - t0))
                echo "FAIL  (${elapsed}s)" | tee -a "$LOG"
                ((fail++)) || true
                fail_dates="$fail_dates $date_str"
            fi
        done
    done
done

echo "" | tee -a "$LOG"
echo "Done: $ok ok, $fail fail, $skip skip" | tee -a "$LOG"
if [ -n "$fail_dates" ]; then
    echo "Failed dates:$fail_dates" | tee -a "$LOG"
fi
