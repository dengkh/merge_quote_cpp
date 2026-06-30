#!/usr/bin/env bash
# Verify all CppGen dates from 2023.01.01 to 2023.12.22 using verify_npy_output.py
set -euo pipefail

CPPGEN_BASE="/mnt/beegfs/quant002/hds_work/CppGen"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY="$SCRIPT_DIR/verify_npy_output.py"
LOG="$SCRIPT_DIR/verify_all.log"

ok=0; fail=0; skip=0
fail_dates=""

: > "$LOG"

for month_dir in "$CPPGEN_BASE"/2023.*/; do
    [ -d "$month_dir" ] || continue
    for day_dir in "$month_dir"2023.*/; do
        [ -d "$day_dir" ] || continue
        # Extract YYYYMMDD from path like .../2023.01/2023.01.03/
        basename_dir=$(basename "$day_dir")       # 2023.01.03
        date_str="${basename_dir//./}"             # 20230103

        # Range check: 20230101 .. 20231222
        if [[ "$date_str" < "20230101" || "$date_str" > "20231222" ]]; then
            continue
        fi

        # Check that output files exist
        if [[ ! -f "$day_dir/default/1/all.parquet" ]]; then
            echo "[$(printf '%3d' $((ok+fail+skip+1)))] $date_str  SKIP (no output)" | tee -a "$LOG"
            ((skip++)) || true
            continue
        fi

        echo -n "[$(printf '%3d' $((ok+fail+skip+1)))] $date_str  " | tee -a "$LOG"

        if python3 "$VERIFY" "$date_str" --output-base "$CPPGEN_BASE" >> "$LOG" 2>&1; then
            echo "OK" | tee -a "$LOG"
            ((ok++)) || true
        else
            echo "FAIL" | tee -a "$LOG"
            ((fail++)) || true
            fail_dates="$fail_dates $date_str"
        fi
    done
done

echo "" | tee -a "$LOG"
echo "Done: $ok ok, $fail fail, $skip skip" | tee -a "$LOG"
if [ -n "$fail_dates" ]; then
    echo "Failed dates:$fail_dates" | tee -a "$LOG"
fi
