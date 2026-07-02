#!/bin/bash
# Run parquet_sort (simple mode: local_time = exchange_time with +1 dedup)
# over all trading days in 2021.01 - 2021.06 (before 2021.07).
#
# Input : /mnt/beegfs_ssd/public/quant002/hds_work/CppGenRaw/YYYY.MM/YYYY.MM.DD/default
# Output: /mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple/YYYY.MM/YYYY.MM.DD/default/{1,2,3}/all.parquet

set -u

BIN=/mnt/data/private/dev002/code/merge_quote_cpp/build/parquet_sort
RAW_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenRaw
OUT_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple
LOG_DIR=/mnt/data/private/dev002/code/merge_quote_cpp/logs_simple_2021h1
SUMMARY="$LOG_DIR/summary.log"

MONTHS="2021.01 2021.02 2021.03 2021.04 2021.05 2021.06"

mkdir -p "$LOG_DIR"
echo "==== run started $(date '+%F %T') ====" | tee -a "$SUMMARY"

n_ok=0
n_fail=0
for m in $MONTHS; do
    for day_dir in "$RAW_BASE/$m/$m."*; do
        [ -d "$day_dir/default" ] || continue
        ymd_dotted=$(basename "$day_dir")           # 2021.01.04
        ymd=${ymd_dotted//./}                        # 20210104
        out_done="$OUT_BASE/$m/$ymd_dotted/default/1/all.parquet"

        if [ -f "$out_done" ]; then
            echo "[$(date '+%F %T')] SKIP $ymd (already done)" | tee -a "$SUMMARY"
            continue
        fi

        echo "[$(date '+%F %T')] START $ymd" | tee -a "$SUMMARY"
        "$BIN" "$ymd" "$day_dir/default" \
            --output-base "$OUT_BASE" \
            --log-file "$LOG_DIR/$ymd.log" \
            > "$LOG_DIR/$ymd.stdout" 2>&1

        rc=$?
        if [ $rc -eq 0 ]; then
            n_ok=$((n_ok+1))
            echo "[$(date '+%F %T')] OK    $ymd" | tee -a "$SUMMARY"
        else
            n_fail=$((n_fail+1))
            echo "[$(date '+%F %T')] FAIL  $ymd (rc=$rc)" | tee -a "$SUMMARY"
        fi
    done
done

echo "==== run finished $(date '+%F %T') : ok=$n_ok fail=$n_fail ====" | tee -a "$SUMMARY"
