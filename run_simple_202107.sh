#!/usr/bin/env bash
set -u
BIN=/mnt/data/private/dev002/code/merge_quote_cpp/build/parquet_sort
RAW_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenRaw
OUT_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple
LOG_DIR=/mnt/data/private/dev002/code/merge_quote_cpp/logs_simple_202107
SUMMARY="$LOG_DIR/summary.log"
: > "$SUMMARY"
echo "==== run started $(date '+%F %T') ====" | tee -a "$SUMMARY"
n_ok=0; n_fail=0
for ymd_dotted in 2021.07.01 2021.07.02 2021.07.05 2021.07.06 2021.07.07; do
    ymd=${ymd_dotted//./}
    day_dir="$RAW_BASE/2021.07/$ymd_dotted"
    echo "[$(date '+%F %T')] START $ymd" | tee -a "$SUMMARY"
    "$BIN" "$ymd" "$day_dir/default" --output-base "$OUT_BASE" --log-file "$LOG_DIR/$ymd.log" > "$LOG_DIR/$ymd.stdout" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then n_ok=$((n_ok+1)); echo "[$(date '+%F %T')] OK    $ymd" | tee -a "$SUMMARY";
    else n_fail=$((n_fail+1)); echo "[$(date '+%F %T')] FAIL  $ymd (rc=$rc)" | tee -a "$SUMMARY"; fi
done
echo "==== run finished $(date '+%F %T') : ok=$n_ok fail=$n_fail ====" | tee -a "$SUMMARY"
