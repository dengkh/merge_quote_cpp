#!/usr/bin/env bash
# Verify all already-generated CppGenSimple days with verify_simple.py.
# A day is considered complete when its order output (3/all.parquet) exists,
# since order is written last by parquet_sort.
#
# Runs PARALLEL days at a time (default 16). Per-day logs in logs_verify_simple/,
# aggregate result in summary.log.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERIFY="$SCRIPT_DIR/verify_simple.py"
OUT_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple
RAW_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenRaw
LOG_DIR="$SCRIPT_DIR/logs_verify_simple"
SUMMARY="$LOG_DIR/summary.log"
STATUS_DIR="$LOG_DIR/status"
PARALLEL="${PARALLEL:-16}"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/*.status 2>/dev/null
: > "$SUMMARY"
echo "==== verify started $(date '+%F %T')  (parallel=$PARALLEL) ====" | tee -a "$SUMMARY"

# Collect completed days (all three types present), chronological.
days=()
for done_file in $(find "$OUT_BASE" -path "*/default/3/all.parquet" 2>/dev/null | sort); do
    day_dir=$(dirname "$(dirname "$(dirname "$done_file")")")  # .../YYYY.MM.DD
    ymd=$(basename "$day_dir"); ymd=${ymd//./}                 # YYYYMMDD
    [ -f "$day_dir/default/1/all.parquet" ] || continue
    [ -f "$day_dir/default/2/all.parquet" ] || continue
    days+=("$ymd")
done
echo "found ${#days[@]} completed days" | tee -a "$SUMMARY"

verify_one() {
    ymd="$1"
    if python3 "$VERIFY" "$ymd" --output-base "$OUT_BASE" --raw-base "$RAW_BASE" \
            > "$LOG_DIR/$ymd.log" 2>&1; then
        echo "PASS" > "$STATUS_DIR/$ymd.status"
        echo "[$(date '+%F %T')] $ymd  PASS"
    else
        echo "FAIL" > "$STATUS_DIR/$ymd.status"
        echo "[$(date '+%F %T')] $ymd  FAIL"
    fi
}
export -f verify_one
export VERIFY OUT_BASE RAW_BASE LOG_DIR STATUS_DIR

printf '%s\n' "${days[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'verify_one "$@"' _ {} \
    | tee -a "$SUMMARY"

# Aggregate
ok=$(grep -l PASS "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
fail=$(grep -l FAIL "$STATUS_DIR"/*.status 2>/dev/null | wc -l)
fail_dates=$(grep -l FAIL "$STATUS_DIR"/*.status 2>/dev/null | xargs -r -n1 basename | sed 's/.status//' | tr '\n' ' ')

echo "" | tee -a "$SUMMARY"
echo "==== verify finished $(date '+%F %T') : pass=$ok fail=$fail ====" | tee -a "$SUMMARY"
[ -n "$fail_dates" ] && echo "FAILED: $fail_dates" | tee -a "$SUMMARY"
