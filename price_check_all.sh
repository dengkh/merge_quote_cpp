#!/usr/bin/env bash
# Run price_check.py over all generated CppGenSimple days (tick price sanity).
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_BASE=/mnt/beegfs_ssd/public/quant002/hds_work/CppGenSimple
LOG_DIR="$SCRIPT_DIR/logs_price_check"
SUMMARY="$LOG_DIR/summary.log"
PARALLEL="${PARALLEL:-16}"

mkdir -p "$LOG_DIR"
: > "$SUMMARY"
echo "==== price check started $(date '+%F %T') (parallel=$PARALLEL) ====" | tee -a "$SUMMARY"

days=()
for f in $(find "$OUT_BASE" -path "*/default/1/all.parquet" 2>/dev/null | sort); do
    d=$(dirname "$(dirname "$(dirname "$f")")"); d=$(basename "$d"); days+=("${d//./}")
done
echo "found ${#days[@]} days" | tee -a "$SUMMARY"

check_one() {
    day="$1"
    out=$(python3 "$SCRIPT_DIR/price_check.py" "$day" 2>&1)
    echo "$out" > "$LOG_DIR/$day.log"
    echo "$out" | tail -1
}
export -f check_one
export SCRIPT_DIR LOG_DIR

printf '%s\n' "${days[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'check_one "$@"' _ {} \
    | sort | tee -a "$SUMMARY"

fail=$(grep -c "FAIL" "$SUMMARY")
echo "" | tee -a "$SUMMARY"
echo "==== price check finished $(date '+%F %T') : fail_days=$fail ====" | tee -a "$SUMMARY"
