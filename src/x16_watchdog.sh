#!/usr/bin/env bash
# x16 watchdog — overnight reliability for the forreal pipeline.
#
# PROBLEM it solves: do_collect() has no retry. When a run OOMs (run 0 already did,
# twice), it logs DIED, records "shards=0", and the worker moves on. That run is
# never collected. With 12 runs needed and a hard morning deadline, silent gaps are
# the failure mode that costs the whole night.
#
# FIX: x16_forreal.sh is idempotent — it skips any run whose log carries the
# "COLLECT] DONE:" marker. So simply RE-LAUNCHING x16 retries exactly the runs that
# failed and nothing else. This watchdog relaunches whenever x16 is not running and
# the pipeline has not reported X16_ALL_DONE.
#
# Also guards the ABORT path (PHASE 1 exits if <20 shards) — a relaunch continues
# collecting instead of leaving the night dead.
set -uo pipefail
LOGD=~/forge_ts/logs
MASTER=$LOGD/x16_master.log
WLOG=$LOGD/x16_watchdog.log
SRC=~/forge_ts/src
CHECK_EVERY=300          # 5 min
MAX_RESTARTS=40
DEADLINE_EPOCH=$(date -d "tomorrow 10:00" +%s 2>/dev/null || echo 0)

log(){ echo "[$(date -Is)] $*" >> "$WLOG"; }

log "==================== watchdog start ===================="
log "will relaunch x16_forreal.sh whenever it is down and not ALL_DONE"

n=0
while [ "$n" -lt "$MAX_RESTARTS" ]; do
  sleep "$CHECK_EVERY"

  # Finished for real? stop watching.
  if grep -q "X16_ALL_DONE" "$MASTER" 2>/dev/null; then
    log "X16_ALL_DONE seen — watchdog exiting"; break
  fi

  # Still alive? nothing to do.
  if pgrep -f "bash x16_forreal.sh" > /dev/null 2>&1; then
    continue
  fi

  # Down but not done -> relaunch (idempotent; completed runs are skipped)
  n=$((n+1))
  done_runs=0
  for i in $(seq 0 11); do
    grep -q "COLLECT\] DONE:" "$LOGD/x16_collect_$i.log" 2>/dev/null && done_runs=$((done_runs+1))
  done
  shards=$(ls /media/data/forge_ts_gear_forreal/_run*/shard_*.npz 2>/dev/null | wc -l)
  log "x16 DOWN (restart #$n) — collect done=$done_runs/12 shards=$shards — relaunching"
  echo "[$(date -Is)] WATCHDOG relaunch #$n (collect done=$done_runs/12)" >> "$MASTER"

  rmdir "$LOGD"/x16_gpulocks/g* 2>/dev/null
  rm -rf "$LOGD"/x16_claims/* 2>/dev/null
  ( cd "$SRC" && setsid nohup bash x16_forreal.sh >> "$LOGD/x16_launch.log" 2>&1 < /dev/null & )
  sleep 30
done

log "watchdog exiting after $n restarts"
