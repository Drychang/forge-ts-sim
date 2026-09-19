#!/usr/bin/env bash
# Back up force_vla_research/checkpoints (32G, dead VLA models) + T-B teachers
# (22G, insurance) to Pluto NAS, then delete ONLY the local checkpoints copy
# AFTER per-file verification passes. Teachers are backed up but NOT deleted
# (they are the live student-distillation source). IsaacLab/ untouched.
set -u
SRC=/extra_home3/user/force_vla_research
DEST=/media/data/force_vla_research_backup
LOG=/home/user/forge_ts/logs/fvr_backup_master.log
log(){ echo "=== [$(date -Is)] $*" >> "$LOG"; }

mkdir -p "$DEST/checkpoints" "$DEST/tb_teachers"
log "START. Pluto avail: $(df -h /media/Pluto | tail -1 | awk '{print $4}')"

# ---------- 1. backup checkpoints (32G) ----------
log "rsync checkpoints START ($(du -sh $SRC/checkpoints 2>/dev/null | cut -f1))"
rsync -a --no-owner --no-group --delete --stats "$SRC/checkpoints/" "$DEST/checkpoints/" >> "$LOG" 2>&1
rc_ck=$?
log "rsync checkpoints rc=$rc_ck"

# ---------- 2. backup T-B teachers (22G, insurance) ----------
log "rsync teachers START"
rsync -a --no-owner --no-group --delete --stats "$SRC/IsaacLab/logs/rl_games/Forge/" "$DEST/tb_teachers/" >> "$LOG" 2>&1
rc_tb=$?
log "rsync teachers rc=$rc_tb"

# ---------- 3. verify checkpoints backup (per-file path+size manifest) ----------
mfsrc=$(cd "$SRC/checkpoints" && find . -type f -printf '%P %s\n' | LC_ALL=C sort)
mfdst=$(cd "$DEST/checkpoints" && find . -type f -printf '%P %s\n' | LC_ALL=C sort)
nsrc=$(echo "$mfsrc" | grep -c . )
ndst=$(echo "$mfdst" | grep -c . )
log "checkpoints verify: src_files=$nsrc dest_files=$ndst"

# ---------- 4. delete local checkpoints ONLY if verified ----------
if [ "$rc_ck" -eq 0 ] && [ "$mfsrc" = "$mfdst" ]; then
  log "checkpoints VERIFY_OK (rc=0, all $nsrc files match path+size) -- deleting local copy"
  rm -rf "$SRC/checkpoints"
  log "LOCAL checkpoints DELETED. extra_home3 now: $(df -h /extra_home3 | tail -1 | awk '{print $4\" free (\"$5\" used)\"}')"
else
  log "checkpoints VERIFY_FAIL -- NOT deleting (rc=$rc_ck, manifest_match=$([ "$mfsrc" = "$mfdst" ] && echo Y || echo N))"
fi

# ---------- 5. verify teachers backup (kept, never deleted) ----------
tmfsrc=$(cd "$SRC/IsaacLab/logs/rl_games/Forge" && find . -type f -printf '%P %s\n' | LC_ALL=C sort)
tmfdst=$(cd "$DEST/tb_teachers" && find . -type f -printf '%P %s\n' | LC_ALL=C sort)
if [ "$rc_tb" -eq 0 ] && [ "$tmfsrc" = "$tmfdst" ]; then
  log "teachers VERIFY_OK (backup on Pluto, kept LIVE on extra_home3, NOT deleted)"
else
  log "teachers VERIFY_WARN (rc=$rc_tb, manifest_match=$([ "$tmfsrc" = "$tmfdst" ] && echo Y || echo N)) -- backup incomplete, teachers untouched locally"
fi

log "FVR_BACKUP_ALL_DONE"
