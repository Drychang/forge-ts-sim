#!/usr/bin/env bash
# ARCH v3 (final fair recipe) — ALL three tasks sequentially on GPU2 only.
# GPU0/GPU1 belong to audreych right now; we never contend for them.
# Order: nut first (it is the one v2 broke), then peg, then gear.
SRC=~/forge_ts/src/arch
cd "$SRC"
for t in nut peg gear; do
  bash run_arch_v3.sh "$t" 2
done
echo "=== [$(date -Is)] ARCH_V3_ALL_DONE ===" >> ~/forge_ts/logs/arch_v3_master.log
