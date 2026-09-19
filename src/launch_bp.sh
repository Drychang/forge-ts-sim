#!/usr/bin/env bash
# Launch the peg breaking-point sweep, then gear+nut. Written to a file and scp'd rather
# than passed inline: the nested quoting of NOISES="7.5 10 15 20" did not survive the ssh
# round trip last time and the loop silently ran zero noise levels.
cd ~/forge_ts/src || exit 1

export TASKS=peg
export NOISES="7.5 10 15 20"
./run_t3_state.sh state > ~/forge_ts/logs/t3_peg_bp.log 2>&1

export TASKS="gear nut"
export NOISES="0 1 2.5 5"
./run_t3_state.sh state > ~/forge_ts/logs/t3_gearnut.log 2>&1

echo "CHAIN_DONE $(date +'%F %T')" >> ~/forge_ts/logs/t3_chain.log
