#!/usr/bin/env bash
# Wait for the sigma=0 tilt chains to finish, then run the SAME tilt sweep at sigma=5 mm.
# Scientific question: the paper shows robustness to 5 mm POSITION error; does it extend to ORIENTATION error,
# and do the two compose? Same protocol, same checkpoints, only --fixed_pos_noise_mm changes.
set -uo pipefail
cd ~/forge_ts/src
while :; do n=$(grep -c "CHAIN_DONE" ~/logs/tilt/master.log 2>/dev/null); [ "$n" -ge 3 ] && break; sleep 120; done
echo "[queue_tilt_n5] sigma=0 chains done, starting sigma=5mm at $(date -Is)"
setsid nohup bash run_tilt_chain.sh 0 ta      "peg gear nut" "0 1 2 3 5 8" 5 256 128 > ~/logs/tilt/chain_gpu0_ta_n5.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash run_tilt_chain.sh 1 student "peg gear"     "0 1 2 3 5 8" 5 128 16  > ~/logs/tilt/chain_gpu1_stu_n5.log 2>&1 < /dev/null &
sleep 5
setsid nohup bash run_tilt_chain.sh 2 student "nut"          "0 1 2 3 5 8" 5 128 16  > ~/logs/tilt/chain_gpu2_stu_n5.log 2>&1 < /dev/null &
wait
