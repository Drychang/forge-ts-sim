#!/usr/bin/env bash
# P1 nut: train both archs then eval, fully sequential (camera RAM exclusion).
SRC=~/forge_ts/src/student
cd "$SRC"
bash run_p1_nut.sh 0
bash run_p1_nut_eval.sh 0
echo "=== [$(date -Is)] P1_NUT_CHAIN_ALL_DONE ===" >> ~/forge_ts/logs/p1_nut_master.log
