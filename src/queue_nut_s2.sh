#!/usr/bin/env bash
# nut aug_v1 seed2, queued behind seed1. Two nut camera trainings at once starved
# the page cache (available RAM 6G, buff/cache 5G) and slowed BOTH to 3417s/epoch
# vs the normal ~530s -- nut is the largest dataset, so it runs one at a time.
while pgrep -f "aug_v1_s[1]" > /dev/null 2>&1; do sleep 300; done
sleep 60
cd ~/forge_ts/src && bash run_aug_seed_one.sh nut 1 2
