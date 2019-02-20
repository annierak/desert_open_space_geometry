#!/bin/bash

# for mag in ($(seq 0.4 0.2 3.8 )); do
for mag in $(seq 0.4 0.2 3.8); do
  python trap_arrival_by_wind_live_coarse_dt.py "$mag" &
done
