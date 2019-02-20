#!/bin/bash

for ang in $(seq 4.71238898038469 0.1308996938995747 5.497787143782138); do
  python trap_arrival_by_wind_live_coarse_dt.py "$ang" &
done
