#!/bin/bash
# Single-Spark memory guard. Run it (e.g. setsid nohup ./memguard-solo.sh &) right after starting vllm_node.
# We set vm.min_free_kbytes=4G, so MemAvailable runs ~4-5 GB below MemFree; guard on MemFree instead.
# Kills vllm_node if truly free RAM < 2 GB. Logs time,avail,free every ~5 s to ~/memguard.csv.
start=$(date +%s); n=0
while docker ps -q --filter name=vllm_node | grep -q . || [ $(( $(date +%s) - start )) -lt 1200 ]; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); fr=$(awk '/MemFree/{print int($2/1048576)}' /proc/meminfo)
  if [ "$fr" -lt 2 ]; then docker kill vllm_node; echo "MEMGUARD KILL vllm_node avail=${a}G free=${fr}G $(date '+%F %T')" >> ~/memguard.log; sync; exit 1; fi
  n=$((n+1)); [ $((n % 5)) -eq 0 ] && { echo "$(date '+%T'),$a,$fr" >> ~/memguard.csv; sync -d ~/memguard.csv; }
  sleep 1
done
