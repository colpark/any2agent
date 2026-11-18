#!/bin/bash
# usage: gpu_gate.sh LOGFILE -- cmd args...   (waits while foreign GPU procs hold > half of 128 GB, polls every 10 min)
LOG=$1; shift; [ "$1" = "--" ] && shift
MYPIDS_RE="^$$\$"
while true; do
  used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{s+=$2} END{print s+0}')
  echo "$(date -u +%FT%TZ) host=$(hostname) foreign_gpu_MiB=$used page-cache-drop=not-permitted(LR-X15)" >> "$LOG"
  [ "$used" -le 65536 ] && break
  echo "$(date -u +%FT%TZ) waiting: > half memory held by other processes" >> "$LOG"; sleep 600
done
echo "$(date -u +%FT%TZ) START $*" >> "$LOG"
nice -n 10 "$@" >> "$LOG" 2>&1; rc=$?
echo "$(date -u +%FT%TZ) END rc=$rc" >> "$LOG"
