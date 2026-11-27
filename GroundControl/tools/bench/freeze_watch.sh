#!/bin/bash
# Freeze diagnosis (Part 2, A0). Runs ON the board, detached:
#   adb push tools/bench/freeze_watch.sh /home/arduino/ && adb shell 'nohup bash ~/freeze_watch.sh 10 > /dev/null 2>&1 &'
# Each round restarts Ground Control and its model runner (the same warm-up a cold boot
# does) and logs, every second until the warm-up finishes (or 400 s): memory, swap, memory
# pressure, CPU temperature, load, and any new kernel OOM/kill/hung-task messages.
# Logs: ~/gc-freeze/round-<n>.log, summary in ~/gc-freeze/summary.txt
N=${1:-10}
APP=/home/arduino/ArduinoApps/ground-control
RUNNER=ground-control-llamacpp-models-runner-1
MAIN=ground-control-main-1
OUT=$HOME/gc-freeze
mkdir -p "$OUT"
echo "round start_utc warm_s outcome peak_used_mb min_avail_mb max_psi_full_avg10 max_temp_c oom_lines" > "$OUT/summary.txt"

temp_c() { cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{printf "%.1f", $1/1000}'; }

for i in $(seq 1 "$N"); do
  LOG="$OUT/round-$i.log"
  K0=$(dmesg 2>/dev/null | wc -l)
  arduino-app-cli app stop "$APP" > /dev/null 2>&1
  docker restart "$RUNNER" > /dev/null 2>&1 || true
  START=$(date +%s)
  arduino-app-cli app start "$APP" > /dev/null 2>&1 &
  echo "t used_mb avail_mb swap_used_mb psi_some10 psi_full10 temp_c load1" > "$LOG"
  PEAK=0; MINAV=999999; PSIMAX=0; TMAX=0; OUTCOME=timeout
  for t in $(seq 1 400); do
    read -r _ _ USED _ _ _ AVAIL < <(free -m | awk '/^Mem:/')
    SWAP=$(free -m | awk '/^Swap:/{print $3}')
    PS=$(awk '/^some/{split($2,a,"="); print a[2]}' /proc/pressure/memory)
    PF=$(awk '/^full/{split($2,a,"="); print a[2]}' /proc/pressure/memory)
    TC=$(temp_c); L1=$(cut -d" " -f1 /proc/loadavg)
    echo "$t $USED $AVAIL $SWAP $PS $PF $TC $L1" >> "$LOG"
    [ "$USED" -gt "$PEAK" ] && PEAK=$USED
    [ "$AVAIL" -lt "$MINAV" ] && MINAV=$AVAIL
    PSIMAX=$(awk -v a="$PSIMAX" -v b="$PF" 'BEGIN{print (b>a)?b:a}')
    TMAX=$(awk -v a="$TMAX" -v b="$TC" 'BEGIN{print (b>a)?b:a}')
    if docker logs --since "${START}" "$MAIN" 2>&1 | grep -q "warm-up done"; then OUTCOME=warm; break; fi
    sleep 1
  done
  WARM=$(( $(date +%s) - START ))
  OOM=$(dmesg 2>/dev/null | tail -n +"$((K0 + 1))" | grep -ciE "oom|killed process|hung_task|blocked for more")
  dmesg 2>/dev/null | tail -n +"$((K0 + 1))" | grep -iE "oom|killed process|hung_task|blocked for more" >> "$LOG"
  echo "$i $(date -u -d @"$START" +%H:%M:%S) $WARM $OUTCOME $PEAK $MINAV $PSIMAX $TMAX $OOM" >> "$OUT/summary.txt"
done
echo done >> "$OUT/summary.txt"
