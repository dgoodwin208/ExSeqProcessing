#!/bin/bash

trap 'kill 0' EXIT

if [ $# -ge 1 ]; then
    CMD="$@"
else
    CMD=
fi

INTERVAL=5
COUNT=12

LOGDIR=logs
if [ ! -d $LOGDIR ]; then
  mkdir $LOGDIR
else
  echo "WARNING: logs dir exists. OK? (y/n)"
  read -sn1 ANSW
  if [ "$ANSW" = "n" -o "$ANSW" = "N" ]; then
    echo "Canceled."
    exit 1
  fi
fi
echo "Starting.."

if [ -z "$(type lsblk 2>&1 | grep 'not found')" ]; then
    lsblk > $LOGDIR/lsblk.txt
fi

TOPLOG=$LOGDIR/top-$(date '+%Y%m%d').log
VMLOG=$LOGDIR/vmstat-$(date '+%Y%m%d').log
IOLOG=$LOGDIR/iostat-$(date '+%Y%m%d').log
NFSIOLOG=$LOGDIR/nfsiostat-$(date '+%Y%m%d').log
GPULOG=$LOGDIR/gpu-$(date '+%Y%m%d').log

export COLUMNS=200

echo "datetime: $(date '+%Y/%m/%d %H:%M:%S')" > $TOPLOG
# NOTE: -c option works as toggle
#top -bc -d $INTERVAL >> $TOPLOG &
top -b -d $INTERVAL >> $TOPLOG &

vmstat -wt -n $INTERVAL > $VMLOG &

#python -u schedule_cmd.py 1m sensors > $SENSORSLOG &

if [ -z "$(type iostat 2>&1 | grep 'not found')" ]; then
    iostat -xdt $INTERVAL > $IOLOG &
fi

if [ -z "$(type nfsiostat 2>&1 | grep 'not found')" ]; then
    while :; do
        echo "datetime: $(date '+%Y/%m/%d %H:%M:%S')"
        nfsiostat -adps $INTERVAL $COUNT
    done > $NFSIOLOG &
fi

if [ -z "$(type nvidia-smi 2>&1 | grep 'not found')" ]; then
    nvidia-smi --query-gpu=timestamp,name,index,pstate,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,fan.speed --format=csv,nounits -l 1 -f $GPULOG &
fi

TOPCOLS=0
while [ $TOPCOLS -eq 0 ]; do
    TOPCOLS=$(grep "\+$" $TOPLOG | head -n 1 | wc -c)
    sleep 1
done

echo "top column size=$TOPCOLS"
if [ $TOPCOLS -le 130 ]; then
    echo "top column size is too small."
    echo "Please check!"
    exit 1
fi

echo "Now recording..."
echo

if [ -n "$CMD" ]; then
    sleep 1
    echo "Command: $CMD"
    bash -c "$CMD" &
    WAIT_PID=$!
    echo "Waiting for PID=$WAIT_PID..."
    wait $WAIT_PID
    echo "Command is done."
    sleep 10
    echo "Recording is done."
else
    wait
fi
