#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

CWD=$PWD
if [ $# -eq 1 ]; then
    CWD="$1"
fi

${SCRIPT_DIR}/summary-matlab-proc-in-top-log.py ${CWD}/top-*.log | tee ${CWD}/summary-top.txt
${SCRIPT_DIR}/summary-nfsiostat-log.py ${CWD}/nfsiostat-*.log > ${CWD}/summary-nfsiostat.csv
${SCRIPT_DIR}/summary-iostat-log.py    ${CWD}/iostat-*.log    > ${CWD}/summary-iostat.csv
${SCRIPT_DIR}/summary-vmstat-log.py    ${CWD}/vmstat-*.log    > ${CWD}/summary-vmstat.csv
${SCRIPT_DIR}/summary-gpu-log.py       ${CWD}/gpu-*.log       > ${CWD}/summary-gpu.csv

cp -af ${SCRIPT_DIR}/perf-measurement-template.ipynb ${CWD}/perf-measurement.ipynb

echo
echo "run a command below"
echo "$ jupyter nbconvert --to notebook --inplace --execute perf-measurement.ipynb"
echo
echo "NOTE: you might have to change nvme device name in perf-measurement.ipynb manually if errors"

