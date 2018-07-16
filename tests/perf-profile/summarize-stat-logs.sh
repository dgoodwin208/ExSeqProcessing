#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

CWD=$PWD
if [ $# -eq 1 ]; then
    CWD="$1"
fi

pushd ${CWD}

${SCRIPT_DIR}/summary-matlab-proc-in-top-log.py top-*.log | tee summary-top.txt
${SCRIPT_DIR}/summary-nfsiostat-log.py nfsiostat-*.log > summary-nfsiostat.csv
${SCRIPT_DIR}/summary-iostat-log.py    iostat-*.log    > summary-iostat.csv
${SCRIPT_DIR}/summary-vmstat-log.py    vmstat-*.log    > summary-vmstat.csv
${SCRIPT_DIR}/summary-gpu-log.py       gpu-*.log       > summary-gpu.csv

cp -af ${SCRIPT_DIR}/perf-measurement-template.ipynb perf-measurement.ipynb
jupyter nbconvert --to notebook --inplace --execute perf-measurement.ipynb

popd

echo
echo "NOTE: you might have to change nvme device name in perf-measurement.ipynb manually if errors"
echo "      then, run a command below again"
echo "$ jupyter nbconvert --to notebook --inplace --execute perf-measurement.ipynb"
echo

