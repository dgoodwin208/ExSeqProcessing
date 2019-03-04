#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

CWD=$PWD
if [ $# -eq 1 ]; then
    CWD="$1"
fi

pushd ${CWD}

TOPLOG=$(\ls -1t top-*.log | head -n 1)
VMLOG=$(\ls -1t nfsiostat-*.log | head -n 1)
IOLOG=$(\ls -1t iostat-*.log | head -n 1)
NFSIOLOG=$(\ls -1t vmstat-*.log | head -n 1)
GPULOG=$(\ls -1t gpu-*.log | head -n 1)

${SCRIPT_DIR}/summary-matlab-proc-in-top-log.py "${TOPLOG}" | tee summary-top.txt
${SCRIPT_DIR}/summary-nfsiostat-log.py "${VMLOG}"    > summary-nfsiostat.csv
${SCRIPT_DIR}/summary-iostat-log.py    "${IOLOG}"    > summary-iostat.csv
${SCRIPT_DIR}/summary-vmstat-log.py    "${NFSIOLOG}" > summary-vmstat.csv
${SCRIPT_DIR}/summary-gpu-log.py       "${GPULOG}"   > summary-gpu.csv

cp -af ${SCRIPT_DIR}/perf-measurement-template.ipynb perf-measurement.ipynb
jupyter nbconvert --to notebook --inplace --execute perf-measurement.ipynb

popd

echo
echo "NOTE: you might have to change nvme device name in perf-measurement.ipynb manually if errors"
echo "      then, run a command below again"
echo "$ jupyter nbconvert --to notebook --inplace --execute perf-measurement.ipynb"
echo

