#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

FUNC=load3DImage_uint16
SIZE_Z=250
for n in 500 1000 1500 2000 2500 3000; do
    echo "===== ${FUNC}(): n = $n"
    TOP_LOG=${TOP_LOG_DIR}/top-${FUNC}-xy-${n}-z-${SIZE_Z}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "x=${n};y=${n};z=${SIZE_Z}; fprintf('size=(%3.1e, %3.1e, %3.1e)\n',x,y,z); \
        d=rand(x,y,z); save3DImage_uint16(d,'test.h5'); pause(1); \
        tic; for i=1:20; disp(i); ret=load3DImage_uint16('test.h5'); ret=[]; end; toc; delete test.h5; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

    TOP_INTERVAL=1

    top -bc -w 400 -d $TOP_INTERVAL -p$MATLAB_PID > $TOP_LOG &
    TOP_PID=$!
    echo "top PID=$TOP_PID"
    echo

    wait $MATLAB_PID
    echo "done"
    kill $TOP_PID
    sleep 1
done

