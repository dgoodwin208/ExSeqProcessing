#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

FUNC=imresize3
SIZE_Z=250
for n in 1000 2000 3000 4000 5000; do
    echo "===== ${FUNC}(): n = $n"
    TOP_LOG=${TOP_LOG_DIR}/top-${FUNC}-0.5-xy-${n}-z-${SIZE_Z}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "x=${n};y=${n};z=${SIZE_Z}; fprintf('size=(%3.1e, %3.1e, %3.1e)\n',x,y,z); d=rand(x,y,z); tic; for i=1:20; disp(i); ret=imresize3(d,1/2,'linear'); ret=[]; end; toc; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

    TOP_INTERVAL=1
#    if [ "$n" == "1e4" ] || [ "$n" == "0.2e5" ] || [ "$n" == "0.4e5" ]; then
#        TOP_INTERVAL=1
#    else
#        TOP_INTERVAL=5
#    fi

    top -bc -w 400 -d $TOP_INTERVAL -p$MATLAB_PID > $TOP_LOG &
    TOP_PID=$!
    echo "top PID=$TOP_PID"
    echo

    wait $MATLAB_PID
    echo "done"
    kill $TOP_PID
    sleep 1
done

