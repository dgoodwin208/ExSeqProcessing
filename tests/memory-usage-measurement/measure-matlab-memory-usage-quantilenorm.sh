#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

# 1e9 causes out-of-memory
for n in 1e7 0.2e8 0.4e8 0.6e8 0.8e8 1e8; do
    echo "===== quantilenorm(): n = $n"
    TOP_LOG=${TOP_LOG_DIR}/top-quantilenorm-nrows-${n}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "n=${n};fprintf('# rows=%3.1e\n',n); x=rand(n,4); tic; for i=1:5; disp(i); ret=quantilenorm(x); ret=[]; end; toc; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

    if [ "$n" == "1e7" ] || [ "$n" == "0.2e8" ]; then
        TOP_INTERVAL=1
    else
        TOP_INTERVAL=5
    fi

    top -bc -w 400 -d $TOP_INTERVAL -p$MATLAB_PID > $TOP_LOG &
    TOP_PID=$!
    echo "top PID=$TOP_PID"
    echo

    wait $MATLAB_PID
    echo "done"
    kill $TOP_PID
    sleep 1
done

