#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

NCOLS=3
N2ROWS=1.4e5
# 1e6 causes out-of-memory
for n1rows in 1e3 0.2e4 0.4e4 0.6e4 0.8e4 1e4; do
    echo "===== pdist2(): n1rows = $n1rows"
    TOP_LOG=${TOP_LOG_DIR}/top-pdist2-ncols-${NCOLS}-n1rows-${n1rows}-n2rows-${N2ROWS}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "ncols=${NCOLS};n1rows=${n1rows};n2rows=${N2ROWS}; fprintf('n1rows=%3.1e, n2rows=%3.1e, ncols=%d\n',n1rows,n2rows,ncols); x=rand(n1rows,ncols); y=rand(n2rows,ncols); tic; for i=1:40; disp(i); ret=pdist2(x,y); ret=[]; end; toc; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

#    if [ "$n1rows" == "1e3" ] || [ "$n1rows" == "0.2e4" ] || [ "$n1rows" == "0.4e4" ]; then
        TOP_INTERVAL=1
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

