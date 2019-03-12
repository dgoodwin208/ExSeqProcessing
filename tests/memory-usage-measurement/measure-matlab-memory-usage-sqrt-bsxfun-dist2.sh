#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

NCOLS=640
# 1e6 causes out-of-memory
for n in 1e4 0.2e5 0.4e5 0.6e5 0.8e5 1e5; do
    echo "===== sqrt-bsxfun-dist2(): n = $n"
    TOP_LOG=${TOP_LOG_DIR}/top-sqrt-bsxfun-dist2-nrows-${n}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "n=${n};ncols=${NCOLS}; fprintf('# rows=%3.1e, # cols=%d\n',n,ncols); x=rand(n,ncols); y=rand(n,ncols); tic; for i=1:5; disp(i); ret=sqrt(bsxfun(@plus,sum(x.^2,2),sum(y.^2,2)') - 2*(x*y')); ret=[]; end; toc; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

    if [ "$n" == "1e4" ] || [ "$n" == "0.2e5" ] || [ "$n" == "0.4e5" ]; then
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

