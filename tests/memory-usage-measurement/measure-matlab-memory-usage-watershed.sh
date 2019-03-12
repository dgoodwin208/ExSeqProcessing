#!/bin/bash

trap 'kill 0' INT TERM

if [ -n "$1" ]; then
    TOP_LOG_DIR="$1"
else
    TOP_LOG_DIR=.
fi
echo "[TOP_LOG_DIR]=${TOP_LOG_DIR}"
echo

FUNC=watershed
SIZE_Z=250
NUM_PT=10000
for n in 500 1000 1500 2000 2500 3000; do
    echo "===== ${FUNC}(): n = $n"
    TOP_LOG=${TOP_LOG_DIR}/top-${FUNC}-xy-${n}-z-${SIZE_Z}.log
    if [ -f ${TOP_LOG} ]; then
        echo "already exists"
        continue
    fi

    matlab -nodisplay -nosplash -r "x=${n};y=${n};z=${SIZE_Z}; fprintf('size=(%3.1e, %3.1e, %3.1e)\n',x,y,z); \
        d=zeros(x,y,z); r=randi([1 x*y*z],${NUM_PT},1); d(r)=1; \
        d=imdilate(d,strel('sphere',2)); D=bwdist(~d); D=-D; D(~d)=Inf; d=[];\
        tic; for i=1:5; disp(i); ret=watershed(D); ret=[]; end; toc; exit" &
    MATLAB_PID=$!
    echo "matlab PID=$MATLAB_PID"

    TOP_INTERVAL=5

    top -bc -w 400 -d $TOP_INTERVAL -p$MATLAB_PID > $TOP_LOG &
    TOP_PID=$!
    echo "top PID=$TOP_PID"
    echo

    wait $MATLAB_PID
    echo "done"
    kill $TOP_PID
    sleep 1
done

