#!/bin/bash

TEST_DIR=$1
RIGHT_DATA_DIR=$2

F1=($(\ls $TEST_DIR/*.tif))
F2=($(\ls $RIGHT_DATA_DIR/*.tif))

if [ ${#F1[*]} -ne ${#F2[*]} ]; then
    echo "# of tiffs in TEST_DIR       = ${#F1[*]}"
    echo "# of tiffs in RIGHT_DATA_DIR = ${#F2[*]}"
    exit 1
fi

for((i=0; i<${#F1[*]}; i++)); do
#    echo ${F1[$i]} ${F2[$i]}
    ret=$(diff ${F1[$i]} ${F2[$i]})
#    echo "ret = $ret"

    if [ "$ret" != "" ]; then
        echo "different!"
        echo "F1: ${F1[$i]}"
        echo "F2: ${F2[$i]}"
        exit 1
    fi
done

exit 0

