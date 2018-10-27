#!/bin/bash
# file: tests/test-runPipeline.sh

if [ ! -f tests/test-helper.sh ]; then
    echo "test-helper.sh not exist"
    exit
fi

. tests/test-helper.sh

SHUNIT2_SRC_DIR=~/works/shunit2
INPUT_IMAGE_DIR=/mp/nas1/share/ExSEQ/AutoSeq2/test-data/xy01/1_deconvolution-links
DECONVOLUTION_DIR=1_deconvolution
TEMP_DIR=./tmp-test
TEMP_DIR_TMP=${TEMP_DIR}-tmp

# =================================================================================================
oneTimeSetUp() {
    if [ ! -d vlfeat-0.9.20 ]; then
        ln -s ~/lib/matlab/vlfeat-0.9.20
    fi

    if [ -f loadParameters.m ]; then
        rm -i loadParameters.m
    fi

    Result_dir=test-results/test-runPipeline-$(date +%Y%m%d_%H%M%S)
    mkdir $Result_dir

    for d in [2-6]_*
    do
        [ -e "$d" ] || continue
        rm -r "$d"
    done
    if [ -d logs ]; then
        rm -r logs
    fi

    sed -i.bak -e "s#\(params.tempDir\) *= *.*;#\1 = '${TEMP_DIR}';#" loadParameters.m.template
    mkdir -p "$TEMP_DIR"
    mkdir -p "$TEMP_DIR_TMP"

    TEMP_DIR=$(cd ${TEMP_DIR} && pwd)
    TEMP_DIR_TMP=$(cd ${TEMP_DIR_TMP} && pwd)
}

oneTimeTearDown() {
    if [ -d $DECONVOLUTION_DIR ]; then
        rm -r $DECONVOLUTION_DIR
    fi
    if [ -h vlfeat-0.9.20 ]; then
        rm ./vlfeat-0.9.20
    fi

    if [ -d "$TEMP_DIR" ]; then
        rm -r "$TEMP_DIR"
    fi
    if [ -d "$TEMP_DIR_TMP" ]; then
        rm -r "$TEMP_DIR_TMP"
    fi
}

# -------------------------------------------------------------------------------------------------

setUp() {
    if [ ! -d $DECONVOLUTION_DIR ]; then
        cp -a $INPUT_IMAGE_DIR $DECONVOLUTION_DIR
    fi

    if [ -f loadParameters.m ]; then
        rm loadParameters.m
    fi
}

tearDown() {
    for d in [2-6]_*
    do
        [ -e "$d" ] || continue
        rm -r "$d"
    done
    if [ -d logs ]; then
        rm -r logs
    fi

    if [ -f loadParameters.m ]; then
        rm loadParameters.m
    fi
}

# =================================================================================================
testRun001_run_pipeline_to_small_data() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    local Log=$Result_dir/$curfunc/output.log

    set -m
    ./runPipeline.sh -B 1 -N 4 -G -s base-calling -y > $Log 2>&1
    set +m

    term_cnt=$(grep -o "Err" "$Log" | wc -l)
    assertEquals 0 $term_cnt
    term_cnt=$(grep -o "No such" "$Log" | wc -l)
    assertEquals 0 $term_cnt

    term_cnt=$(grep -o "Pipeline finished" "$Log" | wc -l)
    assertEquals 1 $term_cnt

    cp -a ./loadParameters.m ./startup.m [1-5]_* logs ${Result_dir}/${curfunc}/
}

# load and run shunit2
. $SHUNIT2_SRC_DIR/shunit2

