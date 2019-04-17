#!/bin/bash

STAGES='color-correction normalization registration puncta-extraction'

stage_idx=2
for s in ${STAGES}; do
    ./runPipeline.sh -y -P -e ${s} | tee run-stage${stage_idx}-${s}-perf.log
    mv logs logs-stage${stage_idx}-${s}

    stage_idx=$(( $stage_idx + 1 ))
done

