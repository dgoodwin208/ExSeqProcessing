#!/bin/bash
# file: tests/test-runPipeline.sh

. tests/test-helper.sh

SHUNIT2_SRC_DIR=~/works/shunit2/source/2.1/src
INPUT_IMAGE_DIR=/mp/nas1/share/ExSEQ/ExSeqCulture-small/input-new
DECONVOLUTION_DIR=1_deconvolution

# =================================================================================================
oneTimeSetUp() {
    if [ ! -d test1_deconv ]; then
        ln -s $INPUT_IMAGE_DIR test1_deconv
    fi
    if [ ! -d vlfeat-0.9.20 ]; then
        ln -s ~/lib/matlab/vlfeat-0.9.20
    fi
    if [ ! -d rajlabimagetools ]; then
        ln -s ~/lib/matlab/rajlabimagetools
    fi

    if [ ! -d test-results ]; then
        mkdir test-results
    fi

    Result_dir=test-results/$(date +%Y%m%d_%H%M%S)
    mkdir $Result_dir

    for d in [2-6]_*
    do
        [ -e "$d" ] || continue
        rm -r "$d"
    done
    if [ -d logs ]; then
        rm -r logs
    fi
}

oneTimeTearDown() {
    if [ -d $DECONVOLUTION_DIR ]; then
        rm $DECONVOLUTION_DIR
    fi
    if [ -d test1_deconv ]; then
        rm test1_deconv
    fi
    if [ -d vlfeat-0.9.20 ]; then
        rm ./vlfeat-0.9.20
    fi
    if [ -d rajlabimagetools ]; then
        rm ./rajlabimagetools
    fi

    for d in [2-6]_* test[2-6]_* test_report
    do
        [ -e "$d" ] || continue
        rm -r "$d"
    done
    if [ -d test_logs ]; then
        rm -r test_logs
    fi
}

# -------------------------------------------------------------------------------------------------

setUp() {
    if [ ! -d $DECONVOLUTION_DIR ]; then
        ln -s $INPUT_IMAGE_DIR $DECONVOLUTION_DIR
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
}

# =================================================================================================
get_values_and_keys() {
    Value[ 1]=$(get_value_by_key "$Log" "# of rounds")
    Value[ 2]=$(get_value_by_key "$Log" "file basename")
    Value[16]=$(get_value_by_key "$Log" "reference round")
    Value[ 3]=$(get_value_by_key "$Log" "processing channels")
    Value[ 4]=$(get_value_by_key "$Log" "registration channel")
    Value[ 5]=$(get_value_by_key "$Log" "warp channels")
    Value[15]=$(get_value_by_key "$Log" "r-1 threshold decision")
    Value[ 6]=$(get_value_by_key "$Log" "deconvolution images")
    Value[18]=$(get_value_by_key "$Log" "color correction images")
    Value[ 7]=$(get_value_by_key "$Log" "normalization images")
    Value[ 8]=$(get_value_by_key "$Log" "registration images")
    Value[ 9]=$(get_value_by_key "$Log" "puncta")
    Value[10]=$(get_value_by_key "$Log" "transcripts")
    Value[11]=$(get_value_by_key "$Log" "Registration project")
    Value[12]=$(get_value_by_key "$Log" "vlfeat lib")
    Value[13]=$(get_value_by_key "$Log" "Raj lab image tools")
    Value[17]=$(get_value_by_key "$Log" "Reporting")
    Value[14]=$(get_value_by_key "$Log" "Log")

    Key[1]=$(get_key_by_value "$Log" "profile-check")
    Key[2]=$(get_key_by_value "$Log" "color-correction")
    Key[3]=$(get_key_by_value "$Log" "normalization")
    Key[4]=$(get_key_by_value "$Log" "registration")
    Key[5]=$(get_key_by_value "$Log" "puncta-extraction")
    Key[6]=$(get_key_by_value "$Log" "transcripts")
    Key[7]=$(get_key_by_value "$Log" "calc-descriptors")
    Key[8]=$(get_key_by_value "$Log" "register-with-descriptors")
}

assert_all_default_values() {
    local skips=()

    if [ $# -ge 2 ]; then
        local check_mode=$1
        shift

        for arg in "$@"
        do
          if [ $check_mode = "skip" ]; then
              skips[$arg]=skip
          fi
        done
    fi

    if [ ! "${skips[1]}" = "skip" ]; then
        assertEquals 12 ${Value[1]}
    fi
    if [ ! "${skips[2]}" = "skip" ]; then
        assertEquals "sa0916dncv" "${Value[2]}"
    fi
    if [ ! "${skips[3]}" = "skip" ]; then
        assertEquals "'chan1','chan2','chan3','chan4'" "${Value[3]}"
    fi
    if [ ! "${skips[4]}" = "skip" ]; then
        assertEquals "summedNorm" "${Value[4]}"
    fi
    if [ ! "${skips[5]}" = "skip" ]; then
        assertEquals "'summedNorm','chan1','chan2','chan3','chan4'" "${Value[5]}"
    fi
    if [ ! "${skips[6]}" = "skip" ]; then
        assertEquals "$PWD/1_deconvolution" "${Value[6]}"
    fi
    if [ ! "${skips[18]}" = "skip" ]; then
        assertEquals "$PWD/2_color-correction" "${Value[18]}"
    fi
    if [ ! "${skips[7]}" = "skip" ]; then
        assertEquals "$PWD/3_normalization" "${Value[7]}"
    fi
    if [ ! "${skips[8]}" = "skip" ]; then
        assertEquals "$PWD/4_registration" "${Value[8]}"
    fi
    if [ ! "${skips[9]}" = "skip" ]; then
        assertEquals "$PWD/5_puncta-extraction" "${Value[9]}"
    fi
    if [ ! "${skips[10]}" = "skip" ]; then
        assertEquals "$PWD/6_transcripts" "${Value[10]}"
    fi
    if [ ! "${skips[11]}" = "skip" ]; then
        reg_proj_dir=$(cd ../Registration && pwd)
        assertEquals "$reg_proj_dir" "${Value[11]}"
    fi
    if [ ! "${skips[12]}" = "skip" ]; then
        vlfeat_dir=$(cd ~/lib/matlab/vlfeat-0.9.20 && pwd)
        assertEquals "$vlfeat_dir" "${Value[12]}"
    fi
    if [ ! "${skips[13]}" = "skip" ]; then
        raj_lab_dir=$(cd ~/lib/matlab/rajlabimagetools && pwd)
        assertEquals "$raj_lab_dir" "${Value[13]}"
    fi
    if [ ! "${skips[14]}" = "skip" ]; then
        log_dir=$(cd ./logs && pwd)
        assertEquals "$log_dir" "${Value[14]}"
    fi
    if [ ! "${skips[15]}" = "skip" ]; then
        assertEquals "auto" ${Value[15]}
    fi
    if [ ! "${skips[16]}" = "skip" ]; then
        assertEquals 1 ${Value[16]}
    fi
    if [ ! "${skips[17]}" = "skip" ]; then
        reporting_dir=$(cd ./logs/imgs && pwd)
        assertEquals "$reporting_dir" "${Value[17]}"
    fi
}

assert_all_default_keys() {
    local skips=()

    if [ $# -ge 2 ]; then
        local check_mode=$1
        shift

        for arg in "$@"
        do
          if [ $check_mode = "skip" ]; then
              skips[$arg]=skip
          fi
        done
    fi

    for((i=1; i<=${#Key[*]}; i++))
    do
        if [ ! "${skips[$i]}" = "skip" ]; then
            assertEquals "" "${Key[$i]}"
        fi
    done
}

# =================================================================================================
testArgument001_default_values() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assert_all_default_values
    assert_all_default_keys
}

# -------------------------------------------------------------------------------------------------
testArgument002_set_roundnum() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -N 8 > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals 8 ${Value[1]}

    # others are default values
    assert_all_default_values skip 1
    assert_all_default_keys
}

testArgument003_set_roundnum_auto() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -N auto > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals 4 ${Value[1]}

    # others are default values
    assert_all_default_values skip 1
    assert_all_default_keys
}

testArgument004_set_file_basename() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -b sa0916slicedncv > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "sa0916slicedncv" "${Value[2]}"

    # others are default values
    assert_all_default_values skip 2
    assert_all_default_keys
}

testArgument005_set_channels() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -c "'ch00','ch01','ch02','ch03'" > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "'ch00','ch01','ch02','ch03'" "${Value[3]}"
    assertEquals "'summedNorm','ch00','ch01','ch02','ch03'" "${Value[5]}"

    # others are default values
    assert_all_default_values skip 3 5
    assert_all_default_keys
}

testArgument006_set_deconvolution_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -d test1_deconv > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test1_deconv" "${Value[6]}"

    # others are default values
    assert_all_default_values skip 6
    assert_all_default_keys
}

testArgument007_set_color_correction_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -C test2_colorcor > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test2_colorcor" "${Value[18]}"

    # others are default values
    assert_all_default_values skip 18
    assert_all_default_keys
}

testArgument008_set_normalization_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -n test3_norm > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test3_norm" "${Value[7]}"

    # others are default values
    assert_all_default_values skip 7
    assert_all_default_keys
}

testArgument009_set_registration_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -r test4_reg > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test4_reg" "${Value[8]}"

    # others are default values
    assert_all_default_values skip 8
    assert_all_default_keys
}

testArgument010_set_puncta_extraction_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -p test5_puncta > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test5_puncta" "${Value[9]}"

    # others are default values
    assert_all_default_values skip 9
    assert_all_default_keys
}

testArgument011_set_set_transcript_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -t test6_transc > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "$PWD/test6_transc" "${Value[10]}"

    # others are default values
    assert_all_default_values skip 10
    assert_all_default_keys
}

testArgument012_set_registration_proj_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -R ../Registration-test > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    reg_proj_dir=$(cd ../Registration-test && pwd)
    assertEquals "$reg_proj_dir" "${Value[11]}"

    # others are default values
    assert_all_default_values skip 11
    assert_all_default_keys
}

testArgument013_set_vlfeat_lib_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -V ./vlfeat-0.9.20 > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    vlfeat_dir=$(cd ./vlfeat-0.9.20 && pwd)
    assertEquals "$vlfeat_dir" "${Value[12]}"

    # others are default values
    assert_all_default_values skip 12
    assert_all_default_keys
}

testArgument014_set_rajlabtools_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -I ./rajlabimagetools > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    rajlabtools_dir=$(cd ./rajlabimagetools && pwd)
    assertEquals "$rajlabtools_dir" "${Value[13]}"

    # others are default values
    assert_all_default_values skip 13
    assert_all_default_keys
}

testArgument015_set_log_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -L test_logs > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    log_dir=$(cd ./test_logs && pwd)
    assertEquals "$log_dir" "${Value[14]}"

    # others are default values
    assert_all_default_values skip 14
    assert_all_default_keys
}

testArgument016_set_threshold_manual_decision_in_round_1() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -m > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "manual" ${Value[15]}

    # others are default values
    assert_all_default_values skip 15
    assert_all_default_keys
}

testArgument017_set_reference_round() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -B 2 > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals 2 ${Value[16]}

    # others are default values
    assert_all_default_values skip 16
    assert_all_default_keys
}

testArgument018_set_reporting_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -i test_report > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    reporting_dir=$(cd ./test_report && pwd)
    assertEquals "$reporting_dir" "${Value[17]}"

    # others are default values
    assert_all_default_values skip 17
    assert_all_default_keys
}

# -------------------------------------------------------------------------------------------------
testArgument100_skip_stage_profile_check() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'profile-check' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 1
}

testArgument101_skip_stage_color_correction() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'color-correction' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[2]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 2
}

testArgument102_skip_stage_normalization() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'normalization' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[3]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 3
}

testArgument103_skip_stage_registration() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'registration' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 4 7 8
}

testArgument104_skip_stage_puncta_extraction() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'puncta-extraction' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[5]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 5
}

testArgument105_skip_stage_transcripts() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'transcripts' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[6]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 6
}

testArgument106_skip_substage_calc_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'calc-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[7]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 7
}

testArgument107_skip_substage_register_with_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'register-with-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 8
}

testArgument108_skip_all_stages() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'profile-check,color-correction,normalization,registration,puncta-extraction,transcripts' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument109_skip_stage_normalization_and_substage_calc_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'normalization,calc-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[7]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 3 7
}

testArgument110_skip_stage_registration_and_substage_calc_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -s 'registration,calc-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
    assert_all_default_keys skip 4 7 8
}


# -------------------------------------------------------------------------------------------------
testArgument111_exec_stage_profile_check() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'profile-check' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals ""     "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument111_exec_stage_color_correction() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'color-correction' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals ""     "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument112_exec_stage_normalization() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'normalization' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals ""     "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument113_exec_stage_registration() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'registration' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals ""     "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals ""     "${Key[7]}"
    assertEquals ""     "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument114_exec_stage_puncta_extraction() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'puncta-extraction' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals ""     "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument115_exec_stage_transcripts() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'transcripts' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals ""     "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument116_exec_substage_calc_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'calc-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals ""     "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals ""     "${Key[7]}"
    assertEquals "skip" "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument117_exec_substage_register_with_descriptors() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'register-with-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals ""     "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals ""     "${Key[8]}"

    # others are default values
    assert_all_default_values
}

testArgument118_exec_stage_normalization_and_registration() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'normalization,registration' > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals ""     "${Key[3]}"
    assertEquals ""     "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals ""     "${Key[7]}"
    assertEquals ""     "${Key[8]}"

    # others are default values
    assert_all_default_values
}

# -------------------------------------------------------------------------------------------------
testArgument200_Error_set_roundnum() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -N a > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "# of rounds is not number" "$Log" | wc -l)
    assertEquals 1 $message
}

testArgument201_Error_no_deconvolution_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    rm $DECONVOLUTION_DIR

    echo 'n' | ./runPipeline.sh > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No deconvolution dir" "$Log" | wc -l)
    assertEquals 1 $message

    ln -s $INPUT_IMAGE_DIR $DECONVOLUTION_DIR
}

testArgument202_Error_no_registration_proj_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -R reg > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No Registration project dir" "$Log" | wc -l)
    assertEquals 1 $message
}

testArgument203_Error_no_registration_proj_matlab_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -R $DECONVOLUTION_DIR > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No MATLAB dir. in Registration project" "$Log" | wc -l)
    assertEquals 1 $message
}

testArgument204_Error_no_registration_proj_scripts_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    mkdir -p dummy_proj/MATLAB

    echo 'n' | ./runPipeline.sh -R dummy_proj > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No scripts dir. in Registration project" "$Log" | wc -l)
    assertEquals 1 $message

    rm -r dummy_proj
}

testArgument205_Error_no_rajlabtools_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -I dummy_proj > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No Raj lab image tools project" "$Log" | wc -l)
    assertEquals 1 $message
}

testArgument206_Error_no_vlfeat_dir() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -V dummy_proj > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No vlfeat library dir" "$Log" | wc -l)
    assertEquals 1 $message
}

testArgument207_Error_no_import_cluster_profiles_sh() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    mkdir -p dummy_proj/{MATLAB,scripts}

    echo 'n' | ./runPipeline.sh -R dummy_proj > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No 'import_cluster_profiles.sh'" "$Log" | wc -l)
    assertEquals 1 $message

    rm -r dummy_proj
}

testArgument208_Error_no_load_experiment_params_m() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    mkdir -p dummy_proj/{MATLAB,scripts}
    touch dummy_proj/scripts/import_cluster_profiles.sh

    echo 'n' | ./runPipeline.sh -R dummy_proj > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No 'loadExperimentParams.m'" "$Log" | wc -l)
    assertEquals 1 $message

    rm -r dummy_proj
}

testArgument209_Error_no_load_params_m() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    mv loadParameters.m{,-orig}

    echo 'n' | ./runPipeline.sh > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "No 'loadParameters.m'" "$Log" | wc -l)
    assertEquals 1 $message

    mv loadParameters.m{-orig,}
}

testArgument210_Error_unacceptable_both_e_and_s_args() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    echo 'n' | ./runPipeline.sh -e 'registration' -s 'calc-descriptors' > $Log 2>&1
    local status=$?
    assertEquals 1 $status

    message=$(grep "cannot use both -e and -s" "$Log" | wc -l)
    assertEquals 1 $message
}

# -------------------------------------------------------------------------------------------------
testRun001_replace_parameters_and_skip_all() {
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    ./runPipeline.sh -y -e ' ' -N 8 -b sa0916slicedncv -B 2 -c "'ch00','ch01','ch02','ch03'" -C test2_colorcor -n test3_norm -r test4_reg -p test5_puncta -t test6_trans -V ./vlfeat-0.9.20 -I ./rajlabimagetools -i test_report > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    get_values_and_keys

    assertEquals "skip" "${Key[1]}"
    assertEquals "skip" "${Key[2]}"
    assertEquals "skip" "${Key[3]}"
    assertEquals "skip" "${Key[4]}"
    assertEquals "skip" "${Key[5]}"
    assertEquals "skip" "${Key[6]}"
    assertEquals "skip" "${Key[7]}"
    assertEquals "skip" "${Key[8]}"


    local param=$(sed -ne 's#params.SAMPLE_NAME = \(.*\);#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "'${Value[2]}_'" "$param"

    local param=$(sed -ne 's#params.DATACHANNEL = \(.*\);#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "'${Value[4]}'" "$param"

    local param=$(sed -ne 's#params.REGISTERCHANNEL = \(.*\);#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "'${Value[4]}'" "$param"

    local param=$(sed -ne 's#params.CHANNELS = \(.*\);.*#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "{${Value[5]}}" "$param"

    local param=$(sed -ne 's#params.INPUTDIR = \(.*\);#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "'${Value[7]}'" "$param"

    local param=$(sed -ne 's#params.OUTPUTDIR = \(.*\);#\1#p' ../Registration/MATLAB/loadExperimentParams.m)
    assertEquals "'${Value[8]}'" "$param"

    local param=$(sed -ne 's#params.deconvolutionImagesDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[6]}'" "$param"

    local param=$(sed -ne 's#params.colorCorrectionImagesDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[18]}'" "$param"

    local param=$(sed -ne 's#params.registeredImagesDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[8]}'" "$param"

    local param=$(sed -ne 's#params.punctaSubvolumeDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[9]}'" "$param"

    local param=$(sed -ne 's#params.transcriptResultsDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[10]}'" "$param"

    local param=$(sed -ne 's#params.reportingDir = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[17]}'" "$param"

    local param=$(sed -ne 's#params.FILE_BASENAME = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "'${Value[2]}'" "$param"

    local param=$(sed -ne 's#params.NUM_ROUNDS = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "${Value[1]}" "$param"

    local param=$(sed -ne 's#params.REFERENCE_ROUND_PUNCTA = \(.*\);#\1#p' ./loadParameters.m)
    assertEquals "${Value[16]}" "$param"

    local param=$(sed -ne "s#run('\(.*\)/toolbox.*#\1#p" ./startup.m)
    assertEquals "${Value[12]}" "$param"

    local param_cnt=$(grep -o "${Value[11]}" ./startup.m | wc -l)
    assertEquals 2 $param_cnt

    local param_cnt=$(grep -o "${Value[13]}" ./startup.m | wc -l)
    assertEquals 1 $param_cnt

    local skip_cnt=$(grep -o 'Skip!' "$Log" | wc -l)
    assertEquals 6 $skip_cnt

    cp -a ../Registration/MATLAB/loadExperimentParams.m ./loadParameters.m ./startup.m ${Result_dir}/${curfunc}/
}

# -------------------------------------------------------------------------------------------------
testRun002_run_pipeline_to_small_data() {
    return
    local curfunc=${FUNCNAME[0]}
    mkdir ${Result_dir}/${curfunc}
    Log=$Result_dir/$curfunc/output.log

    ./runPipeline.sh -N auto -y > $Log 2>&1
    local status=$?
    assertEquals 0 $status

    term_cnt=$(grep -o "pipeline finished" "$Log" | wc -l)
    assertEquals 1 $term_cnt

    cp -a [1-5]_* logs ${Result_dir}/${curfunc}/
}

# load and run shunit2
. $SHUNIT2_SRC_DIR/shunit2

