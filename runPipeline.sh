#!/bin/bash

function trap_exit_handle() {
    echo "(exit_handle)"
    if [ -n "${lock_pid_file}" ]; then
        rm ${lock_pid_file}
    fi

    print_stage_status
    if [ -n "${PID_PERF_PROFILE}" ]; then
        echo "(kill perf-profile)"
        kill -- -${PID_PERF_PROFILE}
    fi
#    kill 0
}

trap trap_exit_handle EXIT


function print_stage_status() {
    if [ "$printed_stage_status" = "false" ] && [ ${#STAGES_STATUS[*]} -gt 0 ]; then
        local min_sec=${STAGE_SECONDS[0]}
        local max_sec=${min_sec}
        echo "========================================================================="
        echo "Stage summary"
        echo "-------------"
        local total_res="SUCCESS"
        for((i=0; i<${#STAGES[*]}; i++))
        do
            j=$(expr $i + 1)
            printf "%-20s : %-7s  (%s)\n" ${STAGES[i]} "${STAGES_STATUS[i]}" $(print_elapsed_time_from_seconds ${STAGE_SECONDS[i]} ${STAGE_SECONDS[j]})

            if [ "${STAGES_STATUS[i]}" = "ERROR" ]; then
                total_res="FAILURE"
            fi
            if [ -n "${STAGE_SECONDS[j]}" ]; then
                if [ "${min_sec}" -gt "${STAGE_SECONDS[j]}" ]; then
                    min_sec=${STAGE_SECONDS[j]}
                fi
                if [ "${max_sec}" -lt "${STAGE_SECONDS[j]}" ]; then
                    max_sec=${STAGE_SECONDS[j]}
                fi
            fi
        done
        echo "------------------------------------------"
        printf "%-20s : %-7s  (%s)\n" "total" "$total_res" $(print_elapsed_time_from_seconds ${min_sec} ${max_sec})
        echo
    fi
    printed_stage_status="true"
}

function print_elapsed_time_from_seconds() {
    local start_sec=$1
    local end_sec=$2

    if [ -z "${start_sec}" ] || [ -z "${end_sec}" ]; then
        return
    fi

    local ss=$(expr ${end_sec} - ${start_sec})

    local hh=$(expr ${ss} / 3600)
    ss=$(expr ${ss} % 3600)
    local mm=$(expr ${ss} / 60)
    ss=$(expr ${ss} % 60)

    printf "%02d:%02d:%02d" ${hh} ${mm} ${ss}
}

function lowercase() {
    echo $(echo "$1" | tr "[A-Z]" "[a-z]")
}

function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --configure     configure using GUI/CUI"
    echo "  --auto-config   set parameters from input images"
    echo "  -I              input data directory"
    echo "  -O              output data directory"
    echo "  -N              # of rounds"
    echo "  -b              file basename"
    echo "  -B              reference round"
    echo "  -i              reporting directory"
    echo "  -T              temp directory"
    echo "  -L              log directory"
    echo "  -A              acceleration (CPU or GPU_CUDA)"
    echo "  -F              file format of intermediate images (tiff or hdf5)"
    echo "  -J              change additional parameters; A=1,B=2,C=\'c\'"
    echo "  -P              mode to get performance profile"
    echo "  -e              execution stages;  exclusively use for skip stages"
    echo "  -s              skip stages;  setup-cluster,color-correction,normalization,registration,puncta-extraction,base-calling"
    echo "  -y              continue interactive questions as 'yes'"
    echo "  -h              print help"
    exit
}

function check_number() {
    local NUM=$1
    expr $NUM + 1 > /dev/null 2>&1
    if [ $? -ge 2 ]; then
        return 1
    fi
    return 0
}


###### initialization

if [ ! -f ./loadParameters.m.template ]; then
    echo "[ERROR] No 'loadParameters.m.template' in ExSeqProcessing MATLAB"
    exit
fi
if [ ! -f ./loadParameters.m ]; then
    echo "No 'loadParameters.m' in ExSeqProcessing MATLAB. Copy from a template file"
    cp -a ./loadParameters.m{.template,}
    python configuration.py
    if [ $? -ne 0 ]; then
        echo
        echo "Canceled."
        exit
    fi
fi

AUTO_CONFIG=false
for arg in "$@"; do
    if [ "$arg" = "--configure" ]; then
        python configuration.py
        if [ $? -ne 0 ]; then
            echo
            echo "Canceled."
            exit
        fi
    elif [ "$arg" = "--auto-config" ]; then
        AUTO_CONFIG=true
    fi
done

PARAMETERS_FILE=./loadParameters.m

INPUT_FILE_PATH=$(sed -ne "s#params.INPUT_FILE_PATH *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
DECONVOLUTION_DIR=$(sed -ne "s#params.deconvolutionImagesDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
COLOR_CORRECTION_DIR=$(sed -ne "s#params.colorCorrectionImagesDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
NORMALIZATION_DIR=$(sed -ne "s#params.normalizedImagesDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
REGISTRATION_DIR=$(sed -ne "s#params.registeredImagesDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
PUNCTA_DIR=$(sed -ne "s#params.punctaSubvolumeDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
BASE_CALLING_DIR=$(sed -ne "s#params.basecallingResultsDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
REPORTING_DIR=$(sed -ne "s#params.reportingDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
LOG_DIR=$(sed -ne "s#params.logDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
TEMP_DIR=$(sed -ne "s#params.tempDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
LOCK_DIR=/tmp/.exseqproc

FILE_BASENAME=$(sed -ne "s#params.FILE_BASENAME *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
ROUND_NUM=$(sed -ne "s#params.NUM_ROUNDS *= *\(.*\);#\1#p" ${PARAMETERS_FILE})
REFERENCE_ROUND=$(sed -ne "s#params.REFERENCE_ROUND_WARP *= *\(.*\);#\1#p" ${PARAMETERS_FILE})
CHAN_STRS=$(sed -ne "s#params.CHAN_STRS *= *{\(.*\)};#\1#p" ${PARAMETERS_FILE})
USE_GPU_CUDA=$(sed -ne "s#params.USE_GPU_CUDA *= *\(.*\);#\1#p" ${PARAMETERS_FILE})
#IMAGE_EXT=$(sed -ne "s#params.IMAGE_EXT *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
INPUT_IMAGE_EXT=$(sed -ne "s#params.INPUT_IMAGE_EXT *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})

CHAN_ARRAY=($(echo ${CHAN_STRS//\'/} | tr ',' ' '))

if [ -z "$LOG_DIR" ]; then
    LOG_DIR="logs"
fi
if [ -z "$TEMP_DIR" ]; then
    TEMP_DIR="tmp"
fi

PERF_PROFILE=false

NUM_LOGICAL_CORES=$(lscpu | grep ^CPU\(s\) | sed -e "s/[^0-9]*\([0-9]*\)/\1/")


###### getopts

PARAM_KEYS=()
PARAM_VALS=()

while getopts N:b:B:I:O:T:i:L:e:s:-:A:F:J:Pyh OPT
do
    case $OPT in
        N)  ROUND_NUM=$OPTARG
            if ! check_number ${ROUND_NUM}; then
                echo "[ERROR] # of rounds is not number; ${ROUND_NUM}"
                exit
            fi
            ;;
        b)  FILE_BASENAME=$OPTARG
            ;;
        B)  REFERENCE_ROUND=$OPTARG
                if ! check_number ${REFERENCE_ROUND}; then
                    echo "[ERROR] reference round is not number; ${REFERENCE_ROUND}"
                    exit
                fi
            ;;
        I)  INPUT_FILE_PATH=$OPTARG
            ;;
        O)  OUTPUT_FILE_PATH=$OPTARG
            basedir=$(basename ${DECONVOLUTION_DIR})
            DECONVOLUTION_DIR=${OUTPUT_FILE_PATH}/${basedir}
            basedir=$(basename ${COLOR_CORRECTION_DIR})
            COLOR_CORRECTION_DIR=${OUTPUT_FILE_PATH}/${basedir}
            basedir=$(basename ${NORMALIZATION_DIR})
            NORMALIZATION_DIR=${OUTPUT_FILE_PATH}/${basedir}
            basedir=$(basename ${REGISTRATION_DIR})
            REGISTRATION_DIR=${OUTPUT_FILE_PATH}/${basedir}
            basedir=$(basename ${PUNCTA_DIR})
            PUNCTA_DIR=${OUTPUT_FILE_PATH}/${basedir}
            basedir=$(basename ${BASE_CALLING_DIR})
            BASE_CALLING_DIR=${OUTPUT_FILE_PATH}/${basedir}
            ;;
        T)  TEMP_DIR=$OPTARG
            ;;
        i)  REPORTING_DIR=$OPTARG
            ;;
        L)  LOG_DIR=$OPTARG
            ;;
        e)  ARG_EXEC_STAGES=$OPTARG
            ;;
        s)  ARG_SKIP_STAGES=$OPTARG
            ;;
        A)  ACCELERATION=$OPTARG
            if [ $(lowercase "${ACCELERATION}") != "cpu" ] && [ $(lowercase "${ACCELERATION}") != "gpu_cuda" ]; then
                echo "[ERROR] Not support acceleration: ${ACCELERATION}"
                exit
            fi
            ;;
        F)  FORMAT=$OPTARG
            if [ $(lowercase "${FORMAT}") != "tiff" ] && [ $(lowercase "${FORMAT}") != "hdf5" ]; then
                echo "[ERROR] Not support image format: ${FORMAT}"
                exit
            fi
            ;;
        J)  IFS=, read -a param_keyvals <<<$OPTARG
            for((i=0; i<${#param_keyvals[*]}; i++))
            do
                IFS='=' read -a keyval <<<${param_keyvals[i]}
                PARAM_KEYS+=( ${keyval[0]} )
                PARAM_VALS+=( ${keyval[1]} )
            done
            ;;
        P)  PERF_PROFILE=true
            ;;
        y)  QUESTION_ANSW='yes'
            ;;
        h)  usage
            ;;
        \?) usage
            ;;
        -)
    esac
done


###### auto configuration

if [ "$AUTO_CONFIG" = "true" ]; then
    echo "===== auto configuration from input images"
    FILE_BASENAME=$(\ls -1 ${INPUT_FILE_PATH}/*_round001_${CHAN_ARRAY[0]}.tif | grep -v downsample | xargs basename -a | head -n 1 | sed -e 's/\([^_]*\)_round.*/\1/')
    ROUND_NUM=$(\ls -1 ${INPUT_FILE_PATH}/${FILE_BASENAME}_round*_${CHAN_ARRAY[0]}.tif | sed -e 's/.*round\([0-9]*\).*/\1/' | wc -l)

    ## check if all files exist
    for ((i=1; i<=${ROUND_NUM}; i++)); do
        for ((j=0; j<${#CHAN_ARRAY[*]}; j++)); do
            f=$(printf "${INPUT_FILE_PATH}/${FILE_BASENAME}_round%03d_%s.tif" $i ${CHAN_ARRAY[j]})
            if [ ! -f "$f" ]; then
                echo "[ERROR] Not exist file: $f"
                exit
            fi
        done
    done

    ## check reference round
    if [ "$REFERENCE_ROUND" -gt "$ROUND_NUM" ]; then
        echo "[ERROR] REFERENCE_ROUND (${REFERENCE_ROUND}) is not included in # of rounds (${ROUND_NUM})"
        exit
    fi
fi


shift $((OPTIND - 1))


if [ "$ACCELERATION" = "gpu_cuda" ]; then
    USE_GPU_CUDA=true
elif [ "$ACCELERATION" = "cpu" ]; then
    USE_GPU_CUDA=false
fi

if [ "$FORMAT" = "hdf5" ]; then
    INPUT_IMAGE_EXT=h5
elif [ "$FORMAT" = "tiff" ]; then
    INPUT_IMAGE_EXT=tif
fi


###### check temporary files
if [ -n "${TEMP_DIR}" ] && [ -n "$(find ${TEMP_DIR} \( -name \*.bin -o -name \*.h5 -o -name \*.tif \))" ]; then
    echo "Temporary dir. is not empty."

    if [ ! "${QUESTION_ANSW}" = 'yes' ]; then
        echo "Delete [${TEMP_DIR}/*.{bin,h5,tif}]? (y/n)"
        read -sn1 ANSW
    else
        ANSW='y'
    fi

    if [ $ANSW = 'y' -o $ANSW = 'Y' ]; then
        echo -n "Deleting.. "
        find ${TEMP_DIR} -type f \( -name *.bin -o -name *.h5 -o -name *.tif \) -exec rm {} +
        echo "done."
        echo
    fi
fi


###### setup directories

if [ ! -d "${DECONVOLUTION_DIR}" ]; then
    echo "No deconvolution dir."
    echo "mkdir ${DECONVOLUTION_DIR}"
    mkdir "${DECONVOLUTION_DIR}"
fi

if [ ! -d "${COLOR_CORRECTION_DIR}" ]; then
    echo "No color correction dir."
    echo "mkdir ${COLOR_CORRECTION_DIR}"
    mkdir "${COLOR_CORRECTION_DIR}"
fi

if [ ! -d "${NORMALIZATION_DIR}" ]; then
    echo "No normalization dir."
    echo "mkdir ${NORMALIZATION_DIR}"
    mkdir "${NORMALIZATION_DIR}"
fi

if [ ! -d "${REGISTRATION_DIR}" ]; then
    echo "No registration dir."
    echo "mkdir ${REGISTRATION_DIR}"
    mkdir "${REGISTRATION_DIR}"
fi

if [ ! -d "${PUNCTA_DIR}" ]; then
    echo "No puncta-extraction dir."
    echo "mkdir ${PUNCTA_DIR}"
    mkdir "${PUNCTA_DIR}"
fi

if [ ! -d "${BASE_CALLING_DIR}" ]; then
    echo "No base calling dir."
    echo "mkdir ${BASE_CALLING_DIR}"
    mkdir "${BASE_CALLING_DIR}"
fi

if [ -n "${TEMP_DIR}" ] && [ ! -d "${TEMP_DIR}" ]; then
    echo "No temp dir."
    echo "mkdir ${TEMP_DIR}"
    mkdir "${TEMP_DIR}"
fi

if [ ! -d "${REPORTING_DIR}" ]; then
    echo "No reporting dir."
    echo "mkdir -p ${REPORTING_DIR}"
    mkdir -p "${REPORTING_DIR}"
fi

if [ ! -d "${LOG_DIR}" ]; then
    echo "No log dir."
    echo "mkdir ${LOG_DIR}"
    mkdir "${LOG_DIR}"
fi

if [ ! -d "${LOCK_DIR}" ]; then
    echo "No lock dir."
    echo "mkdir ${LOCK_DIR}"
    mkdir "${LOCK_DIR}"
    chmod 777 "${LOCK_DIR}"
fi

# exchange paths to absolute paths
DECONVOLUTION_DIR=$(cd "${DECONVOLUTION_DIR}" && pwd)
COLOR_CORRECTION_DIR=$(cd "${COLOR_CORRECTION_DIR}" && pwd)
NORMALIZATION_DIR=$(cd "${NORMALIZATION_DIR}" && pwd)
REGISTRATION_DIR=$(cd "${REGISTRATION_DIR}" && pwd)
PUNCTA_DIR=$(cd "${PUNCTA_DIR}" && pwd)
BASE_CALLING_DIR=$(cd "${BASE_CALLING_DIR}" && pwd)

REPORTING_DIR=$(cd "${REPORTING_DIR}" && pwd)
LOG_DIR=$(cd "${LOG_DIR}" && pwd)
TEMP_DIR=$(cd "${TEMP_DIR}" && pwd)


# prepare symbolic links for input files
if [ -z "$(find ${INPUT_FILE_PATH} -name \*.${INPUT_IMAGE_EXT})" ]; then
    echo "[ERROR] No input ${INPUT_IMAGE_EXT} files"
    exit
fi

for filename in ${INPUT_FILE_PATH}/*.${INPUT_IMAGE_EXT}; do
    basename=$(basename $filename)
    if [ ! -f "${DECONVOLUTION_DIR}/$basename" ]; then
        ln -s $filename ${DECONVOLUTION_DIR}/
    fi
done


STAGES=("setup-cluster" "color-correction" "normalization" "registration" "puncta-extraction" "base-calling")

# check stages to be skipped and executed
if [ ! "${ARG_EXEC_STAGES}" = "" -a ! "${ARG_SKIP_STAGES}" = "" ]; then
    echo "[ERROR] cannot use both -e and -s"
    exit
fi

if [ ! "${ARG_EXEC_STAGES}" = "" ]; then
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ "${ARG_EXEC_STAGES/${STAGES[i]}}" = "${ARG_EXEC_STAGES}" ]; then
            SKIP_STAGES[i]="skip"
        fi
    done
else
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ ! "${ARG_SKIP_STAGES/${STAGES[i]}}" = "${ARG_SKIP_STAGES}" ]; then
            SKIP_STAGES[i]="skip"
        fi
    done
fi


echo "#########################################################################"
echo "Parameters"
echo "  Running on host        :  "`hostname`
echo "  # of rounds            :  ${ROUND_NUM}"
echo "  file basename          :  ${FILE_BASENAME}"
echo "  reference round        :  ${REFERENCE_ROUND}"
echo "  channels               :  ${CHAN_STRS}"
#echo "  shift channels         :  ${SHIFT_CHANNELS}"
echo "  use GPU_CUDA           :  ${USE_GPU_CUDA}"
echo "  input image ext        :  ${INPUT_IMAGE_EXT}"
#echo "  intermediate image ext :  ${IMAGE_EXT}"
echo
echo "Stages"
for((i=0; i<${#STAGES[*]}; i++))
do
    if [ "${SKIP_STAGES[i]}" = "skip" ]; then
        echo -n "                    skip "
    else
        echo -n "                         "
    fi
    echo ":  ${STAGES[i]}"
done
echo
echo "Directories"
echo "  input images           :  ${INPUT_FILE_PATH}"
echo "  deconvolution images   :  ${DECONVOLUTION_DIR}"
echo "  color correction images:  ${COLOR_CORRECTION_DIR}"
echo "  normalization images   :  ${NORMALIZATION_DIR}"
echo "  registration images    :  ${REGISTRATION_DIR}"
echo "  puncta                 :  ${PUNCTA_DIR}"
echo "  base calling           :  ${BASE_CALLING_DIR}"
echo
echo "  Temporal storage       :  ${TEMP_DIR}"
echo
echo "  Reporting              :  ${REPORTING_DIR}"
echo "  Log                    :  ${LOG_DIR}"
echo
echo "Additional parameter changes"
for((i=0; i<${#PARAM_KEYS[*]}; i++))
do
    param=$(sed -ne "s#.*${PARAM_KEYS[i]} *= *\(.*\);#\1#p" ${PARAMETERS_FILE})
    if [ -n "${param}" ]; then
        echo -n "${PARAM_KEYS[i]} : ${param} --> ${PARAM_VALS[i]}"
        if [ "${param}" = "${PARAM_VALS[i]}" ]; then
            echo " [UNCHANGE]"
        else
            echo
        fi
    else
        unset PARAM_KEYS[i]
    fi
done
echo
echo "#########################################################################"

if [ ! "${QUESTION_ANSW}" = 'yes' ]; then
    echo "OK? (y/n)"
    read -sn1 ANSW
    if [ $ANSW = 'n' -o $ANSW = 'N' ]; then
        echo
        echo 'Canceled.'
        exit
    fi
fi
echo

echo "## symbolic link check"
ls -ld ${DECONVOLUTION_DIR}
ls -ld ${COLOR_CORRECTION_DIR}
ls -ld ${NORMALIZATION_DIR}
ls -ld ${REGISTRATION_DIR}
ls -ld ${PUNCTA_DIR}
ls -ld ${BASE_CALLING_DIR}
echo


if [ "${PERF_PROFILE}" = "true" ]; then
    if [ -n "$(type jupyter 2>&1 | grep 'not found')" ]; then
        echo "WARNING: command 'jupyter' is not installed. a notebook of performance profile will be not created."
    fi
    set -m
    ./tests/perf-profile/get-stats.sh $$ &
    set +m
    PID_PERF_PROFILE=$!
    sleep 1
fi


###### setup startup.m

cat << EOF > startup.m
addpath(genpath('$(pwd)'));

EOF


###### setup MATLAB scripts

sed -e "s#\(params.INPUT_FILE_PATH\) *= *.*;#\1 = '${INPUT_FILE_PATH}';#" \
    -e "s#\(params.deconvolutionImagesDir\) *= *.*;#\1 = '${DECONVOLUTION_DIR}';#" \
    -e "s#\(params.colorCorrectionImagesDir\) *= *.*;#\1 = '${COLOR_CORRECTION_DIR}';#" \
    -e "s#\(params.normalizedImagesDir\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(params.registeredImagesDir\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *.*;#\1 = '${PUNCTA_DIR}';#" \
    -e "s#\(params.basecallingResultsDir\) *= *.*;#\1 = '${BASE_CALLING_DIR}';#" \
    -e "s#\(params.reportingDir\) *= *.*;#\1 = '${REPORTING_DIR}';#" \
    -e "s#\(params.logDir\) *= *.*;#\1 = '${LOG_DIR}';#" \
    -e "s#\(params.FILE_BASENAME\) *= *.*;#\1 = '${FILE_BASENAME}';#" \
    -e "s#\(params.NUM_ROUNDS\) *= *.*;#\1 = ${ROUND_NUM};#" \
    -e "s#\(params.REFERENCE_ROUND_WARP\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.REFERENCE_ROUND_PUNCTA\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.tempDir\) *= *.*;#\1 = '${TEMP_DIR}';#" \
    -e "s#\(params.USE_GPU_CUDA\) *= *.*;#\1 = ${USE_GPU_CUDA};#" \
    -e "s#\(params.INPUT_IMAGE_EXT\) *= *.*;#\1 = '${INPUT_IMAGE_EXT}';#" \
    -e "s#\(params.NUM_LOGICAL_CORES\) *= *.*;#\1 = ${NUM_LOGICAL_CORES};#" \
    -i.back \
    ./loadParameters.m

for((i=0; i<${#PARAM_KEYS[*]}; i++))
do
    if [ -n "${PARAM_KEYS[i]}" ]; then
        sed -e "s#\(${PARAM_KEYS[i]}\) *= *[^;]*;#\1 = ${PARAM_VALS[i]};#" -i ./loadParameters.m
    fi
done

###### clean up flocks

echo "===== clean up lockfiles"
echo "remove a lockfile of no running proc"
num_running_procs=0
for p in $(find ${LOCK_DIR}/ -name run.*.lock)
do
    lock_pid=$(cat $p)
    if [ -z "$(find /proc -maxdepth 1 -name ${lock_pid})" ]; then
        echo "rm $p"
        rm -f $p
    else
        num_running_procs=$(expr ${num_running_procs} + 1)
    fi
done

if [ "${num_running_procs}" -eq 0 ]; then
    echo
    echo "remove unused lockfiles"
    for l in $(find ${LOCK_DIR}/ -name *.lock)
    do
        echo "rm $l"
        rm -f $l
    done
fi
echo

lock_pid_file="${LOCK_DIR}/run.$$.lock"
echo "$$" > ${lock_pid_file}

echo "===== set a lockfile"
echo "${lock_pid_file}"
echo


###### run pipeline

ERR_HDL_PRECODE='try;'
ERR_HDL_POSTCODE=' catch ME; disp(ME.getReport); exit(1); end; exit'

stage_idx=0
printed_stage_status="false"


###### setup a cluster profile
echo "========================================================================="
echo "Setup cluster-profile"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    stage_log=${LOG_DIR}/matlab-setup-cluster-profile.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        setup_cluster_profile(); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!
    STAGES_STATUS[$stage_idx]="DONE"
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# downsampling and color correction
echo "========================================================================="
echo "Downsampling & color correction"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    stage_log=${LOG_DIR}/matlab-downsampling-and-color-correction.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        stage_downsampling_and_color_correction(); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        STAGE_SECONDS[$(($stage_idx + 1))]=$SECONDS
        echo "[ERROR] downsampling and color correction failed."
        exit
    else
        STAGES_STATUS[$stage_idx]="DONE"
    fi
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo


stage_idx=$(( $stage_idx + 1 ))

# normalization
echo "========================================================================="
echo "Normalization"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    stage_log=${LOG_DIR}/matlab-normalization.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        stage_normalization(); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!

    grep -A 40 "^Error" ${LOG_DIR}/matlab-normalization-*.log

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        STAGE_SECONDS[$(($stage_idx + 1))]=$SECONDS
        echo "[ERROR] normalization failed."
        exit
    else
        STAGES_STATUS[$stage_idx]="DONE"
    fi
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))

# registration
echo "========================================================================="
echo "Registration"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    stage_log=${LOG_DIR}/matlab-registration.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        stage_registration(); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!

    grep -A 40 "^Error" ${LOG_DIR}/matlab-reg[12]-*.log

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        STAGE_SECONDS[$(($stage_idx + 1))]=$SECONDS
        echo "[ERROR] registration failed."
        exit
    else
        STAGES_STATUS[$stage_idx]="DONE"
    fi
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# puncta extraction
echo "========================================================================="
echo "Puncta extraction"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    stage_log=${LOG_DIR}/matlab-puncta-extraction.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        stage_puncta_extraction(); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        STAGE_SECONDS[$(($stage_idx + 1))]=$SECONDS
        echo "[ERROR] puncta extraction failed."
        exit
    else
        STAGES_STATUS[$stage_idx]="DONE"
    fi
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# base calling
echo "========================================================================="
echo "Base calling"; date
echo

STAGE_SECONDS[$stage_idx]=$SECONDS
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    stage_log=${LOG_DIR}/matlab-base-calling-making.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r " \
        ${ERR_HDL_PRECODE} \
        loadParameters; basecalling_simple; \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        STAGE_SECONDS[$(($stage_idx + 1))]=$SECONDS
        echo "[ERROR] base calling failed."
        exit
    else
        STAGES_STATUS[$stage_idx]="DONE"
    fi
else
    STAGES_STATUS[$stage_idx]="SKIPPED"
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


STAGE_SECONDS[$stage_idx]=$SECONDS

if [ "${PERF_PROFILE}" = "true" ]; then
    echo "========================================================================="
    echo "Profile logs summary"; date
    kill -- -${PID_PERF_PROFILE}
    PID_PERF_PROFILE=
    sleep 1
    tests/perf-profile/summarize-stat-logs.sh logs
fi

echo "========================================================================="
echo "Concurrency size summary"
echo
grep -e '## .*JOBS' -e '## .*POOL' -e '## .*THREADS' ${LOG_DIR}/matlab*.log | sed -e 's/## /  params./' -e 's/.*-\([a-z]*\)\.log/\1/'
echo

print_stage_status

echo "========================================================================="
echo "Pipeline finished"; date
echo

