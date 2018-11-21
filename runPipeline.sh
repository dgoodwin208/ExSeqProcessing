#!/bin/bash

function trap_exit_handle() {
    print_stage_status
    if [ -n "${PID_PERF_PROFILE}" ]; then
        kill -- -${PID_PERF_PROFILE}
    fi
    kill 0
    if [ -n "$(find /dev/shm -name sem.$USER*)" ]; then
        rm /dev/shm/sem.$USER*
    fi
}

function print_stage_status() {
    if [ "$printed_stage_status" = "false" ] && [ ${#STAGES_STATUS[*]} -gt 0 ]; then
        echo "========================================================================="
        echo "Stage summary"
        echo "-------------"
        for((i=0; i<${#STAGES[*]}; i++))
        do
            j=$(expr $i + 1)
            printf "%-20s : %-7s  (%s)\n" ${STAGES[i]} "${STAGES_STATUS[i]}" $(print_elapsed_time_from_seconds ${STAGE_SECONDS[i]} ${STAGE_SECONDS[j]})
        done
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

trap trap_exit_handle EXIT


function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --configure     Configure using GUI"
    echo "  -N              # of rounds"
    echo "  -b              file basename"
    echo "  -B              reference round"
    echo "  -d              deconvolution image directory"
    echo "  -C              color correction image directory"
    echo "  -n              normalization image directory"
    echo "  -r              registration image directory"
    echo "  -p              puncta extraction directory"
    echo "  -t              base calling directory"
    echo "  -i              reporting directory"
    echo "  -T              temp directory (default: not use temp dir)"
    echo "  -L              log directory"
    echo "  -G              use GPU and CUDA (default: no)"
    echo "  -H              use HDF5 format for intermediate files (default: no)"
    echo "  -J              set # of concurrent jobs for color-correction, normalization, calc-desc, reg-with-corr, affine-transform-in-reg, puncta-extraction;  5,10,10,4,4,10"
    echo "  -P              mode to get performance profile"
    echo "  -e              execution stages;  exclusively use for skip stages"
    echo "  -s              skip stages;  setup-cluster,color-correction,normalization,registration,puncta-extraction,base-calling"
    echo "  -y              continue interactive questions"
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

export TZ=America/New_York


REPORTING_DIR=logs/imgs

if [ -f ./loadParameters.m ]; then
    PARAMETERS_FILE=./loadParameters.m
else
    PARAMETERS_FILE=./loadParameters.m.template
fi

DOWN_SAMPLING_MAX_POOL_SIZE=$(sed -ne "s#params.DOWN_SAMPLING_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
COLOR_CORRECTION_MAX_RUN_JOBS=$(sed -ne "s#params.COLOR_CORRECTION_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
COLOR_CORRECTION_MAX_POOL_SIZE=$(sed -ne "s#params.COLOR_CORRECTION_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
COLOR_CORRECTION_MAX_THREADS=$(sed -ne "s#params.COLOR_CORRECTION_MAX_THREADS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
NORMALIZATION_MAX_RUN_JOBS=$(sed -ne "s#params.NORM_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
NORMALIZATION_MAX_POOL_SIZE=$(sed -ne "s#params.NORM_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
NORMALIZATION_MAX_THREADS=0
CALC_DESC_MAX_RUN_JOBS=$(sed -ne "s#params.CALC_DESC_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
CALC_DESC_MAX_POOL_SIZE=$(sed -ne "s#params.CALC_DESC_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
CALC_DESC_MAX_THREADS=0
REG_CORR_MAX_RUN_JOBS=$(sed -ne "s#params.REG_CORR_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
REG_CORR_MAX_POOL_SIZE=$(sed -ne "s#params.REG_CORR_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
REG_CORR_MAX_THREADS=$(sed -ne "s#params.REG_CORR_MAX_THREADS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
AFFINE_MAX_RUN_JOBS=$(sed -ne "s#params.AFFINE_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
AFFINE_MAX_POOL_SIZE=$(sed -ne "s#params.AFFINE_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
AFFINE_MAX_THREADS=$(sed -ne "s#params.AFFINE_MAX_THREADS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
TPS3DWARP_MAX_RUN_JOBS=$(sed -ne "s#params.TPS3DWARP_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
TPS3DWARP_MAX_POOL_SIZE=$(sed -ne "s#params.TPS3DWARP_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
TPS3DWARP_MAX_THREADS=$(sed -ne "s#params.TPS3DWARP_MAX_THREADS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
APPLY3DTPS_MAX_RUN_JOBS=$(sed -ne "s#params.APPLY3DTPS_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
APPLY3DTPS_MAX_POOL_SIZE=$(sed -ne "s#params.APPLY3DTPS_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
APPLY3DTPS_MAX_THREADS=$(sed -ne "s#params.APPLY3DTPS_MAX_THREADS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
PUNCTA_MAX_RUN_JOBS=$(sed -ne "s#params.PUNCTA_MAX_RUN_JOBS *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
PUNCTA_MAX_POOL_SIZE=$(sed -ne "s#params.PUNCTA_MAX_POOL_SIZE *= *\([0-9]*\);#\1#p" ${PARAMETERS_FILE})
PUNCTA_MAX_THREADS=0
REGISTRATION_CHANNELS=$(sed -ne "s#regparams.CHANNELS *= *{\(.*\)};#\1#p" ${PARAMETERS_FILE})



if [ $# -gt 0 ]; then
    if [ $1 = "--configure" ]; then
        python configuration.py
    fi
fi

if [ ! -f ./configuration.cfg ]; then
    python configuration.py
fi


while IFS= read -r line
do
        # display $line or do somthing with $line
    field=$(echo $line | awk -F  "=" '{print $1}')
    value=$(echo $line | awk -F  "=" '{print $2}')
    if [ $field = "input_path" ]; then
        INPUT_FILE_PATH=$value
    elif [ $field = "basename" ]; then
        FILE_BASENAME=$value
    elif [ $field = "channels" ]; then
        CHANNELS=$value
    elif [ $field = "output_path" ]; then
        OUTPUT_FILE_PATH=$value
    elif [ $field = "format" ]; then
        FORMAT=$value
    elif [ $field = "log_path" ]; then
        LOG_DIR=$value
    elif [ $field = "tmp_path" ]; then
        USE_TMP_FILES=true
        TEMP_DIR=$value
    elif [ $field = "total_rounds" ]; then
        ROUND_NUM=$value
    elif [ $field = "reference_round" ]; then
        REFERENCE_ROUND=$value
    elif [ $field = "acceleration" ]; then
        ACCELERATION=$value
    fi
done < configuration.cfg


echo "INPUT_FILE_PATH: $INPUT_FILE_PATH"
echo "FILE_BASENAME: $FILE_BASENAME"
echo "OUTPUT_FILE_PATH: $OUTPUT_FILE_PATH"
echo "FORMAT: $FORMAT"
echo "LOG_DIR: $LOG_DIR"
echo "TEMP_DIR: $TEMP_DIR"
echo "ROUND_NUM: $ROUND_NUM"
echo "REFERENCE_ROUND: $REFERENCE_ROUND"
echo "ACCELERATION: $ACCELERATION"



CHANNEL_ARRAY=($(echo ${CHANNELS//\'/} | tr ',' ' '))
REGISTRATION_CHANNEL_ARRAY=($(echo ${REGISTRATION_CHANNELS//\'/} | tr ',' ' '))
REGISTRATION_SAMPLE=${FILE_BASENAME}_

if [ $ACCELERATION = 'gpu' ]; then
    USE_GPU_CUDA=true
else
    USE_GPU_CUDA=false
fi

if [ $FORMAT = 'tiff' ]; then
    USE_HDF5=false
else
    USE_HDF5=true
fi


PERF_PROFILE=false

NUM_LOGICAL_CORES=$(lscpu | grep ^CPU\(s\) | sed -e "s/[^0-9]*\([0-9]*\)/\1/")

###### getopts

while getopts N:b:B:d:C:n:r:p:t:T:i:L:e:s:-:GHJ:Pyh OPT
do
    case $OPT in
        N)  ROUND_NUM=$OPTARG
            if ! check_number ${ROUND_NUM}; then
                echo "# of rounds is not number; ${ROUND_NUM}"
                exit
            fi
            ;;
        b)  FILE_BASENAME=$OPTARG
            REGISTRATION_SAMPLE=${FILE_BASENAME}_
            ;;
        B)  REFERENCE_ROUND=$OPTARG
                if ! check_number ${REFERENCE_ROUND}; then
                    echo "reference round is not number; ${REFERENCE_ROUND}"
                    exit
                fi
            ;;
        d)  DECONVOLUTION_DIR=$OPTARG
            ;;
        C)  COLOR_CORRECTION_DIR=$OPTARG
            ;;
        n)  NORMALIZATION_DIR=$OPTARG
            ;;
        r)  REGISTRATION_DIR=$OPTARG
            ;;
        p)  PUNCTA_DIR=$OPTARG
            ;;
        t)  BASE_CALLING_DIR=$OPTARG
            ;;
        T)  TEMP_DIR=$OPTARG
            USE_TMP_FILES=true
            ;;
        i)  REPORTING_DIR=$OPTARG
            ;;
        L)  LOG_DIR=$OPTARG
            ;;
        e)  ARG_EXEC_STAGES=$OPTARG
            ;;
        s)  ARG_SKIP_STAGES=$OPTARG
            ;;
        G)  USE_GPU_CUDA=true
            ;;
        H)  USE_HDF5=true
            ;;
        J)  IFS=, read NJOBS_CC NJOBS_NORM NJOBS_CALCD NJOBS_REGC NJOBS_AFFINE TMP <<<$OPTARG
            if [ -n "${NJOBS_CC}" ]; then
                if check_number ${NJOBS_CC}; then
                    if [ ${NJOBS_CC} -eq 0 ]; then
                        echo "# of jobs for color-correction is zero"
                        exit
                    fi
                    COLOR_CORRECTION_MAX_RUN_JOBS=${NJOBS_CC}
                else
                    echo "${NJOBS_CC} is not number"
                    exit
                fi
            fi
            if [ -n "${NJOBS_NORM}" ]; then
                if check_number ${NJOBS_NORM}; then
                    if [ ${NJOBS_NORM} -eq 0 ]; then
                        echo "# of jobs for normalization is zero"
                        exit
                    fi
                    NORMALIZATION_MAX_RUN_JOBS=${NJOBS_NORM}
                else
                    echo "${NJOBS_NORM} is not number"
                    exit
                fi
            fi
            if [ -n "${NJOBS_CALCD}" ]; then
                if check_number ${NJOBS_CALCD}; then
                    if [ ${NJOBS_CALCD} -eq 0 ]; then
                        echo "# of jobs for calc-descriptors is zero"
                        exit
                    fi
                    CALC_DESC_MAX_RUN_JOBS=${NJOBS_CALCD}
                else
                    echo "${NJOBS_CALCD} is not number"
                    exit
                fi
            fi
            if [ -n "${NJOBS_REGC}" ]; then
                if check_number ${NJOBS_REGC}; then
                    if [ ${NJOBS_REGC} -eq 0 ]; then
                        echo "# of jobs for reg-with-correspondences is zero"
                        exit
                    fi
                    REG_CORR_MAX_RUN_JOBS=${NJOBS_REGC}
                else
                    echo "${NJOBS_REGC} is not number"
                    exit
                fi
            fi
            if [ -n "${NJOBS_AFFINE}" ]; then
                if check_number ${NJOBS_AFFINE}; then
                    if [ ${NJOBS_AFFINE} -eq 0 ]; then
                        echo "# of jobs for affine-transform-in-reg is zero"
                        exit
                    fi
                    AFFINE_MAX_RUN_JOBS=${NJOBS_AFFINE}
                else
                    echo "${NJOBS_AFFINE} is not number"
                    exit
                fi
            fi
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

shift $((OPTIND - 1))


###### clear semaphores

if [ -n "$(find /dev/shm -name sem.$USER*)" ]; then
    rm /dev/shm/sem.$USER*
fi


###### check files

if [ ! -f ./loadParameters.m.template ]; then
    echo "No 'loadParameters.m.template' in ExSeqProcessing MATLAB"
    exit
fi
if [ ! -f ./loadParameters.m ]; then
    echo "No 'loadParameters.m' in ExSeqProcessing MATLAB. Copy from a template file"
    cp -a ./loadParameters.m{.template,}
fi


###### check temporary files
if [ -n "$(find ${TEMP_DIR} -type d -not -empty)" ]; then
    echo "Temporary dir. is not empty."

    if [ ! "${QUESTION_ANSW}" = 'yes' ]; then
        echo "Delete [${TEMP_DIR}/*]? (y/n)"
        read -sn1 ANSW
    else
        ANSW='y'
    fi

    if [ $ANSW = 'y' -o $ANSW = 'Y' ]; then
        echo -n "Deleting.. "
        find ${TEMP_DIR} -type f -exec rm '{}' \;
        echo "done."
        echo
    fi
fi


###### setup directories


DECONVOLUTION_DIR=${OUTPUT_FILE_PATH}/1_deconvolution
COLOR_CORRECTION_DIR=${OUTPUT_FILE_PATH}/2_color-correction
NORMALIZATION_DIR=${OUTPUT_FILE_PATH}/3_normalization
REGISTRATION_DIR=${OUTPUT_FILE_PATH}/4_registration
PUNCTA_DIR=${OUTPUT_FILE_PATH}/5_puncta-extraction
BASE_CALLING_DIR=${OUTPUT_FILE_PATH}/6_base-calling

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

if [ ! -d "${TEMP_DIR}" ]; then
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

# exchange paths to absolute paths
DECONVOLUTION_DIR=$(cd "${DECONVOLUTION_DIR}" && pwd)
COLOR_CORRECTION_DIR=$(cd "${COLOR_CORRECTION_DIR}" && pwd)
NORMALIZATION_DIR=$(cd "${NORMALIZATION_DIR}" && pwd)
REGISTRATION_DIR=$(cd "${REGISTRATION_DIR}" && pwd)
PUNCTA_DIR=$(cd "${PUNCTA_DIR}" && pwd)
BASE_CALLING_DIR=$(cd "${BASE_CALLING_DIR}" && pwd)

REPORTING_DIR=$(cd "${REPORTING_DIR}" && pwd)
LOG_DIR=$(cd "${LOG_DIR}" && pwd)

for filename in ${INPUT_FILE_PATH}/*.tif; do
    basename=$(basename $filename)
    if [ ! -f "${DECONVOLUTION_DIR}/$basename" ]; then
        ln -s $filename ${DECONVOLUTION_DIR}/
    fi
done


if [ "$USE_HDF5" = "true" ]; then
    IMAGE_EXT=h5
else
    IMAGE_EXT=tif
fi

STAGES=("setup-cluster" "color-correction" "normalization" "registration" "puncta-extraction" "base-calling")

# check stages to be skipped and executed
if [ ! "${ARG_EXEC_STAGES}" = "" -a ! "${ARG_SKIP_STAGES}" = "" ]; then
    echo "cannot use both -e and -s"
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
echo "  channels               :  ${CHANNELS}"
echo "  use GPU_CUDA           :  ${USE_GPU_CUDA}"
echo "  intermediate image ext :  ${IMAGE_EXT}"
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
echo "  deconvolution images   :  ${DECONVOLUTION_DIR}"
echo "  color correction images:  ${COLOR_CORRECTION_DIR}"
echo "  normalization images   :  ${NORMALIZATION_DIR}"
echo "  registration images    :  ${REGISTRATION_DIR}"
echo "  puncta                 :  ${PUNCTA_DIR}"
echo "  base calling           :  ${BASE_CALLING_DIR}"
echo
echo "  Temporal storage       :  "$(if [ "${USE_TMP_FILES}" = "true" ]; then echo ${TEMP_DIR}; else echo "(on-memory)";fi)
echo
echo "  Reporting              :  ${REPORTING_DIR}"
echo "  Log                    :  ${LOG_DIR}"
echo
echo "========================================================================="
echo "Concurrency: # of parallel jobs, workers/job, threads/worker"
echo "  # of logical cores     :  ${NUM_LOGICAL_CORES}"

printf "  down-sampling          :  --,%2d,--\n" ${DOWN_SAMPLING_MAX_POOL_SIZE}
printf "  color-correction       :  %2d,%2d,%2d\n" ${COLOR_CORRECTION_MAX_RUN_JOBS} ${COLOR_CORRECTION_MAX_POOL_SIZE} ${COLOR_CORRECTION_MAX_THREADS}
printf "  normalization          :  %2d,%2d,%2d\n" ${NORMALIZATION_MAX_RUN_JOBS} ${NORMALIZATION_MAX_POOL_SIZE} ${NORMALIZATION_MAX_THREADS}
printf "  registration           :\n"
printf "    calc-descriptors     :  %2d,%2d,%2d\n" ${CALC_DESC_MAX_RUN_JOBS} ${CALC_DESC_MAX_POOL_SIZE} ${CALC_DESC_MAX_THREADS}
printf "    reg-with-corres.     :  %2d,%2d,%2d\n" ${REG_CORR_MAX_RUN_JOBS} ${REG_CORR_MAX_POOL_SIZE} ${REG_CORR_MAX_THREADS}
printf "    affine-transforms    :  %2d,%2d,%2d\n" ${AFFINE_MAX_RUN_JOBS} ${AFFINE_MAX_POOL_SIZE} ${AFFINE_MAX_THREADS}
printf "    calc-3DTPS-warp      :  %2d,%2d,%2d\n" ${TPS3DWARP_MAX_RUN_JOBS} ${TPS3DWARP_MAX_POOL_SIZE} ${TPS3DWARP_MAX_THREADS}
printf "    apply-3DTPS          :  %2d,%2d,%2d\n" ${APPLY3DTPS_MAX_RUN_JOBS} ${APPLY3DTPS_MAX_POOL_SIZE} ${APPLY3DTPS_MAX_THREADS}
#printf "  puncta-extraction      :  %2d,%2d,%2d\n" ${PUNCTA_MAX_RUN_JOBS} ${PUNCTA_MAX_POOL_SIZE} ${PUNCTA_MAX_THREADS}
echo
echo "#########################################################################"
echo

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
    set -m
    ./tests/perf-profile/get-stats.sh &
    set +m
    PID_PERF_PROFILE=$!
    sleep 1
fi


###### setup startup.m

cat << EOF > startup.m
addpath(genpath('$(pwd)'));

EOF


###### setup MATLAB scripts

sed -e "s#\(regparams.INPUTDIR\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(regparams.OUTPUTDIR\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(regparams.FIXED_RUN\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.deconvolutionImagesDir\) *= *.*;#\1 = '${DECONVOLUTION_DIR}';#" \
    -e "s#\(params.colorCorrectionImagesDir\) *= *.*;#\1 = '${COLOR_CORRECTION_DIR}';#" \
    -e "s#\(params.normalizedImagesDir\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(params.registeredImagesDir\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *.*;#\1 = '${PUNCTA_DIR}';#" \
    -e "s#\(params.basecallingResultsDir\) *= *.*;#\1 = '${BASE_CALLING_DIR}';#" \
    -e "s#\(params.reportingDir\) *= *.*;#\1 = '${REPORTING_DIR}';#" \
    -e "s#\(params.FILE_BASENAME\) *= *.*;#\1 = '${FILE_BASENAME}';#" \
    -e "s#\(params.NUM_ROUNDS\) *= *.*;#\1 = ${ROUND_NUM};#" \
    -e "s#\(params.REFERENCE_ROUND_WARP\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.REFERENCE_ROUND_PUNCTA\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.tempDir\) *= *.*;#\1 = '${TEMP_DIR}';#" \
    -e "s#\(params.USE_TMP_FILES\) *= *.*;#\1 = ${USE_TMP_FILES};#" \
    -e "s#\(params.IMAGE_EXT\) *= *.*;#\1 = '${IMAGE_EXT}';#" \
    -e "s#\(params.NUM_LOGICAL_CORES\) *= *.*;#\1 = ${NUM_LOGICAL_CORES};#" \
    -e "s#\(params.DOWN_SAMPLING_MAX_POOL_SIZE\) *= *.*;#\1 = ${DOWN_SAMPLING_MAX_POOL_SIZE};#" \
    -e "s#\(params.COLOR_CORRECTION_MAX_RUN_JOBS\) *= *.*;#\1 = ${COLOR_CORRECTION_MAX_RUN_JOBS};#" \
    -e "s#\(params.NORM_MAX_RUN_JOBS\) *= *.*;#\1 = ${NORMALIZATION_MAX_RUN_JOBS};#" \
    -e "s#\(params.CALC_DESC_MAX_RUN_JOBS\) *= *.*;#\1 = ${CALC_DESC_MAX_RUN_JOBS};#" \
    -e "s#\(params.REG_CORR_MAX_RUN_JOBS\) *= *.*;#\1 = ${REG_CORR_MAX_RUN_JOBS};#" \
    -e "s#\(params.AFFINE_MAX_RUN_JOBS\) *= *.*;#\1 = ${AFFINE_MAX_RUN_JOBS};#" \
    -i.back \
    ./loadParameters.m


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
    (
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-setup-cluster-profile.log -r "${ERR_HDL_PRECODE} setup_cluster_profile(); ${ERR_HDL_POSTCODE}"
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

    #(
    #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-copy-scopenames-to-regnames.log -r "${ERR_HDL_PRECODE} copy_scope_names_to_reg_names; ${ERR_HDL_POSTCODE}"
    #) & wait $!

    stage_log=${LOG_DIR}/matlab-downsample-all.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
        run('downsample_all.m'); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!
    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        echo "[ERROR] downsample_all failed."
        exit
    fi


    stage_log=${LOG_DIR}/matlab-color-correction.log
    (
    if [ ${USE_GPU_CUDA} == "true" ]; then
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            colorcorrection_3D_cuda(${ROUND_NUM}); \
            postcheck_color_correction(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    else
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            for i=1:${ROUND_NUM}; colorcorrection_3D(i); end; \
            postcheck_color_correction(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    fi
    if ls matlab-color-correction-*.log > /dev/null 2>&1; then
        grep -A 40 Error matlab-color-correction-*.log
        mv matlab-color-correction-*.log ${LOG_DIR}/
    else
        echo "No job log files."
    fi
    ) & wait $!
    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        echo "[ERROR] colorcollrection_3D failed."
        exit
    fi


    stage_log=${LOG_DIR}/matlab-downsample-apply.log
    (
    matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
        run('downsample_applycolorshiftstofullres.m'); \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!
    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        echo "[ERROR] downsample_apply failed."
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

    cwd=$PWD
    pushd ${COLOR_CORRECTION_DIR}
    for f in ${DECONVOLUTION_DIR}/*ch00.${IMAGE_EXT}
    do
        if [ ! -f $(basename $f) ]; then
            ln -s ${f/$cwd/..}
        fi
    done
    popd

    if [ ${USE_GPU_CUDA} == "true" ]; then
        stage_log=${LOG_DIR}/matlab-normalization.log
        (
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            normalization_cuda('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM}); \
            postcheck_normalization(${ROUND_NUM},true); \
            postcheck_normalization(${ROUND_NUM},false); \
            ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            grep -A 40 Error matlab-normalization-*.log
            mv matlab-normalization-*.log ${LOG_DIR}/
        else
            echo "No job log files."
        fi
        ) & wait $!
        # check result
        if [ "$(grep '\[DONE\]' ${stage_log} | wc -l)" -ne 2 ]; then
            STAGES_STATUS[$stage_idx]="ERROR"
            echo "[ERROR] normalization_cuda failed."
            exit
        else
            STAGES_STATUS[$stage_idx]="DONE"
        fi
    else
        stage_log=${LOG_DIR}/matlab-normalization.log
        (
        #Process the full-resolution data
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},false); \
            postcheck_normalization(${ROUND_NUM},false); \
        ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            grep -A 40 Error matlab-normalization-*.log
            mv matlab-normalization-*.log ${LOG_DIR}/
        else
            echo "No job log files."
        fi
        ) & wait $!
        # check result
        if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
            STAGES_STATUS[$stage_idx]="ERROR"
            echo "[ERROR] normalization of full-size images failed."
            exit
        fi

        stage_log=${LOG_DIR}/matlab-normalization-downsample.log
        (
        #Process the downsampled data, if specified
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            loadParameters; if params.DO_DOWNSAMPLE; \
            normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},true);end; \
            postcheck_normalization(${ROUND_NUM},true); \
            ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            mkdir -p ${LOG_DIR}/downsample
            grep -A 40 Error matlab-normalization-*.log
            mv matlab-normalization-*.log ${LOG_DIR}/downsample/
        else
            echo "No job log files."
        fi
        ) & wait $!
        # check result
        if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
            STAGES_STATUS[$stage_idx]="ERROR"
            echo "[ERROR] normalization of downsampled-size images failed."
            exit
        else
            STAGES_STATUS[$stage_idx]="DONE"
        fi
    fi


    # prepare normalized channel images for warp
    for((i=0; i<${#CHANNEL_ARRAY[*]}; i++))
    do
        for f in $(\ls ${COLOR_CORRECTION_DIR}/*_${CHANNEL_ARRAY[i]}.${IMAGE_EXT})
        do
            round_num=$(( $(echo $f | sed -ne "s/.*_round0*\([0-9]\+\)_.*.${IMAGE_EXT}/\1/p") ))
            if [ $round_num -eq 0 ]; then
                echo "round number is wrong."
            fi

            zero_pad_round_num=$(printf "round%03d" $round_num)
            if [[ $f = *"downsample"* ]]; then
                # prepare downsampled normalized channel images for warp
                normalized_ch_downsample_file=${NORMALIZATION_DIR}/${FILE_BASENAME}-downsample_${zero_pad_round_num}_${CHANNEL_ARRAY[i]}.${IMAGE_EXT}

                if [ ! -f $normalized_ch_downsample_file ]; then
                    ln -s ${f/$PWD/..} $normalized_ch_downsample_file
                fi
            else
                # prepare full-sized normalized channel images for warp
                normalized_ch_file=${NORMALIZATION_DIR}/${FILE_BASENAME}_${zero_pad_round_num}_${CHANNEL_ARRAY[i]}.${IMAGE_EXT}

                if [ ! -f $normalized_ch_file ]; then
                    ln -s ${f/$PWD/..} $normalized_ch_file
                fi
            fi
        done
    done
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

    stage_log=${LOG_DIR}/matlab-calcDesc-group.log
    (
    rounds=$(seq -s' ' 1 ${ROUND_NUM})
    # calculateDescriptors for all rounds in parallel
    if [ ${USE_GPU_CUDA} == "true" ]; then
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            calculateDescriptorsCUDAInParallel([$rounds]); \
            postcheck_calculateDescriptors(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    else
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            calculateDescriptorsInParallel([$rounds]); \
            postcheck_calculateDescriptors(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    fi

    if ls matlab-calcDesc-*.log > /dev/null 2>&1; then
        grep -A 40 Error matlab-calcDesc-*.log
        mv matlab-calcDesc-*.log ${LOG_DIR}/
    else
        echo "No log files."
    fi
    ) & wait $!

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        echo "[ERROR] calculateDescriptors in registration failed."
        exit
    fi
#
    stage_log=${LOG_DIR}/matlab-regCorr-group.log
    (
    # make symbolic links of reference-round images because it is not necessary to warp them
    ref_round=$(printf "round%03d" $REFERENCE_ROUND)
    for normalized_file in ${NORMALIZATION_DIR}/${FILE_BASENAME}_${ref_round}_*.${IMAGE_EXT}
    do
        base_file=$(basename "${normalized_file}")
        registered_affine_file=${REGISTRATION_DIR}/${base_file/.${IMAGE_EXT}/_affine.${IMAGE_EXT}}
        registered_tps_file=${REGISTRATION_DIR}/${base_file/.${IMAGE_EXT}/_registered.${IMAGE_EXT}}
        if [ ! -f $registered_affine_file ]; then
            ln -s ${normalized_file/$PWD/..} $registered_affine_file
        fi
        if [ ! -f $registered_tps_file ]; then
            ln -s ${normalized_file/$PWD/..} $registered_tps_file
        fi
    done

    rounds=$(seq -s' ' 1 ${ROUND_NUM})' '
    rounds=${rounds/$REFERENCE_ROUND /}
    echo "Skipping registration of the reference round"
    echo $rounds
    if [ ${USE_GPU_CUDA} == "true" ]; then
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            registerWithCorrespondencesCUDAInParallel([$rounds]); \
            postcheck_registerWithCorrespondences(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    else
        #Because the matching is currently single-threaded, we can parpool it in one loop
        matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
            registerWithCorrespondencesInParallel([$rounds]); \
            postcheck_registerWithCorrespondences(${ROUND_NUM}); \
            ${ERR_HDL_POSTCODE}"
    fi

    if ls matlab-regCorr-*.log > /dev/null 2>&1; then
        grep -A 40 Error matlab-regCorr-*.log
        mv matlab-regCorr-*.log ${LOG_DIR}/
    else
        echo "No regCorr-log files."
    fi
    ) & wait $!

    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
        echo "[ERROR] registerWithCorrespondences in registration failed."
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
    matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} \
        loadParameters; punctafeinder; puncta_roicollect_bgincl; \
        postcheck_puncta_extraction(); \
        ${ERR_HDL_POSTCODE}"
    #matlab -nodisplay -nosplash -logfile ${stage_log} -r "${ERR_HDL_PRECODE} punctafeinder_in_parallel; ${ERR_HDL_POSTCODE}"
    #if ls matlab-puncta-extraction-*.log > /dev/null 2>&1; then
    #    grep -A 40 Error matlab-puncta-extraction-*.log
    #    mv matlab-puncta-extraction-*.log ${LOG_DIR}/
    #else
    #    echo "No job log files."
    #fi
    ) & wait $!
    # check result
    if [ -z "$(grep '\[DONE\]' ${stage_log})" ]; then
        STAGES_STATUS[$stage_idx]="ERROR"
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

    (
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-base-calling-making.log -r "${ERR_HDL_PRECODE} \
        loadParameters; basecalling_simple; \
        ${ERR_HDL_POSTCODE}"
    ) & wait $!
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

print_stage_status

echo "========================================================================="
echo "Pipeline finished"; date
echo

