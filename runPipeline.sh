#!/bin/bash

SEMAPHORE_LIST=(/g0 /g1 /g2 /g3 /qn_c0 /qn_c1 /qn_c2 /qn_c3 /gr)
function clear_semaphores() {

    existed_sem_list=()
    for ((i=0; i<${#SEMAPHORE_LIST[*]}; i++)); do
        sem_status=$(./utils/semaphore/semaphore ${SEMAPHORE_LIST[i]} getvalue | grep OK)
        if [ -n "${sem_status}" ]; then
            existed_sem_list+=( ${SEMAPHORE_LIST[i]} )
        fi
    done

    if [ ${#existed_sem_list[*]} -gt 0 ]; then
        echo "Semaphores have left."
        for ((i=0; i<${#existed_sem_list[*]}; i++)); do
            ./utils/semaphore/semaphore ${existed_sem_list[i]} unlink
        done
    fi
}

function trap_handle() {
    if [ -n "${PID_PERF_PROFILE}" ]; then
        kill -- -${PID_PERF_PROFILE}
    fi
    kill 0
    clear_semaphores
    exit
}

trap trap_handle EXIT


function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -N    # of rounds; 'auto' means # is calculated from files."
    echo "  -b    file basename"
    echo "  -c    channel names; ex. 'chn01','ch02corr'"
    echo "  -B    reference round puncta"
    echo "  -d    deconvolution image directory"
    echo "  -C    color correction image directory"
    echo "  -n    normalization image directory"
    echo "  -r    registration image directory"
    echo "  -p    puncta extraction directory"
    echo "  -t    transcript information directory"
    echo "  -R    registration MATLAB directory"
    echo "  -V    vlfeat lib directory"
    echo "  -I    Raj lab image tools MATLAB directory"
    echo "  -i    reporting directory"
    echo "  -T    temp directory"
    echo "  -L    log directory"
    echo "  -G    use GPUs (default: no)"
    echo "  -H    use HDF5 format for intermediate files (default: no)"
    echo "  -J    set # of concurrent jobs for color-correction, normalization;  5,10"
    echo "  -P    mode to get performance profile"
    echo "  -e    execution stages;  exclusively use for skip stages"
    echo "  -s    skip stages;  setup-cluster,color-correction,normalization,registration,calc-descriptors,register-with-correspondences,puncta-extraction,transcripts"
    echo "  -y    continue interactive questions"
    echo "  -h    print help"
    exit 1
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

ROUND_NUM=20
REFERENCE_ROUND=5

DECONVOLUTION_DIR=1_deconvolution
COLOR_CORRECTION_DIR=2_color-correction
NORMALIZATION_DIR=3_normalization
REGISTRATION_DIR=4_registration
PUNCTA_DIR=5_puncta-extraction
TRANSCRIPT_DIR=6_transcripts

REGISTRATION_PROJ_DIR=registration
VLFEAT_DIR=~/lib/matlab/vlfeat-0.9.20
RAJLABTOOLS_DIR=~/lib/matlab/rajlabimagetools
REPORTING_DIR=logs/imgs
LOG_DIR=logs

if [ -f ./loadParameters.m ]; then
    PARAMETERS_FILE=./loadParameters.m
else
    PARAMETERS_FILE=./loadParameters.m.template
fi

TEMP_DIR=$(sed -ne "s#params.tempDir *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
USE_TMP_FILES_IN_NORM=$(sed -ne "s#params.USE_TMP_FILES_IN_NORM *= *\(.*\);#\1#p" ${PARAMETERS_FILE})

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

FILE_BASENAME=$(sed -ne "s#params.FILE_BASENAME *= *'\(.*\)';#\1#p" ${PARAMETERS_FILE})
CHANNELS=$(sed -ne "s#params.CHAN_STRS *= *{\(.*\)};#\1#p" ${PARAMETERS_FILE})

CHANNEL_ARRAY=($(echo ${CHANNELS//\'/} | tr ',' ' '))
REGISTRATION_SAMPLE=${FILE_BASENAME}_
REGISTRATION_CHANNEL=summedNorm
PUNCTA_EXTRACT_CHANNEL=summedNorm
REGISTRATION_WARP_CHANNELS="'${REGISTRATION_CHANNEL}',${CHANNELS}"

USE_GPU_CUDA=false
USE_HDF5=false
PERF_PROFILE=false

NUM_LOGICAL_CORES=$(lscpu | grep ^CPU\(s\) | sed -e "s/[^0-9]*\([0-9]*\)/\1/")

###### getopts

while getopts N:b:c:B:d:C:n:r:p:t:R:V:I:T:i:L:e:s:GHJ:Pyh OPT
do
    case $OPT in
        N)  ROUND_NUM=$OPTARG
            if [ $ROUND_NUM != "auto" ]; then
                if ! check_number ${ROUND_NUM}; then
                    echo "# of rounds is not number; ${ROUND_NUM}"
                    exit 1
                fi
            fi
            ;;
        b)  FILE_BASENAME=$OPTARG
            REGISTRATION_SAMPLE=${FILE_BASENAME}_
            ;;
        c)  CHANNELS=$OPTARG
            CHANNEL_ARRAY=($(echo ${CHANNELS//\'/} | tr ',' ' '))
            REGISTRATION_WARP_CHANNELS="'${REGISTRATION_CHANNEL}',${CHANNELS}"
            PUNCTA_EXTRACT_CHANNELS="'${REGISTRATION_CHANNEL}'"
            ;;
        B)  REFERENCE_ROUND=$OPTARG
                if ! check_number ${REFERENCE_ROUND}; then
                    echo "reference round is not number; ${REFERENCE_ROUND}"
                    exit 1
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
        t)  TRANSCRIPT_DIR=$OPTARG
            ;;
        R)  REGISTRATION_PROJ_DIR=$OPTARG
            ;;
        V)  VLFEAT_DIR=$OPTARG
            ;;
        I)  RAJLABTOOLS_DIR=$OPTARG
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
        G)  USE_GPU_CUDA=true
            ;;
        H)  USE_HDF5=true
            ;;
        J)  IFS=, read NJOBS_CC NJOBS_NORM TMP <<<$OPTARG
            if [ -n "${NJOBS_CC}" ]; then
                if check_number ${NJOBS_CC}; then
                    if [ ${NJOBS_CC} -eq 0 ]; then
                        echo "# of jobs for color-correction is zero"
                        exit 1
                    fi
                    COLOR_CORRECTION_MAX_RUN_JOBS=${NJOBS_CC}
                else
                    echo "${NJOBS_CC} is not number"
                    exit 1
                fi
            fi
            if [ -n "${NJOBS_NORM}" ]; then
                if check_number ${NJOBS_NORM}; then
                    if [ ${NJOBS_NORM} -eq 0 ]; then
                        echo "# of jobs for normalization is zero"
                        exit 1
                    fi
                    NORMALIZATION_MAX_RUN_JOBS=${NJOBS_NORM}
                else
                    echo "${NJOBS_NORM} is not number"
                    exit 1
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
    esac
done

shift $((OPTIND - 1))


###### clear semaphores

clear_semaphores


###### check directories

if [ ! -d "${DECONVOLUTION_DIR}" ]; then
    echo "No deconvolution dir.: ${DECONVOLUTION_DIR}"
    exit 1
fi

if [ ! -d "${REGISTRATION_PROJ_DIR}" ]; then
    echo "No Registration project dir.: ${REGISTRATION_PROJ_DIR}"
    exit 1
fi

if [ ! -d "${REGISTRATION_PROJ_DIR}"/MATLAB ]; then
    echo "No MATLAB dir. in Registration project: ${REGISTRATION_PROJ_DIR}/MATLAB"
    exit 1
fi

if [ ! -d "${RAJLABTOOLS_DIR}" ]; then
    echo "No Raj lab image tools project dir.: ${RAJLABTOOLS_DIR}"
    exit 1
fi

if [ ! -d "${VLFEAT_DIR}" ]; then
    echo "No vlfeat library dir.: ${VLFEAT_DIR}"
    exit 1
fi


###### check files

if [ ! -f ./loadParameters.m.template ]; then
    echo "No 'loadParameters.m.template' in ExSeqProcessing MATLAB"
    exit 1
fi
if [ ! -f ./loadParameters.m ]; then
    echo "No 'loadParameters.m' in ExSeqProcessing MATLAB. Copy from a template file"
    cp -a ./loadParameters.m{.template,}
fi


###### check temporary files
if [ -n "$(find ${TEMP_DIR} -type d -not -empty)" ]; then
    echo "Temporary dir. is not empty."

    if [ ! "${QUESTION_ANSW}" = 'yes' ]; then
        echo "Delete? (y/n)"
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

if [ ! -d "${TRANSCRIPT_DIR}" ]; then
    echo "No transcript information dir."
    echo "mkdir ${TRANSCRIPT_DIR}"
    mkdir "${TRANSCRIPT_DIR}"
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
TRANSCRIPT_DIR=$(cd "${TRANSCRIPT_DIR}" && pwd)

REGISTRATION_PROJ_DIR=$(cd "${REGISTRATION_PROJ_DIR}" && pwd)
VLFEAT_DIR=$(cd "${VLFEAT_DIR}" && pwd)
RAJLABTOOLS_DIR=$(cd "${RAJLABTOOLS_DIR}" && pwd)

REPORTING_DIR=$(cd "${REPORTING_DIR}" && pwd)
LOG_DIR=$(cd "${LOG_DIR}" && pwd)

if [ $ROUND_NUM = "auto" ]; then
    ROUND_NUM=$(find ${DECONVOLUTION_DIR}/ -name "*_${CHANNEL_ARRAY[0]}.tif" | wc -l)
fi

if [ "$USE_HDF5" = "true" ]; then
    IMAGE_EXT=h5
else
    IMAGE_EXT=tif
fi

STAGES=("setup-cluster" "color-correction" "normalization" "registration" "puncta-extraction" "transcripts")
REG_STAGES=("calc-descriptors" "register-with-correspondences")

# check stages to be skipped and executed
if [ ! "${ARG_EXEC_STAGES}" = "" -a ! "${ARG_SKIP_STAGES}" = "" ]; then
    echo "cannot use both -e and -s"
    exit 1
fi

if [ ! "${ARG_EXEC_STAGES}" = "" ]; then
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ "${ARG_EXEC_STAGES/${STAGES[i]}}" = "${ARG_EXEC_STAGES}" ]; then
            SKIP_STAGES[i]="skip"
        fi
    done
    for((i=0; i<${#REG_STAGES[*]}; i++))
    do
        if [ "${ARG_EXEC_STAGES/registration}" = "${ARG_EXEC_STAGES}" -a "${ARG_EXEC_STAGES/${REG_STAGES[i]}}" = "${ARG_EXEC_STAGES}" ]; then
            SKIP_REG_STAGES[i]="skip"
        else
            SKIP_STAGES[3]=
        fi
    done
else
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ ! "${ARG_SKIP_STAGES/${STAGES[i]}}" = "${ARG_SKIP_STAGES}" ]; then
            SKIP_STAGES[i]="skip"
        fi
    done
    for((i=0; i<${#REG_STAGES[*]}; i++))
    do
        if [ ! "${ARG_SKIP_STAGES/registration}" = "${ARG_SKIP_STAGES}" ]; then
            SKIP_REG_STAGES[i]="skip"
        elif [ ! "${ARG_SKIP_STAGES/${REG_STAGES[i]}}" = "${ARG_SKIP_STAGES}" ]; then
            SKIP_REG_STAGES[i]="skip"
        fi
    done
fi


echo "#########################################################################"
echo "Parameters"
echo "  Running on host        :  "`hostname`
echo "  # of rounds            :  ${ROUND_NUM}"
echo "  file basename          :  ${FILE_BASENAME}"
echo "  reference round        :  ${REFERENCE_ROUND}"
echo "  processing channels    :  ${CHANNELS}"
echo "  registration channel   :  ${REGISTRATION_CHANNEL}"
echo "  warp channels          :  ${REGISTRATION_WARP_CHANNELS}"
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
echo "Registration sub-stages"
for((i=0; i<${#REG_STAGES[*]}; i++))
do
    if [ "${SKIP_REG_STAGES[i]}" = "skip" ]; then
        echo -n "                    skip "
    else
        echo -n "                         "
    fi
    echo ":  ${REG_STAGES[i]}"
done
echo
echo "Directories"
echo "  deconvolution images   :  ${DECONVOLUTION_DIR}"
echo "  color correction images:  ${COLOR_CORRECTION_DIR}"
echo "  normalization images   :  ${NORMALIZATION_DIR}"
echo "  registration images    :  ${REGISTRATION_DIR}"
echo "  puncta                 :  ${PUNCTA_DIR}"
echo "  transcripts            :  ${TRANSCRIPT_DIR}"
echo
echo "  Registration project   :  ${REGISTRATION_PROJ_DIR}"
echo "  vlfeat lib             :  ${VLFEAT_DIR}"
echo "  Raj lab image tools    :  ${RAJLABTOOLS_DIR}"
echo
echo "  Temporal storage       :  ${TEMP_DIR}"
echo
echo "  Reporting              :  ${REPORTING_DIR}"
echo "  Log                    :  ${LOG_DIR}"
echo
echo "========================================================================="
echo "Concurrency: # of parallel jobs, workers/job, threads/worker"
echo "  # of logical cores     :  ${NUM_LOGICAL_CORES}"

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
echo "Parameters in only loadParameters.m"
echo "  temporary data in norm :  "$(if [ "${USE_TMP_FILES_IN_NORM}" = "true" ]; then echo "storage"; else echo "on-memory";fi)
echo "#########################################################################"
echo

if [ ! "${QUESTION_ANSW}" = 'yes' ]; then
    echo "OK? (y/n)"
    read -sn1 ANSW
    if [ $ANSW = 'n' -o $ANSW = 'N' ]; then
        echo
        echo 'Canceled.'
        exit 0
    fi
fi
echo


if [ "${PERF_PROFILE}" = "true" ]; then
    set -m
    ./tests/perf-profile/get-stats.sh &
    set +m
    PID_PERF_PROFILE=$!
    sleep 1
fi

stage_idx=0

###### setup a cluster profile
echo "========================================================================="
echo "Setup cluster-profile"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    (
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-setup-cluster-profile.log -r "${ERR_HDL_PRECODE} setup_cluster_profile(); ${ERR_HDL_POSTCODE}"
    ) & wait $!
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


###### setup MATLAB scripts

sed -e "s#\(regparams.DATACHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(regparams.REGISTERCHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(regparams.CHANNELS\) *= *.*;#\1 = {${REGISTRATION_WARP_CHANNELS}};#" \
    -e "s#\(regparams.INPUTDIR\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(regparams.OUTPUTDIR\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(regparams.FIXED_RUN\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.deconvolutionImagesDir\) *= *.*;#\1 = '${DECONVOLUTION_DIR}';#" \
    -e "s#\(params.colorCorrectionImagesDir\) *= *.*;#\1 = '${COLOR_CORRECTION_DIR}';#" \
    -e "s#\(params.registeredImagesDir\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *.*;#\1 = '${PUNCTA_DIR}';#" \
    -e "s#\(params.transcriptResultsDir\) *= *.*;#\1 = '${TRANSCRIPT_DIR}';#" \
    -e "s#\(params.reportingDir\) *= *.*;#\1 = '${REPORTING_DIR}';#" \
    -e "s#\(params.FILE_BASENAME\) *= *.*;#\1 = '${FILE_BASENAME}';#" \
    -e "s#\(params.NUM_ROUNDS\) *= *.*;#\1 = ${ROUND_NUM};#" \
    -e "s#\(params.REFERENCE_ROUND_WARP\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.REFERENCE_ROUND_PUNCTA\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -e "s#\(params.NUM_CHANNELS\) *= *.*;#\1 = ${#CHANNEL_ARRAY[*]};#" \
    -e "s#\(params.CHAN_STRS\) *= *.*;#\1 = {${CHANNELS}};#" \
    -e "s#\(params.tempDir\) *= *.*;#\1 = '${TEMP_DIR}';#" \
    -e "s#\(params.IMAGE_EXT\) *= *.*;#\1 = '${IMAGE_EXT}';#" \
    -e "s#\(params.NUM_LOGICAL_CORES\) *= *.*;#\1 = ${NUM_LOGICAL_CORES};#" \
    -e "s#\(params.COLOR_CORRECTION_MAX_RUN_JOBS\) *= *.*;#\1 = ${COLOR_CORRECTION_MAX_RUN_JOBS};#" \
    -e "s#\(params.NORM_MAX_RUN_JOBS\) *= *.*;#\1 = ${NORMALIZATION_MAX_RUN_JOBS};#" \
    -i.back \
    ./loadParameters.m


###### setup startup.m


cat << EOF > startup.m
run('${VLFEAT_DIR}/toolbox/vl_setup')

addpath(genpath('${REGISTRATION_PROJ_DIR}/MATLAB'),genpath('${REGISTRATION_PROJ_DIR}/scripts'),genpath('${RAJLABTOOLS_DIR}'),genpath('$(pwd)'));

EOF

###### run pipeline

ERR_HDL_PRECODE='try;'
ERR_HDL_POSTCODE=' catch ME; disp(ME.getReport); exit(1); end; exit'

# downsampling and color correction
echo "========================================================================="
echo "Downsampling & color correction"; date
echo



if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    (
    #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-copy-scopenames-to-regnames.log -r "${ERR_HDL_PRECODE} copy_scope_names_to_reg_names; ${ERR_HDL_POSTCODE}"
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-downsample-all.log -r "${ERR_HDL_PRECODE} run('downsample_all.m'); ${ERR_HDL_POSTCODE}"

    #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-color-correction.log -r "${ERR_HDL_PRECODE} for i=1:${ROUND_NUM};try; colorcorrection_3D_poc(i);catch; colorcorrection_3D(i); end; end; ${ERR_HDL_POSTCODE}"
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-color-correction.log -r "${ERR_HDL_PRECODE} colorcorrection_3D_cuda(${ROUND_NUM}); ${ERR_HDL_POSTCODE}"
    if ls matlab-color-correction-*.log > /dev/null 2>&1; then
        mv matlab-color-correction-*.log ${LOG_DIR}/
    else
        echo "No job log files."
    fi

    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-downsample-apply.log -r "${ERR_HDL_PRECODE} run('downsample_applycolorshiftstofullres.m'); ${ERR_HDL_POSTCODE}"
    ) & wait $!
else
    echo "Skip!"
fi
echo


stage_idx=$(( $stage_idx + 1 ))

# normalization
echo "========================================================================="
echo "Normalization"; date
echo

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
#    cp 1_deconvolution/*ch00.${IMAGE_EXT} 2_color-correction/

    (
    if [ ${USE_GPU_CUDA} == "true" ]; then
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization.log -r "${ERR_HDL_PRECODE} normalization_cuda('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM}); ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            mv matlab-normalization-*.log ${LOG_DIR}/
        else
            echo "No job log files."
        fi
    else
        #Process the full-resolution data
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization.log -r "${ERR_HDL_PRECODE} normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},false); ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            mv matlab-normalization-*.log ${LOG_DIR}/
        else
            echo "No job log files."
        fi

        #Process the downsampled data, if specified
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization-downsample.log -r "${ERR_HDL_PRECODE} loadParameters; if params.DO_DOWNSAMPLE; normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},true);end; ${ERR_HDL_POSTCODE}"

        if ls matlab-normalization-*.log > /dev/null 2>&1; then
            mkdir -p ${LOG_DIR}/downsample
            mv matlab-normalization-*.log ${LOG_DIR}/downsample/
        else
            echo "No job log files."
        fi
    fi
    ) & wait $!


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
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))

# registration
echo "========================================================================="
echo "Registration"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

    echo "-------------------------------------------------------------------------"
    echo "Registration - calculateDescriptors"; date
    echo

    reg_stage_idx=0
    if [ ! "${SKIP_REG_STAGES[$reg_stage_idx]}" = "skip" ]; then
        (
        rounds=$(seq -s' ' 1 ${ROUND_NUM})
        # calculateDescriptors for all rounds in parallel
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-calcDesc-group.log -r "${ERR_HDL_PRECODE} calculateDescriptorsInParallel([$rounds]); ${ERR_HDL_POSTCODE}"

        if ls matlab-calcDesc-*.log > /dev/null 2>&1; then
            mv matlab-calcDesc-*.log ${LOG_DIR}/
        else
            echo "No log files."
        fi
        ) & wait $!
    else
        echo "Skip!"
    fi
    reg_stage_idx=$(( $reg_stage_idx + 1 ))

    echo "-------------------------------------------------------------------------"
    echo "Registration - registerWithCorrespondences"; date
    echo

    if [ ! "${SKIP_REG_STAGES[$reg_stage_idx]}" = "skip" ]; then
        (
        # make symbolic links of reference-round images because it is not necessary to warp them
        for ch in ${REGISTRATION_CHANNEL} ${CHANNEL_ARRAY[*]}
        do
            ref_round=$(printf "round%03d" $REFERENCE_ROUND)
            normalized_file=${NORMALIZATION_DIR/$PWD/..}/${FILE_BASENAME}_${ref_round}_${ch}.${IMAGE_EXT}

            registered_affine_file=${REGISTRATION_DIR}/${FILE_BASENAME}_${ref_round}_${ch}_affine.${IMAGE_EXT}
            registered_tps_file=${REGISTRATION_DIR}/${FILE_BASENAME}_${ref_round}_${ch}_registered.${IMAGE_EXT}
            if [ ! -f $registered_affine_file ]; then
                ln -s $normalized_file $registered_affine_file
            fi
            if [ ! -f $registered_tps_file ]; then
                ln -s $normalized_file $registered_tps_file
            fi
        done

        rounds=$(seq -s' ' 1 ${ROUND_NUM})' '
        rounds=${rounds/$REFERENCE_ROUND /}
        echo "Skipping registration of the reference round"
        echo $rounds
        # registerWithCorrespondences for ${REFERENCE_ROUND} and i
        if [ ${USE_GPU_CUDA} == "true" ]; then
            matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-regCorr-group.log -r "${ERR_HDL_PRECODE} registerWithCorrespondencesCUDAInParallel([$rounds]); ${ERR_HDL_POSTCODE}"
        else
            #Because the matching is currently single-threaded, we can parpool it in one loop
            #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-registerWCorr-${i}.log -r "${ERR_HDL_PRECODE} parpool; parfor i = 1:20; if i==4;fprintf('Skipping reference round\n');continue;end; calcCorrespondences(i);registerWithCorrespondences(i,true);registerWithCorrespondences(i,false); ${ERR_HDL_POSTCODE}"
            matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-regCorr-group.log -r "${ERR_HDL_PRECODE} registerWithCorrespondencesInParallel([$rounds]); ${ERR_HDL_POSTCODE}"
        fi

        if ls matlab-regCorr-*.log > /dev/null 2>&1; then
            mv matlab-regCorr-*.log ${LOG_DIR}/
        else
            echo "No regCorr-log files."
        fi
        ) & wait $!
    else
        echo "Skip!"
    fi
    reg_stage_idx=$(( $reg_stage_idx + 1 ))

else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# puncta extraction
echo "========================================================================="
echo "puncta extraction"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    (
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-puncta-extraction.log -r "${ERR_HDL_PRECODE} loadParameters; punctafeinder_simple; puncta_subvolumes_simple; ${ERR_HDL_POSTCODE}"
    #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-puncta-extraction.log -r "${ERR_HDL_PRECODE} punctafeinder_in_parallel; ${ERR_HDL_POSTCODE}"

    if ls matlab-puncta-extraction-*.log > /dev/null 2>&1; then
        mv matlab-puncta-extraction-*.log ${LOG_DIR}/
    else
        echo "No job log files."
    fi
    ) & wait $!
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# base calling of transcripts
echo "========================================================================="
echo "base calling"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    (
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-transcript-making.log -r "${ERR_HDL_PRECODE} loadParameters; basecalling_simple;  ${ERR_HDL_POSTCODE}"
    ) & wait $!
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))



if [ "${PERF_PROFILE}" = "true" ]; then
    kill -- -${PID_PERF_PROFILE}
    PID_PERF_PROFILE=
    sleep 1
    tests/perf-profile/summarize-stat-logs.sh logs
fi

echo "========================================================================="
echo "pipeline finished"; date
echo

