#!/bin/bash

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -N    # of rounds; 'auto' means # is calculated from files."
    echo "  -b    file basename"
    echo "  -c    channel names; ex. 'chn01','ch02corr'"
    echo "  -d    deconvolution image directory"
    echo "  -n    normalization image directory"
    echo "  -r    registration image directory"
    echo "  -p    puncta extraction directory"
    echo "  -t    transcript information directory"
    echo "  -R    registration MATLAB directory"
    echo "  -V    vlfeat lib directory"
    echo "  -I    Raj lab image tools MATLAB directory"
    echo "  -L    log directory"
    echo "  -e    execution stages which are higher priority than skip stages"
    echo "  -s    skip stages;  profile-check,normalization,registration,calc-descriptors,register-with-descriptors,puncta-extraction,transcripts"
    echo "  -y    continue interactive questions"
    echo "  -h    print help"
    exit 1
}

export TZ=America/New_York

ROUND_NUM=12

DECONVOLUTION_DIR=1_deconvolution
NORMALIZATION_DIR=2_normalization
REGISTRATION_DIR=3_registration
PUNCTA_DIR=4_puncta-extraction
TRANSCRIPT_DIR=5_transcripts

REGISTRATION_PROJ_DIR=../Registration
VLFEAT_DIR=~/lib/matlab/vlfeat-0.9.20
RAJLABTOOLS_DIR=~/lib/matlab/rajlabimagetools
LOG_DIR=./logs

FILE_BASENAME=sa0916dncv
CHANNELS="'chan1','chan2','chan3','chan4'"
CHANNEL_ARRAY=($(echo ${CHANNELS//\'/} | tr ',' ' '))
REGISTRATION_SAMPLE=${FILE_BASENAME}_
REGISTRATION_CHANNEL=summedNorm
REGISTRATION_WARP_CHANNELS="'${REGISTRATION_CHANNEL}',${CHANNELS}"

###### getopts

while getopts N:b:c:d:n:r:p:t:R:V:I:L:e:s:yh OPT
do
    case $OPT in
        N)  ROUND_NUM=$OPTARG
            if [ $ROUND_NUM != "auto" ]
            then
                expr $ROUND_NUM + 1 > /dev/null 2>&1
                if [ $? -ge 2 ]
                then
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
            ;;
        d)  DECONVOLUTION_DIR=$OPTARG
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
        L)  LOG_DIR=$OPTARG
            ;;
        e)  ARG_EXEC_STAGES=$OPTARG
            ;;
        s)  ARG_SKIP_STAGES=$OPTARG
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


###### check directories

if [ ! -d "${DECONVOLUTION_DIR}" ]
then
    echo "No deconvolution dir.: ${DECONVOLUTION_DIR}"
    exit 1
fi

if [ ! -d "${REGISTRATION_PROJ_DIR}" ]
then
    echo "No Registration project dir.: ${REGISTRATION_PROJ_DIR}"
    exit 1
fi

if [ ! -d "${REGISTRATION_PROJ_DIR}"/MATLAB ]
then
    echo "No MATLAB dir. in Registration project: ${REGISTRATION_PROJ_DIR}/MATLAB"
    exit 1
fi

if [ ! -d "${REGISTRATION_PROJ_DIR}"/scripts ]
then
    echo "No scripts dir. in Registration project: ${REGISTRATION_PROJ_DIR}/scripts"
    exit 1
fi

if [ ! -d "${RAJLABTOOLS_DIR}" ]
then
    echo "No Raj lab image tools project dir.: ${RAJLABTOOLS_DIR}"
    exit 1
fi

if [ ! -d "${VLFEAT_DIR}" ]
then
    echo "No vlfeat library dir.: ${VLFEAT_DIR}"
    exit 1
fi


###### check files

echo "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh
if [ ! -f "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh ]
then
    echo "No 'import_cluster_profiles.sh'"
    exit 1
fi

if [ ! -f "${REGISTRATION_PROJ_DIR}"/MATLAB/loadExperimentParams.m ]
then
    echo "No 'loadExperimentParams.m' in Registration MATLAB"
    exit 1
fi

if [ ! -f ./loadParameters.m ]
then
    echo "No 'loadParameters.m' in ExSeqProcessing MATLAB"
    exit 1
fi


###### setup directories

if [ ! -d "${NORMALIZATION_DIR}" ]
then
    echo "No normalization dir."
    echo "mkdir ${NORMALIZATION_DIR}"
    mkdir "${NORMALIZATION_DIR}"
fi

if [ ! -d "${REGISTRATION_DIR}" ]
then
    echo "No registration dir."
    echo "mkdir ${REGISTRATION_DIR}"
    mkdir "${REGISTRATION_DIR}"
fi

if [ ! -d "${PUNCTA_DIR}" ]
then
    echo "No puncta-extraction dir."
    echo "mkdir ${PUNCTA_DIR}"
    mkdir "${PUNCTA_DIR}"
fi

if [ ! -d "${TRANSCRIPT_DIR}" ]
then
    echo "No transcript information dir."
    echo "mkdir ${TRANSCRIPT_DIR}"
    mkdir "${TRANSCRIPT_DIR}"
fi

if [ ! -d "${LOG_DIR}" ]
then
    echo "No log dir."
    echo "mkdir ${LOG_DIR}"
    mkdir "${LOG_DIR}"
fi

# exchange paths to absolute paths
DECONVOLUTION_DIR=$(cd "${DECONVOLUTION_DIR}" && pwd)
NORMALIZATION_DIR=$(cd "${NORMALIZATION_DIR}" && pwd)
REGISTRATION_DIR=$(cd "${REGISTRATION_DIR}" && pwd)
PUNCTA_DIR=$(cd "${PUNCTA_DIR}" && pwd)
TRANSCRIPT_DIR=$(cd "${TRANSCRIPT_DIR}" && pwd)

REGISTRATION_PROJ_DIR=$(cd "${REGISTRATION_PROJ_DIR}" && pwd)
VLFEAT_DIR=$(cd "${VLFEAT_DIR}" && pwd)
RAJLABTOOLS_DIR=$(cd "${RAJLABTOOLS_DIR}" && pwd)

LOG_DIR=$(cd "${LOG_DIR}" && pwd)

if [ $ROUND_NUM = "auto" ]
then
    ROUND_NUM=$(find ${DECONVOLUTION_DIR}/ -name "${FILE_BASENAME}_round*_${CHANNEL_ARRAY[0]}.tif" | wc -l)
fi

STAGES=("profile-check" "normalization" "registration" "puncta-extraction" "transcripts")
REG_STAGES=("calc-descriptors" "register-with-descriptors")

# check stages to be skipped and executed
if [ ! "${ARG_EXEC_STAGES}" = "" -a ! "${ARG_SKIP_STAGES}" = "" ]
then
    echo "cannot use both -e and -s"
    exit 1
fi

if [ ! "${ARG_EXEC_STAGES}" = "" ]
then
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ "${ARG_EXEC_STAGES/${STAGES[i]}}" = "${ARG_EXEC_STAGES}" ]
        then
            SKIP_STAGES[i]="skip"
        fi
    done
    for((i=0; i<${#REG_STAGES[*]}; i++))
    do
        if [ "${ARG_EXEC_STAGES/registration}" = "${ARG_EXEC_STAGES}" -a "${ARG_EXEC_STAGES/${REG_STAGES[i]}}" = "${ARG_EXEC_STAGES}" ]
        then
            SKIP_REG_STAGES[i]="skip"
        else
            SKIP_STAGES[2]=
        fi
    done
else
    for((i=0; i<${#STAGES[*]}; i++))
    do
        if [ ! "${ARG_SKIP_STAGES/${STAGES[i]}}" = "${ARG_SKIP_STAGES}" ]
        then
            SKIP_STAGES[i]="skip"
        fi
    done
    for((i=0; i<${#REG_STAGES[*]}; i++))
    do
        if [ ! "${ARG_SKIP_STAGES/registration}" = "${ARG_SKIP_STAGES}" ]
        then
            SKIP_REG_STAGES[i]="skip"
        elif [ ! "${ARG_SKIP_STAGES/${REG_STAGES[i]}}" = "${ARG_SKIP_STAGES}" ]
        then
            SKIP_REG_STAGES[i]="skip"
        fi
    done
fi


echo "#########################################################################"
echo "Parameters"
echo "  # of rounds            :  ${ROUND_NUM}"
echo "  file basename          :  ${FILE_BASENAME}"
echo "  processing channels    :  ${CHANNELS}"
echo "  registration channel   :  ${REGISTRATION_CHANNEL}"
echo "  warp channels          :  ${REGISTRATION_WARP_CHANNELS}"
echo
echo "Stages"
for((i=0; i<${#STAGES[*]}; i++))
do
    if [ "${SKIP_STAGES[i]}" = "skip" ]
    then
        echo -n "                    skip "
    else
        echo -n "                         "
    fi
    echo ":  ${STAGES[i]}"
done
echo "Registration sub-stages"
for((i=0; i<${#REG_STAGES[*]}; i++))
do
    if [ "${SKIP_REG_STAGES[i]}" = "skip" ]
    then
        echo -n "                    skip "
    else
        echo -n "                         "
    fi
    echo ":  ${REG_STAGES[i]}"
done
echo
echo "Directories"
echo "  deconvolution images   :  ${DECONVOLUTION_DIR}"
echo "  normalization images   :  ${NORMALIZATION_DIR}"
echo "  registration images    :  ${REGISTRATION_DIR}"
echo "  puncta                 :  ${PUNCTA_DIR}"
echo "  transcripts            :  ${TRANSCRIPT_DIR}"
echo
echo "  Registration project   :  ${REGISTRATION_PROJ_DIR}"
echo "  vlfeat lib             :  ${VLFEAT_DIR}"
echo "  Raj lab image tools    :  ${RAJLABTOOLS_DIR}"
echo
echo "  Log                    :  ${LOG_DIR}"
echo "#########################################################################"
echo

if [ ! "${QUESTION_ANSW}" = 'yes' ]
then
    echo "OK? (y/n)"
    read -sn1 ANSW
    if [ $ANSW = 'n' -o $ANSW = 'N' ]
    then
        echo
        echo 'Canceled.'
        exit 0
    fi
fi
echo

stage_idx=0

###### check a cluster profile
echo "========================================================================="
echo "Cluster-profile check"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]
then
    pushd "${REGISTRATION_PROJ_DIR}"
    "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh
    popd
else
    echo "Skip!"
fi

stage_idx=$(( $stage_idx + 1 ))


###### setup MATLAB scripts

# setup for Registration

sed -e "s#\(params.SAMPLE_NAME\) *= *.*;#\1 = '${REGISTRATION_SAMPLE}';#" \
    -e "s#\(params.DATACHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(params.REGISTERCHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(params.CHANNELS\) *= *.*;#\1 = {${REGISTRATION_WARP_CHANNELS}};#" \
    -e "s#\(params.INPUTDIR\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(params.OUTPUTDIR\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -i.back \
    "${REGISTRATION_PROJ_DIR}"/MATLAB/loadExperimentParams.m

# setup for segmentation using Raj lab image tools
set -g mouse-select-window on

sed -e "s#\(params.registeredImagesDir\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *.*;#\1 = '${PUNCTA_DIR}';#" \
    -e "s#\(params.FILE_BASENAME\) *= *.*;#\1 = '${FILE_BASENAME}';#" \
    -e "s#\(params.NUM_ROUNDS\) *= *.*;#\1 = ${ROUND_NUM};#" \
    -i.back \
    ./loadParameters.m

###### setup startup.m

cat << EOF > startup.m
run('${VLFEAT_DIR}/toolbox/vl_setup')

addpath(genpath('${REGISTRATION_PROJ_DIR}/MATLAB'),genpath('${REGISTRATION_PROJ_DIR}/scripts'),genpath('${RAJLABTOOLS_DIR}'),genpath('$(pwd)'));

EOF

###### run pipeline

# normalization
echo "========================================================================="
echo "Normalization"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]
then
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization.log -r "normalization('${DECONVOLUTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM}); exit"
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))

# registration
echo "========================================================================="
echo "Registration"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]
then

    echo "-------------------------------------------------------------------------"
    echo "Registration - calculateDescriptors"; date
    echo

    reg_stage_idx=0
    if [ ! "${SKIP_REG_STAGES[$reg_stage_idx]}" = "skip" ]
    then
        for((i=1; i<=${ROUND_NUM}; i+=2))
        do
            if [ $i -eq ${ROUND_NUM} ]
            then
                rounds=$i
            else
                rounds="$i $(( $i + 1 ))"
            fi
            # calculateDescriptors for two groups of rounds in parallel
            matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-calcDesc-group-${rounds/ /-}.log -r "calculateDescriptorsInParallel([$rounds]); exit"
    
            if ls *.log > /dev/null 2>&1
            then
                mv matlab-calcDesc-*.log ${LOG_DIR}/
            else
                echo "No log files."
            fi
        done
    else
        echo "Skip!"
    fi
    reg_stage_idx=$(( $reg_stage_idx + 1 ))

    echo "-------------------------------------------------------------------------"
    echo "Registration - registerWithDescriptors"; date
    echo

    if [ ! "${SKIP_REG_STAGES[$reg_stage_idx]}" = "skip" ]
    then
        # prepare normalized channel images for warp
        for((i=0; i<${#CHANNEL_ARRAY[*]}; i++))
        do
            for f in $(\ls ${DECONVOLUTION_DIR}/*_${CHANNEL_ARRAY[i]}.tif)
            do
                round_num=$(( $(echo $f | sed -ne 's/.*_round0*\([0-9]\+\)_.*.tif/\1/p') ))
                if [ $round_num -eq 0 ]
                then
                    echo "round number is wrong."
                fi

                normalized_ch_file=$(printf "${NORMALIZATION_DIR}/${FILE_BASENAME}_round%03d_${CHANNEL_ARRAY[i]}.tif" $round_num)

                if [ ! -f $normalized_ch_file ]
                then
                    ln -s $f $normalized_ch_file
                fi
            done
        done

        # make symbolic links of round-1 images because it is not necessary to warp them
        for ch in ${REGISTRATION_CHANNEL} ${CHANNEL_ARRAY[*]}
        do
            normalized_file=${NORMALIZATION_DIR}/${FILE_BASENAME}_round001_${ch}.tif
            registered_file=${REGISTRATION_DIR}/${FILE_BASENAME}_round001_${ch}_registered.tif
            if [ ! -f $registered_file ]
            then
                ln -s $normalized_file $registered_file
            fi
        done

        for((i=2; i<=${ROUND_NUM}; i++))
        do
            # registerWithDescriptors for round 1 and i
            matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-registerWDesc-${i}.log -r "registerWithDescriptors(${i}); exit"

        done
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

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]
then
    for f in $(\ls ${REGISTRATION_DIR}/${FILE_BASENAME}_round*_${REGISTRATION_CHANNEL}_registered.tif)
    do
        round_num=$(( $(basename $f | sed -ne 's/[^_]*_round0*\([0-9]\+\)_.*.tif/\1/p') ))
        if [ $round_num -eq 0 ]
        then
            echo "round number is wrong."
        fi

        input_file=$(printf "alexa%03d.tiff" $round_num)
        if [ ! -f ${PUNCTA_DIR}/$input_file ]
        then
            ln -s $f ${PUNCTA_DIR}/$input_file
        fi
    done

    pushd ${PUNCTA_DIR}
    if [ ! -f ./startup.m ]
    then
        ln -s ../startup.m
    fi

    #matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-puncta-extraction.log -r "makeROIs();improc2.processImageObjects();adjustThresholds();getPuncta;analyzePuncta;makePunctaVolumes; exit"
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-puncta-extraction.log -r "analyzePuncta;makePunctaVolumes; exit"
    popd
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


# prepare base calling of transcripts
echo "========================================================================="
echo "base calling preparation"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]
then
    cp -a ${REGISTRATION_DIR}/${FILE_BASENAME}_round001_${REGISTRATION_CHANNEL}_registered.tif ${TRANSCRIPT_DIR}/alexa001.tiff
    cp -a ${PUNCTA_DIR}/${FILE_BASENAME}_puncta_rois.mat ${TRANSCRIPT_DIR}/
    #ls -l ${TRANSCRIPT_DIR}
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-transcript-making.log -r "normalizePunctaVector; refineBaseCalling; exit"
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))




echo "========================================================================="
echo "pipeline finished"; date
echo

