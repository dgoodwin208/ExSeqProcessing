#!/bin/bash

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -m    microscopy image directory"
    echo "  -n    normalization image directory"
    echo "  -r    registration image directory"
    echo "  -R    Registration MATLAB directory"
    echo "  -V    vlfeat lib directory"
    echo "  -I    Raj lab image tools MATLAB directory"
    echo "  -L    log directory"
    echo "  -s    skip stages;  normalization,registration,puncta-extraction"
    echo "  -y    continue interactive questions"
    exit 1
}

export TZ=America/New_York
MICROSCOPY_DIR=1_microscopy
NORMALIZATION_DIR=2_normalization
REGISTRATION_DIR=3_registration
PUNCTA_DIR=4_puncta-extraction

REGISTRATION_PROJ_DIR=../Registration
VLFEAT_DIR=~/lib/matlab/vlfeat-0.9.20
RAJLABTOOLS_DIR=~/lib/matlab/rajlabimagetools
LOG_DIR=./logs

REGISTRATION_SAMPLE='sa0916dncv_'
REGISTRATION_CHANNEL='summedNorm'
REGISTRATION_WARP_CHANNELS="'summedNorm','chan1Norm','chan2Norm','chan3Norm','chan4Norm'"

###### getopts

while getopts m:n:r:p:R:V:I:L:s:yh OPT
do
    case $OPT in
        m)  MICROSCOPY_DIR=$OPTARG
            ;;
        n)  NORMALIZATION_DIR=$OPTARG
            ;;
        r)  REGISTRATION_DIR=$OPTARG
            ;;
        p)  PUNCTA_DIR=$OPTARG
            ;;
        R)  REGISTRATION_PROJ_DIR=$OPTARG
            ;;
        V)  VLFEAT_DIR=$OPTARG
            ;;
        I)  RAJLABTOOLS_DIR=$OPTARG
            ;;
        L)  LOG_DIR=$OPTARG
            ;;
        s)  SKIP_STAGES=$OPTARG
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

if [ ! -d "${MICROSCOPY_DIR}" ]
then
    echo "No microscopy dir.: ${MICROSCOPY_DIR}"
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

if [ ! -d "${LOG_DIR}" ]
then
    echo "No log dir."
    echo "mkdir ${LOG_DIR}"
    mkdir "${LOG_DIR}"
fi

MICROSCOPY_DIR=$(cd "${MICROSCOPY_DIR}" && pwd)
NORMALIZATION_DIR=$(cd "${NORMALIZATION_DIR}" && pwd)
REGISTRATION_DIR=$(cd "${REGISTRATION_DIR}" && pwd)
PUNCTA_DIR=$(cd "${PUNCTA_DIR}" && pwd)

REGISTRATION_PROJ_DIR=$(cd "${REGISTRATION_PROJ_DIR}" && pwd)
VLFEAT_DIR=$(cd "${VLFEAT_DIR}" && pwd)
RAJLABTOOLS_DIR=$(cd "${RAJLABTOOLS_DIR}" && pwd)

LOG_DIR=$(cd "${LOG_DIR}" && pwd)


echo "#########################################################################"
echo "Directories"
echo "  microscopy images    :  ${MICROSCOPY_DIR}"
echo "  normalization images :  ${NORMALIZATION_DIR}"
echo "  registration images  :  ${REGISTRATION_DIR}"
echo
echo "  Registration project :  ${REGISTRATION_PROJ_DIR}"
echo "  vlfeat lib           :  ${VLFEAT_DIR}"
echo "  Raj lab image tools  :  ${RAJLABTOOLS_DIR}"
echo
echo "  Log                  :  ${LOG_DIR}"
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


###### setup directories in MATLAB scripts

# setup for Registration

pushd "${REGISTRATION_PROJ_DIR}"
#"${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh
popd

sed -e "s#\(params.SAMPLE_NAME\) *= *'.*';#\1 = '${REGISTRATION_SAMPLE}';#" \
    -e "s#\(params.DATACHANNEL\) *= *'.*';#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(params.REGISTERCHANNEL\) *= *'.*';#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(params.CHANNELS\) *= *{'.*'};#\1 = {${REGISTRATION_WARP_CHANNELS}};#" \
    -e "s#\(params.INPUTDIR\) *= *'.*';#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(params.OUTPUTDIR\) *= *'.*';#\1 = '${REGISTRATION_DIR}';#" \
    -i.back \
    "${REGISTRATION_PROJ_DIR}"/MATLAB/loadExperimentParams.m

# setup for segmentation using Raj lab image tools

sed -e "s#\(params.registeredImagesDir\) *= *'.*';#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.rajlabDirectory\) *= *'.*';#\1 = '.';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *'.*';#\1 = '.';#" \
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

if [ "${SKIP_STAGES/normalization/}" = "${SKIP_STAGES}" ]
then
    echo "Do normalization!"
else
    echo "Skip!"
fi
echo

# registration
echo "========================================================================="
echo "Registration"; date
echo

if [ "${SKIP_STAGES/registration/}" = "${SKIP_STAGES}" ]
then
    echo "-------------------------------------------------------------------------"
    echo "Registration - calculateDescriptors"; date
    echo

    for((i=1; i<=12; i+=2)); do
        round1=$i
        round2=$(( $i + 1 ))
        ### calculateDescriptors for two groups of rounds in parallel
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-calcDesc-group-${round1}-${round2}.log -r "calculateDescriptorsInParallel([$round1 $round2]); exit"

        mv matlab-calcDesc-*.log ${LOG_DIR}/
    done

    echo "-------------------------------------------------------------------------"
    echo "Registration - registerWithDescriptors"; date
    echo

    for((i=1; i<=12; i++)); do
    #for((i=1; i<=2; i++)); do
        #### registerWithDescriptors for round 1 and i
        matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-registerWDesc-${i}.log -r "registerWithDescriptors(${i}); exit"

    done
else
    echo "Skip!"
fi
echo


# puncta extraction
echo "========================================================================="
echo "puncta extraction"; date
echo

if [ "${SKIP_STAGES/puncta-extraction/}" = "${SKIP_STAGES}" ]
then
    for f in $(\ls ${REGISTRATION_DIR}/*.tif)
    do
        #input_file=$(printf "%s%03d.tiff" $(basename $f | sed -e 's/\([^_]*\)_round\([0-9]*\)_\(.*\).tif/\3 \2/'))
        input_file=$(printf "alexa%03d.tiff" $(basename $f | sed -e 's/\([^_]*\)_round\([0-9]*\)_\(.*\).tif/\2/'))
        ln -s $f ${PUNCTA_DIR}/$input_file
    done

    pushd ${PUNCTA_DIR}
    if [ ! -f ./startup.m ]
    then
        ln -s ../startup.m
    fi

    %matlab -nodisplay -nosplash -logfile ../${LOG_DIR}/matlab-puncta-extraction.log -r "makeROIs();improc2.processImageObjects();adjustThresholds();getPuncta;analyzePuncta;makePunctaVolumes; exit"
    matlab -nodisplay -nosplash -logfile ../${LOG_DIR}/matlab-puncta-extraction.log -r "makeROIs();improc2.processImageObjects();adjustThresholds();getPuncta;analyzePuncta; exit"

    popd
else
    echo "Skip!"
fi
echo


echo "========================================================================="
echo "pipeline finished"; date
echo

