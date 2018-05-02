#!/bin/bash

usage() {
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
    echo "  -L    log directory"
    echo "  -e    execution stages;  exclusively use for skip stages"
    echo "  -s    skip stages;  profile-check,color-correction,normalization,registration,calc-descriptors,register-with-descriptors,puncta-extraction,transcripts"
    echo "  -y    continue interactive questions"
    echo "  -h    print help"
    exit 1
}

export TZ=America/New_York

ROUND_NUM=12
REFERENCE_ROUND=1

DECONVOLUTION_DIR=1_deconvolution
COLOR_CORRECTION_DIR=2_color-correction
NORMALIZATION_DIR=3_normalization
REGISTRATION_DIR=4_registration
PUNCTA_DIR=5_puncta-extraction
TRANSCRIPT_DIR=6_transcripts

REGISTRATION_PROJ_DIR=../Registration
VLFEAT_DIR=~/lib/matlab/vlfeat-0.9.20
RAJLABTOOLS_DIR=~/lib/matlab/rajlabimagetools
REPORTING_DIR=./logs/imgs
LOG_DIR=./logs

FILE_BASENAME=sa0916dncv
CHANNELS="'chan1','chan2','chan3','chan4'"
CHANNEL_ARRAY=($(echo ${CHANNELS//\'/} | tr ',' ' '))
REGISTRATION_SAMPLE=${FILE_BASENAME}_
REGISTRATION_CHANNEL=summedNorm
PUNCTA_EXTRACT_CHANNEL=summedNorm
REGISTRATION_WARP_CHANNELS="'${REGISTRATION_CHANNEL}',${CHANNELS}"

###### getopts

while getopts N:b:c:B:d:C:n:r:p:t:R:V:I:i:L:e:s:yh OPT
do
    case $OPT in
        N)  ROUND_NUM=$OPTARG
            if [ $ROUND_NUM != "auto" ]; then
                expr $ROUND_NUM + 1 > /dev/null 2>&1
                if [ $? -ge 2 ]; then
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
                expr $REFERENCE_ROUND + 1 > /dev/null 2>&1
                if [ $? -ge 2 ]; then
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
        i)  REPORTING_DIR=$OPTARG
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

if [ ! -d "${REGISTRATION_PROJ_DIR}"/scripts ]; then
    echo "No scripts dir. in Registration project: ${REGISTRATION_PROJ_DIR}/scripts"
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

echo "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh
if [ ! -f "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh ]; then
    echo "No 'import_cluster_profiles.sh'"
    exit 1
fi

if [ ! -f "${REGISTRATION_PROJ_DIR}"/MATLAB/loadExperimentParams.m ]; then
    echo "No 'loadExperimentParams.m' in Registration MATLAB"
    exit 1
fi

if [ ! -f ./loadParameters.m ]; then
    echo "No 'loadParameters.m' in ExSeqProcessing MATLAB"
    exit 1
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

STAGES=("profile-check" "color-correction" "normalization" "registration" "puncta-extraction" "transcripts")
REG_STAGES=("calc-descriptors" "register-with-descriptors")

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
echo "  # of rounds            :  ${ROUND_NUM}"
echo "  file basename          :  ${FILE_BASENAME}"
echo "  reference round        :  ${REFERENCE_ROUND}"
echo "  processing channels    :  ${CHANNELS}"
echo "  registration channel   :  ${REGISTRATION_CHANNEL}"
echo "  warp channels          :  ${REGISTRATION_WARP_CHANNELS}"
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
echo "  Reporting              :  ${REPORTING_DIR}"
echo "  Log                    :  ${LOG_DIR}"
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

stage_idx=0

###### check a cluster profile
echo "========================================================================="
echo "Cluster-profile check"; date
echo

if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    pushd "${REGISTRATION_PROJ_DIR}"
    "${REGISTRATION_PROJ_DIR}"/scripts/import_cluster_profiles.sh
    popd
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))


###### setup MATLAB scripts

# setup for Registration

sed -e "s#\(regparams.DATACHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(regparams.REGISTERCHANNEL\) *= *.*;#\1 = '${REGISTRATION_CHANNEL}';#" \
    -e "s#\(regparams.CHANNELS\) *= *.*;#\1 = {${REGISTRATION_WARP_CHANNELS}};#" \
    -e "s#\(regparams.INPUTDIR\) *= *.*;#\1 = '${NORMALIZATION_DIR}';#" \
    -e "s#\(regparams.OUTPUTDIR\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(regparams.FIXED_RUN\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
    -i.back \
    ./loadParameters.m

# setup for segmentation using Raj lab image tools

# tmux command?
#set -g mouse-select-window on

sed -e "s#\(params.deconvolutionImagesDir\) *= *.*;#\1 = '${DECONVOLUTION_DIR}';#" \
    -e "s#\(params.colorCorrectionImagesDir\) *= *.*;#\1 = '${COLOR_CORRECTION_DIR}';#" \
    -e "s#\(params.registeredImagesDir\) *= *.*;#\1 = '${REGISTRATION_DIR}';#" \
    -e "s#\(params.punctaSubvolumeDir\) *= *.*;#\1 = '${PUNCTA_DIR}';#" \
    -e "s#\(params.transcriptResultsDir\) *= *.*;#\1 = '${TRANSCRIPT_DIR}';#" \
    -e "s#\(params.reportingDir\) *= *.*;#\1 = '${REPORTING_DIR}';#" \
    -e "s#\(params.FILE_BASENAME\) *= *.*;#\1 = '${FILE_BASENAME}';#" \
    -e "s#\(params.NUM_ROUNDS\) *= *.*;#\1 = ${ROUND_NUM};#" \
    -e "s#\(params.REFERENCE_ROUND_PUNCTA\) *= *.*;#\1 = ${REFERENCE_ROUND};#" \
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

echo "========================================================================="
echo "Downsampling"; date
echo


# color correction
#echo "========================================================================="
#echo "Color correction"; date
#echo



if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then

matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-downsample.log -r "run('downsample_all.m');exit;"

    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-color-correction.log -r " for i=1:${ROUND_NUM};try; colorcorrection_3D_poc(i);catch; fprintf('POC FAIL, CATCH:\n'); colorcorrection_3D(i); end; end;exit;"


matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-downsample.log -r "run('downsample_applycolorshiftstofullres.m');exit;"

else
    echo "Skip!"
fi
echo

#echo "========================================================================="
#echo "Applying color correction to downsample"; date
#echo


stage_idx=$(( $stage_idx + 1 ))

# normalization
echo "========================================================================="
echo "Normalization"; date
echo

# Quick fix by Dan 01/29/2018
if [ ! "${SKIP_STAGES[$stage_idx]}" = "skip" ]; then
    
cp 1_deconvolution/*ch00.tif 2_color-correction/

#Process the full-resolution data
matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization.log -r "${ERR_HDL_PRECODE} normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},false); ${ERR_HDL_POSTCODE}"

   if ls matlab-normalization-*.log > /dev/null 2>&1; then
        mv matlab-normalization-*.log ${LOG_DIR}/
    else
        echo "No job log files."
    fi

#Process the downsampled data, if specified
matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-normalization-downsample.log -r "${ERR_HDL_PRECODE} loadParameters; if params.DO_DOWNSAMPLE; normalization('${COLOR_CORRECTION_DIR}','${NORMALIZATION_DIR}','${FILE_BASENAME}',{${CHANNELS}},${ROUND_NUM},true);end; ${ERR_HDL_POSTCODE}"

    if ls matlab-normalization-downsample*.log > /dev/null 2>&1; then
        mv matlab-normalization-downsample*.log ${LOG_DIR}/
    else
        echo "No job log files."
    fi
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
        
        # prepare normalized channel images for warp
        for((i=0; i<${#CHANNEL_ARRAY[*]}; i++))
        do
            for f in $(\ls ${COLOR_CORRECTION_DIR}/*_${CHANNEL_ARRAY[i]}.tif)
            do
                round_num=$(( $(echo $f | sed -ne 's/.*_round0*\([0-9]\+\)_.*.tif/\1/p') ))
        	if [[ $f = *"downsample"* ]]; then
  		    #Only create links to full-res images in this loop
 		    #echo "picked up a downsample on accident!, skipping"
                    continue
		fi 
                if [ $round_num -eq 0 ]; then
                    echo "round number is wrong."
                fi

                normalized_ch_file=$(printf "${NORMALIZATION_DIR}/${FILE_BASENAME}_round%03d_${CHANNEL_ARRAY[i]}.tif" $round_num)

                if [ ! -f $normalized_ch_file ]; then
                    ln -s $f $normalized_ch_file
                fi
            done
        done
        # prepare downsampled normalized channel images for warp
        for((i=0; i<${#CHANNEL_ARRAY[*]}; i++))
        do
            for f in $(\ls ${COLOR_CORRECTION_DIR}/*downsample*_${CHANNEL_ARRAY[i]}.tif)
            do
                round_num=$(( $(echo $f | sed -ne 's/.*_round0*\([0-9]\+\)_.*.tif/\1/p') ))
                if [ $round_num -eq 0 ]; then
                    echo "round number is wrong."
                fi

                normalized_ch_downsample_file=$(printf "${NORMALIZATION_DIR}/${FILE_BASENAME}-downsample_round%03d_${CHANNEL_ARRAY[i]}.tif" $round_num)

                if [ ! -f $normalized_ch_downsample_file ]; then
                    ln -s $f $normalized_ch_downsample_file
                fi
            done
        done

        for((i=1; i<=${ROUND_NUM}; i+=2))
        do
            if [ $i -eq ${ROUND_NUM} ]; then
                rounds=$i
            else
                rounds="$i $(( $i + 1 ))"
            fi
            # calculateDescriptors for two groups of rounds in parallel
            matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-calcDesc-group-${rounds/ /-}.log -r "${ERR_HDL_PRECODE} calculateDescriptorsInParallel([$rounds]); ${ERR_HDL_POSTCODE}"
    
            if ls matlab-calcDesc-*.log > /dev/null 2>&1; then
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

    if [ ! "${SKIP_REG_STAGES[$reg_stage_idx]}" = "skip" ]; then
        
        # make symbolic links of round-5 images because it is not necessary to warp them
        #TODO: Change this to be any round, currently hardcoding to 5
        for ch in ${REGISTRATION_CHANNEL} ${CHANNEL_ARRAY[*]}
        do
            normalized_file=${NORMALIZATION_DIR}/${FILE_BASENAME}_round004_${ch}.tif

            registered_tps_file=${REGISTRATION_DIR}/${FILE_BASENAME}_round004_${ch}_registered.tif
            registered_affine_file=${REGISTRATION_DIR}/${FILE_BASENAME}_round004_${ch}_affine.tif
            if [ ! -f $registered_affine_file ]; then
                ln -s $normalized_file $registered_affine_file
                ln -s $normalized_file $registered_tps_file
            fi
        done

#Because the matching is currently single-threaded, we can parpool it in one loop
matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-registerWDesc-${i}.log -r "parpool; parfor i = 1:20; if i==4;fprintf('Skipping reference round\n');continue;end; calcCorrespondences(i);registerWithCorrespondences(i,true);registerWithCorrespondences(i,false);end;exit;"

        #ifor((i=1; i<=${ROUND_NUM}; i++))
        #do
        #    if [ ${REFERENCE_ROUND} -eq $i ]; then
        #    echo "Skipping registration of the reference round"
        #    continue   
        #    fi
        #    # registerWithDescriptors for ${REFERENCE_ROUND} and i
        #    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-registerWDesc-${i}.log -r "${ERR_HDL_PRECODE} calcCorrespondences(${i});registerWithCorrespondences(${i}); ${ERR_HDL_POSTCODE}"

        #done
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
matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-puncta-extraction.log -r "${ERR_HDL_PRECODE} loadParameters; punctafeinder_simple; puncta_subvolumes_simple;${ERR_HDL_POSTCODE}"    
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
    matlab -nodisplay -nosplash -logfile ${LOG_DIR}/matlab-transcript-making.log -r "${ERR_HDL_PRECODE} loadParameters; basecalling_simple;  ${ERR_HDL_POSTCODE}"
else
    echo "Skip!"
fi
echo

stage_idx=$(( $stage_idx + 1 ))




echo "========================================================================="
echo "pipeline finished"; date
echo

