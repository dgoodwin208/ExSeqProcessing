matlab -nodisplay -logfile ../logs/colorCorrectionProcessing.log -r "run('../4_puncta-extraction/startup.m'); parpool(5); parfor roundnum=1:20; interpolateDataVolume(roundnum,.5/.17); end;" 
