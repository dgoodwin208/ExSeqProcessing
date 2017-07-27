matlab -nodisplay -logfile logs/colorCorrectionProcessing.log -r " parpool(5); parfor roundnum=1:20; colorcorrection_3D(roundnum); end;" 
