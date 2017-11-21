% INPUTS:
% run_num is the index of the experiment for the specified sample
% target_idx is the index of subvolumes to calculate keypoints and descripors
function result = testCalcDesc(run_num, target_idx)

    profile on -detail builtin
    result = 1;

    disp(['batch: ', num2str(run_num), ' - ', num2str(target_idx)])
    calculateDescriptors(run_num, target_idx, target_idx);

    profile off
    profsave(profile('info'), ['profile_results-',num2str(run_num),'-',num2str(target_idx)])
    result = 0; % success
end
