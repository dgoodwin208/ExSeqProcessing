% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = calculateDescriptorsInParallel(run_num_list)

    success_code = batch_process('calcDesc', @calculateDescriptors, run_num_list, []);

end

