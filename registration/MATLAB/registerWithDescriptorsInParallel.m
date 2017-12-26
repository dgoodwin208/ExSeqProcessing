% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
function success_code = registerWithDescriptorsInParallel(run_num_list)
    
    success_code = batch_process('regDesc', @registerWithDescriptors, run_num_list, []);

end
