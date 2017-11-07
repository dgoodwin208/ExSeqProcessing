% INPUTS:
% run_num_list is the index list of the experiment for the specified sample
% job_idx is the int number assigned to each job 
% desc_size is the total number of discretized sections of the image
function [run_num, target_idx] = getJobIds(run_num_list, job_idx, desc_size)
    % determine run_number idx 
    run_num = run_num_list(ceil(job_idx / desc_size));

    % convert job_idx 1 - 9
    target_idx = mod(job_idx, desc_size);
    if target_idx == 0
        target_idx = desc_size;
    end
end
