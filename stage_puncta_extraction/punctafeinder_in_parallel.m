function success_code = punctafeinder()

    loadParameters;

    run_num_list = 1:params.NUM_ROUNDS;
    arg_list = {};
    postfix_list = {};
    for run_num = run_num_list
        arg_list{end+1} = {run_num};
        postfix_list{end+1} = num2str(run_num);
    end

    %centroids are the location
    [success_code,centroids] = batch_process('puncta-extraction', @punctafeinder_round, run_num_list, arg_list,...
        postfix_list, params.PUNCTA_MAX_POOL_SIZE, params.NUM_ROUNDS, params.PUNCTA_MAX_RUN_JOBS, params.WAIT_SEC, 1, params.NUM_CHANNELS);

    makeCentroidsAndVoxels(centroids);

end
