function setup_cluster_profile()

    loadParameters;

    try
        cprof = parcluster('local_logical_cores');
    catch
        cprof_local = parcluster('local');
        saveAsProfile(cprof_local,'local_logical_cores');
        cprof = parcluster('local_logical_cores');
    end

    cprof.NumWorkers = params.NUM_LOGICAL_CORES;
    cprof.NumThreads = 1;

    saveProfile(cprof);

    parallel.defaultClusterProfile('local_logical_cores');

end

