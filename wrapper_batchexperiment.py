#!/usr/bin/env python
import yaml
import argparse
import sys
import subprocess
import os
from pprint import pprint
from time import gmtime, strftime
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--nsplit', type=int, default=1)
    parser.add_argument('--splitid', type=int, default=0)
    parser.add_argument('--skipsteps', type=str, default=None)
    parser.set_defaults(exec=False)


    args = vars(parser.parse_args())
    filename = args['config']
    if filename is None:
        print("Must give a parameter to --config")

    with open(f'{filename}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config.update(args)

    #Generate the list of all fovs that must be processed
    total_fovs = np.arange(np.prod(config['montage_size']))
    fovs_skip = config['fovs_to_skip']
    fovs_skip_mask = [(f not in fovs_skip) for f in total_fovs]
    fovs_to_process = total_fovs[fovs_skip_mask]

    #Is this list of fovs_to_process being split over several instances?
    fov_batches = np.array_split(fovs_to_process,config['nsplit'])
    fovs_willprocess = fov_batches[config['splitid']]



    base_dir = config['base_dir'] 
    base_filename = f"{config['basename']}"
    skipsteps = config['skipsteps']

    # Loop over the specified FOVs
    for fov in fovs_willprocess:
        # Create FOV string for directory and base name
        # e.g i fov=1 then fov_name = "F001"
        fov_name = f"F{fov:03d}"
        # parent directory of FOV containing the ExSeq directories
        # (1_deconvolution, 2_color-correction, ...)
        output_dir = os.path.join(base_dir  , fov_name)
        # images are stored in each FOV folder in a 0_raw folder
        input_dir  = os.path.join(output_dir, "0_raw")
        # get current time and use that for a sub directory in the log directory
        # this makes it so there is a log folders per run of the pipeline
        # (reusing a log folder would overwrite old logs)
        logDirWithDateAppended = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        log_dir    = os.path.join(output_dir, "logs", logDirWithDateAppended)
        report_dir = os.path.join(log_dir   , "imgs")


        # In this case want to store and create a log and
        # reporting directory per FOV
        os.makedirs(log_dir   )
        os.makedirs(report_dir)

        # A convention we use is to also append the FOV to experiment
        # base name (experiment-F001 etc.)
        final_base = base_filename + "-" + fov_name

        # Create the exact string that would run the pipeline
        # specifying the appropriate folders and importantly
        # forcing yes to all interactive prompts.
        # The last line here redirects all output (standard and error)
        # to a file as well as to the screen.
        runPipelineLogFile = os.path.join(log_dir, "runPipeline.log")
        
        #TODO: There was a very strange issue with some whitespace that 
        #necessitated the hardocded '-F hdf5'. That must be changed in the future
        if skipsteps is None:
            commandStr = "./runPipeline.sh -F hdf5" + \
                         " -I " + input_dir  + \
                         " -O " + output_dir + \
                         " -b " + final_base + \
                         " -L " + log_dir    + \
                         " -i " + report_dir + \
                         " -y "              + \
                         f" -B {config['ref_round']}" + \
                         " 2>&1 | tee " + runPipelineLogFile
        else:
            commandStr = "./runPipeline.sh -F hdf5" + \
                     " -I " + input_dir  + \
                     " -O " + output_dir + \
                     " -b " + final_base + \
                     " -L " + log_dir    + \
                     " -i " + report_dir + \
                     f' -s "{skipsteps}"' + \
                     " -y "              + \
                     f" -B {config['ref_round']}" + \
                     " 2>&1 | tee " + runPipelineLogFile

        # Run the pipeline waiting for return
        # and piping output through
        print("=========== Processing " + fov_name + " ===========")
        #print(commandStr)
        subprocess.run(                        \
          commandStr, shell=True, check=True,  \
          stderr=sys.stderr, stdout=sys.stdout \
        )

