#!/usr/bin/env python
import sys
import subprocess
import os
from pprint import pprint
from time import gmtime, strftime

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(                 \
         "Error expecting: " + \
         "starting_fov "     + \
         "num_fovs "         + \
         "base_dir "         + \
         "base_filename "    + \
         "as parameters"       \
        )
        exit(1)
    # read command line parameters converting where necessary
    starting_fov  =             int(sys.argv[1])
    num_fovs      =             int(sys.argv[2])
    base_dir      = os.path.abspath(sys.argv[3])
    base_filename =                 sys.argv[4]
    # Loop over the specified FOVs
    for fov in range(starting_fov,starting_fov+num_fovs):
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
        commandStr = "./runPipeline.sh " + \
                     " -I " + input_dir  + \
                     " -O " + output_dir + \
                     " -b " + final_base + \
                     " -L " + log_dir    + \
                     " -i " + report_dir + \
                     " -y "              + \
                     " 2>&1 | tee " + runPipelineLogFile

        # Run the pipeline waiting for return
        # and piping output through
        print("=========== Processing " + fov_name + " ===========")
        subprocess.run(                        \
          commandStr, shell=True, check=True,  \
          stderr=sys.stderr, stdout=sys.stdout \
        )
