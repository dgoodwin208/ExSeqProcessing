#include <vector>
#include <string>

#include "spdlog/spdlog.h"
#include "mex.h"
#include "mex-utils/tiffs.h"
#include "mex-utils/hdf5.h"

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        //spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr);
        spdlog::set_level(spdlog::level::trace);
        logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex-check.log");
        }
        //logger->flush_on(spdlog::level::err);
        logger->flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        mexErrMsgIdAndTxt("MATLAB:check_mexutils:logNoInit", "Log initialization failed: %s", ex.what());
    }

    try {
        logger->info("{:=>50}", " check_mexutils start");

        /* Check for proper number of input and output arguments */
        if (nrhs != 1) {
            mexErrMsgIdAndTxt( "MATLAB:check_mexutils:minrhs", "1 input argument required.");
        } 
        if (nlhs > 0) {
            mexErrMsgIdAndTxt( "MATLAB:check_mexutils:maxrhs", "Too many output arguments.");
        }

        /* make sure input arguments are expected types */
        if ( !mxIsCell(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:check_mexutils:notCell", "1st input arg must be type cell.");
        }

        const mxArray *root_cell_ptr = prhs[0];
        mwSize total_num_cells = mxGetNumberOfElements(root_cell_ptr);
        if (total_num_cells == 0) {
            mexErrMsgIdAndTxt("MATLAB:check_mexutils:invalidInput", "Filename is not given.");
        }

        std::vector<std::string> fnames;
        for (int i = 0; i < total_num_cells; i++) {
            const mxArray *elem_ptr = mxGetCell(root_cell_ptr, i);
            if (elem_ptr == NULL) {
                mexErrMsgIdAndTxt("MATLAB:check_mexutils:invalidInput", "Empty cell.");
            }
            if ( !mxIsChar(elem_ptr)) {
                mexErrMsgIdAndTxt("MATLAB:check_mexutils:invalidInput", "Invalid input filename.");
            }

            std::string fname(mxArrayToString(elem_ptr));
            fnames.push_back(fname);
            logger->debug("in[{}] = {}", i, fnames[i].c_str());

            if (fname.substr(fname.size() - 4, 4) != ".tif" && fname.substr(fname.size() - 3, 3) != ".h5") {
                mexErrMsgIdAndTxt("MATLAB:check_mexutils:invalidInput", "Invalid input filetype.");
            }
        }

        size_t image_width, image_height, num_slices;
        bool use_hdf5;
        if (fnames[0].substr(fnames[0].size() - 4, 4) == ".tif") {
            mexutils::gettiffinfo(fnames[0], image_width, image_height, num_slices);
            use_hdf5 = false;
        } else if (fnames[0].substr(fnames[0].size() - 3, 3) == ".h5") {
            mexutils::gethdf5finfo(fnames[0], image_width, image_height, num_slices);
            use_hdf5 = true;
        }

        if (use_hdf5) {
            for (auto h5_file : fnames) {
                logger->info("loadhdf5 start {}", h5_file);
                std::shared_ptr<std::vector<uint16_t>> image = std::make_shared<std::vector<uint16_t>>();

                mexutils::loadhdf5(h5_file, 0, num_slices - 1, image_height, image_width, image);

                logger->info("slice ({}), image={}", num_slices, image->size());
                logger->info("loadhdf5 end   {}", h5_file);
            }
        } else {
            for (auto tif_file : fnames) {
                logger->info("loadtiff start {}", tif_file);
                std::shared_ptr<std::vector<uint16_t>> image = std::make_shared<std::vector<uint16_t>>();

                mexutils::loadtiff(tif_file, 0, num_slices, image);

                logger->info("slice ({}), image={}", image->size());
                logger->info("loadtiff end   {}", tif_file);
            }
        }

        logger->info("{:=>50}", " check_mexutils end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

