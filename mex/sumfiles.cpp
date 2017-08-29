/*=================================================================
 * sumfiles.cpp - sum image files to one file
 *
 *  sumfiles(outputdir, sumfile_list, out_file)
 *
 *  outputdir(char):  directory to be stored out file
 *  sumfile_list(cell):  a list of quartets; (in1file, in2file, ..)
 *  out_file(char):  output file name
 *
 *=================================================================*/
 

#include <thread>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <vector>

#include "mex.h"

const auto FILECHECK_INTERVAL_SEC = std::chrono::seconds(1);
const auto FILECHECK_TIMEOUT_SEC  = std::chrono::seconds(60*300);
const unsigned int FILE_BUFSIZE = 1024*4;
const size_t NUM_ELEMENTS = 2;
const size_t IDX_SUM = 1;

int
sumFiles(
    const std::string& outputdir,
    const std::vector<std::string>& inpostfixes,
    const std::string& outpostfix)
{
    std::vector<std::string> in_filenames;
    std::string tmp_out_filename = outputdir + "/.tmp." + outpostfix;
    for (int i = 0; i < inpostfixes.size(); i++) {
        std::string in_fname = outputdir + "/" + inpostfixes[i];
        in_filenames.push_back(in_fname);
    }

    std::vector<std::ifstream> ifs(inpostfixes.size());
    auto timer_start = std::chrono::steady_clock::now();
    while (1) {
        bool is_all_open = true;
        for (int i = 0; i < ifs.size(); i++) {
            if (ifs[i].fail() || ! ifs[i].is_open()) {
                ifs[i].close();
                ifs[i].open(in_filenames[i], std::ifstream::in | std::ifstream::binary);
            }
            if (ifs[i].fail() || ! ifs[i].is_open()) {
                is_all_open = false;
                mexPrintf("cannot open %s\n", in_filenames[i].c_str());
            }
        }

        if (is_all_open)
            break;

        std::this_thread::sleep_for(FILECHECK_INTERVAL_SEC);

        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timer_start);
        if (elapsed_time > FILECHECK_TIMEOUT_SEC) {
            mexPrintf("timeout of waiting for a file to be mean.\n");
            return -1;
        }
    }

    std::ofstream of(tmp_out_filename);
    if (! of.is_open()) {
        mexPrintf("cannot open an output file for mean.\n");
        return -1;
    }

    std::vector<double> in_buf(FILE_BUFSIZE / sizeof(double));
    std::vector<double> out_buf(FILE_BUFSIZE / sizeof(double) / NUM_ELEMENTS);
    size_t read_buf_size = 0;
    size_t prev_buf_size = 0;
    size_t total_write_bytes = 0;

    int count = 0;
    while (1) {
        for (int i = 0; i < read_buf_size / NUM_ELEMENTS; i++) {
            out_buf[i] = 0.0;
        }

        for (int i = 0; i < ifs.size(); i++) {
            read_buf_size = ifs[i].read((char*)in_buf.data(), FILE_BUFSIZE).gcount();
            read_buf_size /= sizeof(double);

            if (i == 0) {
                prev_buf_size = read_buf_size;
            } else if (prev_buf_size != read_buf_size) {
                mexPrintf("read buffer size is different.\n");
                return -1;
            }

            for (int j = 0; j < read_buf_size / NUM_ELEMENTS; j++) {
                out_buf[j] += in_buf[j * NUM_ELEMENTS + IDX_SUM];
            }
        }
        if (read_buf_size == 0) {
            break;
        }

        of.write((char*)out_buf.data(), read_buf_size / NUM_ELEMENTS * sizeof(double));
        total_write_bytes += read_buf_size / NUM_ELEMENTS * sizeof(double);
    }

    of.close();

    std::string out_filename = outputdir + "/" + outpostfix;
    int ret = rename(tmp_out_filename.c_str(), out_filename.c_str());
    if (ret != 0) {
        mexPrintf("cannot rename the tmp output file to tmp output file.\n");
        return -1;
    }
//    mexPrintf("[%010lu] total write(%ld) end. %s\n", std::this_thread::get_id(), total_write_bytes, outpostfix.c_str());

    for (int i = 0; i < ifs.size(); i++) {
        ifs[i].close();
    }

    return 0;
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    /* Check for proper number of input and output arguments */
    if (nrhs != 3) {
        mexErrMsgIdAndTxt( "MATLAB:meanfiles:minrhs","3 input arguments required.");
    } 
    if (nlhs > 0) {
        mexErrMsgIdAndTxt( "MATLAB:meanfiles:maxrhs","Too many output arguments.");
    }

    /* make sure input arguments are expected types */
    if ( !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:meanfiles:notChar","1st input arg must be type char.");
    }
    if ( !mxIsCell(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:meanfiles:notCell","2nd input arg must be type cell.");
    }
    if ( !mxIsChar(prhs[2])) {
        mexErrMsgIdAndTxt("MATLAB:meanfiles:notChar","3rd input arg must be type char.");
    }


    std::string outputdir = std::string(mxArrayToString(prhs[0]));
    mexPrintf("%s\n", outputdir.c_str());

    const mxArray *root_cell_ptr = prhs[1];
    mwSize total_num_cells = mxGetNumberOfElements(root_cell_ptr);
    mexPrintf("total_num_cells = %d\n", total_num_cells);

    std::vector<std::string> inpostfixes;
    for (int i = 0; i < total_num_cells; i++) {
        const mxArray *elem_ptr = mxGetCell(root_cell_ptr, i);
        if (elem_ptr == NULL) {
            mexPrintf("skip empty cell.\n");
            continue;
        }
        if ( !mxIsChar(elem_ptr)) {
            mexPrintf("skip invalid input file name.\n");
            break;
        }
        std::string in_name(mxArrayToString(elem_ptr));
        inpostfixes.push_back(in_name);
        mexPrintf("in[%d] = %s\n", i, inpostfixes[i].c_str());
    }
    std::string outpostfix = std::string(mxArrayToString(prhs[2]));
    mexPrintf("out = %s\n", outpostfix.c_str());

    sumFiles(outputdir, inpostfixes, outpostfix);

    mexPrintf("done\n");

    return;
}

