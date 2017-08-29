/*=================================================================
 * mergesortfiles.cpp - merge two sorted data files to one file
 *                  the first column is sorted
 *
 *  mergesortfiles(outputdir, num_sort_elements, mergefile_list)
 *
 *  outputdir(char):  directory to be stored data files
 *  num_sort_elements(scalar):  # of elements in one sorted group
 *  mergefile_list(cell):  a list of triplets; (in1file, in2file, outfile)
 *
 *=================================================================*/
 

#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

#include <string.h>
#include "mex.h"

const auto FILECHECK_INTERVAL_SEC = std::chrono::seconds(1);
const auto FILECHECK_TIMEOUT_SEC  = std::chrono::seconds(60*300);
//const auto FILECHECK_TIMEOUT_SEC = std::chrono::seconds(10);
const unsigned int FILEREAD_BUFSIZE  = 1024*4;
const unsigned int FILEWRITE_BUFSIZE = 1024*4;

int
mergeTwoFiles(const std::string& outputdir, const std::string& in1postfix, const std::string& in2postfix, const std::string& outpostfix)
{
    std::string in1_filename = outputdir + "/" + in1postfix;
    std::string in2_filename = outputdir + "/" + in2postfix;
    std::string tmp_out_filename = outputdir + "/.tmp." + outpostfix;

    std::ifstream if1(in1_filename, std::ios::in | std::ios::binary);
    std::ifstream if2(in2_filename, std::ios::in | std::ios::binary);
    auto timer_start = std::chrono::steady_clock::now();
    while (if1.fail() || ! if1.is_open() || if2.fail() || ! if2.is_open()) {
        std::this_thread::sleep_for(FILECHECK_INTERVAL_SEC);

        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timer_start);
        if (elapsed_time > FILECHECK_TIMEOUT_SEC) {
            //mexErrMsgIdAndTxt("MATLAB:mergesortfiles:timeout","Timeout of waiting for a file to be merged.");
            mexPrintf("timeout of waiting for a file to be merged.\n");
            return -1;
        }

        if1.close();
        if2.close();
        if1.open(in1_filename, std::ifstream::in | std::ifstream::binary);
        if2.open(in2_filename, std::ifstream::in | std::ifstream::binary);
    }

    std::ofstream of(tmp_out_filename);
    if (! of.is_open()) {
        mexPrintf("cannot open an output file for merge.\n");
        return -1;
    }

    std::vector<double> in1_buf(FILEREAD_BUFSIZE / sizeof(double));
    std::vector<double> in2_buf(FILEREAD_BUFSIZE / sizeof(double));
    std::vector<double> out_buf(FILEWRITE_BUFSIZE / sizeof(double));
    size_t in1_buf_i = 0;
    size_t in2_buf_i = 0;
    size_t out_buf_i = 0;
    size_t in1_buf_size  = 0;
    size_t in2_buf_size  = 0;
    size_t read_buf_size = 0;

    size_t in1_read_bytes = 0;
    size_t in2_read_bytes = 0;
    size_t total_write_bytes = 0;

    while (in1_buf_i != -1 || in2_buf_i != -1) {
        if (in1_buf_i == in1_buf_size) {
            read_buf_size = if1.read((char*)in1_buf.data(), FILEREAD_BUFSIZE).gcount();
            if (read_buf_size > 0) {
                in1_buf_size = read_buf_size / sizeof(double);
                in1_buf_i = 0;
                in1_read_bytes += read_buf_size;
            } else {
                in1_buf_i = -1;
            }
        }
        if (in2_buf_i == in2_buf_size) {
            read_buf_size = if2.read((char*)in2_buf.data(), FILEREAD_BUFSIZE).gcount();
            if (read_buf_size > 0) {
                in2_buf_size = read_buf_size / sizeof(double);
                in2_buf_i = 0;
                in2_read_bytes += read_buf_size;
            } else {
                in2_buf_i = -1;
            }
        }

        if (in1_buf_i != -1 && in2_buf_i != -1) {
            if (in1_buf[in1_buf_i] <= in2_buf[in2_buf_i]) {
                // copy two elements
                out_buf[out_buf_i++] = in1_buf[in1_buf_i++];
                out_buf[out_buf_i++] = in1_buf[in1_buf_i++];
            } else {
                out_buf[out_buf_i++] = in2_buf[in2_buf_i++];
                out_buf[out_buf_i++] = in2_buf[in2_buf_i++];
            }

            if (out_buf_i == out_buf.size()) {
                of.write((char*)out_buf.data(), out_buf.size() * sizeof(double));
                total_write_bytes += out_buf.size() * sizeof(double);
                out_buf_i = 0;
            }
        } else {
            if (out_buf_i > 0) {
                of.write((char*)out_buf.data(), out_buf_i * sizeof(double));
                total_write_bytes += out_buf_i * sizeof(double);
                out_buf_i = 0;
            }

            if (in1_buf_i != -1) {
                of.write((char*)&in1_buf[in1_buf_i], (in1_buf_size - in1_buf_i) * sizeof(double));
                total_write_bytes += (in1_buf_size - in1_buf_i) * sizeof(double);
                read_buf_size = if1.read((char*)in1_buf.data(), FILEREAD_BUFSIZE).gcount();
                int count = 1;
                while (read_buf_size > 0) {
                    in1_read_bytes += read_buf_size;
                    of.write((char*)in1_buf.data(), read_buf_size);
                    total_write_bytes += read_buf_size;
                    read_buf_size = if1.read((char *)in1_buf.data(), FILEREAD_BUFSIZE).gcount();
                    count++;
                }
                in1_buf_i = -1;
            } else if (in2_buf_i != -1) {
                of.write((char*)&in2_buf[in2_buf_i], (in2_buf_size - in2_buf_i) * sizeof(double));
                total_write_bytes += (in2_buf_size - in2_buf_i) * sizeof(double);
                read_buf_size = if2.read((char*)in2_buf.data(), FILEREAD_BUFSIZE).gcount();
                int count = 1;
                while (read_buf_size > 0) {
                    in2_read_bytes += read_buf_size;
                    of.write((char*)in2_buf.data(), read_buf_size);
                    total_write_bytes += read_buf_size;
                    read_buf_size = if2.read((char*)in2_buf.data(), FILEREAD_BUFSIZE).gcount();
                    count++;
                }
                in2_buf_i = -1;
            }
        }
    }
//    mexPrintf("[%010lu] total write(%ld), read(%ld, %ld) end. %s, %s\n", std::this_thread::get_id(),
//        total_write_bytes, in1_read_bytes, in2_read_bytes, in1postfix.c_str(), in2postfix.c_str());

    of.close();

    std::string out_filename = outputdir + "/" + outpostfix;
    int ret = rename(tmp_out_filename.c_str(), out_filename.c_str());
    if (ret != 0) {
        mexPrintf("cannot rename the tmp output file to tmp output file.\n");
        return -1;
    }

    if1.close();
    if2.close();

    remove(in1_filename.c_str());
    remove(in2_filename.c_str());

    return 0;
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    /* Check for proper number of input and output arguments */
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "MATLAB:mergesortfiles:minrhs","Two input arguments required.");
    } 
    if (nlhs > 0) {
        mexErrMsgIdAndTxt( "MATLAB:mergesortfiles:maxrhs","Too many output arguments.");
    }

    /* make sure input arguments are expected types */
    if ( !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:mergesortfiles:notChar","1st input arg must be type char.");
    }
    if ( !mxIsCell(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:mergesortfiles:notCell","2nd input arg must be type cell.");
    }


    std::string outputdir = std::string(mxArrayToString(prhs[0]));
    mexPrintf("%s\n", outputdir.c_str());

    const mxArray *root_cell_ptr = prhs[1];
    mwSize total_num_cells = mxGetNumberOfElements(root_cell_ptr);
    mexPrintf("total_num_cells = %d\n", total_num_cells);

    std::vector<std::future<int>> futures;
    for (int i = 0; i < total_num_cells; i++) {
        const mxArray *triplet_cell_ptr = mxGetCell(root_cell_ptr, i);
        if (triplet_cell_ptr == NULL) {
            mexPrintf("skip empty triplet cell.\n");
            continue;
        }

        mwSize num_triplet_cells = mxGetNumberOfElements(triplet_cell_ptr);
//        mexPrintf("num_triplet_cells = %d\n", num_triplet_cells);
        if (num_triplet_cells != 3) {
            mexErrMsgIdAndTxt("MATLAB:mergesortfiles:invalidInputSize","Input cell triplet must be 3.");
        }

        const mxArray *in1_elem_ptr = mxGetCell(triplet_cell_ptr, 0);
        const mxArray *in2_elem_ptr = mxGetCell(triplet_cell_ptr, 1);
        const mxArray *out_elem_ptr = mxGetCell(triplet_cell_ptr, 2);
        if (in1_elem_ptr == NULL || in2_elem_ptr == NULL || out_elem_ptr == NULL) {
            mexPrintf("skip invalid file names.\n");
            continue;
        }

        if ( !mxIsChar(in1_elem_ptr)) {
            mexPrintf("skip invalid input.1 file name.\n");
            break;
        }
        if ( !mxIsChar(in2_elem_ptr)) {
            mexPrintf("skip invalid input.2 file name.\n");
            break;
        }
        if ( !mxIsChar(out_elem_ptr)) {
            mexPrintf("skip invalid output file name.\n");
            break;
        }
        std::string in1_name(mxArrayToString(in1_elem_ptr));
        std::string in2_name(mxArrayToString(in2_elem_ptr));
        std::string out_name(mxArrayToString(out_elem_ptr));

        futures.push_back(std::async(std::launch::async, mergeTwoFiles, outputdir, in1_name, in2_name, out_name));
        mexPrintf("%s, %s -> %s\n", in1_name.c_str(), in2_name.c_str(), out_name.c_str());
    }

    mexPrintf("waiting...\n");
    for (int i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        mexPrintf("[%d] done - %d\n", i, ret);
    }
    mexPrintf("done\n");

    return;
}

