/*=================================================================
 * substituteToNormValues.cpp - substitute sorted values to mean values in quantilenorm_simple
 *
 *  substituteToNormValues(outputdir, mean_file, in_file_list, out_file_list)
 *
 *  outputdir(char):  directory to be stored out file
 *  mean_file(char):  mean file name
 *  in_file_list(cell):  a list of input file names
 *  out_file_list(cell):  a list of output file names
 *
 *=================================================================*/
 

#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>

#include "mex.h"

const auto FILECHECK_INTERVAL_SEC = std::chrono::seconds(1);
const auto FILECHECK_TIMEOUT_SEC  = std::chrono::seconds(60*300);

void
writeMeanValues(
    std::vector<double>& mean_value,
    std::vector<double>& mean_index,
    std::ofstream &of,
    size_t &total_write_bytes)
{
    double sum = 0.0;
    for_each(mean_value.begin(), mean_value.end(),
        [&sum] (double &v) { sum += v; });
    double mean = sum / (double)mean_value.size();

    for (int i = 0; i < mean_value.size(); i++) {
        of.write((char*)&mean_index[i], sizeof(double));
        of.write((char*)&mean, sizeof(double));
        total_write_bytes += sizeof(double) * 2;
//        mexPrintf("write[%d] - mean(%f,%f)\n", i, mean_index[i], mean);
    }

    mean_value.clear();
    mean_index.clear();
}

int
substituteToNormValues(
    const std::string& outputdir,
    const std::string& meanpostfix,
    const std::string& inpostfix,
    const std::string& outpostfix)
{
    std::string mean_filename    = outputdir + "/" + meanpostfix;
    std::string in_filename      = outputdir + "/" + inpostfix;
    std::string tmp_out_filename = outputdir + "/.tmp." + outpostfix;
    mexPrintf("%s, %s\n",mean_filename.c_str(),in_filename.c_str());

    std::ifstream mnf(mean_filename, std::ios::in | std::ios::binary);
    std::ifstream inf(in_filename,   std::ios::in | std::ios::binary);
    auto timer_start = std::chrono::steady_clock::now();
    while (mnf.fail() || ! mnf.is_open() || inf.fail() || ! inf.is_open()) {
        std::this_thread::sleep_for(FILECHECK_INTERVAL_SEC);

//        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timer_start);
//        if (elapsed_time > FILECHECK_TIMEOUT_SEC) {
//            mexPrintf("timeout of waiting for a file.\n");
//            return -1;
//        }

        mnf.close();
        inf.close();
        mnf.open(mean_filename, std::ifstream::in | std::ifstream::binary);
        inf.open(in_filename,   std::ifstream::in | std::ifstream::binary);
    }

    std::ofstream of(tmp_out_filename);
    if (! of.is_open()) {
        mexPrintf("cannot open an output file; %s.\n", tmp_out_filename.c_str());
        return -1;
    }

    double in_buf[2][2];
    double mean_buf[2];
    size_t read_in_size = 0;
    size_t read_mean_size = 0;
    size_t total_write_bytes = 0;
    int cur_buf_i = 0;
    int nxt_buf_i = 1;
    std::vector<double> mean_value;
    std::vector<double> mean_index;

    read_in_size   = inf.read((char*)&in_buf[cur_buf_i][0], sizeof(double) * 2).gcount();
    read_mean_size = mnf.read((char*)&mean_buf[cur_buf_i], sizeof(double)).gcount();

    while (read_mean_size != 0 && read_in_size != 0) {
//        mexPrintf("cur_buf_i = %d, nxt_buf_i = %d\n", cur_buf_i, nxt_buf_i);
//        mexPrintf("in_buf[%d][0] = %f\n", cur_buf_i, in_buf[cur_buf_i][0]);
//        mexPrintf("in_buf[%d][1] = %f\n", cur_buf_i, in_buf[cur_buf_i][1]);
//        mexPrintf("mean_buf[%d]  = %f\n", cur_buf_i, mean_buf[cur_buf_i]);

        read_in_size   = inf.read((char*)&in_buf[nxt_buf_i][0], sizeof(double) * 2).gcount();
        read_mean_size = mnf.read((char*)&mean_buf[nxt_buf_i], sizeof(double)).gcount();
//        mexPrintf("in_buf[%d][0] = %f\n", nxt_buf_i, in_buf[nxt_buf_i][0]);
//        mexPrintf("in_buf[%d][1] = %f\n", nxt_buf_i, in_buf[nxt_buf_i][1]);
//        mexPrintf("mean_buf[%d]  = %f\n", nxt_buf_i, mean_buf[nxt_buf_i]);
        if (read_mean_size == 0 && read_in_size == 0) {
            in_buf[nxt_buf_i][0] = -1.0; // dummy for difference intently as the last position
        } else if (read_in_size != read_mean_size * 2) {
            mexPrintf("file size is different.\n");
            return -1;
        }

        // ([*][0], [*][1]) .. (actual value, index)
//        mexPrintf("cur: %f, nxt: %f\n", in_buf[cur_buf_i][0], in_buf[nxt_buf_i][0]);
        if (in_buf[cur_buf_i][0] != in_buf[nxt_buf_i][0]) {
            if (mean_value.empty()) {
//                mexPrintf("copy - mean(%f,%f)\n", in_buf[cur_buf_i][1], mean_buf[cur_buf_i]);
                of.write((char*)&in_buf[cur_buf_i][1], sizeof(double));
                of.write((char*)&mean_buf[cur_buf_i],  sizeof(double));
                total_write_bytes += sizeof(double) * 2;
            } else {
//                mexPrintf("write - mean\n");
                mean_value.push_back(mean_buf[cur_buf_i]);
                mean_index.push_back(in_buf[cur_buf_i][1]);
                writeMeanValues(mean_value, mean_index, of, total_write_bytes);
            }
        } else {
            mean_value.push_back(mean_buf[cur_buf_i]);
            mean_index.push_back(in_buf[cur_buf_i][1]);
//            mexPrintf("push - mean = %f, index = %f\n", mean_buf[cur_buf_i], in_buf[cur_buf_i][1]);
        }

        cur_buf_i = cur_buf_i ^ 1;
        nxt_buf_i = nxt_buf_i ^ 1;
    }

    of.close();

    std::string out_filename = outputdir + "/" + outpostfix;
    int ret = rename(tmp_out_filename.c_str(), out_filename.c_str());
    if (ret != 0) {
        mexPrintf("cannot rename the tmp output file to tmp output file.\n");
        return -1;
    }

    mnf.close();
    inf.close();

//    mexPrintf("[%010lu] total write(%ld) end. %s\n",std::this_thread::get_id(),total_write_bytes,out_filename.c_str());

    return 0;
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    /* Check for proper number of input and output arguments */
    if (nrhs != 4) {
        mexErrMsgIdAndTxt( "MATLAB:substituteToNormValues:minrhs","4 input arguments required.");
    } 
    if (nlhs > 0) {
        mexErrMsgIdAndTxt( "MATLAB:substituteToNormValues:maxrhs","Too many output arguments.");
    }

    /* make sure input arguments are expected types */
    if ( !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:substituteToNormValues:notChar","1st input arg must be type char.");
    }
    if ( !mxIsChar(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:substituteToNormValues:notChar","2nd input arg must be type char.");
    }
    if ( !mxIsCell(prhs[2])) {
        mexErrMsgIdAndTxt("MATLAB:substituteToNormValues:notCell","3rd input arg must be type cell.");
    }
    if ( !mxIsCell(prhs[3])) {
        mexErrMsgIdAndTxt("MATLAB:substituteToNormValues:notCell","4th input arg must be type cell.");
    }


    std::string outputdir = std::string(mxArrayToString(prhs[0]));
    mexPrintf("%s\n", outputdir.c_str());

    std::string meanpostfix = std::string(mxArrayToString(prhs[1]));
    mexPrintf("mean = %s\n", meanpostfix.c_str());

    const mxArray *in_cell_ptr = prhs[2];
    mwSize total_num_in_cells = mxGetNumberOfElements(in_cell_ptr);
//    mexPrintf("total_num_in_cells = %d\n", total_num_in_cells);

    const mxArray *out_cell_ptr = prhs[3];
    mwSize total_num_out_cells = mxGetNumberOfElements(out_cell_ptr);
//    mexPrintf("total_num_out_cells = %d\n", total_num_out_cells);
    if (total_num_in_cells != total_num_out_cells) {
        mexErrMsgIdAndTxt("MATLAB:substituteToNormValues:invalidSize","The numbers are different between input and output filenames.");
    }

    std::vector<std::string> inpostfixes;
    for (int i = 0; i < total_num_in_cells; i++) {
        const mxArray *elem_ptr = mxGetCell(in_cell_ptr, i);
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

    std::vector<std::string> outpostfixes;
    for (int i = 0; i < total_num_out_cells; i++) {
        const mxArray *elem_ptr = mxGetCell(out_cell_ptr, i);
        if (elem_ptr == NULL) {
            mexPrintf("skip empty cell.\n");
            continue;
        }
        if ( !mxIsChar(elem_ptr)) {
            mexPrintf("skip invalid output file name.\n");
            break;
        }
        std::string out_name(mxArrayToString(elem_ptr));
        outpostfixes.push_back(out_name);
        mexPrintf("out[%d] = %s\n", i, outpostfixes[i].c_str());
    }

    std::vector<std::future<int>> futures;
    for (int i = 0; i < total_num_in_cells; i++) {
        futures.push_back(std::async(std::launch::async, substituteToNormValues, outputdir, meanpostfix, inpostfixes[i], outpostfixes[i]));
    }

    mexPrintf("waiting...\n");
    for (int i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        mexPrintf("[%d] done - %d\n", i, ret);
    }
    mexPrintf("done\n");

    return;
}


