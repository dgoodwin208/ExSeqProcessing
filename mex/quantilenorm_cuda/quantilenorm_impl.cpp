#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <exception>
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "quantilenorm_impl.h"
#include "utils/filebuffers.h"
#include "mex-utils/tiffs.h"
#include "mex-utils/hdf5.h"
#include "radixsort.h"
#include "gpulock.h"
#include "gpudevice.h"

QuantileNormImpl::QuantileNormImpl()
    : datadir_(""),
      basename_(""),
      in_fnames_(),
      image_width_(0),
      image_height_(0),
      num_slices_(0),
      num_gpus_(0),
      num_channels_(0),
      use_hdf5_(false),
      use_tmp_files_(true) {
    logger_ = spdlog::get("mex_logger");
}

QuantileNormImpl::QuantileNormImpl(const std::string& datadir,
                                   const std::string& basename,
                                   const std::vector<std::string>& in_fnames,
                                   const size_t image_width,
                                   const size_t image_height,
                                   const size_t num_slices,
                                   const size_t num_gpus,
                                   const bool use_hdf5,
                                   const bool use_tmp_files)
    : datadir_(datadir),
      basename_(basename),
      in_fnames_(in_fnames),
      image_width_(image_width),
      image_height_(image_height),
      num_slices_(num_slices),
      num_gpus_(num_gpus),
      num_channels_(in_fnames.size()),
      use_hdf5_(use_hdf5),
      use_tmp_files_(use_tmp_files)
{
    user_name_ = std::getenv("USER");

    logger_ = spdlog::get("mex_logger");

    size_t free_size;
    cudautils::get_gpu_mem_size(free_size, gpu_mem_total_);
}

void
QuantileNormImpl::run() {
    logger_->info("[{}] ##### sort 1", basename_);

    setupFileList();

    listTmpDataBuffersKeys();
    logger_->debug("[{}] {}", basename_, getStatMem());

    if (allDataExist(sorted_file1_list_)) {
        logger_->info("[{}] already exists sorted files 1.", basename_);
    } else {
        mergeSort1();
        radixSort1();

        waitForTasks("radixsort1", radixsort1_futures_);
        waitForTasks("mergesort1", mergesort1_futures_);
    }

    listTmpDataBuffersKeys();
    logger_->debug("[{}] {}", basename_, getStatMem());

    logger_->info("[{}] ##### sum", basename_);
    if (oneDataExists(summed_file_)) {
        logger_->info("[{}] already exists summed files.", basename_);
    } else {
        if (use_tmp_files_) {
            sumSortedFiles();
        } else {
            sumSortedBuffers();
        }
    }

    listTmpDataBuffersKeys();
    logger_->debug("[{}] {}", basename_, getStatMem());

    logger_->info("[{}] ##### substitute to normal values", basename_);
    if (allDataExist(substituted_file_list_)) {
        logger_->info("[{}] already exists substituted files.", basename_);
    } else {
        substituteValues();

        waitForTasks("substitute-values", substitute_values_futures_);

#ifndef DEBUG_FILEOUT
        std::string summed_filepath = datadir_ + "/" + summed_file_;
        if (use_tmp_files_) {
            remove(summed_filepath.c_str());
        } else {
            std::lock_guard<std::mutex> lock(tmp_data_mutex_);
            //DEBUG
            std::shared_ptr<std::vector<unsigned int>> summed_data = tmp_data_buffers_[summed_filepath];
            logger_->debug("[{}] summed#={}", basename_, summed_data.use_count());

            tmp_data_buffers_.erase(summed_filepath);
        }
#endif
    }

    listTmpDataBuffersKeys();
    logger_->debug("[{}] {}", basename_, getStatMem());

    logger_->info("[{}] ##### sort 2", basename_);
    if (allDataExist(sorted_file2_list_)) {
        logger_->info("[{}] already exists sorted files 2.", basename_);
    } else {
        mergeSort2();
        radixSort2();

        waitForTasks("radixsort2", radixsort2_futures_);
        waitForTasks("mergesort2", mergesort2_futures_);

#ifndef DEBUG_FILEOUT
        for (size_t i = 0; i < radixsort2_file_list_.size(); i++) {
            std::string subst_filepath     = datadir_ + "/" + std::get<2>(radixsort2_file_list_[i]);
            std::string sort1_idx_filepath = datadir_ + "/" + std::get<3>(radixsort2_file_list_[i]);

            if (use_tmp_files_) {
                remove(subst_filepath.c_str());
                remove(sort1_idx_filepath.c_str());
            } else {
                std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                //DEBUG
                std::shared_ptr<std::vector<float>> subst_data = tmp_data_buffers_[subst_filepath];
                std::shared_ptr<std::vector<unsigned int>> sort1_idx_data = tmp_data_buffers_[sort1_idx_filepath];
                logger_->debug("[{}] subst#={},sort1_idx#={}", basename_, subst_data.use_count(), sort1_idx_data.use_count());

                tmp_data_buffers_.erase(subst_filepath);
                tmp_data_buffers_.erase(sort1_idx_filepath);
            }

            logger_->debug("[{}] remove: ({}) subst filepath     = {}", basename_, i, subst_filepath);
            logger_->debug("[{}] remove: ({}) sort1 idx filepath = {}", basename_, i, sort1_idx_filepath);
        }

        if (mergesort2_file_list_.empty()) {
            for (size_t i = 0; i < radixsort2_file_list_.size(); i++) {
                std::string sort2_idx_filepath = datadir_ + "/idx_" + std::get<4>(radixsort2_file_list_[i]);

                if (use_tmp_files_) {
                    remove(sort2_idx_filepath.c_str());
                } else {
                    std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                    //DEBUG
                    std::shared_ptr<std::vector<unsigned int>> sort2_idx_data = tmp_data_buffers_[sort2_idx_filepath];
                    logger_->debug("[{}] sort2_idx#={}", basename_, sort2_idx_data.use_count());

                    tmp_data_buffers_.erase(sort2_idx_filepath);
                }

                logger_->debug("[{}] remove.1: ({}) sort2 idx filepath = {}", basename_, i, sort2_idx_filepath);
            }
        } else {
            size_t size_per_channel = mergesort2_file_list_.size() / num_channels_;
            for (size_t i = size_per_channel - 1; i < mergesort2_file_list_.size(); i+=size_per_channel) {
                std::string sort2_idx_filepath = datadir_ + "/" + mergesort2_file_list_[i][2];

                if (use_tmp_files_) {
                    remove(sort2_idx_filepath.c_str());
                } else {
                    std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                    //DEBUG
                    std::shared_ptr<std::vector<unsigned int>> sort2_idx_data = tmp_data_buffers_[sort2_idx_filepath];
                    logger_->debug("[{}] sort2_idx#={}", basename_, sort2_idx_data.use_count());

                    tmp_data_buffers_.erase(sort2_idx_filepath);
                }

                logger_->debug("[{}] remove.2: ({}) sort2 idx filepath = {}", basename_, i, sort2_idx_filepath);
            }
        }
#endif
    }

    listTmpDataBuffersKeys();
    logger_->debug("[{}] {}", getStatMem());

    logger_->info("[{}] ##### done", basename_);
}

std::vector<std::shared_ptr<std::vector<float>>>
QuantileNormImpl::getNormResult() {
    std::vector<std::shared_ptr<std::vector<float>>> result(sorted_file2_list_.size());
    for (size_t i = 0; i < sorted_file2_list_.size(); i++) {
        std::string out_filepath = datadir_ + "/" + sorted_file2_list_[i];
        result[i] = tmp_data_buffers_[out_filepath];
    }

    return result;
}


// ========== protected member functions
void
QuantileNormImpl::setupFileList() {
    std::string sorted_file1_prefix     = basename_ + "_1_sort1_c";
    std::string sorted_idx_file1_prefix = "idx_" + sorted_file1_prefix;
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::ostringstream sout;
        sout << "-000-" << std::setw(3) << std::setfill('0') << num_slices_ - 1 << ".bin";
        std::string sorted_file1 = sorted_file1_prefix + std::to_string(c_i) + sout.str();
        sorted_file1_list_.push_back(sorted_file1);
        logger_->info("[{}] [sorted file 1] {}", basename_, sorted_file1);
    }

    size_t one_slice_mem_usage = image_height_ * image_width_ * (sizeof(uint16_t) + sizeof(unsigned int))
        * THRUST_RADIXSORT_MEMORY_USAGE_RATIO;
    size_t num_sub_slices = gpu_mem_total_ * GPU_USER_MEMORY_USAGE_RATIO / one_slice_mem_usage;
    logger_->debug("[{}] gpu total mem = {}, data(uint16_t) + idx(unsigned int) * {} for one slice = {}, # sub slices = {}", basename_, gpu_mem_total_, THRUST_RADIXSORT_MEMORY_USAGE_RATIO, one_slice_mem_usage, num_sub_slices);
    if (num_sub_slices == 0) {
        logger_->error("[{}] one slice mem usage is over gpu total memory.", basename_);
        num_sub_slices = 1;
    }

    std::vector<std::tuple<size_t, size_t>> idx_sort1;
    for (size_t i = 0; i < num_slices_; i += num_sub_slices) {
        size_t next_i = i + num_sub_slices - 1;
        if (next_i >= num_slices_) {
            next_i = num_slices_ - 1;
        }
        idx_sort1.push_back(std::make_tuple(i, next_i));
//        logger_->debug("[{}] base idx: ({}, {})", basename_, i, next_i);

        for (size_t c_i = 0; c_i < num_channels_; c_i++) {
            std::ostringstream sout;
            sout << "-" << std::setw(3) << std::setfill('0') << i
                 << "-" << std::setw(3) << std::setfill('0') << next_i << ".bin";
            std::string radixsort1_file = sorted_file1_prefix + std::to_string(c_i) + sout.str();
            radixsort1_file_list_.push_back(std::make_tuple(i, next_i, in_fnames_[c_i], radixsort1_file));
            logger_->info("[{}] [radixsort 1] {} -> {}", basename_, in_fnames_[c_i], radixsort1_file);
        }
    }

    makeMergeSortFileList(sorted_file1_prefix, sorted_idx_file1_prefix, idx_sort1, mergesort1_file_list_);

    summed_file_ = basename_ + "_2_sort1_sum.bin";

    std::string subst_file_prefix = basename_ + "_3_subst_c";
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::ostringstream sout;
        sout << "-000-" << std::setw(3) << std::setfill('0') << num_slices_ - 1 << ".bin";
        std::string subst_file = subst_file_prefix + std::to_string(c_i) + sout.str();
        substituted_file_list_.push_back(subst_file);
        logger_->info("[{}] [subst_file] {}", basename_, subst_file);
    }

    std::string sorted_file2_prefix     = basename_ + "_4_sort2_c";
    std::string sorted_idx_file2_prefix = "idx_" + sorted_file2_prefix;

    size_t unit_data_size  = gpu_mem_total_ * GPU_USER_MEMORY_USAGE_RATIO
        / ((sizeof(float) + sizeof(unsigned int)) * THRUST_RADIXSORT_MEMORY_USAGE_RATIO);

    size_t total_data_size = num_slices_ * image_height_ * image_width_;
    size_t num_sub_data    = total_data_size / unit_data_size;
    if (total_data_size % unit_data_size != 0) {
        num_sub_data++;
    }
    logger_->debug("[{}] gpu total mem = {}, data(float) + idx(unsigned int) * {} for unit data = {}, total data = {}, # sub slices = {}", basename_, gpu_mem_total_, THRUST_RADIXSORT_MEMORY_USAGE_RATIO, unit_data_size, total_data_size, num_sub_data);
    if (num_sub_data == 0) {
        logger_->error("[{}] unit data size is over gpu total memory.", basename_);
        num_sub_data = 1;
    }

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::ostringstream sout;
        sout << "-000-" << std::setw(3) << std::setfill('0') << num_sub_data - 1 << ".bin";
        std::string sorted_file2 = sorted_file2_prefix + std::to_string(c_i) + sout.str();
        sorted_file2_list_.push_back(sorted_file2);
        logger_->info("[{}] [sorted file 2] {}", basename_, sorted_file2);
    }

    std::vector<std::tuple<size_t, size_t>> idx_sort2;
    for (size_t i = 0; i < num_sub_data; i++) {
        idx_sort2.push_back(std::make_tuple(i, i));
        size_t num_data_start = unit_data_size * i;
        size_t data_size      = unit_data_size;
        if (num_data_start + data_size > total_data_size) {
            data_size = total_data_size - num_data_start;
        }

        for (size_t c_i = 0; c_i < num_channels_; c_i++) {
            std::string subst_file = substituted_file_list_[c_i];
            std::string idx_file   = "idx_" + sorted_file1_list_[c_i];

            std::ostringstream sout;
            sout << "-" << std::setw(3) << std::setfill('0') << i
                 << "-" << std::setw(3) << std::setfill('0') << i << ".bin";
            std::string radixsort2_file = sorted_file2_prefix + std::to_string(c_i) + sout.str();
            radixsort2_file_list_.push_back(std::make_tuple(num_data_start, data_size, subst_file, idx_file, radixsort2_file));
            logger_->info("[{}] [radixsort 2] c{} ({}-{}) {}, {} -> {}", basename_, c_i, num_data_start, data_size, subst_file, idx_file, radixsort2_file);
        }
    }

    makeMergeSortFileList(sorted_idx_file2_prefix, sorted_file2_prefix, idx_sort2, mergesort2_file_list_);
}

void
QuantileNormImpl::makeMergeSortFileList(const std::string& file_prefix,
                                        const std::string& dep_file_prefix,
                                        const std::vector<std::tuple<size_t, size_t>>& idx,
                                        std::vector<std::vector<std::string>>& mergesort_file_list) {
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::vector<std::tuple<size_t, size_t>> sub_idx[2];
        int cur_sub_idx = 0;
        int nxt_sub_idx = 1;

        std::copy(idx.begin(), idx.end(), std::back_inserter(sub_idx[cur_sub_idx]));

        bool is_odd = false;
        do {
            size_t start_j = 0;
            if (is_odd && (sub_idx[cur_sub_idx].size() % 2 == 1)) {
                sub_idx[nxt_sub_idx].push_back(sub_idx[cur_sub_idx][0]);
                start_j = 1;
            }
            for (size_t j = start_j; j < sub_idx[cur_sub_idx].size(); j += 2) {
                size_t cur_j = j;
                size_t nxt_j = j + 1;
                size_t sub_idx_end = sub_idx[cur_sub_idx].size();
                if (nxt_j == sub_idx_end) {
                    sub_idx[nxt_sub_idx].push_back(sub_idx[cur_sub_idx][sub_idx_end - 1]);
                    is_odd = true;
                    break;
                }
                size_t in1_start = std::get<0>(sub_idx[cur_sub_idx][cur_j]);
                size_t in1_end   = std::get<1>(sub_idx[cur_sub_idx][cur_j]);
                size_t in2_start = std::get<0>(sub_idx[cur_sub_idx][nxt_j]);
                size_t in2_end   = std::get<1>(sub_idx[cur_sub_idx][nxt_j]);

                std::ostringstream sout;
                sout << "-" << std::setw(3) << std::setfill('0') << in1_start
                     << "-" << std::setw(3) << std::setfill('0') << in1_end;
                std::string in1_file = file_prefix + std::to_string(c_i) + sout.str() + ".bin";
                std::string in1_dep_file = dep_file_prefix + std::to_string(c_i) + sout.str() + ".bin";

                sout.str("");
                sout << "-" << std::setw(3) << std::setfill('0') << in2_start
                     << "-" << std::setw(3) << std::setfill('0') << in2_end;
                std::string in2_file = file_prefix + std::to_string(c_i) + sout.str() + ".bin";
                std::string in2_dep_file = dep_file_prefix + std::to_string(c_i) + sout.str() + ".bin";

                sout.str("");
                sout << "-" << std::setw(3) << std::setfill('0') << in1_start
                     << "-" << std::setw(3) << std::setfill('0') << in2_end;
                std::string out_file = file_prefix + std::to_string(c_i) + sout.str() + ".bin";
                std::string out_dep_file = dep_file_prefix + std::to_string(c_i) + sout.str() + ".bin";

                sub_idx[nxt_sub_idx].push_back(std::make_tuple(in1_start, in2_end));

                logger_->info("[{}] [mergesort] {}, {} -> {}", basename_, in1_file, in2_file, out_file);
                mergesort_file_list.push_back({in1_file, in2_file, out_file, in1_dep_file, in2_dep_file, out_dep_file});
            }

            cur_sub_idx = cur_sub_idx ^ 1;
            nxt_sub_idx = nxt_sub_idx ^ 1;

            sub_idx[nxt_sub_idx].clear();
        } while(sub_idx[cur_sub_idx].size() > 1);
    }
}

bool
QuantileNormImpl::allDataExist(const std::vector<std::string>& file_list) {
    bool all_data_exist = true;
    for (auto filename : file_list) {
        if (! oneDataExists(filename)) {
            all_data_exist = false;
            break;
        }
    }

    return all_data_exist;
}

bool
QuantileNormImpl::oneDataExists(const std::string& filename) {
    std::string filepath = datadir_ + "/" + filename;

    bool one_data_exists;
    if (use_tmp_files_) {
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        one_data_exists = (fin.is_open());
        fin.close();
    } else {
        std::lock_guard<std::mutex> lock(tmp_data_mutex_);
        one_data_exists = (tmp_data_buffers_.find(filename) != tmp_data_buffers_.end());
    }

    return one_data_exists;
}

void
QuantileNormImpl::listTmpDataBuffersKeys()
{
    logger_->debug("===== listTmpDataBuffersKeys start");
    for (auto itr = tmp_data_buffers_.begin(); itr != tmp_data_buffers_.end(); itr++) {
        logger_->debug("{}", itr->first);
    }
    logger_->debug("===== listTmpDataBuffersKeys end");
}

std::string
QuantileNormImpl::getStatMem()
{
    std::string line;
    std::ifstream fin("/proc/self/statm");
    std::getline(fin, line);

    line = "statm: " + line;

    return line;
}


void
QuantileNormImpl::radixSort1() {
#ifdef DEBUG_NO_THREADING
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = std::launch::async;
#endif
    radixsort1_futures_.push_back(std::async(policy, &QuantileNormImpl::radixSort1FromData, this));

    if (use_hdf5_) {
        loadRadixSort1Hdf5Data();
    } else {
        loadRadixSort1TiffData();
    }
}

int
QuantileNormImpl::loadRadixSort1Hdf5Data() {

    for (auto radixsort_info : radixsort1_file_list_) {
        size_t slice_start  = std::get<0>(radixsort_info);
        size_t slice_end    = std::get<1>(radixsort_info);
        std::string h5_file = std::get<2>(radixsort_info);

        logger_->info("[{}] loadRadixSort1Hdf5Data: loadhdf5 start {}", basename_, h5_file);
        std::shared_ptr<RadixSort1Info> data = std::make_shared<RadixSort1Info>();
        data->slice_start  = slice_start;
        data->image        = std::make_shared<std::vector<uint16_t>>();
        data->out_filename = std::get<3>(radixsort_info);

        mexutils::loadhdf5(h5_file, slice_start, slice_end, image_height_, image_width_, data->image);

        logger_->info("slice ({} - {}), image={}, out={}", data->slice_start, slice_end, data->image->size(), data->out_filename);
        logger_->info("[{}] loadRadixSort1Hdf5Data: loadhdf5 end   {}", basename_, h5_file);

        radixsort1_queue_.push(data);
    }

    radixsort1_queue_.close();

    return 0;
}

int
QuantileNormImpl::loadRadixSort1TiffData() {
    for (auto radixsort_info : radixsort1_file_list_) {
        size_t slice_start   = std::get<0>(radixsort_info);
        size_t slice_end     = std::get<1>(radixsort_info);
        std::string tif_file = std::get<2>(radixsort_info);

        logger_->info("[{}] loadRadixSort1TiffData: loadtiff start {}", basename_, tif_file);
        std::shared_ptr<RadixSort1Info> data = std::make_shared<RadixSort1Info>();
        data->slice_start  = slice_start;
        data->image        = std::make_shared<std::vector<uint16_t>>();
        data->out_filename = std::get<3>(radixsort_info);

        mexutils::loadtiff(tif_file, slice_start, slice_end, data->image);

        logger_->info("slice ({} - {}), image={}, out={}", data->slice_start, slice_end, data->image->size(), data->out_filename);
        logger_->info("[{}] loadRadixSort1TiffData: loadtiff end   {}", basename_, tif_file);

        radixsort1_queue_.push(data);
    }

    radixsort1_queue_.close();

    return 0;
}

int
QuantileNormImpl::radixSort1FromData() {

    logger_->info("[{}] radixSort1FromData: selectGPU", basename_);
    cudautils::GPULock lock(num_gpus_);
    int idx_gpu = lock.trylock();
    if (idx_gpu >= 0) {
        logger_->info("[{}] radixSort1FromData: lock idx_gpu = {}", basename_, idx_gpu);
    } else {
        logger_->error("[{}] radixSort1FromData: failed to lock gpus", basename_);
        return -1;
    }

    while(1) {
        std::shared_ptr<RadixSort1Info> info;
        bool has_data = radixsort1_queue_.pop(info);
        if (! has_data) {
            logger_->info("[{}] radixSort1FromData: data end", basename_);
            break;
        }
        logger_->info("[{}] radixSort1FromData: start {} ({})", basename_, info->out_filename, info->slice_start);


        try {
            unsigned int idx_start = info->slice_start * image_height_ * image_width_;
            std::shared_ptr<std::vector<unsigned int>> idx(new std::vector<unsigned int>(info->image->size()));
            thrust::sequence(thrust::host, idx->begin(), idx->end(), idx_start);

            cudautils::radixsort(*(info->image), *idx);

            int ret;
            std::string out_idx_filename = "idx_" + info->out_filename;
            if (use_tmp_files_) {
                ret = saveDataToFile(datadir_, out_idx_filename, idx);
                ret = saveDataToFile(datadir_, info->out_filename, info->image);
            } else {
                ret = saveDataToBuffer(datadir_, out_idx_filename, idx);
                ret = saveDataToBuffer(datadir_, info->out_filename, info->image);
            }

        } catch (std::exception& ex) {
            logger_->error("[{}] end - {}", basename_, ex.what());
            if (idx_gpu != -1) {
                lock.unlock();
                logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
            }
            return -1;
        } catch (...) {
            logger_->error("[{}] end - unknown error..", basename_);
            if (idx_gpu != -1) {
                lock.unlock();
                logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
            }
            return -1;
        }

        logger_->info("[{}] radixSort1FromData: end   {} ({})", basename_, info->out_filename, info->slice_start);
    }

    lock.unlock();
    logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);

    return 0;
}

void
QuantileNormImpl::radixSort2() {
#ifdef DEBUG_NO_THREADING
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = std::launch::async;
#endif
    for (size_t i = 0; i < radixsort2_file_list_.size(); i++) {
        radixsort2_futures_.push_back(std::async(policy, &QuantileNormImpl::radixSort2FromData, this, i));
    }
}

int
QuantileNormImpl::radixSort2FromData(const size_t idx_radixsort) {
    cudautils::GPULock lock(num_gpus_);
    int idx_gpu = -1;
    try {
        logger_->info("[{}] radixSort2FromData: start ({})", basename_, idx_radixsort);

        size_t num_data_start = std::get<0>(radixsort2_file_list_[idx_radixsort]);
        size_t data_size      = std::get<1>(radixsort2_file_list_[idx_radixsort]);

        std::string in_subst_filepath = datadir_ + "/" + std::get<2>(radixsort2_file_list_[idx_radixsort]);
        std::string in_idx_filepath   = datadir_ + "/" + std::get<3>(radixsort2_file_list_[idx_radixsort]);

        std::string out_file = std::get<4>(radixsort2_file_list_[idx_radixsort]);
        std::string out_filepath      = datadir_ + "/"     + out_file;
        std::string out_idx_filepath  = datadir_ + "/idx_" + out_file;

        logger_->debug("[{}] radixSort2FromData: ({}) ({}-{})", basename_, idx_radixsort, num_data_start, data_size);
        logger_->debug("[{}] radixSort2FromData: ({}) in subst filepath = {}", basename_, idx_radixsort, in_subst_filepath);
        logger_->debug("[{}] radixSort2FromData: ({}) in idx filepath   = {}", basename_, idx_radixsort, in_idx_filepath);
        logger_->debug("[{}] radixSort2FromData: ({}) out filepath      = {}", basename_, idx_radixsort, out_filepath);
        logger_->debug("[{}] radixSort2FromData: ({}) out idx filepath  = {}", basename_, idx_radixsort, out_idx_filepath);

        std::shared_ptr<std::vector<float>> data;
        std::shared_ptr<std::vector<unsigned int>> index;
        if (use_tmp_files_) {
            data  = loadDataFromFile<float>(in_subst_filepath, num_data_start, data_size);
            index = loadDataFromFile<unsigned int>(in_idx_filepath, num_data_start, data_size);
        } else {
            data  = loadDataFromBuffer<float>(in_subst_filepath, num_data_start, data_size);
            index = loadDataFromBuffer<unsigned int>(in_idx_filepath, num_data_start, data_size);
        }
        if (data == nullptr) {
            logger_->debug("[{}] radixSort2FromData: ({}) failed to load data file", basename_, idx_radixsort);
            return -1;
        }
        if (index == nullptr) {
            logger_->debug("[{}] radixSort2FromData: ({}) failed to load index file", basename_, idx_radixsort);
            return -1;
        }
        logger_->debug("[{}] radixSort2FromData: after loadin data {}", basename_, getStatMem());

        int ret;
        logger_->info("[{}] radixSort2FromData: ({}) selectGPU", basename_, idx_radixsort);
        idx_gpu = lock.trylock();
        if (idx_gpu >= 0) {
            logger_->info("[{}] radixSort2FromData: ({}) lock idx_gpu = {}", basename_, idx_radixsort, idx_gpu);
        } else {
            logger_->error("[{}] radixSort2FromData: ({}) failed to lock gpus", basename_, idx_radixsort);
            return -1;
        }

        try {
            cudautils::radixsort<unsigned int, float>(*index, *data);
        } catch (std::exception& ex) {
            logger_->debug("[{}] radixSort2FromData: {}", basename_, ex.what());
            cudaError err = cudaGetLastError();
            if (err != cudaSuccess) {
                logger_->error("[{}] {}", basename_, cudaGetErrorString(err));
            }
            lock.unlock();
            logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
            return -1;
        } catch (...) {
            logger_->debug("[{}] radixSort2FromData: unknown error", basename_);
            cudaError err = cudaGetLastError();
            if (err != cudaSuccess) {
                logger_->error("[{}] {}", basename_, cudaGetErrorString(err));
            }
            lock.unlock();
            logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
            return -1;
        }

        lock.unlock();
        logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);

        std::string out_idx_file = "idx_" + out_file;
        if (use_tmp_files_) {
            ret = saveDataToFile(datadir_, out_idx_file, index);
            ret = saveDataToFile(datadir_, out_file, data);
        } else {
            ret = saveDataToBuffer(datadir_, out_idx_file, index);
            ret = saveDataToBuffer(datadir_, out_file, data);
        }

        logger_->debug("[{}] radixSort2FromData: before return {}", basename_, getStatMem());
        logger_->info("[{}] radixSort2FromData: end   ({})", basename_, idx_radixsort);
        return ret;
    } catch (std::exception& ex) {
        logger_->error("[{}] end - {}", basename_, ex.what());
        if (idx_gpu != -1) {
            lock.unlock();
            logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
        }
        return -1;
    } catch (...) {
        logger_->error("[{}] end - unknown error..", basename_);
        if (idx_gpu != -1) {
            lock.unlock();
            logger_->info("[{}] unlock idx_gpu = {}", basename_, idx_gpu);
        }
        return -1;
    }

    return 0;
}

template <typename T>
int
QuantileNormImpl::saveDataToFile(const std::string& datadir, const std::string& out_filename, const std::shared_ptr<std::vector<T>> data) {
    std::string tmp_out_filepath = datadir + "/.tmp." + out_filename;
    std::ofstream fout(tmp_out_filepath, std::ios::out | std::ios::binary);
    if (! fout.is_open()) {
        logger_->error("[{}] cannot open an output file for radixsort.", basename_);
        return -1;
    }

    int data_size = (int)data->size();
    int buffer_size = FILEWRITE_BUFSIZE / sizeof(T);
    for (int i = 0; i < data_size; i += buffer_size) {
        if (i + buffer_size > data_size) {
            buffer_size = data_size - i;
        }

        fout.write((char*)&(*data)[i], buffer_size * sizeof(T));
    }
    fout.close();

    std::string out_filepath = datadir + "/" + out_filename;
    int ret = rename(tmp_out_filepath.c_str(), out_filepath.c_str());
    if (ret != 0) {
        logger_->error("[{}] cannot rename output file; {} -> {}", basename_, tmp_out_filepath, out_filepath);
        return -1;
    }

    return 0;
}

template <typename T>
int
QuantileNormImpl::saveDataToBuffer(const std::string& datadir, const std::string& out_filename, const std::shared_ptr<std::vector<T>> data) {

    std::lock_guard<std::mutex> lock(tmp_data_mutex_);

    std::string out_filepath = datadir + "/" + out_filename;
    tmp_data_buffers_.insert(std::make_pair(out_filepath, TmpDataBuffer(data)));

    return 0;
}

template <typename T>
std::shared_ptr<std::vector<T>>
QuantileNormImpl::loadDataFromFile(const std::string& in_filepath, const size_t num_data_start, const size_t data_size) {
    utils::FileBufferReader<T> fb_reader(in_filepath, FILEREAD_BUFSIZE);
    try {
        fb_reader.open();
    } catch (std::runtime_error& ex) {
        logger_->error("[{}] in loadfile, {}", basename_, in_filepath);
        logger_->error("[{}] {}", basename_, ex.what());
        return nullptr;
    }

    std::shared_ptr<std::vector<T>> data(new std::vector<T>(data_size));
    fb_reader.readFileToBufferPartially(num_data_start, data);
    fb_reader.close();

    return data;
}

template <typename T>
std::shared_ptr<std::vector<T>>
QuantileNormImpl::loadDataFromBuffer(const std::string& in_filepath, const size_t num_data_start, const size_t data_size) {

    std::lock_guard<std::mutex> lock(tmp_data_mutex_);

    auto find_itr = tmp_data_buffers_.find(in_filepath);

    if (find_itr == tmp_data_buffers_.end()) {
        logger_->error("[{}] in loadfile, {}", basename_, in_filepath);
        return nullptr;
    }

    std::shared_ptr<std::vector<T>> data = find_itr->second;
    std::shared_ptr<std::vector<T>> partial_data(new std::vector<T>(data_size));

    std::copy(&(*data)[num_data_start], &(*data)[num_data_start] + data_size, partial_data->data());

    return partial_data;
}

//int
//QuantileNormImpl::selectGPU() {
//    int idx_gpu = -1;
//    for (size_t i = 0; i < num_gpus_; i++) {
//        std::string sem_name = "/" + user_name_ + ".g" + std::to_string(i);
////        logger_->trace("[{}] sem_name = {}", basename_, sem_name);
//        sem_t *sem;
//        sem = sem_open(sem_name.c_str(), O_RDWR);
//        int ret = errno;
//        if (sem == SEM_FAILED) {
//            logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
//            continue;
//        }
//
//        ret = sem_trywait(sem);
//        if (ret == 0) {
//            logger_->trace("[{}] selectGPU {}", basename_, sem_name);
//            idx_gpu = i;
//            cudaSetDevice(idx_gpu);
//            break;
//        }
//    }
//
//    return idx_gpu;
//}

//void
//QuantileNormImpl::unselectGPU(const int idx_gpu) {
//    std::string sem_name = "/" + user_name_ + ".g" + std::to_string(idx_gpu);
//    sem_t *sem;
//    sem = sem_open(sem_name.c_str(), O_RDWR);
//    int ret = errno;
//    if (sem == SEM_FAILED) {
//        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
//        return;
//    }
//
//    cudaDeviceReset();
//
//    logger_->trace("[{}] unselectGPU {}", basename_, sem_name);
//    ret = sem_post(sem);
//    if (ret != 0) {
//        logger_->error("[{}] cannot post semaphore of {}", basename_, sem_name);
//        return;
//    }
//}

//void
//QuantileNormImpl::selectCore(const int idx_core_group) {
//    std::string sem_name = "/" + user_name_ + ".qn_c" + std::to_string(idx_core_group);
////    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
//    sem_t *sem;
//    sem = sem_open(sem_name.c_str(), O_RDWR);
//    int ret = errno;
//    if (sem == SEM_FAILED) {
//        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
//        return;
//    }
//
//    auto interval_sec = std::chrono::seconds(1);
//    int count = 1;
//    while (1) {
//        ret = sem_trywait(sem);
//        if (ret == 0) {
//            break;
//        }
//        std::this_thread::sleep_for(interval_sec);
//        count++;
//    }
//    logger_->trace("[{}] selectCore: {} ({})", basename_, sem_name, count);
//}
//
//int
//QuantileNormImpl::selectCoreNoblock(const int idx_core_group) {
//    std::string sem_name = "/" + user_name_ + ".qn_c" + std::to_string(idx_core_group);
////    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
//    sem_t *sem;
//    sem = sem_open(sem_name.c_str(), O_RDWR);
//    int ret = errno;
//    if (sem == SEM_FAILED) {
//        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
//        return -1;
//    }
//
//    ret = sem_trywait(sem);
//    if (ret == 0) {
//        logger_->trace("[{}] selectCoreNoblock: {}", basename_, sem_name);
//    }
//
//    return ret;
//}
//
//void
//QuantileNormImpl::unselectCore(const int idx_core_group) {
//    std::string sem_name = "/" + user_name_ + ".qn_c" + std::to_string(idx_core_group);
////    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
//    sem_t *sem;
//    sem = sem_open(sem_name.c_str(), O_RDWR);
//    int ret = errno;
//    if (sem == SEM_FAILED) {
//        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
//        return;
//    }
//
//    ret = sem_post(sem);
//    if (ret == -1) {
//        logger_->error("[{}] unselect failed; {}", basename_, sem_name);
//    }
//    logger_->trace("[{}] unselectCore: {}", basename_, sem_name);
//}

void
QuantileNormImpl::mergeSort1() {
#ifdef DEBUG_NO_THREADING
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = std::launch::async;
#endif

    for (size_t i = 0; i < mergesort1_file_list_.size(); i++) {
        if (use_tmp_files_) {
            mergesort1_futures_.push_back(std::async(policy, &QuantileNormImpl::mergeSortTwoFiles<uint16_t, unsigned int>, this, i, mergesort1_file_list_));
        } else {
            mergesort1_futures_.push_back(std::async(policy, &QuantileNormImpl::mergeSortTwoBuffers<uint16_t, unsigned int>, this, i, mergesort1_file_list_));
        }
    }
}

void
QuantileNormImpl::mergeSort2() {
#ifdef DEBUG_NO_THREADING
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = std::launch::async;
#endif

    for (size_t i = 0; i < mergesort2_file_list_.size(); i++) {
        if (use_tmp_files_) {
            mergesort2_futures_.push_back(std::async(policy, &QuantileNormImpl::mergeSortTwoFiles<unsigned int, float>, this, i, mergesort2_file_list_));
        } else {
            mergesort2_futures_.push_back(std::async(policy, &QuantileNormImpl::mergeSortTwoBuffers<unsigned int, float>, this, i, mergesort2_file_list_));
        }
    }
}

template <typename T1, typename T2>
int
QuantileNormImpl::mergeSortTwoFiles(const size_t idx, const std::vector<std::vector<std::string>>& mergesort_file_list) {
    logger_->trace("[{}] mergeSortTwoFiles: start ({})", basename_, idx);
    try {
//        assert(mergesort_file_list[idx].size() == 6);

        const std::string in1_filename = mergesort_file_list[idx][0];
        const std::string in2_filename = mergesort_file_list[idx][1];
        const std::string out_filename = mergesort_file_list[idx][2];

        const std::string in1_dep_filename = mergesort_file_list[idx][3];
        const std::string in2_dep_filename = mergesort_file_list[idx][4];
        const std::string out_dep_filename = mergesort_file_list[idx][5];

        std::string in1_filepath = datadir_ + "/" + in1_filename;
        std::string in2_filepath = datadir_ + "/" + in2_filename;
        std::string tmp_out_filepath = datadir_ + "/.tmp." + out_filename;

        std::string in1_dep_filepath = datadir_ + "/" + in1_dep_filename;
        std::string in2_dep_filepath = datadir_ + "/" + in2_dep_filename;
        std::string tmp_out_dep_filepath = datadir_ + "/.tmp." + out_dep_filename;

        utils::FileBufferReader<T1> in1_fb_reader(in1_filepath, FILEREAD_BUFSIZE);
        utils::FileBufferReader<T1> in2_fb_reader(in2_filepath, FILEREAD_BUFSIZE);

        utils::FileBufferReader<T2> in1_dep_fb_reader(in1_dep_filepath, FILEREAD_BUFSIZE);
        utils::FileBufferReader<T2> in2_dep_fb_reader(in2_dep_filepath, FILEREAD_BUFSIZE);

        in1_fb_reader.tryToOpen();
        in2_fb_reader.tryToOpen();

        in1_dep_fb_reader.tryToOpen();
        in2_dep_fb_reader.tryToOpen();

        utils::FileBufferWriter<T1> out_fb_writer(tmp_out_filepath, FILEWRITE_BUFSIZE);
        utils::FileBufferWriter<T2> out_dep_fb_writer(tmp_out_dep_filepath, FILEWRITE_BUFSIZE);

        try {
            out_fb_writer.open();
            out_dep_fb_writer.open();
        } catch (std::runtime_error& ex) {
            logger_->error("[{}] in mergeSortTwoFiles,", basename_);
            logger_->error("[{}] {}", basename_, ex.what());
            return -1;
        }


        in1_fb_reader.readFileToBuffer();
        in2_fb_reader.readFileToBuffer();
        in1_dep_fb_reader.readFileToBuffer();
        in2_dep_fb_reader.readFileToBuffer();

//        int count = 1;
        while (! in1_fb_reader.finishedReadingAll() && ! in2_fb_reader.finishedReadingAll()) {
//            assert(! in1_dep_fb_reader.finishedReadingAll());
//            assert(! in2_dep_fb_reader.finishedReadingAll());

            const T1 val1 = in1_fb_reader.get();
            const T1 val2 = in2_fb_reader.get();
            if (val1 <= val2) {
//                logger_->debug("[{}] {} (val1, val2): {} <= {}", basename_, count++, val1, val2);
                in1_fb_reader.next();
                out_fb_writer.set(val1);

                const T2 dep_val1 = in1_dep_fb_reader.get();
                in1_dep_fb_reader.next();
                out_dep_fb_writer.set(dep_val1);
            } else {
//                logger_->debug("[{}] {} (val1, val2): {} >  {}", basename_, count++, val1, val2);
                in2_fb_reader.next();
                out_fb_writer.set(val2);

                const T2 dep_val2 = in2_dep_fb_reader.get();
                in2_dep_fb_reader.next();
                out_dep_fb_writer.set(dep_val2);
            }
        }
//        logger_->debug("1. total bytes");
//        logger_->debug("in1 = {}, in2 = {}, out = {}",
//            in1_fb_reader.getTotalReadBytes(), in2_fb_reader.getTotalReadBytes(),
//            out_fb_writer.getTotalWriteBytes());
//        logger_->debug("in1_dep = {}, in2_dep = {}, out_dep = {}",
//            in1_dep_fb_reader.getTotalReadBytes(), in2_dep_fb_reader.getTotalReadBytes(),
//            out_dep_fb_writer.getTotalWriteBytes());

        if (out_fb_writer.hasBufferData()) {
//            assert(out_dep_fb_writer.hasBufferData());

            out_fb_writer.writeFileFromBuffer();
            out_dep_fb_writer.writeFileFromBuffer();
        }

        if (! in1_fb_reader.finishedReadingAll()) {
//            assert(! in1_dep_fb_reader.finishedReadingAll());

            out_fb_writer.writeFileFromReaderBufferDirectly(in1_fb_reader);
            out_dep_fb_writer.writeFileFromReaderBufferDirectly(in1_dep_fb_reader);
        } else if (! in2_fb_reader.finishedReadingAll()) {
//            assert(! in2_dep_fb_reader.finishedReadingAll());

            out_fb_writer.writeFileFromReaderBufferDirectly(in2_fb_reader);
            out_dep_fb_writer.writeFileFromReaderBufferDirectly(in2_dep_fb_reader);
        }

        out_fb_writer.close();
        out_dep_fb_writer.close();
//        logger_->debug("2. total bytes");
//        logger_->debug("in1 = {}, in2 = {}, out = {}",
//            in1_fb_reader.getTotalReadBytes(), in2_fb_reader.getTotalReadBytes(),
//            out_fb_writer.getTotalWriteBytes());
//        logger_->debug("in1_dep = {}, in2_dep = {}, out_dep = {}",
//            in1_dep_fb_reader.getTotalReadBytes(), in2_dep_fb_reader.getTotalReadBytes(),
//            out_dep_fb_writer.getTotalWriteBytes());

        std::string out_filepath = datadir_ + "/" + out_filename;
        int ret = rename(tmp_out_filepath.c_str(), out_filepath.c_str());
        if (ret != 0) {
            logger_->error("[{}] cannot rename the tmp output file to output file. {} -> {}",
                    basename_, tmp_out_filepath, out_filepath);
            logger_->error("[{}] errno = {}, {}", basename_, errno, strerror(errno));
            return -1;
        }
        std::string out_dep_filepath = datadir_ + "/" + out_dep_filename;
        ret = rename(tmp_out_dep_filepath.c_str(), out_dep_filepath.c_str());
        if (ret != 0) {
            logger_->error("[{}] cannot rename the tmp output dep file to output dep file. {} -> {}",
                    basename_, tmp_out_dep_filepath, out_dep_filepath);
            logger_->error("[{}] errno = {}, {}", basename_, errno, strerror(errno));
            return -1;
        }

        in1_fb_reader.close();
        in2_fb_reader.close();
        in1_dep_fb_reader.close();
        in2_dep_fb_reader.close();

        remove(in1_filepath.c_str());
        remove(in2_filepath.c_str());
        remove(in1_dep_filepath.c_str());
        remove(in2_dep_filepath.c_str());
    } catch (std::exception& ex) {
        logger_->error("[{}] mergeSortTwoFiles: ({}) {}", basename_, idx, ex.what());
        return -1;
    } catch (...) {
        logger_->error("[{}] mergeSortTwoFiles: ({}) unknown error..", basename_, idx);
        return -1;
    }

    logger_->trace("[{}] mergeSortTwoFiles: end   ({})", basename_, idx);
    return 0;
}

template <typename T1, typename T2>
int
QuantileNormImpl::mergeSortTwoBuffers(const size_t idx, const std::vector<std::vector<std::string>>& mergesort_file_list) {
    logger_->trace("[{}] mergeSortTwoBuffers: start ({})", basename_, idx);
    try {
        const std::string in1_filename = mergesort_file_list[idx][0];
        const std::string in2_filename = mergesort_file_list[idx][1];
        const std::string out_filename = mergesort_file_list[idx][2];

        const std::string in1_dep_filename = mergesort_file_list[idx][3];
        const std::string in2_dep_filename = mergesort_file_list[idx][4];
        const std::string out_dep_filename = mergesort_file_list[idx][5];

        std::string in1_filepath = datadir_ + "/" + in1_filename;
        std::string in2_filepath = datadir_ + "/" + in2_filename;

        std::string in1_dep_filepath = datadir_ + "/" + in1_dep_filename;
        std::string in2_dep_filepath = datadir_ + "/" + in2_dep_filename;


        std::shared_ptr<std::vector<T1>> in1_data;
        std::shared_ptr<std::vector<T1>> in2_data;
        std::shared_ptr<std::vector<T2>> in1_dep_data;
        std::shared_ptr<std::vector<T2>> in2_dep_data;

        bool in1_exists = false;
        bool in2_exists = false;
        bool in1_dep_exists = false;
        bool in2_dep_exists = false;
        while (! in1_exists || ! in2_exists || ! in1_dep_exists || ! in2_dep_exists) {
            if (!in1_exists) {
                std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                auto find_itr = tmp_data_buffers_.find(in1_filepath);
                if (find_itr != tmp_data_buffers_.end()) {
                    in1_exists = true;
                    in1_data = find_itr->second;
                }
            }
            if (!in2_exists) {
                std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                auto find_itr = tmp_data_buffers_.find(in2_filepath);
                if (find_itr != tmp_data_buffers_.end()) {
                    in2_exists = true;
                    in2_data = find_itr->second;
                }
            }
            if (!in1_dep_exists) {
                std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                auto find_itr = tmp_data_buffers_.find(in1_dep_filepath);
                if (find_itr != tmp_data_buffers_.end()) {
                    in1_dep_exists = true;
                    in1_dep_data = find_itr->second;
                }
            }
            if (!in2_dep_exists) {
                std::lock_guard<std::mutex> lock(tmp_data_mutex_);
                auto find_itr = tmp_data_buffers_.find(in2_dep_filepath);
                if (find_itr != tmp_data_buffers_.end()) {
                    in2_dep_exists = true;
                    in2_dep_data = find_itr->second;
                }
            }

            std::this_thread::sleep_for(utils::FILECHECK_INTERVAL_SEC);
        }

        if (in1_data->size() != in1_dep_data->size()) {
            logger_->error("[{}] mergeSortTwoBuffers: in1 arrays have different size. {}", in1_filepath);
            return -1;
        }
        if (in2_data->size() != in2_dep_data->size()) {
            logger_->error("[{}] mergeSortTwoBuffers: in2 arrays have different size. {}", in2_filepath);
            return -1;
        }

        size_t out_data_size = in1_data->size() + in2_data->size();
        std::shared_ptr<std::vector<T1>> out_data     = std::make_shared<std::vector<T1>>(out_data_size);
        std::shared_ptr<std::vector<T2>> out_dep_data = std::make_shared<std::vector<T2>>(out_data_size);

        size_t in1_idx = 0;
        size_t in2_idx = 0;
        size_t out_idx = 0;
        while (in1_idx < in1_data->size() && in2_idx < in2_data->size()) {
            const T1 val1 = (*in1_data)[in1_idx];
            const T1 val2 = (*in2_data)[in2_idx];
            if (val1 <= val2) {
//                logger_->debug("[{}] {} (val1, val2): {} <= {}", basename_, out_idx, val1, val2);
                (*out_data)[out_idx] = val1;

                const T2 dep_val1 = (*in1_dep_data)[in1_idx];
                (*out_dep_data)[out_idx] = dep_val1;
                in1_idx++;
            } else {
//                logger_->debug("[{}] {} (val1, val2): {} >  {}", basename_, out_idx, val1, val2);
                (*out_data)[out_idx] = val2;

                const T2 dep_val2 = (*in2_dep_data)[in2_idx];
                (*out_dep_data)[out_idx] = dep_val2;
                in2_idx++;
            }
            out_idx++;
        }
//        logger_->debug("1. total bytes");
//        logger_->debug("in1 = {}, in2 = {}, out = {}",
//            in1_fb_reader.getTotalReadBytes(), in2_fb_reader.getTotalReadBytes(),
//            out_fb_writer.getTotalWriteBytes());
//        logger_->debug("in1_dep = {}, in2_dep = {}, out_dep = {}",
//            in1_dep_fb_reader.getTotalReadBytes(), in2_dep_fb_reader.getTotalReadBytes(),
//            out_dep_fb_writer.getTotalWriteBytes());

        if (in1_idx < in1_data->size()) {
            std::copy(&(*in1_data)[in1_idx], in1_data->data() + in1_data->size(), &(*out_data)[out_idx]);
            std::copy(&(*in1_dep_data)[in1_idx], in1_dep_data->data() + in1_dep_data->size(), &(*out_dep_data)[out_idx]);
        }
        if (in2_idx < in2_dep_data->size()) {
            std::copy(&(*in2_data)[in2_idx], in2_data->data() + in2_data->size(), &(*out_data)[out_idx]);
            std::copy(&(*in2_dep_data)[in2_idx], in2_dep_data->data() + in2_dep_data->size(), &(*out_dep_data)[out_idx]);
        }

//        logger_->debug("2. total bytes");
//        logger_->debug("in1 = {}, in2 = {}, out = {}",
//            in1_fb_reader.getTotalReadBytes(), in2_fb_reader.getTotalReadBytes(),
//            out_fb_writer.getTotalWriteBytes());
//        logger_->debug("in1_dep = {}, in2_dep = {}, out_dep = {}",
//            in1_dep_fb_reader.getTotalReadBytes(), in2_dep_fb_reader.getTotalReadBytes(),
//            out_dep_fb_writer.getTotalWriteBytes());

        std::string out_filepath     = datadir_ + "/" + out_filename;
        std::string out_dep_filepath = datadir_ + "/" + out_dep_filename;

        listTmpDataBuffersKeys();
        logger_->debug("[{}] mergesort - before erase {}", basename_, getStatMem());
        {
            std::lock_guard<std::mutex> lock(tmp_data_mutex_);
            //DEBUG
            logger_->debug("[{}] in1#={}, in2#={}, in1_dep#={}, in2_dep={}", basename_,
                    in1_data.use_count(), in2_data.use_count(), in1_dep_data.use_count(), in2_dep_data.use_count());

            tmp_data_buffers_.erase(in1_filepath);
            tmp_data_buffers_.erase(in2_filepath);
            tmp_data_buffers_.erase(in1_dep_filepath);
            tmp_data_buffers_.erase(in2_dep_filepath);

            tmp_data_buffers_.insert(std::make_pair(out_filepath, TmpDataBuffer(out_data)));
            tmp_data_buffers_.insert(std::make_pair(out_dep_filepath, TmpDataBuffer(out_dep_data)));

        }
        listTmpDataBuffersKeys();
        logger_->debug("[{}] mergesort - after erase {}", basename_, getStatMem());

    } catch (std::exception& ex) {
        logger_->error("[{}] mergeSortTwoBuffers: ({}) {}", basename_, idx, ex.what());
        return -1;
    } catch (...) {
        logger_->error("[{}] mergeSortTwoBuffers: ({}) unknown error..", basename_, idx);
        return -1;
    }

    logger_->trace("[{}] mergeSortTwoBuffers: end   ({})", basename_, idx);
    return 0;
}

void
QuantileNormImpl::sumSortedFiles() {
    logger_->info("[{}] sumSortedFiles: start {}", basename_, summed_file_);

    std::string tmp_summed_filepath = datadir_ + "/.tmp." + summed_file_;
    std::shared_ptr<utils::FileBufferReader<uint16_t>> in_fb_reader[num_channels_];
    utils::FileBufferWriter<unsigned int> out_fb_writer(tmp_summed_filepath, FILEWRITE_BUFSIZE);

    try {
        for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
            std::string in_filepath = datadir_ + "/" + sorted_file1_list_[i];
            in_fb_reader[i] = std::shared_ptr<utils::FileBufferReader<uint16_t>>(new utils::FileBufferReader<uint16_t>(in_filepath, FILEREAD_BUFSIZE));
            in_fb_reader[i]->open();
        }

        out_fb_writer.open();
    } catch (std::runtime_error& ex) {
        logger_->error("[{}] in sumSortedFiles,", basename_);
        logger_->error("[{}] {}", basename_, ex.what());
        throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failOpen", ex.what());
    }

    for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
        in_fb_reader[i]->readFileToBuffer();
    }

//    int count = 0;
    while (1) {
//    while (count++ < 1000) {
        size_t count_finished = 0;
        for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
            if (in_fb_reader[i]->finishedReadingAll()) {
                count_finished++;
            }
        }
        if (count_finished == sorted_file1_list_.size()) {
//            logger_->debug("[{}] all finished", basename_);
            break;
        } else if (count_finished > 0) {
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:invalidInput", "invalid different length of data between files");
        }

        unsigned int sum = 0;
        for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
            sum += in_fb_reader[i]->get();
            in_fb_reader[i]->next();
        }
//        logger_->debug("[{}] [{}] sum = {}", basename_, count, sum);
        out_fb_writer.set(sum);
    }

    if (out_fb_writer.hasBufferData()) {
        out_fb_writer.writeFileFromBuffer();
    }

    out_fb_writer.close();
    for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
        in_fb_reader[i]->close();
    }

    std::string summed_filepath = datadir_ + "/" + summed_file_;
    int ret = rename(tmp_summed_filepath.c_str(), summed_filepath.c_str());
    if (ret != 0) {
        throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failedRename", "cannot rename the tmp summed file to summed file.");
    }

    logger_->info("[{}] sumSortedFiles: end   {}", basename_, summed_file_);
}

void
QuantileNormImpl::sumSortedBuffers() {
    logger_->info("[{}] sumSortedBuffers: start {}", basename_, summed_file_);
    // only one thread treats tmp_data_buffers_

    std::shared_ptr<std::vector<uint16_t>> in_data[num_channels_];
    for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
        std::string in_filepath = datadir_ + "/" + sorted_file1_list_[i];

        auto in_find_itr = tmp_data_buffers_.find(in_filepath);
        if (in_find_itr == tmp_data_buffers_.end()) {
            logger_->error("[{}] in sumSortedBuffers,", basename_);
            logger_->error("[{}] {} not exist in tmp_data_buffers_", basename_, in_filepath);
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failGet", in_filepath + " not exist in tmp_data_buffers_");
        }

        in_data[i] = in_find_itr->second;

        if (i > 0 && (in_data[0]->size() != in_data[i]->size())) {
            logger_->error("[{}] in sumSortedBuffers,", basename_);
            logger_->error("[{}] {} size is wrong", basename_, in_filepath);
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:wrongSize", in_filepath + " size is wrong");
        }
    }

    std::shared_ptr<std::vector<unsigned int>> out_data = std::make_shared<std::vector<unsigned int>>(in_data[0]->size());

//    int count = 0;
    for (size_t j = 0; j < in_data[0]->size(); j++) {
        unsigned int sum = 0;
        for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
            sum += (*in_data[i])[j];
        }
//        logger_->debug("[{}] [{}] sum = {}", basename_, count, sum);
        (*out_data)[j] = sum;
    }

    std::string summed_filepath = datadir_ + "/" + summed_file_;
    tmp_data_buffers_.insert(std::make_pair(summed_filepath, TmpDataBuffer(out_data)));

    logger_->info("[{}] sumSortedBuffers: end   {}", basename_, summed_file_);
}

void
QuantileNormImpl::substituteValues() {
#ifdef DEBUG_NO_THREADING
    std::launch policy = std::launch::deferred;
#else
    std::launch policy = std::launch::async;
#endif

    for (size_t i = 0; i < substituted_file_list_.size(); i++) {
        if (use_tmp_files_) {
            substitute_values_futures_.push_back(std::async(policy, &QuantileNormImpl::substituteToNormValuesWithStorage, this, i));
        } else {
            substitute_values_futures_.push_back(std::async(policy, &QuantileNormImpl::substituteToNormValuesWithMemory, this, i));
        }
    }
}

int
QuantileNormImpl::substituteToNormValuesWithStorage(const size_t idx) {
    logger_->info("[{}] substituteToNormValuesWithStorage: start {} {}", basename_, idx, substituted_file_list_[idx]);

    //TODO: at first copy summed file to tmp subst file, then substitute the same values to mean value.
    std::string summed_filepath = datadir_ + "/" + summed_file_;
    std::string sorted_filepath = datadir_ + "/" + sorted_file1_list_[idx];
    std::string tmp_subst_filepath = datadir_ + "/.tmp." + substituted_file_list_[idx];

    utils::FileBufferReader<unsigned int> summed_fb_reader(summed_filepath, FILEREAD_BUFSIZE);
    utils::FileBufferReader<uint16_t> sorted_fb_reader(sorted_filepath, FILEREAD_BUFSIZE);
    utils::FileBufferWriter<float> subst_fb_writer(tmp_subst_filepath, FILEWRITE_BUFSIZE);
    try {
        summed_fb_reader.open();
        sorted_fb_reader.open();
        subst_fb_writer.open();
    } catch (std::runtime_error& ex) {
        logger_->error("[{}] in substituteToNormValuesWithStorage,", basename_);
        logger_->error("[{}] {}", basename_, ex.what());
        return -1;
    }

    float num_channels = (float)num_channels_;
    std::vector<unsigned int> vals_for_mean;
    summed_fb_reader.readFileToBuffer();
    sorted_fb_reader.readFileToBuffer();
    unsigned int cur_summed_val = summed_fb_reader.get();
    uint16_t     cur_sorted_val = sorted_fb_reader.get();
    summed_fb_reader.next();
    sorted_fb_reader.next();

    unsigned int nxt_summed_val;
    uint16_t     nxt_sorted_val;
    bool loop_continued = true;
    int count = 1;
    while (loop_continued) {
        if (summed_fb_reader.finishedReadingAll() && sorted_fb_reader.finishedReadingAll()) {
            loop_continued = false;
            nxt_sorted_val = cur_sorted_val + 1;
            // for process a value at the last position
        } else {
            assert(! summed_fb_reader.finishedReadingAll());
            assert(! sorted_fb_reader.finishedReadingAll());

            nxt_summed_val = summed_fb_reader.get();
            nxt_sorted_val = sorted_fb_reader.get();
            summed_fb_reader.next();
            sorted_fb_reader.next();
        }

        if (cur_sorted_val != nxt_sorted_val) {
            if (vals_for_mean.empty()) {
                float mean_val = (float)cur_summed_val / num_channels;
                subst_fb_writer.set(mean_val);
            } else {
                vals_for_mean.push_back(cur_summed_val);

                size_t mid_i = vals_for_mean.size() / 2;
                float mean_val = 0.0;
                if (vals_for_mean.size() % 2 == 0) {
                    mean_val = (float)(vals_for_mean[mid_i] + vals_for_mean[mid_i-1]) * 0.5 / num_channels;
                } else {
                    mean_val = (float)vals_for_mean[mid_i] / num_channels;
                }

                for (size_t i = 0; i < vals_for_mean.size(); i++) {
                    subst_fb_writer.set(mean_val);
                }

                vals_for_mean.clear();
            }
        } else {
             vals_for_mean.push_back(cur_summed_val);
        }
        count++;

        cur_summed_val = nxt_summed_val;
        cur_sorted_val = nxt_sorted_val;
    }

    if (subst_fb_writer.hasBufferData()) {
        subst_fb_writer.writeFileFromBuffer();
    }

    subst_fb_writer.close();

    std::string subst_filepath = datadir_ + "/" + substituted_file_list_[idx];
    int ret = rename(tmp_subst_filepath.c_str(), subst_filepath.c_str());
    if (ret != 0) {
        logger_->error("[{}] cannot rename the tmp substituted file to substituted file.", basename_);
        return -1;
    }

    summed_fb_reader.close();
    sorted_fb_reader.close();

#ifndef DEBUG_FILEOUT
    remove(sorted_filepath.c_str());
#endif

    logger_->info("[{}] substituteToNormValuesWithStorage: end   {} {}", basename_, idx, substituted_file_list_[idx]);
    return 0;
}

int
QuantileNormImpl::substituteToNormValuesWithMemory(const size_t idx) {
    logger_->info("[{}] substituteToNormValuesWithMemory: start {} {}", basename_, idx, substituted_file_list_[idx]);

    std::string summed_filepath = datadir_ + "/" + summed_file_;
    std::string sorted_filepath = datadir_ + "/" + sorted_file1_list_[idx];

    std::shared_ptr<std::vector<unsigned int>> summed_data;
    std::shared_ptr<std::vector<uint16_t>> sorted_data;
    {
        std::lock_guard<std::mutex> lock(tmp_data_mutex_);
        auto find_summed_itr = tmp_data_buffers_.find(summed_filepath);
        if (find_summed_itr == tmp_data_buffers_.end()) {
            logger_->error("[{}] in substituteToNormValuesWithMemory,", basename_);
            logger_->error("[{}] no summed data", basename_);
            return -1;
        }
        summed_data = find_summed_itr->second;

        auto find_sorted_itr = tmp_data_buffers_.find(sorted_filepath);
        if (find_sorted_itr == tmp_data_buffers_.end()) {
            logger_->error("[{}] in substituteToNormValuesWithMemory,", basename_);
            logger_->error("[{}] no sorted data", basename_);
            return -1;
        }
        sorted_data = find_sorted_itr->second;
    }

    if (summed_data->size() != sorted_data->size()) {
        logger_->error("[{}] in substituteToNormValuesWithMemory,", basename_);
        logger_->error("[{}] summed data and sorted data are different size", basename_);
        return -1;
    }

    std::shared_ptr<std::vector<float>> subst_data = std::make_shared<std::vector<float>>(summed_data->size());


    float num_channels = (float)num_channels_;
    std::vector<unsigned int> vals_for_mean;
    size_t in_idx = 0;
    size_t out_idx = 0;
    unsigned int cur_summed_val = (*summed_data)[in_idx];
    uint16_t     cur_sorted_val = (*sorted_data)[in_idx];
    in_idx++;

    unsigned int nxt_summed_val;
    uint16_t     nxt_sorted_val;
    bool loop_continued = true;
    int count = 1;
    while (loop_continued) {
        if (in_idx == summed_data->size()) {
            loop_continued = false;
            nxt_sorted_val = cur_sorted_val + 1;
            // for process a value at the last position
        } else {
            nxt_summed_val = (*summed_data)[in_idx];
            nxt_sorted_val = (*sorted_data)[in_idx];
            in_idx++;
        }

        if (cur_sorted_val != nxt_sorted_val) {
            if (vals_for_mean.empty()) {
                float mean_val = (float)cur_summed_val / num_channels;
                (*subst_data)[out_idx++] = mean_val;
            } else {
                vals_for_mean.push_back(cur_summed_val);

                size_t mid_i = vals_for_mean.size() / 2;
                float mean_val = 0.0;
                if (vals_for_mean.size() % 2 == 0) {
                    mean_val = (float)(vals_for_mean[mid_i] + vals_for_mean[mid_i-1]) * 0.5 / num_channels;
                } else {
                    mean_val = (float)vals_for_mean[mid_i] / num_channels;
                }

                for (size_t i = 0; i < vals_for_mean.size(); i++) {
                    (*subst_data)[out_idx++] = mean_val;
                }

                vals_for_mean.clear();
            }
        } else {
             vals_for_mean.push_back(cur_summed_val);
        }
        count++;

        cur_summed_val = nxt_summed_val;
        cur_sorted_val = nxt_sorted_val;
    }

    std::string subst_filepath = datadir_ + "/" + substituted_file_list_[idx];
    {
        std::lock_guard<std::mutex> lock(tmp_data_mutex_);
#ifndef DEBUG_FILEOUT
        //DEBUG
        logger_->debug("[{}] sort#={}", basename_, sorted_data.use_count());

        tmp_data_buffers_.erase(sorted_filepath);
#endif

        tmp_data_buffers_.insert(std::make_pair(subst_filepath, TmpDataBuffer(subst_data)));
    }

    logger_->info("[{}] substituteToNormValuesWithMemory: end   {} {}", basename_, idx, substituted_file_list_[idx]);
    return 0;
}

void
QuantileNormImpl::waitForTasks(const std::string& task_name, std::vector<std::future<int>>& futures) {
    logger_->info("[{}] {} waiting...", basename_, task_name);
    for (size_t i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] {} [{}] failed - {}", basename_, task_name, i, ret);
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failTasks - ", task_name + " failed.");
        }
    }
    logger_->info("[{}] {} done", basename_, task_name);
}

