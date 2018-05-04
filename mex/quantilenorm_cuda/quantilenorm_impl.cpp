#include <sstream>
#include <iomanip>
#include <exception>
#include <chrono>
#include <cassert>
#include <cstring>
#include <semaphore.h>
#include <fcntl.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "quantilenorm_impl.h"
#include "utils/filebuffers.h"
#include "mex-utils/tiffs.h"
#include "cuda-utils/radixsort.h"

QuantileNormImpl::QuantileNormImpl()
    : datadir_(""),
      basename_(""),
      tif_fnames_(),
      image_width_(0),
      image_height_(0),
      num_slices_(0),
      num_gpus_(0),
      num_channels_(0),
      max_data_size_for_gpu_(MAX_DATA_SIZE_FOR_GPU) {
    logger_ = spdlog::get("mex_logger");
}

QuantileNormImpl::QuantileNormImpl(const std::string& datadir,
                                   const std::string& basename,
                                   const std::vector<std::string>& tif_fnames,
                                   const size_t image_width,
                                   const size_t image_height,
                                   const size_t num_slices,
                                   const size_t num_gpus,
                                   const size_t max_data_size_for_gpu)
    : datadir_(datadir),
      basename_(basename),
      tif_fnames_(tif_fnames),
      image_width_(image_width),
      image_height_(image_height),
      num_slices_(num_slices),
      num_gpus_(num_gpus),
      num_channels_(tif_fnames.size()),
      max_data_size_for_gpu_(max_data_size_for_gpu)
{
    logger_ = spdlog::get("mex_logger");
}

void
QuantileNormImpl::run() {
    logger_->info("[{}] ##### sort 1", basename_);

    setupFileList();

    if (filesExists(sorted_file1_list_)) {
        logger_->info("[{}] already exists sorted files 1.", basename_);
    } else {
        mergeSort1();
        radixSort1();

#ifndef SEQUENTIAL_RUN
        waitForTasks("radixsort1", radixsort1_futures_);
        waitForTasks("mergesort1", mergesort1_futures_);
#endif
    }

    logger_->info("[{}] ##### sum", basename_);
    if (oneFileExists(summed_file_)) {
        logger_->info("[{}] already exists summed files.", basename_);
    } else {
        sumSortedFiles();
    }

    logger_->info("[{}] ##### substitute to normal values", basename_);
    if (filesExists(substituted_file_list_)) {
        logger_->info("[{}] already exists substituted files.", basename_);
    } else {
        substituteValues();

#ifndef SEQUENTIAL_RUN
        waitForTasks("substitute-values", substitute_values_futures_);
#endif
    }

    logger_->info("[{}] ##### sort 2", basename_);
    if (filesExists(sorted_file2_list_)) {
        logger_->info("[{}] already exists sorted files 2.", basename_);
    } else {
        mergeSort2();
        radixSort2();

#ifndef SEQUENTIAL_RUN
        waitForTasks("radixsort2", radixsort2_futures_);
        waitForTasks("mergesort2", mergesort2_futures_);
#endif
    }

    logger_->info("[{}] ##### done", basename_);
}


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

    size_t one_slice_size = image_height_ * image_width_ * sizeof(uint16_t);
    size_t num_sub_slices = max_data_size_for_gpu_ / one_slice_size;
    logger_->debug("[{}] data(uint16_t) - max data size = {}, one slice size = {}, # sub slices = {}", basename_, max_data_size_for_gpu_, one_slice_size, num_sub_slices);
    if (num_sub_slices == 0) {
        logger_->warn("[{}] MAX DATA SIZE_FOR_GPU is smaller than one slice size.", basename_);
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
            radixsort1_file_list_.push_back(std::make_tuple(i, next_i, tif_fnames_[c_i], radixsort1_file));
            logger_->info("[{}] [radixsort 1] {} -> {}", basename_, tif_fnames_[c_i], radixsort1_file);
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

    size_t total_data_size = num_slices_ * image_height_ * image_width_;
    size_t unit_data_size  = max_data_size_for_gpu_ / sizeof(double);
    size_t num_sub_data    = total_data_size / unit_data_size;
    if (total_data_size % unit_data_size != 0) {
        num_sub_data++;
    }
    logger_->debug("[{}] data(double) - max data size = {}, total data size = {}, unit data size = {}, sub data size = {}", basename_, max_data_size_for_gpu_, total_data_size, unit_data_size, num_sub_data);
    if (num_sub_data == 0) {
        logger_->warn("[{}] MAX_DATA_SIZE_FOR_GPU is larger than total data size.", basename_);
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
QuantileNormImpl::filesExists(const std::vector<std::string>& file_list) {
    bool exists_files = true;
    for (auto filename : file_list) {
        std::string filepath = datadir_ + "/" + filename;
        std::ifstream fin(filepath, std::ios::in | std::ios::binary);
        if (! fin.is_open()) {
            exists_files = false;
            break;
        }
        fin.close();
    }

    return exists_files;
}

bool
QuantileNormImpl::oneFileExists(const std::string& filename) {
    std::string filepath = datadir_ + "/" + filename;
    std::ifstream fin(filepath, std::ios::in | std::ios::binary);
    bool exists_file = (fin.is_open());
    fin.close();

    return exists_file;
}

void
QuantileNormImpl::radixSort1() {
    for (auto radixsortfile : radixsort1_file_list_) {
        size_t slice_start   = std::get<0>(radixsortfile);
        size_t slice_end     = std::get<1>(radixsortfile);
        std::string tif_file = std::get<2>(radixsortfile);
        std::string out_file = std::get<3>(radixsortfile);

        selectCore(0);

        logger_->info("[{}] radixSort1: loadtiff start {}", basename_, tif_file);
        std::shared_ptr<std::vector<uint16_t>> image(new std::vector<uint16_t>());
        mexutils::loadtiff(tif_file, slice_start, slice_end, image);
        logger_->info("[{}] radixSort1: loadtiff end   {}", basename_, tif_file);

#ifdef SEQUENTIAL_RUN
        QuantileNormImpl::radixSort1FromData(image, slice_start, out_file);
#else
        radixsort1_futures_.push_back(std::async(std::launch::async, &QuantileNormImpl::radixSort1FromData, this, image, slice_start, out_file));
#endif
    }
}

int
QuantileNormImpl::radixSort1FromData(std::shared_ptr<std::vector<uint16_t>> image, const size_t slice_start, const std::string& out_filename) {
    logger_->info("[{}] radixSort1FromData: start {}", basename_, out_filename);
    unsigned int idx_start = slice_start * image_height_ * image_width_;
    std::shared_ptr<std::vector<unsigned int>> idx(new std::vector<unsigned int>(image->size()));
    thrust::sequence(thrust::host, idx->begin(), idx->end(), idx_start);

    auto interval_sec = std::chrono::seconds(1);
    logger_->info("[{}] radixSort1FromData: selectGPU {}", basename_, out_filename);
    while (1) {
        int idx_gpu = selectGPU();
        if (idx_gpu >= 0) {
            logger_->info("[{}] idx_gpu = {}", basename_, idx_gpu);
            cudautils::radixsort(*image, *idx);

            unselectGPU(idx_gpu);
            break;
        } else {
//            int ret = selectCoreNoblock(1);
//            if (ret == 0) {
//                logger_->info("[{}] core", basename_);
//                cudautils::radixsort_host(*image, *idx);
//                unselectCore(1);
//                break;
//            } else {
                std::this_thread::sleep_for(interval_sec);
//            }
        }
    }

    int ret;
    std::string out_idx_filename = "idx_" + out_filename;
    ret = savefile(datadir_, out_idx_filename, idx);
    ret = savefile(datadir_, out_filename, image);

    unselectCore(0);
    logger_->info("[{}] radixSort1FromData: end   {}", basename_, out_filename);
    return ret;
}

void
QuantileNormImpl::radixSort2() {
    for (size_t i = 0; i < radixsort2_file_list_.size(); i++) {
#ifdef SEQUENTIAL_RUN
        QuantileNormImpl::radixSort2FromData(i);
#else
        radixsort2_futures_.push_back(std::async(std::launch::async, &QuantileNormImpl::radixSort2FromData, this, i));
#endif
    }
}

int
QuantileNormImpl::radixSort2FromData(const size_t idx_radixsort) {
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

        selectCore(0);
        std::shared_ptr<std::vector<double>> data;
        std::shared_ptr<std::vector<unsigned int>> index;
        data  = loadfile<double>(in_subst_filepath, num_data_start, data_size);
        index = loadfile<unsigned int>(in_idx_filepath, num_data_start, data_size);
        if (data == nullptr) {
            logger_->debug("[{}] radixSort2FromData: ({}) failed to load data file", basename_, idx_radixsort);
            return -1;
        }
        if (index == nullptr) {
            logger_->debug("[{}] radixSort2FromData: ({}) failed to load index file", basename_, idx_radixsort);
            return -1;
        }

        int ret;
        auto interval_sec = std::chrono::seconds(1);
//        logger_->info("[{}] radixSort2FromData: ({}) selectGPU", basename_, idx_radixsort);
        while (1) {
            int idx_gpu = selectGPU();
            if (idx_gpu >= 0) {
                logger_->info("[{}] radixSort2FromData: ({}) idx_gpu = {}", basename_, idx_radixsort, idx_gpu);

                try {
                    cudautils::radixsort<unsigned int, double>(*index, *data);
                } catch (std::exception& ex) {
                    logger_->debug("[{}] radixSort2FromData: {}", basename_, ex.what());
                    cudaError err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        logger_->error("[{}] {}", basename_, cudaGetErrorString(err));
                    }
                    unselectGPU(idx_gpu);
                    std::this_thread::sleep_for(interval_sec);
                    continue;
                } catch (...) {
                    logger_->debug("[{}] radixSort2FromData: unknown error", basename_);
                    cudaError err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        logger_->error("[{}] {}", basename_, cudaGetErrorString(err));
                    }
                    unselectGPU(idx_gpu);
                    std::this_thread::sleep_for(interval_sec);
                    continue;
                }

                unselectGPU(idx_gpu);
                break;
            } else {
//                ret = selectCoreNoblock(3);
//                if (ret == 0) {
//                    logger_->info("[{}] radixSort2FromData: ({}) core", basename_, idx_radixsort);
//                    cudautils::radixsort_host(*index, *data);
//                    unselectCore(3);
//                    break;
//                } else {
                    std::this_thread::sleep_for(interval_sec);
//                }
            }
        }

        std::string out_idx_file = "idx_" + out_file;
        ret = savefile(datadir_, out_idx_file, index);
        ret = savefile(datadir_, out_file, data);

        unselectCore(0);
        logger_->info("[{}] radixSort2FromData: end   ({})", basename_, idx_radixsort);
        return ret;
    } catch (std::exception& ex) {
            logger_->error("[{}] end - {}", basename_, ex.what());
            return -1;
    } catch (...) {
            logger_->error("[{}] end - unknown error..", basename_);
            return -1;
    }

    return 0;
}

template<typename T>
int
QuantileNormImpl::savefile(const std::string& datadir, const std::string& out_filename, const std::shared_ptr<std::vector<T>> data) {
    std::string tmp_out_filepath = datadir + "/.tmp." + out_filename;
    std::ofstream fout(tmp_out_filepath, std::ios::out | std::ios::binary);
    if (! fout.is_open()) {
        logger_->error("[{}] cannot open an output file for radixsort.", basename_);
        return -1;
    }

    int buffer_size = FILEWRITE_BUFSIZE / sizeof(T);
    for (int i = 0; i < data->size(); i += buffer_size) {
        if (i + buffer_size > data->size()) {
            buffer_size = data->size() - i;
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

template<typename T>
std::shared_ptr<std::vector<T>>
QuantileNormImpl::loadfile(const std::string& in_filepath, const size_t num_data_start, const size_t data_size) {
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

int
QuantileNormImpl::selectGPU() {
    int idx_gpu = -1;
    for (size_t i = 0; i < num_gpus_; i++) {
        std::string sem_name = "/g" + std::to_string(i);
//        logger_->trace("[{}] sem_name = {}", basename_, sem_name);
        sem_t *sem;
        sem = sem_open(sem_name.c_str(), O_RDWR);
        int ret = errno;
        if (sem == SEM_FAILED) {
            logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
            continue;
        }

        ret = sem_trywait(sem);
        if (ret == 0) {
            logger_->trace("[{}] selectGPU {}", basename_, sem_name);
            idx_gpu = i;
            cudaSetDevice(idx_gpu);
            break;
        }
    }

    return idx_gpu;
}

void
QuantileNormImpl::unselectGPU(const int idx_gpu) {
    std::string sem_name = "/g" + std::to_string(idx_gpu);
    sem_t *sem;
    sem = sem_open(sem_name.c_str(), O_RDWR);
    int ret = errno;
    if (sem == SEM_FAILED) {
        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
        return;
    }

    cudaDeviceReset();

    logger_->trace("[{}] unselectGPU {}", basename_, sem_name);
    ret = sem_post(sem);
    if (ret != 0) {
        logger_->error("[{}] cannot post semaphore of {}", basename_, sem_name);
        return;
    }
}

void
QuantileNormImpl::selectCore(const int idx_core_group) {
    std::string sem_name = "/qn_c" + std::to_string(idx_core_group);
//    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
    sem_t *sem;
    sem = sem_open(sem_name.c_str(), O_RDWR);
    int ret = errno;
    if (sem == SEM_FAILED) {
        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
        return;
    }

    auto interval_sec = std::chrono::seconds(1);
    int count = 1;
    while (1) {
        ret = sem_trywait(sem);
        if (ret == 0) {
            break;
        }
        std::this_thread::sleep_for(interval_sec);
        count++;
    }
    logger_->trace("[{}] selectCore: {} ({})", basename_, sem_name, count);
}

int
QuantileNormImpl::selectCoreNoblock(const int idx_core_group) {
    std::string sem_name = "/qn_c" + std::to_string(idx_core_group);
//    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
    sem_t *sem;
    sem = sem_open(sem_name.c_str(), O_RDWR);
    int ret = errno;
    if (sem == SEM_FAILED) {
        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
        return -1;
    }

    ret = sem_trywait(sem);
    if (ret == 0) {
        logger_->trace("[{}] selectCoreNoblock: {}", basename_, sem_name);
    }

    return ret;
}

void
QuantileNormImpl::unselectCore(const int idx_core_group) {
    std::string sem_name = "/qn_c" + std::to_string(idx_core_group);
//    logger_->trace("[{}] sem_name = {}", basename_, sem_name);
    sem_t *sem;
    sem = sem_open(sem_name.c_str(), O_RDWR);
    int ret = errno;
    if (sem == SEM_FAILED) {
        logger_->error("[{}] cannot open semaphore of {}", basename_, sem_name);
        return;
    }

    ret = sem_post(sem);
    if (ret == -1) {
        logger_->error("[{}] unselect failed; {}", basename_, sem_name);
    }
    logger_->trace("[{}] unselectCore: {}", basename_, sem_name);
}

void
QuantileNormImpl::mergeSort1() {
    for (size_t i = 0; i < mergesort1_file_list_.size(); i++) {
#ifdef SEQUENTIAL_RUN
        mergeSortTwoFiles<uint16_t, unsigned int>(i, mergesort1_file_list_);
#else
        mergesort1_futures_.push_back(std::async(std::launch::async, &QuantileNormImpl::mergeSortTwoFiles<uint16_t, unsigned int>, this, i, mergesort1_file_list_));
#endif
    }
}

void
QuantileNormImpl::mergeSort2() {
    for (size_t i = 0; i < mergesort2_file_list_.size(); i++) {
#ifdef SEQUENTIAL_RUN
        mergeSortTwoFiles<unsigned int, double>(i, mergesort2_file_list_);
#else
        mergesort2_futures_.push_back(std::async(std::launch::async, &QuantileNormImpl::mergeSortTwoFiles<unsigned int, double>, this, i, mergesort2_file_list_));
#endif
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

void
QuantileNormImpl::sumSortedFiles(){
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

    std::string summed_filepath = datadir_ + "/" + summed_file_;
    int ret = rename(tmp_summed_filepath.c_str(), summed_filepath.c_str());
    if (ret != 0) {
        throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failedRename", "cannot rename the tmp summed file to summed file.");
    }

    logger_->info("[{}] sumSortedFiles: end   {}", basename_, summed_file_);
}

void
QuantileNormImpl::substituteValues() {
    for (size_t i = 0; i < substituted_file_list_.size(); i++) {
#ifdef SEQUENTIAL_RUN
        QuantileNormImpl::substituteToNormValues(i);
#else
        substitute_values_futures_.push_back(std::async(std::launch::async, &QuantileNormImpl::substituteToNormValues, this, i));
#endif
    }
}

int
QuantileNormImpl::substituteToNormValues(const size_t idx) {
    logger_->info("[{}] substituteToNormValues: start {} {}", basename_, idx, substituted_file_list_[idx]);

    //TODO: at first copy summed file to tmp subst file, then substitute the same values to mean value.
    std::string summed_filepath = datadir_ + "/" + summed_file_;
    std::string sorted_filepath = datadir_ + "/" + sorted_file1_list_[idx];
    std::string tmp_subst_filepath = datadir_ + "/.tmp." + substituted_file_list_[idx];

    utils::FileBufferReader<unsigned int> summed_fb_reader(summed_filepath, FILEREAD_BUFSIZE);
    utils::FileBufferReader<uint16_t> sorted_fb_reader(sorted_filepath, FILEREAD_BUFSIZE);
    utils::FileBufferWriter<double> subst_fb_writer(tmp_subst_filepath, FILEWRITE_BUFSIZE);
    try {
        summed_fb_reader.open();
        sorted_fb_reader.open();
        subst_fb_writer.open();
    } catch (std::runtime_error& ex) {
        logger_->error("[{}] in substituteToNormValues,", basename_);
        logger_->error("[{}] {}", basename_, ex.what());
        return -1;
    }

    double num_channels = (double)num_channels_;
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
                double mean_val = (double)cur_summed_val / num_channels;
                subst_fb_writer.set(mean_val);
            } else {
                vals_for_mean.push_back(cur_summed_val);

                size_t mid_i = vals_for_mean.size() / 2;
                double mean_val = 0.0;
                if (vals_for_mean.size() % 2 == 0) {
                    mean_val = (double)(vals_for_mean[mid_i] + vals_for_mean[mid_i-1]) * 0.5 / num_channels;
                } else {
                    mean_val = (double)vals_for_mean[mid_i] / num_channels;
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

    logger_->info("[{}] substituteToNormValues: end   {} {}", basename_, idx, substituted_file_list_[idx]);
    return 0;
}

void
QuantileNormImpl::waitForTasks(const std::string& task_name, std::vector<std::future<int>>& futures) {
    logger_->info("[{}] {} waiting...", basename_, task_name);
    for (size_t i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] {} [{}] failed - {}", basename_, task_name, i, ret);
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failTasks", task_name + "failed.");
        }
    }
    logger_->info("[{}] {} done", basename_, task_name);
}

