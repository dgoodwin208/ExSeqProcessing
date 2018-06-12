#ifndef __QUANTILENORM_IMPL_H__
#define __QUANTILENORM_IMPL_H__

#include <string>
#include <tuple>
#include <vector>
#include <future>
#include <thread>

#include "spdlog/spdlog.h"

#define GPU_USER_MEMORY_USAGE_RATIO 0.7
#define THRUST_RADIXSORT_MEMORY_USAGE_RATIO 2.0

//#define DEBUG_FILEOUT
//#define DEBUG_NO_THREADING

class QuantileNormImpl {
protected:
    std::string datadir_;
    std::string basename_;
    std::vector<std::string> tif_fnames_;
    size_t image_width_;
    size_t image_height_;
    size_t num_slices_;
    size_t num_gpus_;
    size_t gpu_mem_total_;

    size_t num_channels_;

    std::vector<std::tuple<size_t, size_t, std::string, std::string>>              radixsort1_file_list_;
    std::vector<std::vector<std::string>>                                          mergesort1_file_list_;
    std::vector<std::tuple<std::string, std::string>>                              subsitute_values_file_list_;
    std::vector<std::tuple<size_t, size_t, std::string, std::string, std::string>> radixsort2_file_list_;
    std::vector<std::vector<std::string>>                                          mergesort2_file_list_;

    std::vector<std::future<int>> radixsort1_futures_;
    std::vector<std::future<int>> mergesort1_futures_;
    std::vector<std::future<int>> substitute_values_futures_;
    std::vector<std::future<int>> radixsort2_futures_;
    std::vector<std::future<int>> mergesort2_futures_;

    std::vector<std::string> sorted_file1_list_;
    std::vector<std::string> substituted_file_list_;
    std::vector<std::string> sorted_file2_list_;


    std::string summed_file_;

    std::shared_ptr<spdlog::logger> logger_;

    const size_t FILEREAD_BUFSIZE  = 1024*256;
    const size_t FILEWRITE_BUFSIZE = 1024*256;


public:
    QuantileNormImpl();
    QuantileNormImpl(const std::string& datadir,
                     const std::string& basename,
                     const std::vector<std::string>& tif_fnames,
                     const size_t image_width,
                     const size_t image_height,
                     const size_t num_slices,
                     const size_t num_gpus);

    void run();

    std::vector<std::string>& getResult() { return sorted_file2_list_; }


protected:
    void setupFileList();
    void makeMergeSortFileList(const std::string& file_prefix,
                               const std::string& dep_file_prefix,
                               const std::vector<std::tuple<size_t, size_t>>& idx,
                               std::vector<std::vector<std::string>>& mergesort_file_list);

    bool filesExists(const std::vector<std::string>& file_list);
    bool oneFileExists(const std::string& filename);

    void radixSort1();
    int radixSort1FromData(std::shared_ptr<std::vector<uint16_t>> image, const size_t slice_start, const std::string& out_filename);

    void radixSort2();
    int radixSort2FromData(const size_t idx_radixsort);

    template<typename T>
    int savefile(const std::string& datadir, const std::string& out_filename, const std::shared_ptr<std::vector<T>> data);

    template<typename T>
    std::shared_ptr<std::vector<T>> loadfile(const std::string& in_filepath, const size_t num_data_start, const size_t data_size);

    int selectGPU();
    void unselectGPU(const int idx_gpu);

    void selectCore(const int idx_core_group);
    int selectCoreNoblock(const int idx_core_group);
    void unselectCore(const int idx_core_group);

    void mergeSort1();
    void mergeSort2();

    template <typename T1, typename T2>
    int mergeSortTwoFiles(const size_t idx, const std::vector<std::vector<std::string>>& mergesort_file_list);

    void sumSortedFiles();

    void substituteValues();
    int substituteToNormValues(size_t idx);

    void waitForTasks(const std::string& task_name, std::vector<std::future<int>>& futures);
};


class ExceptionToMATLAB {
    std::string matlab_id_;
    std::string message_;
public:
    ExceptionToMATLAB(const std::string& matlab_id, const std::string& message)
        : matlab_id_(matlab_id),
          message_(message) {

    }

    const std::string& getMatlabId() const { return matlab_id_; }
    const std::string& getMessage() const { return message_; }
};

#endif // __QUANTILENORM_IMPL_H__

