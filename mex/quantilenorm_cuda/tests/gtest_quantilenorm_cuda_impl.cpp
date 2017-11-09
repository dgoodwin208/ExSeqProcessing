#include "gtest/gtest.h"

#include "quantilenorm_impl.h"
#include "filebuffers.h"
#include "spdlog/spdlog.h"

#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

// mock of mexutils
namespace mexutils {

#define WRITE_BUFFER_SIZE 1024*1024*512
#define READ_BUFFER_SIZE  1024*1024*512
#define IMAGE_X 100
#define IMAGE_Y 100
#define IMAGE_Z 10
#define GPU_NUM 2

void loadtiff(const std::string& filename, const size_t slice_start, const size_t slice_end, std::shared_ptr<std::vector<uint16_t>> image)
{
    size_t idx_start = IMAGE_X * IMAGE_Y * slice_start;
    size_t idx_end   = IMAGE_X * IMAGE_Y * (slice_end + 1) - 1;

    utils::FileBufferReader<uint16_t> fb_reader(filename, READ_BUFFER_SIZE);
    fb_reader.open();

    fb_reader.readFileToBuffer();

    size_t count = 0;
    while (! fb_reader.finishedReadingAll()) {
        uint16_t val = fb_reader.get();
        if (count >= idx_start && count <= idx_end) {
            (*image).push_back(val);
        }
        count++;
        fb_reader.next();
    }

    fb_reader.close();
}

}

namespace {

class QuantileNormCudaTest : public QuantileNormImpl, public ::testing::Test {
protected:
    QuantileNormCudaTest()
        : QuantileNormImpl(".", "test", {}, IMAGE_X, IMAGE_Y, IMAGE_Z, GPU_NUM, IMAGE_X * IMAGE_Y * 4 * sizeof(uint16_t)) {
        std::shared_ptr<spdlog::logger> logger;
        logger_ = spdlog::get("mex_logger");
        if (logger_ == nullptr) {
            logger_ = spdlog::basic_logger_mt("mex_logger", "mex.log");
        }

        mode_t old_umask = umask(0);
        for (size_t i = 0; i < num_gpus_; i++) {
            std::string sem_name = "/g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
            sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, 1);
        }
        umask(old_umask);

        num_channels_ = 4;
        std::default_random_engine generator(1);
        std::normal_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < 4; i++) {
            std::string fname = "img_" + std::to_string(i) + ".tif";
            tif_fnames_.push_back(fname);

            utils::FileBufferWriter<uint16_t> fb_writer(fname, WRITE_BUFFER_SIZE);
            fb_writer.open();
            for (int z = 0; z < IMAGE_Z; z++) {
                for (int y = 0; y < IMAGE_Y; y++) {
                    for (int x = 0; x < IMAGE_X; x++) {
                        double v = 1000.0 + 10000.0 * distribution(generator);
                        if (v < 0.0) v = 0.0;
                        uint16_t v_ui16 = (uint16_t)v;
                        fb_writer.set(v_ui16);
                    }
                }
            }
            fb_writer.writeFileFromBuffer();
            fb_writer.close();
        }
    }
    virtual ~QuantileNormCudaTest() {
        for (int i = 0; i < 4; i++) {
            remove(tif_fnames_[i].c_str());
        }

        for (size_t i = 0; i < num_gpus_; i++) {
            std::string sem_name = "/g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
        }
    }
};


TEST_F(QuantileNormCudaTest, MakeMergeSortFileListTest) {
    std::string file_prefix = "data";
    std::string dep_file_prefix = "data_dep";
    std::vector<std::tuple<size_t, size_t>> idx;

    // all are even
    // 1   2   3   4
    // 1 = 2   3 = 4
    // 1 - 2 = 3 - 4
    for (int i = 1; i <= 4; i++) {
        idx.push_back(std::make_tuple(i, i));
    }

    // run
    std::vector<std::vector<std::string>> mergesort_file_list;
    makeMergeSortFileList(file_prefix, dep_file_prefix, idx, mergesort_file_list);

    // check
    ASSERT_EQ(12, mergesort_file_list.size());
    ASSERT_EQ(6,  mergesort_file_list[0].size());

    std::string prefix;
    for (int i = 0; i < 4; i++) {
        prefix = file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 3][0]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 3][1]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 3][2]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 3][0]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 3][1]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 3][2]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[2 + i * 3][0]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[2 + i * 3][1]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[2 + i * 3][2]);

        prefix = dep_file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 3][3]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 3][4]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 3][5]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 3][3]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 3][4]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 3][5]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[2 + i * 3][3]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[2 + i * 3][4]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[2 + i * 3][5]);
    }

    // odd firstly
    // 1   2   3   4   5
    // 1 = 2   3 = 4   5
    // 1 - 2   3 - 4 = 5
    // 1 - 2 = 3 - 4 - 5
    idx.clear();
    for (int i = 1; i <= 5; i++) {
        idx.push_back(std::make_tuple(i, i));
    }

    // run
    mergesort_file_list.clear();
    makeMergeSortFileList(file_prefix, dep_file_prefix, idx, mergesort_file_list);

    // check
    ASSERT_EQ(16, mergesort_file_list.size());
    ASSERT_EQ(6,  mergesort_file_list[0].size());

    for (int i = 0; i < 4; i++) {
        prefix = file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 4][0]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 4][1]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 4][2]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 4][0]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 4][1]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 4][2]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[2 + i * 4][0]);
        ASSERT_EQ(prefix + "-005-005.bin", mergesort_file_list[2 + i * 4][1]);
        ASSERT_EQ(prefix + "-003-005.bin", mergesort_file_list[2 + i * 4][2]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[3 + i * 4][0]);
        ASSERT_EQ(prefix + "-003-005.bin", mergesort_file_list[3 + i * 4][1]);
        ASSERT_EQ(prefix + "-001-005.bin", mergesort_file_list[3 + i * 4][2]);

        prefix = dep_file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 4][3]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 4][4]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 4][5]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 4][3]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 4][4]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 4][5]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[2 + i * 4][3]);
        ASSERT_EQ(prefix + "-005-005.bin", mergesort_file_list[2 + i * 4][4]);
        ASSERT_EQ(prefix + "-003-005.bin", mergesort_file_list[2 + i * 4][5]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[3 + i * 4][3]);
        ASSERT_EQ(prefix + "-003-005.bin", mergesort_file_list[3 + i * 4][4]);
        ASSERT_EQ(prefix + "-001-005.bin", mergesort_file_list[3 + i * 4][5]);
    }

    // even firstly, then odd, and even lastly
    // 1   2   3   4   5   6
    // 1 = 2   3 = 4   5 = 6
    // 1 - 2 = 3 - 4   5 - 6
    // 1 - 2 - 3 - 4 = 5 - 6
    idx.clear();
    for (int i = 1; i <= 6; i++) {
        idx.push_back(std::make_tuple(i, i));
    }

    // run
    mergesort_file_list.clear();
    makeMergeSortFileList(file_prefix, dep_file_prefix, idx, mergesort_file_list);

    // check
    ASSERT_EQ(20, mergesort_file_list.size());
    ASSERT_EQ(6,  mergesort_file_list[0].size());

    for (int i = 0; i < 4; i++) {
        prefix = file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 5][0]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 5][1]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 5][2]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 5][0]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 5][1]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 5][2]);
        ASSERT_EQ(prefix + "-005-005.bin", mergesort_file_list[2 + i * 5][0]);
        ASSERT_EQ(prefix + "-006-006.bin", mergesort_file_list[2 + i * 5][1]);
        ASSERT_EQ(prefix + "-005-006.bin", mergesort_file_list[2 + i * 5][2]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[3 + i * 5][0]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[3 + i * 5][1]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[3 + i * 5][2]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[4 + i * 5][0]);
        ASSERT_EQ(prefix + "-005-006.bin", mergesort_file_list[4 + i * 5][1]);
        ASSERT_EQ(prefix + "-001-006.bin", mergesort_file_list[4 + i * 5][2]);

        prefix = dep_file_prefix + std::to_string(i);
        ASSERT_EQ(prefix + "-001-001.bin", mergesort_file_list[0 + i * 5][3]);
        ASSERT_EQ(prefix + "-002-002.bin", mergesort_file_list[0 + i * 5][4]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[0 + i * 5][5]);
        ASSERT_EQ(prefix + "-003-003.bin", mergesort_file_list[1 + i * 5][3]);
        ASSERT_EQ(prefix + "-004-004.bin", mergesort_file_list[1 + i * 5][4]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[1 + i * 5][5]);
        ASSERT_EQ(prefix + "-005-005.bin", mergesort_file_list[2 + i * 5][3]);
        ASSERT_EQ(prefix + "-006-006.bin", mergesort_file_list[2 + i * 5][4]);
        ASSERT_EQ(prefix + "-005-006.bin", mergesort_file_list[2 + i * 5][5]);
        ASSERT_EQ(prefix + "-001-002.bin", mergesort_file_list[3 + i * 5][3]);
        ASSERT_EQ(prefix + "-003-004.bin", mergesort_file_list[3 + i * 5][4]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[3 + i * 5][5]);
        ASSERT_EQ(prefix + "-001-004.bin", mergesort_file_list[4 + i * 5][3]);
        ASSERT_EQ(prefix + "-005-006.bin", mergesort_file_list[4 + i * 5][4]);
        ASSERT_EQ(prefix + "-001-006.bin", mergesort_file_list[4 + i * 5][5]);
    }
}

TEST_F(QuantileNormCudaTest, FilesExistsTest) {
    ASSERT_TRUE(filesExists(tif_fnames_));

    std::vector<std::string> file_list(tif_fnames_.begin(), tif_fnames_.end());
    file_list.push_back("dummy_img_0.tif");
    ASSERT_FALSE(filesExists(file_list));
}

TEST_F(QuantileNormCudaTest, OneFileExistsTest) {
    std::string filepath = datadir_ + "/" + "img_0.tif";
    ASSERT_TRUE(oneFileExists(filepath));

    filepath = datadir_ + "/" + "dummy_img_0.tif";
    ASSERT_FALSE(oneFileExists(filepath));
}

TEST_F(QuantileNormCudaTest, SetUpFileListTest) {
    // run
    setupFileList();

    // check
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::string prefix = basename_ + "_1_sort1_c" + std::to_string(c_i);
        std::string filename = prefix + "-000-00" + std::to_string(IMAGE_Z - 1) + ".bin";
        EXPECT_EQ(sorted_file1_list_[c_i], filename);

        EXPECT_EQ(0,                       std::get<0>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(3,                       std::get<1>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(tif_fnames_[c_i],        std::get<2>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(prefix + "-000-003.bin", std::get<3>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(4,                       std::get<0>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(7,                       std::get<1>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(tif_fnames_[c_i],        std::get<2>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(prefix + "-004-007.bin", std::get<3>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(8,                       std::get<0>(radixsort1_file_list_[c_i + 8]));
        EXPECT_EQ(9,                       std::get<1>(radixsort1_file_list_[c_i + 8]));
        EXPECT_EQ(tif_fnames_[c_i],        std::get<2>(radixsort1_file_list_[c_i + 8]));
        EXPECT_EQ(prefix + "-008-009.bin", std::get<3>(radixsort1_file_list_[c_i + 8]));

        EXPECT_EQ(prefix + "-000-003.bin", mergesort1_file_list_[0 + c_i * 2][0]);
        EXPECT_EQ(prefix + "-004-007.bin", mergesort1_file_list_[0 + c_i * 2][1]);
        EXPECT_EQ(prefix + "-000-007.bin", mergesort1_file_list_[0 + c_i * 2][2]);
        EXPECT_EQ(prefix + "-000-007.bin", mergesort1_file_list_[1 + c_i * 2][0]);
        EXPECT_EQ(prefix + "-008-009.bin", mergesort1_file_list_[1 + c_i * 2][1]);
        EXPECT_EQ(prefix + "-000-009.bin", mergesort1_file_list_[1 + c_i * 2][2]);
        EXPECT_EQ("idx_" + prefix + "-000-003.bin", mergesort1_file_list_[0 + c_i * 2][3]);
        EXPECT_EQ("idx_" + prefix + "-004-007.bin", mergesort1_file_list_[0 + c_i * 2][4]);
        EXPECT_EQ("idx_" + prefix + "-000-007.bin", mergesort1_file_list_[0 + c_i * 2][5]);
        EXPECT_EQ("idx_" + prefix + "-000-007.bin", mergesort1_file_list_[1 + c_i * 2][3]);
        EXPECT_EQ("idx_" + prefix + "-008-009.bin", mergesort1_file_list_[1 + c_i * 2][4]);
        EXPECT_EQ("idx_" + prefix + "-000-009.bin", mergesort1_file_list_[1 + c_i * 2][5]);
    }

    EXPECT_EQ(summed_file_, basename_ + "_2_sort1_sum.bin");

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::string filename = basename_ + "_3_subst_c" + std::to_string(c_i) + "-000-00" + std::to_string(IMAGE_Z - 1) + ".bin";
        EXPECT_EQ(filename, substituted_file_list_[c_i]);
    }

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::string prefix = basename_ + "_4_sort2_c" + std::to_string(c_i);
        std::string filename = prefix + "-000-009.bin";
        std::string idx_filename = "idx_" + basename_ + "_1_sort1_c" + std::to_string(c_i) + "-000-009.bin";
        EXPECT_EQ(filename, sorted_file2_list_[c_i]);

        EXPECT_EQ(0,                           std::get<0>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(prefix + "-000-000.bin",     std::get<4>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(100*100,                     std::get<0>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(prefix + "-001-001.bin",     std::get<4>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(2*100*100,                   std::get<0>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(prefix + "-002-002.bin",     std::get<4>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(3*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(prefix + "-003-003.bin",     std::get<4>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(4*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 16]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 16]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 16]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 16]));
        EXPECT_EQ(prefix + "-004-004.bin",     std::get<4>(radixsort2_file_list_[c_i + 16]));
        EXPECT_EQ(5*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 20]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 20]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 20]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 20]));
        EXPECT_EQ(prefix + "-005-005.bin",     std::get<4>(radixsort2_file_list_[c_i + 20]));
        EXPECT_EQ(6*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 24]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 24]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 24]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 24]));
        EXPECT_EQ(prefix + "-006-006.bin",     std::get<4>(radixsort2_file_list_[c_i + 24]));
        EXPECT_EQ(7*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 28]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 28]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 28]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 28]));
        EXPECT_EQ(prefix + "-007-007.bin",     std::get<4>(radixsort2_file_list_[c_i + 28]));
        EXPECT_EQ(8*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 32]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 32]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 32]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 32]));
        EXPECT_EQ(prefix + "-008-008.bin",     std::get<4>(radixsort2_file_list_[c_i + 32]));
        EXPECT_EQ(9*100*100,                   std::get<0>(radixsort2_file_list_[c_i + 36]));
        EXPECT_EQ(100*100,                     std::get<1>(radixsort2_file_list_[c_i + 36]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 36]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 36]));
        EXPECT_EQ(prefix + "-009-009.bin",     std::get<4>(radixsort2_file_list_[c_i + 36]));

        EXPECT_EQ("idx_" + prefix + "-000-000.bin", mergesort2_file_list_[0 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-001-001.bin", mergesort2_file_list_[0 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-000-001.bin", mergesort2_file_list_[0 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-002-002.bin", mergesort2_file_list_[1 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-003-003.bin", mergesort2_file_list_[1 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-002-003.bin", mergesort2_file_list_[1 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-004-004.bin", mergesort2_file_list_[2 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-005-005.bin", mergesort2_file_list_[2 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-004-005.bin", mergesort2_file_list_[2 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-006-006.bin", mergesort2_file_list_[3 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-007-007.bin", mergesort2_file_list_[3 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-006-007.bin", mergesort2_file_list_[3 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-008-008.bin", mergesort2_file_list_[4 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-009-009.bin", mergesort2_file_list_[4 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-008-009.bin", mergesort2_file_list_[4 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-000-001.bin", mergesort2_file_list_[5 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-002-003.bin", mergesort2_file_list_[5 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-000-003.bin", mergesort2_file_list_[5 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-004-005.bin", mergesort2_file_list_[6 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-006-007.bin", mergesort2_file_list_[6 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-004-007.bin", mergesort2_file_list_[6 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-004-007.bin", mergesort2_file_list_[7 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-008-009.bin", mergesort2_file_list_[7 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-004-009.bin", mergesort2_file_list_[7 + c_i * 9][2]);
        EXPECT_EQ("idx_" + prefix + "-000-003.bin", mergesort2_file_list_[8 + c_i * 9][0]);
        EXPECT_EQ("idx_" + prefix + "-004-009.bin", mergesort2_file_list_[8 + c_i * 9][1]);
        EXPECT_EQ("idx_" + prefix + "-000-009.bin", mergesort2_file_list_[8 + c_i * 9][2]);
        EXPECT_EQ(prefix + "-000-000.bin", mergesort2_file_list_[0 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-001-001.bin", mergesort2_file_list_[0 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-000-001.bin", mergesort2_file_list_[0 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-002-002.bin", mergesort2_file_list_[1 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-003-003.bin", mergesort2_file_list_[1 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-002-003.bin", mergesort2_file_list_[1 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-004-004.bin", mergesort2_file_list_[2 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-005-005.bin", mergesort2_file_list_[2 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-004-005.bin", mergesort2_file_list_[2 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-006-006.bin", mergesort2_file_list_[3 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-007-007.bin", mergesort2_file_list_[3 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-006-007.bin", mergesort2_file_list_[3 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-008-008.bin", mergesort2_file_list_[4 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-009-009.bin", mergesort2_file_list_[4 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-008-009.bin", mergesort2_file_list_[4 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-000-001.bin", mergesort2_file_list_[5 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-002-003.bin", mergesort2_file_list_[5 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-000-003.bin", mergesort2_file_list_[5 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-004-005.bin", mergesort2_file_list_[6 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-006-007.bin", mergesort2_file_list_[6 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-004-007.bin", mergesort2_file_list_[6 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-004-007.bin", mergesort2_file_list_[7 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-008-009.bin", mergesort2_file_list_[7 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-004-009.bin", mergesort2_file_list_[7 + c_i * 9][5]);
        EXPECT_EQ(prefix + "-000-003.bin", mergesort2_file_list_[8 + c_i * 9][3]);
        EXPECT_EQ(prefix + "-004-009.bin", mergesort2_file_list_[8 + c_i * 9][4]);
        EXPECT_EQ(prefix + "-000-009.bin", mergesort2_file_list_[8 + c_i * 9][5]);
    }
}

TEST_F(QuantileNormCudaTest, SaveAndLoadFileTest) {
    std::string out_file = "data.bin";

    std::shared_ptr<std::vector<unsigned int>> data(new std::vector<unsigned int>());
    for(size_t i = 0; i < 100; i++) {
        (*data).push_back(i);
    }

    // run
    savefile(".", out_file, data);

    // check
    utils::FileBufferReader<unsigned int> fb_reader(out_file, READ_BUFFER_SIZE);
    fb_reader.open();
    fb_reader.readFileToBuffer();

    for (size_t i = 0; i < 100; i++) {
        unsigned int val = fb_reader.get();
        ASSERT_EQ(i, val);
        fb_reader.next();
    }
    ASSERT_TRUE(fb_reader.finishedReadingAll());
    fb_reader.close();

    // run
    std::shared_ptr<std::vector<unsigned int>> data2 = loadfile<unsigned int>(out_file, 0, 100);

    // check
    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ((*data)[i], (*data2)[i]);
    }

    // clean up
    remove(out_file.c_str());
}

TEST_F(QuantileNormCudaTest, RadixSort1Test) {
    std::shared_ptr<std::vector<uint16_t>> image(new std::vector<uint16_t>());
    mexutils::loadtiff(tif_fnames_[0], 0, IMAGE_Z - 1, image);
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*image).size());

    // run
    std::string out_file = "out.bin";
    radixSort1FromData(image, 0, out_file);

    // check
    utils::FileBufferReader<uint16_t> fb_reader(out_file, READ_BUFFER_SIZE);
    fb_reader.open();
    ASSERT_TRUE(fb_reader.isOpen());
    fb_reader.readFileToBuffer();
    logger_->info("5\n");

    uint16_t cur_val = fb_reader.get();
    fb_reader.next();
    while (! fb_reader.finishedReadingAll()) {
        uint16_t nxt_val = fb_reader.get();
        fb_reader.next();
        ASSERT_LE(cur_val, nxt_val);
    }
    fb_reader.close();
    logger_->info("6\n");

    std::string idx_out_file = "idx_out.bin";
    ASSERT_TRUE(oneFileExists(datadir_ + "/" + idx_out_file));

    utils::FileBufferReader<uint16_t> idx_fb_reader(idx_out_file, READ_BUFFER_SIZE);
    idx_fb_reader.open();
    ASSERT_TRUE(idx_fb_reader.isOpen());
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z * sizeof(unsigned int), idx_fb_reader.getFileSize());
    idx_fb_reader.close();

    // clean up
    remove(out_file.c_str());
    remove(idx_out_file.c_str());
}

TEST_F(QuantileNormCudaTest, MergeSort1SequentiallyTest) {
    size_t data_size = 100;
    for (size_t i = 0; i < 5; i++) {
        std::string out_file     = "data_"     + std::to_string(i) + ".bin";
        std::string idx_out_file = "data_idx_" + std::to_string(i) + ".bin";
        utils::FileBufferWriter<uint16_t>     fb_writer(out_file, data_size * sizeof(uint16_t));
        utils::FileBufferWriter<unsigned int> idx_fb_writer(idx_out_file, data_size * sizeof(unsigned int));
        fb_writer.open();
        idx_fb_writer.open();

        for (size_t j = 0; j < data_size; j++) {
            fb_writer.set(5 * i + j);
            unsigned int idx = 10000 + 5 * i + j;
            idx_fb_writer.set(idx);
        }
        fb_writer.writeFileFromBuffer();
        idx_fb_writer.writeFileFromBuffer();
        fb_writer.close();
        idx_fb_writer.close();
    }

    mergesort1_file_list_.push_back({"data_0.bin","data_1.bin","data_01.bin","data_idx_0.bin","data_idx_1.bin","data_idx_01.bin"});
    mergesort1_file_list_.push_back({"data_2.bin","data_3.bin","data_23.bin","data_idx_2.bin","data_idx_3.bin","data_idx_23.bin"});
    mergesort1_file_list_.push_back({"data_23.bin","data_4.bin","data_24.bin","data_idx_23.bin","data_idx_4.bin","data_idx_24.bin"});
    mergesort1_file_list_.push_back({"data_01.bin","data_24.bin","data_04.bin","data_idx_01.bin","data_idx_24.bin","data_idx_04.bin"});

    // run
    mergeSortTwoFiles<uint16_t, unsigned int>(0, mergesort1_file_list_);
    mergeSortTwoFiles<uint16_t, unsigned int>(1, mergesort1_file_list_);
    mergeSortTwoFiles<uint16_t, unsigned int>(2, mergesort1_file_list_);
    mergeSortTwoFiles<uint16_t, unsigned int>(3, mergesort1_file_list_);

    // check
    utils::FileBufferReader<uint16_t> fb_reader("data_04.bin", data_size * sizeof(uint16_t));
    utils::FileBufferReader<unsigned int> idx_fb_reader("data_idx_04.bin", data_size * sizeof(unsigned int));
    fb_reader.open();
    idx_fb_reader.open();

    fb_reader.readFileToBuffer();
    idx_fb_reader.readFileToBuffer();
    uint16_t     cur_val = fb_reader.get();
    unsigned int cur_idx = idx_fb_reader.get();
    fb_reader.next();
    idx_fb_reader.next();

    size_t count= 0;
    while(! fb_reader.finishedReadingAll()) {
        uint16_t     nxt_val = fb_reader.get();
        unsigned int nxt_idx = idx_fb_reader.get();
        ASSERT_LE(cur_val, nxt_val);
        ASSERT_LE(cur_idx, nxt_idx);
        ASSERT_EQ(10000 + 5 * cur_val, cur_idx);

        fb_reader.next();
        idx_fb_reader.next();
        count++;
    }
    fb_reader.close();
    idx_fb_reader.close();

    ASSERT_EQ(data_size * 5 - 1, count);

    // clean up
    remove("data_04.bin");
    remove("data_idx_04.bin");
}

TEST_F(QuantileNormCudaTest, RadixSort1andMergeSortInParallelTest) {
    setupFileList();

    // run
    mergeSort1();
    radixSort1();
    waitForTasks("radixsort1", radixsort1_futures_);
    waitForTasks("mergesort1", mergesort1_futures_);

    // check
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        utils::FileBufferReader<uint16_t> fb_reader(sorted_file1_list_[c_i], READ_BUFFER_SIZE);
        fb_reader.open();
        fb_reader.readFileToBuffer();

        uint16_t cur_val = fb_reader.get();
        fb_reader.next();
        while (! fb_reader.finishedReadingAll()) {
            uint16_t nxt_val = fb_reader.get();
            fb_reader.next();
            ASSERT_LE(cur_val, nxt_val);
        }
        fb_reader.close();
    }

    // clean up
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        remove(sorted_file1_list_[c_i].c_str());
        remove(("idx_" + sorted_file1_list_[c_i]).c_str());
    }
}

TEST_F(QuantileNormCudaTest, SumSortedFilesTest) {
    setupFileList();

    for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
        utils::FileBufferWriter<uint16_t> fb_writer(sorted_file1_list_[i], WRITE_BUFFER_SIZE);
        fb_writer.open();
        for (size_t j = 0; j < 10; j++) {
            fb_writer.set(10 * i + j);
        }
        fb_writer.writeFileFromBuffer();
        fb_writer.close();
    }

    // run
    sumSortedFiles();

    // check
    utils::FileBufferReader<unsigned int> fb_reader(summed_file_, READ_BUFFER_SIZE);
    fb_reader.open();
    fb_reader.readFileToBuffer();
    for (size_t i = 0; i < 10; i++) {
        unsigned int val = fb_reader.get();
        ASSERT_EQ(60 + i * 4, val);
        fb_reader.next();
    }
    fb_reader.close();

    // clean up
    for (size_t i = 0; i < sorted_file1_list_.size(); i++) {
        remove(sorted_file1_list_[i].c_str());
    }
    remove(summed_file_.c_str());
}

TEST_F(QuantileNormCudaTest, SubstituteToNormValuesTest) {
    setupFileList();

    utils::FileBufferWriter<unsigned int> fb_writer(summed_file_, WRITE_BUFFER_SIZE);
    fb_writer.open();
    for (size_t i = 0; i < 10; i++) {
        fb_writer.set(i);
    }
    fb_writer.writeFileFromBuffer();
    fb_writer.close();

    utils::FileBufferWriter<uint16_t> fb_writer2(sorted_file1_list_[0], WRITE_BUFFER_SIZE);
    fb_writer2.open();
    fb_writer2.set(1);
    fb_writer2.set(1);
    fb_writer2.set(1);
    fb_writer2.set(2);
    fb_writer2.set(3);
    fb_writer2.set(4);
    fb_writer2.set(6);
    fb_writer2.set(6);
    fb_writer2.set(8);
    fb_writer2.set(9);
    fb_writer2.writeFileFromBuffer();
    fb_writer2.close(); 

    // run
    substituteToNormValues(0);

    // check
    utils::FileBufferReader<double> fb_reader(substituted_file_list_[0], READ_BUFFER_SIZE);
    fb_reader.open();
    fb_reader.readFileToBuffer();
    EXPECT_EQ(1.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(1.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(1.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(3.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(4.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(5.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(6.5/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(6.5/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(8.0/4.0, fb_reader.get()); fb_reader.next();
    EXPECT_EQ(9.0/4.0, fb_reader.get()); fb_reader.next();
    fb_reader.close();

    // clean up
    remove(summed_file_.c_str());
    remove(sorted_file1_list_[0].c_str());
    remove(substituted_file_list_[0].c_str());
}

TEST_F(QuantileNormCudaTest, RadixSort2Test) {
    setupFileList();

    std::get<0>(radixsort2_file_list_[0]) = 0;
    std::get<1>(radixsort2_file_list_[0]) = 10;

    std::string in_subst_file = std::get<2>(radixsort2_file_list_[0]);
    std::string in_idx_file   = std::get<3>(radixsort2_file_list_[0]);
    utils::FileBufferWriter<unsigned int> fb_writer1(in_idx_file, WRITE_BUFFER_SIZE);
    utils::FileBufferWriter<double>       fb_writer2(in_subst_file, WRITE_BUFFER_SIZE);
    fb_writer1.open();
    fb_writer2.open();
    for (size_t i = 0; i < 10; i++) {
        fb_writer1.set(10 - 1 - i);
        fb_writer2.set((double)i);
    }
    fb_writer1.writeFileFromBuffer();
    fb_writer2.writeFileFromBuffer();
    fb_writer1.close();
    fb_writer2.close();

    // run
    radixSort2FromData(0);

    // check
    std::string out_file     = std::get<4>(radixsort2_file_list_[0]);
    std::string idx_out_file = "idx_" + out_file;
    utils::FileBufferReader<unsigned int> fb_reader1(idx_out_file, READ_BUFFER_SIZE);
    utils::FileBufferReader<double>       fb_reader2(out_file, READ_BUFFER_SIZE);
    fb_reader1.open();
    fb_reader2.open();
    ASSERT_TRUE(fb_reader1.isOpen());
    ASSERT_TRUE(fb_reader2.isOpen());
    fb_reader1.readFileToBuffer();
    fb_reader2.readFileToBuffer();

    for (size_t i = 0; i < 10; i++) {
        unsigned int val1 = fb_reader1.get();
        double       val2 = fb_reader2.get();
        ASSERT_EQ(i, val1);
        ASSERT_EQ((double)(10 - 1 - i), val2);
        fb_reader1.next();
        fb_reader2.next();
    }
    fb_reader1.close();
    fb_reader2.close();

    // clean up
    remove(in_subst_file.c_str());
    remove(in_idx_file.c_str());
    remove(out_file.c_str());
    remove(idx_out_file.c_str());
}

TEST_F(QuantileNormCudaTest, AllRunTest) {
    system("cp -a img_0.tif img_1.tif");
    system("cp -a img_0.tif img_2.tif");
    system("cp -a img_0.tif img_3.tif");

    // run
    run();

    // check
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        utils::FileBufferReader<double> fb_reader1(sorted_file2_list_[c_i], READ_BUFFER_SIZE);
        fb_reader1.open();
        fb_reader1.readFileToBuffer();

        std::string in_file = "img_" + std::to_string(c_i) + ".tif";
        utils::FileBufferReader<uint16_t> fb_reader2(in_file, READ_BUFFER_SIZE);
        fb_reader2.open();
        fb_reader2.readFileToBuffer();

        for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
            double   val1 = fb_reader1.get();
            uint16_t val2 = fb_reader2.get();
            ASSERT_EQ(val1, (double)val2);
            fb_reader1.next();
            fb_reader2.next();
        }
        fb_reader1.close();
        fb_reader2.close();
    }

    // clean up
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        remove(sorted_file1_list_[c_i].c_str());
        remove(("idx_" + sorted_file1_list_[c_i]).c_str());
        remove(summed_file_.c_str());
        remove(substituted_file_list_[c_i].c_str());
        remove(sorted_file2_list_[c_i].c_str());
        remove(("idx_" + sorted_file2_list_[c_i]).c_str());
    }
}

}

