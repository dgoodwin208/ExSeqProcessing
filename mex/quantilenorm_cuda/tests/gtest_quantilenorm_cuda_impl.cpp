#include "gtest/gtest.h"

#include "quantilenorm_impl.h"
#include "filebuffers.h"
#include "spdlog/spdlog.h"

#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <iostream>
#include <cstdlib>
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

void loadhdf5(const std::string& filename, const size_t slice_start, const size_t slice_end, const size_t image_height, const size_t image_width, std::shared_ptr<std::vector<uint16_t>> image)
{
    size_t idx_start = image_width * image_height * slice_start;
    size_t idx_end   = image_width * image_height * (slice_end + 1) - 1;

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
        : QuantileNormImpl(".", "test", {}, IMAGE_X, IMAGE_Y, IMAGE_Z, GPU_NUM) {
        std::shared_ptr<spdlog::logger> logger;
        logger_ = spdlog::get("mex_logger");
        if (logger_ == nullptr) {
            logger_ = spdlog::basic_logger_mt("mex_logger", "mex.log");
        }

        // overwritten for TEST
        gpu_mem_total_ = 700000; // = 680K

        std::string user_name = std::getenv("USER");

        mode_t old_umask = umask(0);
        for (size_t i = 0; i < num_gpus_; i++) {
            std::string sem_name = "/" + user_name + ".g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
            sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, 1);
        }
        umask(old_umask);

        num_channels_ = 4;
        std::default_random_engine generator(1);
        std::normal_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < 4; i++) {
            std::string fname = "img_" + std::to_string(i) + ".tif";
            in_fnames_.push_back(fname);

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

        system("cp -a img_0.tif img_0.h5");
        system("cp -a img_1.tif img_1.h5");
        system("cp -a img_2.tif img_2.h5");
        system("cp -a img_3.tif img_3.h5");
    }
    virtual ~QuantileNormCudaTest() {
        for (int i = 0; i < 4; i++) {
            remove(in_fnames_[i].c_str());
        }
        remove("img_0.h5");
        remove("img_1.h5");
        remove("img_2.h5");
        remove("img_3.h5");

        std::string user_name = std::getenv("USER");
        for (size_t i = 0; i < num_gpus_; i++) {
            std::string sem_name = "/" + user_name + ".g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
        }
    }

    std::shared_ptr<std::vector<std::vector<float>>> quantilenormFromTifFiles(std::vector<std::string>& file_list) {
        size_t num_files = file_list.size();

        // load image data
        std::shared_ptr<std::vector<uint16_t>> img_data[num_files];

        for (size_t i = 0; i < num_files; i++) {
            img_data[i] = std::make_shared<std::vector<uint16_t>>();
            mexutils::loadtiff(file_list[i], 0, IMAGE_Z, img_data[i]);
        }

        return quantilenorm(img_data, num_files);
    }

    std::shared_ptr<std::vector<std::vector<float>>> quantilenorm(std::shared_ptr<std::vector<uint16_t>> img_data[], size_t num_img_data) {
        // sort each data (1)
        std::multimap<uint16_t, unsigned int> sort1[num_img_data]; // (val, idx)

        for (size_t i = 0; i < num_img_data; i++) {
            for (size_t idx = 0; idx < (*img_data[i]).size(); idx++) {
                sort1[i].insert(std::make_pair((*img_data[i])[idx], idx));
            }

//            for (auto itr = sort1[i].begin(); itr != sort1[i].end(); itr++) {
//                std::cout << "i=" << i << "; (" << itr->first << ", " << itr->second << ")" << std::endl;
//            }
        }

        size_t data_size = (*img_data[0]).size();

        // mean of all sorted data
        std::vector<float> mean_data(data_size);
        std::multimap<uint16_t, unsigned int>::iterator sort1_itr[num_img_data];
        for (size_t i = 0; i < num_img_data; i++) {
            sort1_itr[i] = sort1[i].begin();
        }

        for (size_t d_i = 0; d_i < data_size; d_i++) {
            mean_data[d_i] = 0.0;
            for (size_t i = 0; i < num_img_data; i++) {
                mean_data[d_i] += float(sort1_itr[i]->first); // img_val
                sort1_itr[i]++;
            }
            mean_data[d_i] = mean_data[d_i] / num_img_data;

//            std::cout << "mean[" << d_i << "] " << mean_data[d_i] << std::endl;
        }

        // substitute mean values to each data, and sort (2)
        std::shared_ptr<std::vector<std::vector<float>>> sort2 = std::make_shared<std::vector<std::vector<float>>>(num_img_data);

        for (size_t i = 0; i < num_img_data; i++) {
            (*sort2)[i].resize(data_size);

            auto sort1_cur_itr = sort1[i].begin();
            auto sort1_nxt_itr = sort1_cur_itr;
            sort1_nxt_itr++;
            auto sort1_mark_itr = sort1[i].end();
            size_t d_mark_i = 0;
            size_t same_val_count = 0;

            for (size_t d_i = 0; d_i < data_size; d_i++) {
//                std::cout << "i=" << i << " d_i[" << d_i << "] (" << sort1_cur_itr->first << ", " << sort1_cur_itr->second << ") <-> ("
//                          << sort1_nxt_itr->first << ", " << sort1_nxt_itr->second << ")" << std::endl;
                if (d_i == data_size - 1 || sort1_cur_itr->first != sort1_nxt_itr->first) { // img_val is different or the last
//                    std::cout << "    diff." << std::endl;
                    if (sort1_mark_itr == sort1[i].end()) {
//                        std::cout << "        no marking. mean_val=" << mean_data[d_i] << std::endl;
                        (*sort2)[i][sort1_cur_itr->second] = mean_data[d_i];
                    } else {
//                        std::cout << "        marked." << std::endl;
                        same_val_count++;

                        size_t mid_d_i = d_mark_i + same_val_count / 2;
                        float mean_val = 0.0;
                        if (same_val_count % 2 == 0) {
                            mean_val = (mean_data[mid_d_i] + mean_data[mid_d_i - 1]) * 0.5;
                        } else {
                            mean_val = mean_data[mid_d_i];
                        }

//                        std::cout << "set mean_val=" << mean_val << std::endl;
                        for (size_t j = 0; j < same_val_count; j++) {
                            (*sort2)[i][sort1_mark_itr->second] = mean_val;
                            sort1_mark_itr++;
                        }

                        sort1_mark_itr = sort1[i].end();
                        d_mark_i = 0;
                        same_val_count = 0;
                    }
                } else {
//                    std::cout << "    same." << std::endl;
                    if (sort1_mark_itr == sort1[i].end()) {
//                        std::cout << "    1st -> marking!" << std::endl;
                        sort1_mark_itr = sort1_cur_itr;
                        d_mark_i = d_i;
                    }
                    same_val_count++;
                }
                sort1_cur_itr++;
                sort1_nxt_itr++;
            }
        }

        return sort2;
    }
};


TEST_F(QuantileNormCudaTest, HelperQuantileNormTest) {
    std::shared_ptr<std::vector<uint16_t>> img_data[4];
    for (size_t i = 0; i < 4; i++) {
        img_data[i] = std::make_shared<std::vector<uint16_t>>();
    }

    //     [5  4  3  6]                     [5.75  5.25  1.75  5.75]
    // x = [2  1  4  3],  quantilenorm(x) = [1.75  1.75  3.00  3.00]
    //     [3  4  6  1]                     [3.00  5.25  4.75  1.75]
    //     [4  2  8  5]                     [4.75  3.00  5.75  4.75]

    (*img_data[0]).push_back(5);
    (*img_data[0]).push_back(2);
    (*img_data[0]).push_back(3);
    (*img_data[0]).push_back(4);

    (*img_data[1]).push_back(4);
    (*img_data[1]).push_back(1);
    (*img_data[1]).push_back(4);
    (*img_data[1]).push_back(2);

    (*img_data[2]).push_back(3);
    (*img_data[2]).push_back(4);
    (*img_data[2]).push_back(6);
    (*img_data[2]).push_back(8);

    (*img_data[3]).push_back(6);
    (*img_data[3]).push_back(3);
    (*img_data[3]).push_back(1);
    (*img_data[3]).push_back(5);

    // run
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenorm(img_data, 4);

    // check
    ASSERT_EQ(5.75, (*result)[0][0]);
    ASSERT_EQ(1.75, (*result)[0][1]);
    ASSERT_EQ(3.00, (*result)[0][2]);
    ASSERT_EQ(4.75, (*result)[0][3]);

    ASSERT_EQ(5.25, (*result)[1][0]);
    ASSERT_EQ(1.75, (*result)[1][1]);
    ASSERT_EQ(5.25, (*result)[1][2]);
    ASSERT_EQ(3.00, (*result)[1][3]);

    ASSERT_EQ(1.75, (*result)[2][0]);
    ASSERT_EQ(3.00, (*result)[2][1]);
    ASSERT_EQ(4.75, (*result)[2][2]);
    ASSERT_EQ(5.75, (*result)[2][3]);

    ASSERT_EQ(5.75, (*result)[3][0]);
    ASSERT_EQ(3.00, (*result)[3][1]);
    ASSERT_EQ(1.75, (*result)[3][2]);
    ASSERT_EQ(4.75, (*result)[3][3]);
}

TEST_F(QuantileNormCudaTest, HelperQuantileNormFromFilesTest) {
    std::vector<std::string> tif_list(4);
    tif_list[0] = "img_0.tif";
    tif_list[1] = "img_0.tif";
    tif_list[2] = "img_0.tif";
    tif_list[3] = "img_0.tif";

    // run
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenormFromTifFiles(tif_list);

    // check
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*result)[c_i].size());
    }

    utils::FileBufferReader<uint16_t> fb_reader1("img_0.tif", READ_BUFFER_SIZE);
    fb_reader1.open();
    fb_reader1.readFileToBuffer();

    for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
        float val1 = (float)fb_reader1.get();
        for (size_t c_i = 0; c_i < num_channels_; c_i++) {
            float val2 = (*result)[c_i][i];
            ASSERT_EQ(val1, val2);
        }
        fb_reader1.next();
    }
    fb_reader1.close();
}


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

TEST_F(QuantileNormCudaTest, AllDataExistTest) {
    ASSERT_TRUE(allDataExist(in_fnames_));

    std::vector<std::string> file_list(in_fnames_.begin(), in_fnames_.end());
    file_list.push_back("dummy_img_0.tif");
    ASSERT_FALSE(allDataExist(file_list));
}

TEST_F(QuantileNormCudaTest, OneDataExistsTest) {
    std::string filepath = datadir_ + "/" + "img_0.tif";
    ASSERT_TRUE(oneDataExists(filepath));

    filepath = datadir_ + "/" + "dummy_img_0.tif";
    ASSERT_FALSE(oneDataExists(filepath));
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
        EXPECT_EQ(in_fnames_[c_i],         std::get<2>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(prefix + "-000-003.bin", std::get<3>(radixsort1_file_list_[c_i + 0]));
        EXPECT_EQ(4,                       std::get<0>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(7,                       std::get<1>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(in_fnames_[c_i],         std::get<2>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(prefix + "-004-007.bin", std::get<3>(radixsort1_file_list_[c_i + 4]));
        EXPECT_EQ(8,                       std::get<0>(radixsort1_file_list_[c_i + 8]));
        EXPECT_EQ(9,                       std::get<1>(radixsort1_file_list_[c_i + 8]));
        EXPECT_EQ(in_fnames_[c_i],         std::get<2>(radixsort1_file_list_[c_i + 8]));
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

    size_t unit_data_size  = gpu_mem_total_ * GPU_USER_MEMORY_USAGE_RATIO
        / ((sizeof(float) + sizeof(unsigned int)) * THRUST_RADIXSORT_MEMORY_USAGE_RATIO);
    size_t remain_size = IMAGE_X * IMAGE_Y * IMAGE_Z % (3 * unit_data_size);

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        std::string prefix = basename_ + "_4_sort2_c" + std::to_string(c_i);
        std::string sorted_filename = prefix + "-000-003.bin";
        std::string idx_filename = "idx_" + basename_ + "_1_sort1_c" + std::to_string(c_i) + "-000-009.bin";
        EXPECT_EQ(sorted_filename, sorted_file2_list_[c_i]);

        EXPECT_EQ(0,                           std::get<0>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(unit_data_size,              std::get<1>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(prefix + "-000-000.bin",     std::get<4>(radixsort2_file_list_[c_i +  0]));
        EXPECT_EQ(unit_data_size,              std::get<0>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(unit_data_size,              std::get<1>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(prefix + "-001-001.bin",     std::get<4>(radixsort2_file_list_[c_i +  4]));
        EXPECT_EQ(unit_data_size*2,            std::get<0>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(unit_data_size,              std::get<1>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(prefix + "-002-002.bin",     std::get<4>(radixsort2_file_list_[c_i +  8]));
        EXPECT_EQ(unit_data_size*3,            std::get<0>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(remain_size,                 std::get<1>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(substituted_file_list_[c_i], std::get<2>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(idx_filename,                std::get<3>(radixsort2_file_list_[c_i + 12]));
        EXPECT_EQ(prefix + "-003-003.bin",     std::get<4>(radixsort2_file_list_[c_i + 12]));

        EXPECT_EQ("idx_" + prefix + "-000-000.bin", mergesort2_file_list_[0 + c_i * 3][0]);
        EXPECT_EQ("idx_" + prefix + "-001-001.bin", mergesort2_file_list_[0 + c_i * 3][1]);
        EXPECT_EQ("idx_" + prefix + "-000-001.bin", mergesort2_file_list_[0 + c_i * 3][2]);
        EXPECT_EQ("idx_" + prefix + "-002-002.bin", mergesort2_file_list_[1 + c_i * 3][0]);
        EXPECT_EQ("idx_" + prefix + "-003-003.bin", mergesort2_file_list_[1 + c_i * 3][1]);
        EXPECT_EQ("idx_" + prefix + "-002-003.bin", mergesort2_file_list_[1 + c_i * 3][2]);
        EXPECT_EQ("idx_" + prefix + "-000-001.bin", mergesort2_file_list_[2 + c_i * 3][0]);
        EXPECT_EQ("idx_" + prefix + "-002-003.bin", mergesort2_file_list_[2 + c_i * 3][1]);
        EXPECT_EQ("idx_" + prefix + "-000-003.bin", mergesort2_file_list_[2 + c_i * 3][2]);
        EXPECT_EQ(prefix + "-000-000.bin", mergesort2_file_list_[0 + c_i * 3][3]);
        EXPECT_EQ(prefix + "-001-001.bin", mergesort2_file_list_[0 + c_i * 3][4]);
        EXPECT_EQ(prefix + "-000-001.bin", mergesort2_file_list_[0 + c_i * 3][5]);
        EXPECT_EQ(prefix + "-002-002.bin", mergesort2_file_list_[1 + c_i * 3][3]);
        EXPECT_EQ(prefix + "-003-003.bin", mergesort2_file_list_[1 + c_i * 3][4]);
        EXPECT_EQ(prefix + "-002-003.bin", mergesort2_file_list_[1 + c_i * 3][5]);
        EXPECT_EQ(prefix + "-000-001.bin", mergesort2_file_list_[2 + c_i * 3][3]);
        EXPECT_EQ(prefix + "-002-003.bin", mergesort2_file_list_[2 + c_i * 3][4]);
        EXPECT_EQ(prefix + "-000-003.bin", mergesort2_file_list_[2 + c_i * 3][5]);
    }
}

TEST_F(QuantileNormCudaTest, SaveAndLoadFileTest) {
    std::string out_file = "data.bin";

    std::shared_ptr<std::vector<unsigned int>> data(new std::vector<unsigned int>());
    for(size_t i = 0; i < 100; i++) {
        (*data).push_back(i);
    }

    // run
    saveDataToFile(".", out_file, data);

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
    std::shared_ptr<std::vector<unsigned int>> data2 = loadDataFromFile<unsigned int>(out_file, 0, 100);

    // check
    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ((*data)[i], (*data2)[i]);
    }

    // clean up
    remove(out_file.c_str());
}

TEST_F(QuantileNormCudaTest, SaveAndLoadBufferTest) {
    std::string out_file = "data.bin";

    std::shared_ptr<std::vector<unsigned int>> data = std::make_shared<std::vector<unsigned int>>();
    for(size_t i = 0; i < 100; i++) {
        (*data).push_back(i);
    }

    // run
    saveDataToBuffer(".", out_file, data);

    // check
    auto find_itr = tmp_data_buffers_.find("./" + out_file);
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<unsigned int>> ptr = find_itr->second;
    ASSERT_EQ(100, (*ptr).size());

    for (size_t i = 0; i < 100; i++) {
        unsigned int val = (*ptr)[i];
        ASSERT_EQ(i, val);
    }

    // run
    std::shared_ptr<std::vector<unsigned int>> data2 = loadDataFromBuffer<unsigned int>("./" + out_file, 0, 100);

    // check
    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ((*data)[i], (*data2)[i]);
    }

    // clean up
    tmp_data_buffers_.clear();
}

TEST_F(QuantileNormCudaTest, RadixSort1Test) {
    std::shared_ptr<RadixSort1Info> info = std::make_shared<RadixSort1Info>();

    info->slice_start = 0;
    info->image = std::make_shared<std::vector<uint16_t>>();
    info->out_filename = std::string("out.bin");

    mexutils::loadtiff(in_fnames_[0], 0, IMAGE_Z - 1, info->image);// MOCK
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*info->image).size());

    radixsort1_queue_.push(info);
    radixsort1_queue_.close();

    // run
    radixSort1FromData();

    // check
    utils::FileBufferReader<uint16_t> fb_reader(info->out_filename, READ_BUFFER_SIZE);
    fb_reader.open();
    ASSERT_TRUE(fb_reader.isOpen());
    fb_reader.readFileToBuffer();

    uint16_t cur_val = fb_reader.get();
    fb_reader.next();
    while (! fb_reader.finishedReadingAll()) {
        uint16_t nxt_val = fb_reader.get();
        fb_reader.next();
        ASSERT_LE(cur_val, nxt_val);
    }
    fb_reader.close();

    std::string idx_out_file = "idx_out.bin";
    ASSERT_TRUE(oneDataExists(datadir_ + "/" + idx_out_file));

    utils::FileBufferReader<uint16_t> idx_fb_reader(idx_out_file, READ_BUFFER_SIZE);
    idx_fb_reader.open();
    ASSERT_TRUE(idx_fb_reader.isOpen());
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z * sizeof(unsigned int), idx_fb_reader.getFileSize());
    idx_fb_reader.close();

    // clean up
    remove(info->out_filename.c_str());
    remove(idx_out_file.c_str());
}

TEST_F(QuantileNormCudaTest, RadixSort1TmpOnMemoryTest) {
    use_tmp_files_ = false;

    std::shared_ptr<RadixSort1Info> info = std::make_shared<RadixSort1Info>();

    info->slice_start = 0;
    info->image = std::make_shared<std::vector<uint16_t>>();
    info->out_filename = std::string("out.bin");

    mexutils::loadtiff(in_fnames_[0], 0, IMAGE_Z - 1, info->image);// MOCK
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*info->image).size());

    radixsort1_queue_.push(info);
    radixsort1_queue_.close();

    // run
    radixSort1FromData();

    // check
    auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + info->out_filename);
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<uint16_t>> ptr = find_itr->second;
    ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*ptr).size());

    uint16_t cur_val = (*ptr)[0];
    for (size_t i = 1; i < (*ptr).size(); i++) {
        uint16_t nxt_val = (*ptr)[i];
        ASSERT_LE(cur_val, nxt_val);
        cur_val = nxt_val;
    }

    std::string idx_out_file = "idx_out.bin";
    auto idx_find_itr = tmp_data_buffers_.find(datadir_ + "/" + idx_out_file);
    ASSERT_TRUE(idx_find_itr != tmp_data_buffers_.end());

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
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
            fb_writer.set(data_size * i + j);
            unsigned int idx = 10000 + data_size * i + j;
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
        ASSERT_EQ(10000 + cur_val, cur_idx);

        fb_reader.next();
        idx_fb_reader.next();
        cur_val = nxt_val;
        cur_idx = nxt_idx;
        count++;
    }
    fb_reader.close();
    idx_fb_reader.close();

    ASSERT_EQ(data_size * 5 - 1, count);

    // clean up
    remove("data_04.bin");
    remove("data_idx_04.bin");
}

TEST_F(QuantileNormCudaTest, MergeSort1SequentiallyTmpOnMemoryTest) {
    use_tmp_files_ = false;

    size_t data_size = 100;
    for (size_t i = 0; i < 5; i++) {
        std::string out_file     = "data_"     + std::to_string(i) + ".bin";
        std::string idx_out_file = "data_idx_" + std::to_string(i) + ".bin";

        std::shared_ptr<std::vector<uint16_t>>     data = std::make_shared<std::vector<uint16_t>>();
        std::shared_ptr<std::vector<unsigned int>> idx  = std::make_shared<std::vector<unsigned int>>();
        for (size_t j = 0; j < data_size; j++) {
            (*data).push_back(data_size * i + j);
            (*idx) .push_back(10000 + data_size * i + j);
        }

        saveDataToBuffer(datadir_, out_file, data);
        saveDataToBuffer(datadir_, idx_out_file, idx);
    }

    mergesort1_file_list_.push_back({"data_0.bin","data_1.bin","data_01.bin","data_idx_0.bin","data_idx_1.bin","data_idx_01.bin"});
    mergesort1_file_list_.push_back({"data_2.bin","data_3.bin","data_23.bin","data_idx_2.bin","data_idx_3.bin","data_idx_23.bin"});
    mergesort1_file_list_.push_back({"data_23.bin","data_4.bin","data_24.bin","data_idx_23.bin","data_idx_4.bin","data_idx_24.bin"});
    mergesort1_file_list_.push_back({"data_01.bin","data_24.bin","data_04.bin","data_idx_01.bin","data_idx_24.bin","data_idx_04.bin"});

    // run
    mergeSortTwoBuffers<uint16_t, unsigned int>(0, mergesort1_file_list_);
    mergeSortTwoBuffers<uint16_t, unsigned int>(1, mergesort1_file_list_);
    mergeSortTwoBuffers<uint16_t, unsigned int>(2, mergesort1_file_list_);
    mergeSortTwoBuffers<uint16_t, unsigned int>(3, mergesort1_file_list_);

    // check
    auto find_itr = tmp_data_buffers_.find(datadir_ + "/data_04.bin");
    auto idx_find_itr = tmp_data_buffers_.find(datadir_ + "/data_idx_04.bin");
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());
    ASSERT_TRUE(idx_find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<uint16_t>> merged_data = find_itr->second;
    std::shared_ptr<std::vector<unsigned int>> merged_idx = idx_find_itr->second;
    ASSERT_EQ(data_size * 5, (*merged_data).size());
    ASSERT_EQ(data_size * 5, (*merged_idx).size());

    uint16_t     cur_val = (*merged_data)[0];
    unsigned int cur_idx = (*merged_idx)[0];

    for (size_t i = 1; i < data_size * 5; i++) {
        uint16_t nxt_val = (*merged_data)[i];
        unsigned int nxt_idx = (*merged_idx)[i];
        ASSERT_LE(cur_val, nxt_val);
        ASSERT_LE(cur_idx, nxt_idx);
        ASSERT_EQ(10000 + cur_val, cur_idx);

        cur_val = nxt_val;
        cur_idx = nxt_idx;
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
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
            cur_val = nxt_val;
        }
        fb_reader.close();
    }

    // clean up
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        remove(sorted_file1_list_[c_i].c_str());
        remove(("idx_" + sorted_file1_list_[c_i]).c_str());
    }
}

TEST_F(QuantileNormCudaTest, RadixSort1andMergeSortInParallelTmpOnMemoryTest) {
    use_tmp_files_ = false;

    setupFileList();

    // run
    mergeSort1();
    radixSort1();
    waitForTasks("radixsort1", radixsort1_futures_);
    waitForTasks("mergesort1", mergesort1_futures_);

    // check
    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + sorted_file1_list_[c_i]);
        ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

        std::shared_ptr<std::vector<uint16_t>> sorted_data = find_itr->second;
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*sorted_data).size());

        uint16_t cur_val = (*sorted_data)[0];
        for (size_t i = 1; i < (*sorted_data).size(); i++) {
            uint16_t nxt_val = (*sorted_data)[i];
            ASSERT_LE(cur_val, nxt_val);
            cur_val = nxt_val;
        }
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
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

TEST_F(QuantileNormCudaTest, SumSortedBuffersOnMemoryTest) {
    use_tmp_files_ = false;

    setupFileList();

    for (size_t i = 0; i < sorted_file1_list_.size(); i++) { // size = 4
        std::shared_ptr<std::vector<uint16_t>> data = std::make_shared<std::vector<uint16_t>>();
        for (size_t j = 0; j < 10; j++) {
            (*data).push_back(10 * i + j);
        }
        saveDataToBuffer(datadir_, sorted_file1_list_[i], data);
    }

    // run
    sumSortedBuffers();

    // check
    auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + summed_file_);
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<unsigned int>> summed_data = find_itr->second;
    ASSERT_EQ(10, (*summed_data).size());

    for (size_t i = 0; i < 10; i++) {
        unsigned int val = (*summed_data)[i];
        ASSERT_EQ(60 + i * 4, val); // 60 equals to 0 + 10 + 20 + 30, 4 equals to list size
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = false;
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
    substituteToNormValuesWithStorage(0);

    // check
    utils::FileBufferReader<float> fb_reader(substituted_file_list_[0], READ_BUFFER_SIZE);
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

TEST_F(QuantileNormCudaTest, SubstituteToNormValuesTmpOnMemoryTest) {
    use_tmp_files_ = false;

    setupFileList();

    std::shared_ptr<std::vector<unsigned int>> data = std::make_shared<std::vector<unsigned int>>();
    for (size_t i = 0; i < 10; i++) {
        (*data).push_back(i);
    }
    saveDataToBuffer(datadir_, summed_file_, data);

    std::shared_ptr<std::vector<uint16_t>> data2 = std::make_shared<std::vector<uint16_t>>();
    (*data2).push_back(1);
    (*data2).push_back(1);
    (*data2).push_back(1);
    (*data2).push_back(2);
    (*data2).push_back(3);
    (*data2).push_back(4);
    (*data2).push_back(6);
    (*data2).push_back(6);
    (*data2).push_back(8);
    (*data2).push_back(9);
    saveDataToBuffer(datadir_, sorted_file1_list_[0], data2);

    // run
    substituteToNormValuesWithMemory(0);

    // check
    auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + substituted_file_list_[0]);
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<float>> subst_data = find_itr->second;
    ASSERT_EQ(10, (*subst_data).size());

    EXPECT_EQ(1.0/4.0, (*subst_data)[0]);
    EXPECT_EQ(1.0/4.0, (*subst_data)[1]);
    EXPECT_EQ(1.0/4.0, (*subst_data)[2]);
    EXPECT_EQ(3.0/4.0, (*subst_data)[3]);
    EXPECT_EQ(4.0/4.0, (*subst_data)[4]);
    EXPECT_EQ(5.0/4.0, (*subst_data)[5]);
    EXPECT_EQ(6.5/4.0, (*subst_data)[6]);
    EXPECT_EQ(6.5/4.0, (*subst_data)[7]);
    EXPECT_EQ(8.0/4.0, (*subst_data)[8]);
    EXPECT_EQ(9.0/4.0, (*subst_data)[9]);

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = false;
}

TEST_F(QuantileNormCudaTest, RadixSort2Test) {
    setupFileList();

    std::get<0>(radixsort2_file_list_[0]) = 0;
    std::get<1>(radixsort2_file_list_[0]) = 10;

    std::string in_subst_file = std::get<2>(radixsort2_file_list_[0]);
    std::string in_idx_file   = std::get<3>(radixsort2_file_list_[0]);
    utils::FileBufferWriter<unsigned int> fb_writer1(in_idx_file, WRITE_BUFFER_SIZE);
    utils::FileBufferWriter<float>        fb_writer2(in_subst_file, WRITE_BUFFER_SIZE);
    fb_writer1.open();
    fb_writer2.open();
    for (size_t i = 0; i < 10; i++) {
        fb_writer1.set(10 - 1 - i);
        fb_writer2.set((float)i);
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
    utils::FileBufferReader<float>        fb_reader2(out_file, READ_BUFFER_SIZE);
    fb_reader1.open();
    fb_reader2.open();
    ASSERT_TRUE(fb_reader1.isOpen());
    ASSERT_TRUE(fb_reader2.isOpen());
    fb_reader1.readFileToBuffer();
    fb_reader2.readFileToBuffer();

    for (size_t i = 0; i < 10; i++) {
        unsigned int val1 = fb_reader1.get();
        float        val2 = fb_reader2.get();
        ASSERT_EQ(i, val1);
        ASSERT_EQ((float)(10 - 1 - i), val2);
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

TEST_F(QuantileNormCudaTest, RadixSort2TmpOnMemoryTest) {
    use_tmp_files_ = false;

    setupFileList();

    std::get<0>(radixsort2_file_list_[0]) = 0;
    std::get<1>(radixsort2_file_list_[0]) = 10;

    std::string in_subst_file = std::get<2>(radixsort2_file_list_[0]);
    std::string in_idx_file   = std::get<3>(radixsort2_file_list_[0]);
    std::shared_ptr<std::vector<unsigned int>> in_idx = std::make_shared<std::vector<unsigned int>>();
    std::shared_ptr<std::vector<float>> in_subst_data = std::make_shared<std::vector<float>>();
    for (size_t i = 0; i < 10; i++) {
        (*in_idx).push_back(10 - 1 - i);
        (*in_subst_data).push_back((float)i);
    }
    saveDataToBuffer(datadir_, in_idx_file, in_idx);
    saveDataToBuffer(datadir_, in_subst_file, in_subst_data);

    // run
    radixSort2FromData(0);

    // check
    std::string out_file     = std::get<4>(radixsort2_file_list_[0]);
    std::string idx_out_file = "idx_" + out_file;

    auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + out_file);
    auto idx_find_itr = tmp_data_buffers_.find(datadir_ + "/" + idx_out_file);
    ASSERT_TRUE(find_itr != tmp_data_buffers_.end());
    ASSERT_TRUE(idx_find_itr != tmp_data_buffers_.end());

    std::shared_ptr<std::vector<float>> out_data = find_itr->second;
    std::shared_ptr<std::vector<unsigned int>> out_idx = idx_find_itr->second;
    ASSERT_EQ(10, (*out_data).size());
    ASSERT_EQ(10, (*out_idx).size());

    for (size_t i = 0; i < 10; i++) {
        unsigned int val1 = (*out_idx)[i];
        double       val2 = (*out_data)[i];
        ASSERT_EQ(i, val1);
        ASSERT_EQ((float)(10 - 1 - i), val2);
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
}

TEST_F(QuantileNormCudaTest, AllRunTest) {

    // run
    run();

    // check
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenormFromTifFiles(in_fnames_);

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*result)[c_i].size());

        utils::FileBufferReader<float> fb_reader1(sorted_file2_list_[c_i], READ_BUFFER_SIZE);
        fb_reader1.open();
        fb_reader1.readFileToBuffer();

        for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
            float val1 = fb_reader1.get();
            float val2 = (*result)[c_i][i];
            ASSERT_EQ(val1, val2);
            fb_reader1.next();
        }
        fb_reader1.close();
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

TEST_F(QuantileNormCudaTest, AllRunTmpOnMemoryTest) {
    use_tmp_files_ = false;

    // run
    run();

    // check
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenormFromTifFiles(in_fnames_);

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*result)[c_i].size());

        auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + sorted_file2_list_[c_i]);
        ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

        std::shared_ptr<std::vector<float>> sorted_data2 = find_itr->second;
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*sorted_data2).size());

        for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
            float val1 = (*sorted_data2)[i];
            float val2 = (*result)[c_i][i];
            ASSERT_EQ(val1, val2);
        }
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
}

TEST_F(QuantileNormCudaTest, AllRunWithHdf5Test) {
    use_hdf5_ = true;
    std::vector<std::string> backup_fnames(in_fnames_);
    in_fnames_[0] = "img_0.h5";
    in_fnames_[1] = "img_1.h5";
    in_fnames_[2] = "img_2.h5";
    in_fnames_[3] = "img_3.h5";

    // run
    run();

    // check
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenormFromTifFiles(in_fnames_);

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*result)[c_i].size());

        utils::FileBufferReader<float> fb_reader1(sorted_file2_list_[c_i], READ_BUFFER_SIZE);
        fb_reader1.open();
        fb_reader1.readFileToBuffer();

        for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
            float val1 = fb_reader1.get();
            float val2 = (*result)[c_i][i];
            ASSERT_EQ(val1, val2);
            fb_reader1.next();
        }
        fb_reader1.close();
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

    use_hdf5_ = false;
    in_fnames_ = backup_fnames;
}

TEST_F(QuantileNormCudaTest, AllRunWithHdf5TmpOnMemoryTest) {
    use_hdf5_ = true;
    use_tmp_files_ = false;
    std::vector<std::string> backup_fnames(in_fnames_);
    in_fnames_[0] = "img_0.h5";
    in_fnames_[1] = "img_1.h5";
    in_fnames_[2] = "img_2.h5";
    in_fnames_[3] = "img_3.h5";

    // run
    run();

    // check
    std::shared_ptr<std::vector<std::vector<float>>> result = quantilenormFromTifFiles(in_fnames_);

    for (size_t c_i = 0; c_i < num_channels_; c_i++) {
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*result)[c_i].size());

        auto find_itr = tmp_data_buffers_.find(datadir_ + "/" + sorted_file2_list_[c_i]);
        ASSERT_TRUE(find_itr != tmp_data_buffers_.end());

        std::shared_ptr<std::vector<float>> sorted_data2 = find_itr->second;
        ASSERT_EQ(IMAGE_X * IMAGE_Y * IMAGE_Z, (*sorted_data2).size());

        for (size_t i = 0; i < IMAGE_X * IMAGE_Y * IMAGE_Z; i++) {
            float val1 = (*sorted_data2)[i];
            float val2 = (*result)[c_i][i];
            ASSERT_EQ(val1, val2);
        }
    }

    // clean up
    tmp_data_buffers_.clear();

    use_tmp_files_ = true;
    use_hdf5_ = false;
    in_fnames_ = backup_fnames;
}


}

