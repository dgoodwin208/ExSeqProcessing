#include "gtest/gtest.h"

#include "gpudevice.h"
#include "nnsearch2.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <random>

namespace {

class GpuDeviceTest : public ::testing::Test {
protected:
    GpuDeviceTest() {
        auto logger = spdlog::basic_logger_mt("mex_logger", "mex.log");
    }
    virtual ~GpuDeviceTest() {
    }
};

TEST_F(GpuDeviceTest, GetGpuNumTest) {
    int gpu_num = cudautils::get_gpu_num();

    EXPECT_EQ(2, gpu_num);
}

class NearestNeighborSearchTest : public ::testing::Test {
protected:
    std::shared_ptr<spdlog::logger> logger_;
    std::string testdata_dir_;

    NearestNeighborSearchTest() {
        logger_ = spdlog::get("console");
        if (! logger_) {
            logger_ = spdlog::stdout_logger_mt("console");
        }

        testdata_dir_ = "/mp/nas1/share/tests/testdata/nnsearch2_cuda/";
    }
    virtual ~NearestNeighborSearchTest() {
    }

    void print_parameters(
            const unsigned int m,
            const unsigned int n,
            const unsigned int k,
            const unsigned int dm,
            const unsigned int dn,
            const unsigned int num_gpus,
            const unsigned int num_streams) {
        logger_->info("m={},n={},k={},dm={},dn={}", m, n, k, dm, dn);
        logger_->info("num_gpus={},num_streams={}", num_gpus, num_streams);
    }
};

TEST_F(NearestNeighborSearchTest, SmallSize1Test) {

    // small size and # of subdomain is 0
    unsigned int m = 6;
    unsigned int n = 4;
    unsigned int k = 3;
    unsigned int dm = 6;
    unsigned int dn = 4;

    unsigned int num_gpus = 1;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, SmallSize2Test) {

    // small size and # of subdomain is 2 x 2
    unsigned int m = 6;
    unsigned int n = 4;
    unsigned int k = 3;
    unsigned int dm = 3;
    unsigned int dn = 2;

    unsigned int num_gpus = 1;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, SmallSize3Test) {

    // small size and # of subdomain is 2 x 2 with 2 gpus
    unsigned int m = 6;
    unsigned int n = 4;
    unsigned int k = 3;
    unsigned int dm = 3;
    unsigned int dn = 2;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, SmallSize4Test) {

    // small size and # of subdomain is 2 x 2 with 2 gpus and 2 streams
    unsigned int m = 6;
    unsigned int n = 4;
    unsigned int k = 3;
    unsigned int dm = 3;
    unsigned int dn = 2;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 2;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, SmallSize5Test) {

    // small size with extra size for thread size
    unsigned int m = 20;
    unsigned int n = 40;
    unsigned int k = 3;
    unsigned int dm = 20;
    unsigned int dn = 40;

    unsigned int num_gpus = 1;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, MiddleSize1Test) {

    // middle size and # of loops in two tops of mins is over 2
    unsigned int m = 200;
    unsigned int n = 200;
    unsigned int k = 3;
    unsigned int dm = 200;
    unsigned int dn = 200;

    unsigned int num_gpus = 1;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, MiddleSize2Test) {

    // middle size and multi-subdomains with 2gpus and multi streams
    unsigned int m = 200;
    unsigned int n = 200;
    unsigned int k = 3;
    unsigned int dm = 20;
    unsigned int dn = 20;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 4;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, MiddleSize3Test) {

    // middle size and single subdomain with large k
    unsigned int m = 20;
    unsigned int n = 20;
    unsigned int k = 600;
    unsigned int dm = 20;
    unsigned int dn = 30;

    unsigned int num_gpus = 1;
    unsigned int num_streams = 1;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, MiddleSize4Test) {

    // middle size, uneven dn and multi-subdomains with 2gpus and multi streams
    unsigned int m = 20;
    unsigned int n = 200;
    unsigned int k = 3;
    unsigned int dm = 20;
    unsigned int dn = 150;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 4;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}


TEST_F(NearestNeighborSearchTest, LargeSize1Test) {

    // large size
    unsigned int m = 2000;
    unsigned int n = 2000;
    unsigned int k = 2;
    unsigned int dm = 1000;
    unsigned int dn = 1000;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 2;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, LargeSize2Test) {

    // large size
    unsigned int m = 10000;
    unsigned int n = 10000;
    unsigned int k = 2;
    unsigned int dm = 1000;
    unsigned int dn = 1000;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 4;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.generateSequences();

    nns.run();

    ASSERT_TRUE(nns.checkResult());
}

TEST_F(NearestNeighborSearchTest, LargeSize3Test) {

    // large size
//    unsigned int m = 77871;
//    unsigned int n = 68668;
//    unsigned int k = 640;
//    unsigned int dm = 1000;
//    unsigned int dn = 50000;
//
//    unsigned int num_gpus = 2;
//    unsigned int num_streams = 10;

    unsigned int m;
    unsigned int n;
    unsigned int k;

    std::string file1 = testdata_dir_ + "testdata01/sift_x_dim.bin";
    std::string file2 = testdata_dir_ + "testdata01/sift_y_dim.bin";
    FILE* fp1 = fopen(file1.c_str(), "rb");
    FILE* fp2 = fopen(file2.c_str(), "rb");

    fread(&m, sizeof(unsigned int), 1, fp1);
    fread(&k, sizeof(unsigned int), 1, fp1);
    fread(&n, sizeof(unsigned int), 1, fp2);

    fclose(fp1);
    fclose(fp2);

    double *data1 = new double[m * k];
    double *data2 = new double[n * k];

    file1 = testdata_dir_ + "testdata01/sift_x.bin";
    file2 = testdata_dir_ + "testdata01/sift_y.bin";
    fp1 = fopen(file1.c_str(), "rb");
    fp2 = fopen(file2.c_str(), "rb");

    fread(data1, sizeof(double), m * k, fp1);
    fread(data2, sizeof(double), n * k, fp2);

    fclose(fp1);
    fclose(fp2);

//    std::cout << "data1[0]=" << data1[0] << ",data1[1]=" << data1[1] << std::endl;
//    std::cout << "data2[0]=" << data2[0] << ",data2[1]=" << data2[1] << std::endl;

    unsigned int dm = 1000;
    unsigned int dn = 50000;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 10;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.setInput(data1, data2);

    nns.run();

    std::string file = testdata_dir_ + "testdata01/sift_dist2_dim.bin";
    FILE *fp = fopen(file.c_str(), "rb");
    unsigned int dist2_m;
    unsigned int dist2_n;
    fread(&dist2_m, sizeof(unsigned int), 1, fp);
    fread(&dist2_n, sizeof(unsigned int), 1, fp);
    fclose(fp);
    ASSERT_EQ(m, dist2_m);
    ASSERT_EQ(n, dist2_n);

    file = testdata_dir_ + "testdata01/sift_dist2.bin";
    fp = fopen(file.c_str(), "rb");
    double *dist2 = new double[size_t(m) * n];
    fread(dist2, sizeof(double), size_t(m) * n, fp);
    fclose(fp);

//    for (unsigned int i = 0; i < 10; i++) {
//        std::cout << "dist2[" << i << "]=" << dist2[i] << std::endl;
//    }

    ASSERT_TRUE(nns.checkDist2(dist2));
//    double *val = new double[2 * m];
//    unsigned int *idx = new unsigned int[2 * m];
//
//    nns.getResult(&val, &idx);
//
//    ASSERT_TRUE(nns.checkResult());

    delete[] data1;
    delete[] data2;
    delete[] dist2;
//    delete[] val;
//    delete[] idx;
}

TEST_F(NearestNeighborSearchTest, LargeSize4Test) {

    // large size
    unsigned int m;
    unsigned int n;
    unsigned int k;

    std::string file1 = testdata_dir_ + "testdata02/sift_x_dim.bin";
    std::string file2 = testdata_dir_ + "testdata02/sift_y_dim.bin";
    FILE* fp1 = fopen(file1.c_str(), "rb");
    FILE* fp2 = fopen(file2.c_str(), "rb");

    fread(&m, sizeof(unsigned int), 1, fp1);
    fread(&k, sizeof(unsigned int), 1, fp1);
    fread(&n, sizeof(unsigned int), 1, fp2);

    fclose(fp1);
    fclose(fp2);

    double *data1 = new double[m * k];
    double *data2 = new double[n * k];

    file1 = testdata_dir_ + "testdata02/sift_x.bin";
    file2 = testdata_dir_ + "testdata02/sift_y.bin";
    fp1 = fopen(file1.c_str(), "rb");
    fp2 = fopen(file2.c_str(), "rb");

    fread(data1, sizeof(double), m * k, fp1);
    fread(data2, sizeof(double), n * k, fp2);

    fclose(fp1);
    fclose(fp2);

//    std::cout << "data1[0]=" << data1[0] << ",data1[1]=" << data1[1] << std::endl;
//    std::cout << "data2[0]=" << data2[0] << ",data2[1]=" << data2[1] << std::endl;

    unsigned int dm = 1000;
    unsigned int dn = (n > 50000) ? 50000 : n;

    unsigned int num_gpus = 2;
    unsigned int num_streams = 10;

    print_parameters(m, n, k, dm, dn, num_gpus, num_streams);
    cudautils::NearestNeighborSearch nns(m, n, k, dm, dn, num_gpus, num_streams);

    nns.setInput(data1, data2);

    nns.run();

    std::string file = testdata_dir_ + "testdata02/sift_dist2_dim.bin";
    FILE *fp = fopen(file.c_str(), "rb");
    unsigned int dist2_m;
    unsigned int dist2_n;
    fread(&dist2_m, sizeof(unsigned int), 1, fp);
    fread(&dist2_n, sizeof(unsigned int), 1, fp);
    fclose(fp);
    ASSERT_EQ(m, dist2_m);
    ASSERT_EQ(n, dist2_n);

    file = testdata_dir_ + "testdata02/sift_dist2.bin";
    fp = fopen(file.c_str(), "rb");
    double *dist2 = new double[size_t(m) * n];
    fread(dist2, sizeof(double), size_t(m) * n, fp);
    fclose(fp);

//    for (unsigned int i = 0; i < 10; i++) {
//        std::cout << "dist2[" << i << "]=" << dist2[i] << std::endl;
//    }

    ASSERT_TRUE(nns.checkDist2(dist2));


//    for (unsigned int i = 0; i < m; i++) {
//        for (unsigned int j = 0; j < n; j++) {
//            EXPECT_NEAR(dist2[j + i * n], nns.getDist2(i, j), 1e-10);
//        }
//    }

//    double *val = new double[2 * m];
//    unsigned int *idx = new unsigned int[2 * m];
//
//    nns.getResult(&val, &idx);
//
//    ASSERT_TRUE(nns.checkResult());

    delete[] data1;
    delete[] data2;
    delete[] dist2;
//    delete[] val;
//    delete[] idx;
}

}

