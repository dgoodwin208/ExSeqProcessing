/*=================================================================
 * cmd_nnsearch2_cuda.cpp - nearest-neighbor search (k=2)
 *
 *  Input:
 *    filename of x : m x k matrix; lists of k-dimensional vectors
 *    filename of y : n x k matrix; lists of k-dimensional vectors
 *
 *  Output:
 *    r    : m x 2 matrix; lists of 2 tops of minimums
 *    idx  : m x 2 matrix; lists of 2 indices of the 2 tops which show along the n-direction
 *
 *=================================================================*/


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "spdlog/spdlog.h"
#include "cuda-utils/nnsearch2.h"
#include "cuda-utils/gpudevice.h"
#include "cuda_task_executor.h"


int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [in file1] [in file2] [out file]" << std::endl;
        return 1;
    }

    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        spdlog::set_level(spdlog::level::trace);
        logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
        }
        logger->flush_on(spdlog::level::err);
        //logger->flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
        return 1;
    }

    try {
        logger->info("{:=>50}", " nnsearch2_cuda start");

        std::string in_filename1(argv[1]);
        std::string in_filename2(argv[2]);
        std::string out_filename(argv[3]);

        unsigned int m, n, k, k1;
        std::ifstream fin1(in_filename1, std::ios::binary);
        fin1.read((char*)&m, sizeof(unsigned int));
        fin1.read((char*)&k, sizeof(unsigned int));

        std::ifstream fin2(in_filename2, std::ios::binary);
        fin2.read((char*)&n,  sizeof(unsigned int));
        fin2.read((char*)&k1, sizeof(unsigned int));

        if (k != k1) {
            logger->error("k size of data1 and data2 is not the same.");
            fin1.close();
            fin2.close();
            return 1;
        }

        std::vector<double> in_data1(m * k);
        std::vector<double> in_data2(n * k);
        fin1.read((char*)in_data1.data(), m * k * sizeof(double));
        fin2.read((char*)in_data2.data(), n * k * sizeof(double));
        fin1.close();
        fin2.close();

        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus={}", num_gpus);

        std::vector<double> outMinsVal(2 * m);
        std::vector<unsigned int> outMinsIdx(2 * m);

        float round_up_num = 1000.0;
        unsigned int dm = 1000;
        unsigned int num_dn_blocks = cudautils::get_num_blocks(n, 50000);
        unsigned int dn = (n <= 50000) ? n : (unsigned int)(std::ceil(float(n) / float(num_dn_blocks) / round_up_num) * round_up_num);
        int num_streams = 10;
        logger->info("m={},n={},dm={},dn={},# of dn blocks={},# of streams={}", m, n, dm, dn, num_dn_blocks, num_streams);

        //TODO check the max of GPU memory usage!

        try {
            std::shared_ptr<cudautils::NearestNeighborSearch> nns =
                std::make_shared<cudautils::NearestNeighborSearch>(m, n, k, dm, dn, num_gpus, num_streams);

            parallelutils::CudaTaskExecutor executor(num_gpus, num_streams, nns);

            nns->setInput(in_data1,in_data2);

            executor.run();

            nns->getResult(outMinsVal, outMinsIdx);

            unsigned int num_2 = 2;
            std::ofstream fout(out_filename, std::ios::binary);
            fout.write((char*)&m,     sizeof(unsigned int));
            fout.write((char*)&num_2, sizeof(unsigned int));

            fout.write((char*)outMinsVal.data(), num_2 * m * sizeof(double));
            fout.write((char*)outMinsIdx.data(), num_2 * m * sizeof(unsigned int));
            fout.close();

        } catch (...) {
            logger->error("internal unknown error occurred");
        }

        logger->info("{:=>50}", " nnsearch2_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return 0;
}

