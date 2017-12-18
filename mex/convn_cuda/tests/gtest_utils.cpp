#include "gtest/gtest.h"
#include "filebuffers.h"

#include <fstream>
#include <thread>
#include <vector>
#include <memory>


namespace {

const unsigned int DATA_SIZE = 100;

class FileBufferTest : public ::testing::Test {
protected:
    FileBufferTest() {
        std::ofstream fout("test.bin", std::ofstream::out | std::ofstream::binary);
        for (unsigned int i = 0; i < DATA_SIZE; i++) {
            fout.write((char *)&i, sizeof(i));
        }
        fout.close();
    }
    virtual ~FileBufferTest() {
        remove("test.bin");
    }
};

TEST_F(FileBufferTest, ReaderOpenTest) {
    std::string filename = "test.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));
    ASSERT_FALSE(fb_reader.isOpen());

    fb_reader.open();
    ASSERT_TRUE(fb_reader.isOpen());

    fb_reader.close();
}

TEST_F(FileBufferTest, ReaderTryToOpenTest) {
    std::string filename = "test1.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));

    std::thread wait_thread([&filename]() {
        std::chrono::seconds wait_seconds(1);
        std::this_thread::sleep_for(wait_seconds);

        std::string cmd = "touch " + filename;
        system(cmd.c_str());
    });
    ASSERT_FALSE(fb_reader.isOpen());


    fb_reader.tryToOpen();
    ASSERT_TRUE(fb_reader.isOpen());
    wait_thread.join();

    remove(filename.c_str());

    fb_reader.close();
}

TEST_F(FileBufferTest, ReaderGetFileSizeTest) {
    std::string filename = "test.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));
    fb_reader.open();
    ASSERT_EQ(DATA_SIZE * sizeof(unsigned int), fb_reader.getFileSize());

    fb_reader.close();
}

TEST_F(FileBufferTest, ReaderGetAndNextTest) {
    std::string filename = "test.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 24 * sizeof(unsigned int));
    fb_reader.open();

    ASSERT_TRUE (fb_reader.finishedBufferReading());
    ASSERT_FALSE(fb_reader.finishedReadingAll());
    ASSERT_FALSE(fb_reader.hasBufferData());

    fb_reader.readFileToBuffer();
    ASSERT_EQ(24 * sizeof(unsigned int), fb_reader.getTotalReadBytes());

    ASSERT_FALSE(fb_reader.finishedBufferReading());
    ASSERT_FALSE(fb_reader.finishedReadingAll());
    ASSERT_TRUE (fb_reader.hasBufferData());

    unsigned int count = 0;
    while (! fb_reader.finishedReadingAll()) {
        unsigned int val = fb_reader.get();
        ASSERT_EQ(count++, val);
        fb_reader.next();
    }
    ASSERT_EQ(100 * sizeof(unsigned int), fb_reader.getTotalReadBytes());

    ASSERT_FALSE(fb_reader.finishedBufferReading());
    ASSERT_TRUE (fb_reader.finishedReadingAll());
    ASSERT_FALSE(fb_reader.hasBufferData());

    fb_reader.close();
}

TEST_F(FileBufferTest, ReaderLargerBufferTest) {
    std::string filename = "test.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 200 * sizeof(unsigned int));
    fb_reader.open();

    fb_reader.readFileToBuffer();
    ASSERT_EQ(100 * sizeof(unsigned int), fb_reader.getTotalReadBytes());

    fb_reader.close();
}

TEST_F(FileBufferTest, ReaderReadBufferToPartiallyTest) {
    std::string filename = "test.bin";
    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));
    fb_reader.open();

    size_t pos = 10;
    std::shared_ptr<std::vector<unsigned int>> buf(new std::vector<unsigned int>(25));
    fb_reader.readFileToBufferPartially(pos, buf);
    for (int i = 0; i < 25; i++) {
        ASSERT_EQ(i + 10, (*buf)[i]);
    }

    fb_reader.close();
}

TEST_F(FileBufferTest, WriterOpenTest) {
    std::string filename = "test_out.bin";
    utils::FileBufferWriter<unsigned int> fb_writer(filename, 25 * sizeof(unsigned int));
    ASSERT_FALSE(fb_writer.isOpen());

    fb_writer.open();
    ASSERT_TRUE(fb_writer.isOpen());

    fb_writer.close();
}

TEST_F(FileBufferTest, WriterSetTest) {
    std::string filename = "test_out.bin";
    utils::FileBufferWriter<unsigned int> fb_writer(filename, 25 * sizeof(unsigned int));
    fb_writer.open();

    for (unsigned int i = 0; i < 50; i++) {
        fb_writer.set(i*10);
    }

    fb_writer.close();

    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));
    fb_reader.open();

    fb_reader.readFileToBuffer();
    for (unsigned int i = 0; i < 50; i++) {
        unsigned int val = fb_reader.get();
        ASSERT_EQ(i*10, val);
        fb_reader.next();
    }

    fb_reader.close();

    remove(filename.c_str());
}

TEST_F(FileBufferTest, WriterWriteBufferTest) {
    std::string filename = "test_out.bin";
    utils::FileBufferWriter<unsigned int> fb_writer(filename, 25 * sizeof(unsigned int));
    fb_writer.open();

    ASSERT_FALSE(fb_writer.finishedBufferWriting());
    ASSERT_FALSE(fb_writer.hasBufferData());

    for (unsigned int i = 0; i < 20; i++) {
        fb_writer.set(i*10);
    }

    ASSERT_FALSE(fb_writer.finishedBufferWriting());
    ASSERT_TRUE (fb_writer.hasBufferData());

    fb_writer.writeFileFromBuffer();

    ASSERT_FALSE(fb_writer.finishedBufferWriting());
    ASSERT_FALSE(fb_writer.hasBufferData());

    ASSERT_EQ(20 * sizeof(unsigned int), fb_writer.getTotalWriteBytes());

    fb_writer.close();

    utils::FileBufferReader<unsigned int> fb_reader(filename, 25 * sizeof(unsigned int));
    fb_reader.open();

    fb_reader.readFileToBuffer();
    for (unsigned int i = 0; i < 20; i++) {
        unsigned int val = fb_reader.get();
        ASSERT_EQ(i*10, val);
        fb_reader.next();
    }

    fb_reader.close();

    remove(filename.c_str());
}

TEST_F(FileBufferTest, WriterWriteReaderBufferDirectlyTest) {
    std::string in_filename  = "test.bin";
    std::string out_filename = "test_out.bin";
    utils::FileBufferReader<unsigned int> fb_reader(in_filename , 25 * sizeof(unsigned int));
    utils::FileBufferWriter<unsigned int> fb_writer(out_filename, 25 * sizeof(unsigned int));
    fb_reader.open();
    fb_writer.open();

    fb_writer.writeFileFromReaderBufferDirectly(fb_reader);

    fb_reader.close();
    fb_writer.close();

    utils::FileBufferReader<unsigned int> fb_reader2(out_filename, 25 * sizeof(unsigned int));
    fb_reader2.open();

    fb_reader2.readFileToBuffer();
    for (unsigned int i = 0; i < 100; i++) {
        unsigned int val = fb_reader2.get();
        ASSERT_EQ(i, val);
        fb_reader2.next();
    }

    fb_reader2.close();

    remove(out_filename.c_str());
}

TEST_F(FileBufferTest, DirectWriterOpenTest) {
    std::string filename = "test_out.bin";
    utils::FileDirectWriter<double> fb_writer(filename);
    ASSERT_FALSE(fb_writer.isOpen());

    fb_writer.open();
    ASSERT_TRUE(fb_writer.isOpen());

    fb_writer.close();
}

TEST_F(FileBufferTest, DirectWriterWriteTest) {
    std::string filename = "test_out.bin";
    utils::FileDirectWriter<double> fb_writer(filename);

    fb_writer.open();
    fb_writer.seek(10);

    double val = 1.1;
    fb_writer.write(val);

    fb_writer.close();

    utils::FileBufferReader<double> fb_reader(filename, 25 * sizeof(double));
    fb_reader.open();

    ASSERT_EQ(11 * sizeof(double), fb_reader.getFileSize());

    fb_reader.readFileToBuffer();
    for (unsigned int i = 0; i < 10; i++) {
        val = fb_reader.get();
        ASSERT_EQ(0.0, val);
        fb_reader.next();
    }
    val = fb_reader.get();
    ASSERT_EQ(1.1, val);

    //ASSERT_EQ(11 * sizeof(double), fb_reader.getFileSize());
    //TODO not correct at this position..

    fb_reader.close();
}

}

