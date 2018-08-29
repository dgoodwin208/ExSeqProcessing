#ifndef __FILEBUFFERS_H__
#define __FILEBUFFERS_H__

#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <memory>

namespace utils {

template <typename T> class FileBufferWriter;

static constexpr auto FILECHECK_INTERVAL_SEC = std::chrono::seconds(1);

template <typename T>
class FileBufferReader {
    const int MAX_BUF_SIZE;
    std::vector<T> buf_;
    int buf_i_;
    int buf_size_;
    int read_size_;
    long long total_read_bytes_;

    std::string filename_;
    std::ifstream fin_;
        
public:
    FileBufferReader(const std::string& filename, const size_t max_buffer_size);
    ~FileBufferReader();

    void open();
    void close();
    bool isOpen() { return (! fin_.fail() && fin_.is_open()); }
    bool tryToOpen(const std::chrono::seconds& wait_seconds = FILECHECK_INTERVAL_SEC);

    size_t getFileSize();

    const T& get();
    void next();

    void readFileToBuffer();
    void readFileToBufferPartially(const size_t pos_start, std::shared_ptr<std::vector<T>> buffer);

    bool finishedBufferReading() { return buf_i_ == buf_size_; }
    bool finishedReadingAll() { return buf_i_ == -1; }
    bool hasBufferData() { return buf_size_ > 0; }

    long long getTotalReadBytes() { return total_read_bytes_; }

    friend class FileBufferWriter<T>;
};

template <typename T>
class FileBufferWriter {
    const int MAX_BUF_SIZE;
    std::vector<T> buf_;
    int buf_i_;
    int buf_size_;
    int write_size_;
    long long total_write_bytes_;

    std::string filename_;
    std::ofstream fout_;

public:
    FileBufferWriter(const std::string& filename, const size_t max_buffer_size);
    ~FileBufferWriter();

    void open();
    void close() { fout_.close(); }
    bool isOpen() { return (! fout_.fail() && fout_.is_open()); }

    void set(const T& val);

    void writeFileFromBuffer();
    void writeFileFromReaderBufferDirectly(FileBufferReader<T>& reader);

    bool finishedBufferWriting() { return buf_i_ == buf_.size(); }
    bool hasBufferData() { return buf_i_ > 0; }

    long long getTotalWriteBytes() { return total_write_bytes_; }

};

template <typename T>
class FileDirectWriter {
    std::string filename_;
    std::ofstream fout_;

public:
    FileDirectWriter(const std::string& filename);
    ~FileDirectWriter();

    void open();
    void close();
    bool isOpen() { return (! fout_.fail() && fout_.is_open()); }

    void seek(const std::streampos pos);
    void write(const T val);

};

}

#endif // __FILEBUFFERS_H__

