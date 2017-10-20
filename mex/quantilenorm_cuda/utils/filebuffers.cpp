#include <cstdint>
#include <exception>

#include "filebuffers.h"
#include "spdlog/spdlog.h"

namespace utils {

template <typename T>
FileBufferReader<T>::FileBufferReader(const std::string& filename, const size_t max_buffer_size)
    : MAX_BUF_SIZE(max_buffer_size),
      buf_(max_buffer_size / sizeof(T)),
      buf_i_(0), buf_size_(0), read_size_(0), total_read_bytes_(0),
      filename_(filename) {
}

template <typename T>
FileBufferReader<T>::~FileBufferReader() {
    close();
}

template <typename T>
void
FileBufferReader<T>::open() {
    fin_.open(filename_, std::ios::in | std::ios::binary);
    if (! isOpen()) {
        std::string msg = "cannot open the input file; " + filename_;
        throw std::runtime_error(msg.c_str());
    }
}

template <typename T>
void
FileBufferReader<T>::close() {
    if (isOpen()) {
        fin_.close();
    }
}

template <typename T>
bool
FileBufferReader<T>::tryToOpen(const std::chrono::seconds& wait_seconds) {
    do {
        try {
            open();
        } catch (std::runtime_error& ex) {
            // not yet exist
            std::this_thread::sleep_for(wait_seconds);
        }
    } while (! isOpen());
}

template <typename T>
size_t
FileBufferReader<T>::getFileSize() {
    size_t cur_pos = fin_.tellg();

    fin_.seekg(0, std::ios_base::end);
    size_t file_size = fin_.tellg();

    fin_.seekg(cur_pos);

    return file_size;
}

template <typename T>
const T&
FileBufferReader<T>::get() {
    assert(buf_size_ > 0);

    return buf_[buf_i_];
}

template <typename T>
void
FileBufferReader<T>::next() {
//    auto logger = spdlog::get("mex_logger");
    buf_i_++;

    if (finishedBufferReading()) {
        readFileToBuffer();
    }
//    logger->debug("next - {}: buf_i = {}", filename_.c_str(), buf_i_);
}

template <typename T>
void
FileBufferReader<T>::readFileToBuffer() {
//    auto logger = spdlog::get("mex_logger");
    read_size_ = fin_.read((char*)buf_.data(), MAX_BUF_SIZE).gcount();
    if (read_size_ > 0) {
        buf_size_ = read_size_ / sizeof(T);
        buf_i_ = 0;
        total_read_bytes_ += read_size_;
    } else {
        buf_size_ = 0;
        buf_i_ = -1;
    }
//    logger->debug("readFileToBuffer - {}: read_size = {}, buf_size = {}, buf_i = {}, total = {}", filename_.c_str(), read_size_, buf_size_, buf_i_, total_read_bytes_);
}

template <typename T>
void
FileBufferReader<T>::readFileToBufferPartially(const size_t num_data_start, std::shared_ptr<std::vector<T>> buffer) {
//    auto logger = spdlog::get("mex_logger");
    fin_.seekg(num_data_start * sizeof(T));
//    logger->debug("num_data_start = {}, buffer size = {}", num_data_start, buffer->size());
//    while (! finishedReadingAll()) {
        read_size_ = fin_.read((char*)((*buffer).data()) + total_read_bytes_, buffer->size() * sizeof(T)).gcount();
//        if (read_size_ > 0) {
//            buf_size_ = read_size_ / sizeof(T);
//            buf_i_ = 0;
//            total_read_bytes_ += read_size_;
//        } else {
//            buf_i_ = -1;
//        }
//    }
//    logger->debug("readFileToBuffer - {}: read_size = {}, buf_size = {}, buf_i = {}, total = {}", filename_.c_str(), read_size_, buf_size_, buf_i_, total_read_bytes_);
}


template <typename T>
FileBufferWriter<T>::FileBufferWriter(const std::string& filename, const size_t max_buffer_size)
    : MAX_BUF_SIZE(max_buffer_size),
      buf_(max_buffer_size / sizeof(T)),
      buf_i_(0), buf_size_(0), write_size_(0), total_write_bytes_(0),
      filename_(filename) {
//    auto logger = spdlog::get("mex_logger");
//    logger->debug("MAX = {}, buf size = {}, fname = {}", MAX_BUF_SIZE, buf_.size(), filename_.c_str());
}

template <typename T>
FileBufferWriter<T>::~FileBufferWriter() {
    if (isOpen()) {
        fout_.close();
    }
}

template <typename T>
void
FileBufferWriter<T>::open() {
    fout_.open(filename_, std::ios::out | std::ios::binary);
    if (! isOpen()) {
        std::string msg = "cannot open the output file; " + filename_;
        throw std::runtime_error(msg.c_str());
    }
}

template <typename T>
void
FileBufferWriter<T>::set(const T& val) {
//    auto logger = spdlog::get("mex_logger");
    buf_[buf_i_++] = val;

    if (finishedBufferWriting()) {
        writeFileFromBuffer();
    }
//    logger->debug("set - {}: val = {}, buf_i = {}", filename_.c_str(), val, buf_i_);
}

template <typename T>
void
FileBufferWriter<T>::writeFileFromBuffer() {
//    auto logger = spdlog::get("mex_logger");
    fout_.write((char*)buf_.data(), buf_i_ * sizeof(T));
    total_write_bytes_ += buf_i_ * sizeof(T);
    buf_i_ = 0;
//    logger->debug("writeFileFromBuffer - {}: buf_i = {}, write_size = {}, total = {}", filename_.c_str(), buf_i_, buf_.size() * sizeof(T), total_write_bytes_);
}

template <typename T>
void
FileBufferWriter<T>::writeFileFromReaderBufferDirectly(FileBufferReader<T>& reader) {
//    auto logger = spdlog::get("mex_logger");
    fout_.write((char*)&reader.buf_[reader.buf_i_], (reader.buf_size_ - reader.buf_i_) * sizeof(T));
    total_write_bytes_ += (reader.buf_size_ - reader.buf_i_) * sizeof(T);
//    logger->debug("writeFileFromReaderBufferDirectly - {}: buf_size = {}, write_size = {}, reader.buf_i = {}", filename_.c_str(), reader.buf_.size(), (reader.buf_size_ - reader.buf_i_) * sizeof(T), reader.buf_i_);
    reader.readFileToBuffer();
    int count = 1;
    while (reader.hasBufferData()) {
        fout_.write((char*)reader.buf_.data(), reader.buf_size_ * sizeof(T));
        total_write_bytes_ += reader.buf_size_ * sizeof(T);
//        logger->debug("writeFileFromReaderBufferDirectly - {}: count = {}, buf_size = {}, write_size = {}, reader.buf_i = {}", filename_.c_str(), count, reader.buf_.size(), reader.buf_size_ * sizeof(T), reader.buf_i_);
        reader.readFileToBuffer();
        count++;
    }
}

template <typename T>
FileDirectWriter<T>::FileDirectWriter(const std::string& filename)
    : filename_(filename) {
//    auto logger = spdlog::get("mex_logger");
//    logger->debug("fname = {}", filename_.c_str());
}

template <typename T>
FileDirectWriter<T>::~FileDirectWriter() {
    close();
}

template <typename T>
void
FileDirectWriter<T>::open() {
    fout_.open(filename_, std::ios::out | std::ios::binary);
    if (! isOpen()) {
        std::string msg = "cannot open the output file; " + filename_;
        throw std::runtime_error(msg.c_str());
    }
}

template <typename T>
void
FileDirectWriter<T>::close() {
    if (isOpen()) {
        fout_.close();
    }
}

template <typename T>
void
FileDirectWriter<T>::seek(const std::streampos pos) {
    fout_.seekp(pos * sizeof(T));
}

template <typename T>
void
FileDirectWriter<T>::write(const T val) {
    fout_.write((char *)&val, sizeof(T));
}


template class FileBufferReader<uint16_t>;
template class FileBufferReader<unsigned int>;
template class FileBufferReader<double>;

template class FileBufferWriter<uint16_t>;
template class FileBufferWriter<unsigned int>;
template class FileBufferWriter<double>;

template class FileDirectWriter<double>;

}

