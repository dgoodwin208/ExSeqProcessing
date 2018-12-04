#ifndef __FILELOCK_H__
#define __FILELOCK_H__

#include <string>

#include "spdlog/spdlog.h"

namespace cudautils {

class FileLock {
    const std::string lock_filename_;

    int locked_fd_;

    std::shared_ptr<spdlog::logger> logger_;

public:
    FileLock(const std::string& lock_filename);
    ~FileLock();

    int trylock();
    int unlock();

    bool isLocked() { return (locked_fd_ != -1); }
};

}

#endif

