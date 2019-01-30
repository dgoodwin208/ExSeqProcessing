#include <string>
#include <fstream>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>

#include "filelock.h"

#include "spdlog/spdlog.h"


namespace cudautils {

FileLock::FileLock(const std::string& lock_filename)
    : lock_filename_(lock_filename), locked_fd_(-1) {

    struct stat s;
    if (stat("logs", &s) < 0) {
        mkdir("logs", 0755);
    }

    logger_ = spdlog::get("mex_logger");
    if (logger_ == nullptr) {
        logger_ = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
    }
}

FileLock::~FileLock() {
    if (locked_fd_ != -1) {
        unlock();
    }
}

int FileLock::trylock() {
    if (locked_fd_ != -1) {
        logger_->warn("already locked: lock={}", lock_filename_);
        return 0;
    }

    mode_t old_umask = umask(0);
    int fd = open(lock_filename_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        logger_->error("cannot open: {}", lock_filename_);
        umask(old_umask);
        return -1;
    }

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        umask(old_umask);
        return -1;
    }

    locked_fd_ = fd;
    umask(old_umask);

    return 0;
}

int FileLock::unlock() {
    if (locked_fd_ == -1) {
        logger_->error("not locked");
        return -1;
    }

    int ret = 0;
    if (flock(locked_fd_, LOCK_UN) != 0) {
        logger_->error("failed to unlock: lock={}, ERR={}", lock_filename_, strerror(errno));
        ret = -1;
    }

    close(locked_fd_);
    locked_fd_ = -1;

    return ret;
}

}

