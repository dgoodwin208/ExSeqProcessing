#ifndef __ASYNC_QUEUE_H__
#define __ASYNC_QUEUE_H__

#include <deque>
#include <mutex>
#include <condition_variable>

namespace utils {

template<typename T>
class AsyncQueue {

    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_noempty_;
    bool closed_ = false;

public:
    void push(T data) {
        std::unique_lock<std::mutex> lk(mutex_);

        if (closed_) {
            //throw
            return;
        }

        bool do_notify = queue_.empty();
        queue_.push_back(data);
        if (do_notify) {
            cv_noempty_.notify_one();
        }
    }

    bool pop(T& data) {
        std::unique_lock<std::mutex> lk(mutex_);

        cv_noempty_.wait(lk, [&] { return !queue_.empty() || closed_; });

        if (queue_.empty() && closed_) {
            return false;
        }

        data = queue_.front();
        queue_.pop_front();

        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mutex_);

        closed_ = true;
        cv_noempty_.notify_all();
    }

};

}

#endif // __ASYNC_QUEUE_H__

