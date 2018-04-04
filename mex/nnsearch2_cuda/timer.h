#include <iostream>
#include <chrono>

class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;

public:
    Timer() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    void print_laptime() {
        auto end_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_);
        std::cout << "duration: " << duration.count() << std::endl;
    }
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    float get_laptime() {
        auto end_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_ - start_);
        return (float) duration.count();
    }
};

