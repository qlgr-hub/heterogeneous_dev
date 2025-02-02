#pragma once

#include <chrono>


namespace Utils {

template <typename TT = std::chrono::high_resolution_clock>
class TimeStats {
    std::chrono::time_point<TT> _start;

public:
    TimeStats() {
        _start = TT::now();
    }

public:
    void reset() {
        _start = TT::now();
    }

public:
    double lap_nano() const {
        using namespace std::chrono;
        auto elapse{ duration_cast<nanoseconds>(TT::now() - _start).count() };
        return static_cast<double>(elapse);
    }

    double lap_micro() const {
        return lap_nano() / 1000.0;
    }

    double lap_milli() const {
        return lap_micro() / 1000.0;
    }

    double lap_sec() const {
        return lap_milli() / 1000.0;
    }
};

}