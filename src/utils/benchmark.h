// src/utils/benchmark.h
#pragma once

#include "../core/types.h"
#include <chrono>
#include <vector>

namespace learned_index {

/**
 * @brief Benchmarking utilities for performance measurement
 */
class Benchmark {
public:
    /// Start timer
    static void start();
    
    /// Stop timer and return elapsed time in nanoseconds
    static uint64_t stop();
    
    /// Get current timestamp in nanoseconds
    static uint64_t now();
    
    /**
     * @brief Run operation N times and collect latency samples
     * @tparam F Function type
     * @param op Operation to benchmark
     * @param iterations Number of iterations
     * @return Vector of latency samples in nanoseconds
     */
    template<typename F>
    static std::vector<uint64_t> profile(F op, size_t iterations) {
        std::vector<uint64_t> samples;
        samples.reserve(iterations);
        
        for (size_t i = 0; i < iterations; i++) {
            auto t1 = now();
            op();
            auto t2 = now();
            samples.push_back(t2 - t1);
        }
        
        return samples;
    }
    
    /**
     * @brief Compute percentile latency
     * @param samples Latency samples
     * @param percentile 0-100
     * @return Latency in nanoseconds
     */
    static uint64_t percentile(std::vector<uint64_t> samples, 
                              double percentile);
    
private:
    static std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace learned_index
