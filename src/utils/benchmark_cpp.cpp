// src/utils/benchmark.cpp
#include "benchmark.h"
#include <algorithm>

namespace learned_index {

std::chrono::high_resolution_clock::time_point Benchmark::start_time_;

void Benchmark::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

uint64_t Benchmark::stop() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time_);
    return duration.count();
}

uint64_t Benchmark::now() {
    auto current = std::chrono::high_resolution_clock::now();
    auto epoch = current.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
}

uint64_t Benchmark::percentile(std::vector<uint64_t> samples, 
                               double p) {
    if (samples.empty()) return 0;
    
    std::sort(samples.begin(), samples.end());
    
    if (p < 0.0) p = 0.0;
    if (p > 100.0) p = 100.0;
    
    size_t index = static_cast<size_t>((p / 100.0) * (samples.size() - 1));
    return samples[index];
}

} // namespace learned_index
