#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "core/types.h"
#include "utils/statistics.h"

using namespace learned_index;

// ============================================================================
// PHASE 2: SIMPLE BINARY SEARCH BENCHMARK
// Works with your actual codebase
// ============================================================================

std::vector<KeyValuePair> generateData(size_t size) {
    std::vector<KeyValuePair> data;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 1000000);
    
    for (size_t i = 0; i < size; ++i) {
        data.push_back({static_cast<Key>(dis(gen)), static_cast<Value>(i)});
    }
    
    // Sort by key
    std::sort(data.begin(), data.end(), 
             [](const KeyValuePair& a, const KeyValuePair& b) {
                 return a.key < b.key;
             });
    
    return data;
}

// Simple binary search implementation
size_t binarySearch(const std::vector<KeyValuePair>& data, Key key) {
    size_t left = 0, right = data.size();
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (data[mid].key < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    if (left < data.size() && data[left].key == key) {
        return left;
    }
    return data.size();
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "  PHASE 2: BINARY SEARCH LATENCY BENCHMARK\n";
    std::cout << "  Testing Search Performance\n";
    std::cout << "================================================================================\n\n";

    std::vector<size_t> test_sizes = {10000, 50000, 100000, 500000, 1000000};
    
    for (size_t size : test_sizes) {
        std::cout << "Testing size: " << std::setw(7) << size << " keys... ";
        std::cout.flush();
        
        // Generate data
        auto data = generateData(size);
        
        // Build index (just create the sorted array)
        auto start_build = std::chrono::high_resolution_clock::now();
        // Data is already sorted, so "building" is instant
        auto end_build = std::chrono::high_resolution_clock::now();
        auto build_ms = std::chrono::duration<double, std::milli>(end_build - start_build).count();
        
        // Select test keys (stratified sampling)
        std::vector<Key> test_keys;
        for (size_t i = 0; i < std::min(size_t(5000), size); i += std::max(size_t(1), size / 5000)) {
            test_keys.push_back(data[i].key);
        }
        
        // Benchmark searches
        std::vector<long long> latencies;
        latencies.reserve(test_keys.size());
        
        size_t hits = 0;
        
        for (const auto& test_key : test_keys) {
            // Warm-up (don't measure)
            volatile auto dummy = binarySearch(data, test_key);
            (void)dummy;
            
            // Actual measurement
            auto t1 = std::chrono::high_resolution_clock::now();
            size_t pos = binarySearch(data, test_key);
            auto t2 = std::chrono::high_resolution_clock::now();
            
            long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            latencies.push_back(ns);
            
            if (pos < data.size()) {
                hits++;
            }
        }
        
        // Calculate statistics
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            
            long long p50 = latencies[latencies.size() / 2];
            long long p95 = latencies[(latencies.size() * 95) / 100];
            long long p99 = latencies[(latencies.size() * 99) / 100];
            
            long long sum = 0;
            for (auto l : latencies) {
                sum += l;
            }
            long long avg = sum / latencies.size();
            
            double accuracy = (100.0 * hits) / test_keys.size();
            
            std::cout << "Done\n";
            std::cout << "  Build:   " << std::fixed << std::setprecision(1) << build_ms << " ms\n";
            std::cout << "  P50:     " << p50 << " ns\n";
            std::cout << "  P95:     " << p95 << " ns\n";
            std::cout << "  P99:     " << p99 << " ns\n";
            std::cout << "  Average: " << avg << " ns\n";
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(1) << accuracy << "%\n";
            std::cout << "\n";
        }
    }
    
    std::cout << "================================================================================\n";
    std::cout << "Benchmark completed successfully!\n";
    std::cout << "================================================================================\n\n";
    
    return 0;
}
