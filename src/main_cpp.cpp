// src/main.cpp
/**
 * @file main.cpp
 * @brief Main entry point for Neural-Enhanced Hybrid Learned Index
 * 
 * This program demonstrates and benchmarks the learned index implementation
 * against traditional B+ tree baseline.
 */

#include "core/types.h"
#include "core/sorted_array.h"
#include "baseline/linear_model.h"
#include "baseline/simple_learned_index.h"
#include "baseline/btree.h"
#include "utils/benchmark.h"
#include "utils/statistics.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace learned_index;

/**
 * @brief Generate synthetic dataset
 */
std::vector<KeyValuePair> generateData(size_t size, bool uniform = true) {
    std::vector<KeyValuePair> data;
    data.reserve(size);
    
    std::mt19937 gen(42);
    
    if (uniform) {
        std::uniform_int_distribution<Key> dis(0, UINT64_MAX);
        for (size_t i = 0; i < size; i++) {
            data.emplace_back(dis(gen), i);
        }
    } else {
        // Skewed distribution (Zipf-like)
        for (size_t i = 0; i < size; i++) {
            data.emplace_back(i * i % UINT64_MAX, i);
        }
    }
    
    return data;
}

/**
 * @brief Benchmark lookup operations
 */
void benchmarkLookups(const std::vector<KeyValuePair>& data,
                     size_t num_queries = 100000) {
    std::cout << "\n=== Lookup Benchmark ===" << std::endl;
    std::cout << "Dataset size: " << data.size() << std::endl;
    std::cout << "Queries: " << num_queries << std::endl;
    
    // Prepare query keys
    std::vector<Key> query_keys;
    query_keys.reserve(num_queries);
    std::mt19937 gen(123);
    std::uniform_int_distribution<size_t> dis(0, data.size() - 1);
    
    for (size_t i = 0; i < num_queries; i++) {
        query_keys.push_back(data[dis(gen)].key);
    }
    
    // B+ Tree
    std::cout << "\n--- B+ Tree ---" << std::endl;
    BTree btree;
    btree.bulkLoad(data);
    
    auto btree_samples = Benchmark::profile([&]() {
        volatile auto result = btree.lookup(query_keys[0]);
        (void)result;
    }, num_queries);
    
    std::cout << "P50: " << std::fixed << std::setprecision(2) 
              << Benchmark::percentile(btree_samples, 50) << " ns" << std::endl;
    std::cout << "P95: " << Benchmark::percentile(btree_samples, 95) << " ns" << std::endl;
    std::cout << "P99: " << Benchmark::percentile(btree_samples, 99) << " ns" << std::endl;
    
    double btree_avg = 0;
    for (auto s : btree_samples) btree_avg += s;
    btree_avg /= btree_samples.size();
    std::cout << "Avg: " << btree_avg << " ns" << std::endl;
    
    // Learned Index
    std::cout << "\n--- Simple Learned Index ---" << std::endl;
    SimpleLearnedIndex learned_idx;
    learned_idx.bulkLoad(std::vector<KeyValuePair>(data));
    
    auto learned_samples = Benchmark::profile([&]() {
        volatile auto result = learned_idx.lookup(query_keys[0]);
        (void)result;
    }, num_queries);
    
    std::cout << "P50: " << Benchmark::percentile(learned_samples, 50) << " ns" << std::endl;
    std::cout << "P95: " << Benchmark::percentile(learned_samples, 95) << " ns" << std::endl;
    std::cout << "P99: " << Benchmark::percentile(learned_samples, 99) << " ns" << std::endl;
    
    double learned_avg = 0;
    for (auto s : learned_samples) learned_avg += s;
    learned_avg /= learned_samples.size();
    std::cout << "Avg: " << learned_avg << " ns" << std::endl;
    
    std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) 
              << (btree_avg / learned_avg) << "x" << std::endl;
    
    std::cout << "\n--- Model Statistics ---" << std::endl;
    std::cout << "MAE: " << learned_idx.avgError() << std::endl;
    std::cout << "Max Error: " << learned_idx.maxError() << std::endl;
    std::cout << "RMSE: " << learned_idx.rmse() << std::endl;
    std::cout << "R²: " << learned_idx.rSquared() << std::endl;
}

/**
 * @brief Correctness test
 */
void testCorrectness() {
    std::cout << "\n=== Correctness Test ===" << std::endl;
    
    auto data = generateData(10000);
    SimpleLearnedIndex idx;
    idx.bulkLoad(data);
    
    int correct = 0;
    for (const auto& kv : data) {
        auto result = idx.lookup(kv.key);
        if (result.found && result.value == kv.value) {
            correct++;
        }
    }
    
    std::cout << "Correct lookups: " << correct << " / " << data.size() << std::endl;
    std::cout << "Accuracy: " << (100.0 * correct / data.size()) << "%" << std::endl;
}

int main() {
    std::cout << "Neural-Enhanced Hybrid Learned Index - Phase 1" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    // Run tests
    testCorrectness();
    
    // Small benchmark
    auto data = generateData(100000);
    benchmarkLookups(data, 50000);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
