#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "core/types.h"
#include "utils/statistics.h"

using namespace learned_index;

// ============================================================================
// PHASE 2: COMPREHENSIVE BENCHMARKING SUITE
// Large datasets, detailed metrics, research-grade output
// ============================================================================

/**
 * Data generation with multiple distributions
 */
enum DataDistribution {
    UNIFORM,
    ZIPFIAN,
    NORMAL
};

std::vector<KeyValuePair> generateData(size_t size, DataDistribution dist) {
    std::vector<KeyValuePair> data;
    std::mt19937 gen(42);
    
    if (dist == UNIFORM) {
        std::uniform_int_distribution<> dis(1, 1000000000);
        for (size_t i = 0; i < size; ++i) {
            data.push_back({static_cast<Key>(dis(gen)), static_cast<Value>(i)});
        }
    } else if (dist == ZIPFIAN) {
        // Zipfian distribution (skewed)
        std::uniform_real_distribution<> urd(0.0, 1.0);
        double alpha = 1.5;
        double c = 0.0;
        for (size_t i = 1; i <= 10000; ++i) {
            c += 1.0 / std::pow(i, alpha);
        }
        for (size_t i = 0; i < size; ++i) {
            double u = urd(gen);
            Key key = 1;
            double sum = 1.0 / c;
            while (u > sum && key < 1000000000) {
                ++key;
                sum += (1.0 / (std::pow(key, alpha) * c));
            }
            data.push_back({key, static_cast<Value>(i)});
        }
    } else { // NORMAL
        std::normal_distribution<> dis(500000000, 100000000);
        for (size_t i = 0; i < size; ++i) {
            Key key = static_cast<Key>(std::max(1.0, std::min(1000000000.0, dis(gen))));
            data.push_back({key, static_cast<Value>(i)});
        }
    }
    
    std::sort(data.begin(), data.end(), 
             [](const KeyValuePair& a, const KeyValuePair& b) {
                 return a.key < b.key;
             });
    
    return data;
}

/**
 * Binary search implementation
 */
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

/**
 * Detailed benchmark result
 */
struct DetailedResult {
    size_t dataset_size;
    std::string distribution;
    size_t num_operations;
    long long p50, p95, p99, p999;
    long long min_latency, max_latency;
    double avg_latency;
    double stddev_latency;
    size_t correct_hits;
    double accuracy;
    double throughput_ops_per_sec;
};

/**
 * Calculate statistics
 */
DetailedResult benchmarkSearch(const std::vector<KeyValuePair>& data, 
                               const std::string& dist_name,
                               size_t num_ops) {
    // Select test keys uniformly
    std::vector<Key> test_keys;
    for (size_t i = 0; i < data.size(); i += std::max(size_t(1), data.size() / num_ops)) {
        test_keys.push_back(data[i].key);
    }
    if (test_keys.size() > num_ops) {
        test_keys.resize(num_ops);
    }
    
    std::vector<long long> latencies;
    latencies.reserve(test_keys.size());
    
    size_t hits = 0;
    
    auto start_total = std::chrono::high_resolution_clock::now();
    
    for (const auto& key : test_keys) {
        // Warm-up
        volatile auto dummy = binarySearch(data, key);
        (void)dummy;
        
        // Measure
        auto t1 = std::chrono::high_resolution_clock::now();
        size_t pos = binarySearch(data, key);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        latencies.push_back(ns);
        
        if (pos < data.size()) hits++;
    }
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto total_time_sec = std::chrono::duration<double>(end_total - start_total).count();
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    
    DetailedResult res;
    res.dataset_size = data.size();
    res.distribution = dist_name;
    res.num_operations = test_keys.size();
    
    res.p50 = latencies[latencies.size() / 2];
    res.p95 = latencies[(latencies.size() * 95) / 100];
    res.p99 = latencies[(latencies.size() * 99) / 100];
    res.p999 = latencies[(latencies.size() * 999) / 1000];
    
    res.min_latency = latencies[0];
    res.max_latency = latencies.back();
    
    long long sum = 0;
    for (auto l : latencies) sum += l;
    res.avg_latency = static_cast<double>(sum) / latencies.size();
    
    // Standard deviation
    double variance = 0;
    for (auto l : latencies) {
        double diff = l - res.avg_latency;
        variance += diff * diff;
    }
    variance /= latencies.size();
    res.stddev_latency = std::sqrt(variance);
    
    res.correct_hits = hits;
    res.accuracy = (100.0 * hits) / test_keys.size();
    res.throughput_ops_per_sec = test_keys.size() / total_time_sec;
    
    return res;
}

/**
 * Print detailed results
 */
void printDetailedResults(const std::vector<DetailedResult>& results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "DETAILED LATENCY ANALYSIS\n";
    std::cout << "================================================================================\n\n";
    
    for (const auto& r : results) {
        std::cout << std::left
                 << "Dataset Size: " << r.dataset_size << " keys\n"
                 << "Distribution: " << r.distribution << "\n"
                 << "Operations:   " << r.num_operations << "\n"
                 << "Latency (ns):   Min=" << std::setw(8) << r.min_latency
                 << " P50=" << std::setw(8) << r.p50
                 << " P95=" << std::setw(8) << r.p95
                 << " P99=" << std::setw(8) << r.p99
                 << " P99.9=" << std::setw(8) << r.p999
                 << " Max=" << std::setw(8) << r.max_latency << "\n"
                 << "Statistics:   Avg=" << std::fixed << std::setprecision(2) << r.avg_latency
                 << " ns, StdDev=" << r.stddev_latency << " ns\n"
                 << "Accuracy:     " << r.accuracy << "%\n"
                 << "Throughput:   " << std::scientific << r.throughput_ops_per_sec 
                 << " ops/sec\n"
                 << "\n";
    }
}

/**
 * Export to CSV
 */
void exportToCSV(const std::vector<DetailedResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "Dataset_Size,Distribution,Operations,P50_ns,P95_ns,P99_ns,P999_ns,"
         << "Min_ns,Max_ns,Avg_ns,StdDev_ns,Accuracy_Percent,Throughput_ops_sec\n";
    
    for (const auto& r : results) {
        file << r.dataset_size << "," << r.distribution << "," << r.num_operations << ","
             << r.p50 << "," << r.p95 << "," << r.p99 << "," << r.p999 << ","
             << r.min_latency << "," << r.max_latency << "," 
             << std::fixed << std::setprecision(2) << r.avg_latency << ","
             << r.stddev_latency << "," << r.accuracy << ","
             << std::scientific << r.throughput_ops_per_sec << "\n";
    }
    
    file.close();
    std::cout << "Results exported to " << filename << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "PHASE 2: COMPREHENSIVE BINARY SEARCH BENCHMARKING\n";
    std::cout << "Research-Grade Performance Analysis\n";
    std::cout << "================================================================================\n\n";

    std::vector<DetailedResult> all_results;
    
    // Test configurations
    std::vector<size_t> dataset_sizes = {
        10000,      // 10K
        100000,     // 100K
        1000000,    // 1M
        10000000    // 10M
    };
    
    std::vector<std::pair<DataDistribution, std::string>> distributions = {
        {UNIFORM, "Uniform"},
        {ZIPFIAN, "Zipfian"},
        {NORMAL, "Normal"}
    };
    
    size_t config_count = 0;
    size_t total_configs = dataset_sizes.size() * distributions.size();
    
    std::cout << "Testing " << total_configs << " configurations...\n\n";
    
    for (auto size : dataset_sizes) {
        for (const auto& [dist, dist_name] : distributions) {
            config_count++;
            
            std::cout << "[" << config_count << "/" << total_configs << "] ";
            std::cout << "Size: " << std::setw(8) << size 
                     << " | Distribution: " << std::setw(8) << dist_name << " ... ";
            std::cout.flush();
            
            // Generate data
            auto data = generateData(size, dist);
            
            // Benchmark
            size_t num_operations = std::min(size_t(10000), size);
            auto result = benchmarkSearch(data, dist_name, num_operations);
            all_results.push_back(result);
            
            std::cout << "P50=" << result.p50 << "ns | Acc=" 
                     << std::fixed << std::setprecision(1) << result.accuracy << "%\n";
        }
    }
    
    // Print detailed results
    printDetailedResults(all_results);
    
    // Export to CSV
    exportToCSV(all_results, "benchmark_results.csv");
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "KEY FINDINGS\n";
    std::cout << "================================================================================\n";
    std::cout << "✓ Benchmarked " << all_results.size() << " configurations\n";
    std::cout << "✓ Tested dataset sizes from 10K to 10M keys\n";
    std::cout << "✓ Analyzed 3 distributions (Uniform, Zipfian, Normal)\n";
    std::cout << "✓ Collected latency percentiles (P50, P95, P99, P99.9)\n";
    std::cout << "✓ Measured throughput and accuracy\n";
    std::cout << "✓ Results exported to CSV for analysis\n";
    std::cout << "================================================================================\n\n";
    
    return 0;
}
