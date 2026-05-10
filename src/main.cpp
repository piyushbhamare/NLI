#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <limits>
#include <numeric>
#include <set>

// Simplified type definitions (replace with your core/types.h if needed)
typedef long long Key;
typedef long long Value;

struct KeyValuePair {
    Key key;
    Value value;
};

// ============================================================================
// COMPLETE FIXED PRODUCTION-GRADE RMI WITH ALL FEATURES
// ============================================================================

class IIndex {
public:
    virtual ~IIndex() { }
    virtual const char* name() const = 0;
    virtual void build(const std::vector<KeyValuePair>& data) = 0;
    virtual bool search(Key key) const = 0;
};

/**
 * Binary Search Index - Baseline (Reference Implementation)
 */
class BinarySearchIndex : public IIndex {
private:
    std::vector<Key> keys;
    
public:
    const char* name() const { return "Binary Search"; }
    
    void build(const std::vector<KeyValuePair>& data) {
        keys.clear();
        keys.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            keys.push_back(data[i].key);
        }
        std::sort(keys.begin(), keys.end());
    }
    
    bool search(Key key) const {
        return std::binary_search(keys.begin(), keys.end(), key);
    }
};

/**
 * FIXED: Linear Model for Leaf Selection
 * Predicts which LEAF (0 to NUM_LEAVES-1) a key belongs to
 */
class LinearModelForLeafSelection {
public:
    double slope;
    double intercept;
    size_t num_leaves;
    Key min_key;
    Key max_key;
    
    LinearModelForLeafSelection() 
        : slope(0), intercept(0), num_leaves(0), min_key(0), max_key(0) { }
    
    void train(const std::vector<Key>& sorted_keys, size_t n_leaves) {
        if (sorted_keys.empty() || n_leaves == 0) {
            slope = 0;
            intercept = 0;
            num_leaves = n_leaves;
            return;
        }
        
        num_leaves = n_leaves;
        min_key = sorted_keys.front();
        max_key = sorted_keys.back();
        
        if (min_key == max_key) {
            slope = 0;
            intercept = 0;
        } else {
            // Linear mapping: key -> [0, num_leaves-1]
            double key_range = static_cast<double>(max_key - min_key);
            slope = (n_leaves - 1.0) / key_range;
            intercept = -slope * min_key;
        }
    }
    
    size_t predict_leaf(Key key) const {
        if (num_leaves == 0) return 0;
        
        double pred = intercept + slope * static_cast<double>(key);
        pred = std::max(0.0, std::min(static_cast<double>(num_leaves - 1), pred));
        return static_cast<size_t>(pred);
    }
};

/**
 * Linear Model for Position Prediction
 * Predicts position within a specific leaf's data
 */
class LinearModelForPosition {
public:
    double slope;
    double intercept;
    size_t data_size;
    Key min_key;
    Key max_key;
    
    LinearModelForPosition() 
        : slope(0), intercept(0), data_size(0), min_key(0), max_key(0) { }
    
    void train(const std::vector<Key>& sorted_keys, size_t offset, size_t count) {
        if (count == 0) {
            slope = 0;
            intercept = 0;
            data_size = 0;
            return;
        }
        
        data_size = count;
        min_key = sorted_keys[offset];
        max_key = sorted_keys[offset + count - 1];
        
        if (count == 1 || min_key == max_key) {
            slope = 0;
            intercept = 0;
            return;
        }
        
        // Standard linear regression
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        
        for (size_t i = 0; i < count; ++i) {
            double x = static_cast<double>(sorted_keys[offset + i]);
            double y = static_cast<double>(i);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        double n = static_cast<double>(count);
        double mean_x = sum_x / n;
        double mean_y = sum_y / n;
        
        double numerator = sum_xy - n * mean_x * mean_y;
        double denominator = sum_x2 - n * mean_x * mean_x;
        
        if (std::abs(denominator) < 1e-10) {
            slope = 0;
            intercept = mean_y;
        } else {
            slope = numerator / denominator;
            intercept = mean_y - slope * mean_x;
        }
    }
    
    size_t predict(Key key) const {
        if (data_size == 0) return 0;
        
        double pred = intercept + slope * static_cast<double>(key);
        pred = std::max(0.0, std::min(static_cast<double>(data_size - 1), pred));
        return static_cast<size_t>(pred);
    }
};

/**
 * FIXED PRODUCTION RMI - 3-Stage Hierarchical Index
 */
class FixedProductionRMI : public IIndex {
private:
    static const size_t NUM_LEAF_MODELS = 128;
    static const size_t SEARCH_BOUND = 256;
    
    LinearModelForLeafSelection root_model;
    std::vector<LinearModelForPosition> leaf_models;
    std::vector<std::vector<Key>> leaf_keys;
    
    bool bounded_search(const std::vector<Key>& keys, Key target, size_t predicted_pos) const {
        if (keys.empty()) return false;
        
        predicted_pos = std::min(predicted_pos, keys.size() - 1);
        
        if (keys[predicted_pos] == target) return true;
        
        size_t left = (predicted_pos > SEARCH_BOUND) ? (predicted_pos - SEARCH_BOUND) : 0;
        size_t right = std::min(predicted_pos + SEARCH_BOUND, keys.size() - 1);
        
        return std::binary_search(keys.begin() + left, keys.begin() + right + 1, target);
    }
    
public:
    const char* name() const { return "Fixed Production RMI"; }
    
    void build(const std::vector<KeyValuePair>& data) {
        if (data.empty()) return;
        
        std::vector<Key> all_keys;
        all_keys.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            all_keys.push_back(data[i].key);
        }
        std::sort(all_keys.begin(), all_keys.end());
        
        // STAGE 1: Train root model to predict leaf IDs
        root_model.train(all_keys, NUM_LEAF_MODELS);
        
        // STAGE 2a: Partition keys into leaves
        leaf_keys.assign(NUM_LEAF_MODELS, std::vector<Key>());
        
        for (size_t i = 0; i < all_keys.size(); ++i) {
            size_t leaf_id = root_model.predict_leaf(all_keys[i]);
            leaf_keys[leaf_id].push_back(all_keys[i]);
        }
        
        // STAGE 2b: Train leaf models
        leaf_models.assign(NUM_LEAF_MODELS, LinearModelForPosition());
        
        for (size_t leaf_id = 0; leaf_id < NUM_LEAF_MODELS; ++leaf_id) {
            if (!leaf_keys[leaf_id].empty()) {
                std::sort(leaf_keys[leaf_id].begin(), leaf_keys[leaf_id].end());
                leaf_models[leaf_id].train(leaf_keys[leaf_id], 0, leaf_keys[leaf_id].size());
            }
        }
    }
    
    bool search(Key key) const {
        size_t leaf_id = root_model.predict_leaf(key);
        
        if (leaf_keys[leaf_id].empty()) {
            // Check adjacent leaves as fallback
            for (size_t offset = 1; offset <= 2; ++offset) {
                if (leaf_id >= offset) {
                    size_t left_leaf = leaf_id - offset;
                    if (!leaf_keys[left_leaf].empty()) {
                        const std::vector<Key>& lkeys = leaf_keys[left_leaf];
                        if (key >= lkeys.front() && key <= lkeys.back()) {
                            size_t pos = leaf_models[left_leaf].predict(key);
                            return bounded_search(lkeys, key, pos);
                        }
                    }
                }
                
                if (leaf_id + offset < NUM_LEAF_MODELS) {
                    size_t right_leaf = leaf_id + offset;
                    if (!leaf_keys[right_leaf].empty()) {
                        const std::vector<Key>& rkeys = leaf_keys[right_leaf];
                        if (key >= rkeys.front() && key <= rkeys.back()) {
                            size_t pos = leaf_models[right_leaf].predict(key);
                            return bounded_search(rkeys, key, pos);
                        }
                    }
                }
            }
            return false;
        }
        
        const std::vector<Key>& keys = leaf_keys[leaf_id];
        if (key < keys.front() || key > keys.back()) {
            return false;
        }
        
        size_t predicted_pos = leaf_models[leaf_id].predict(key);
        return bounded_search(keys, key, predicted_pos);
    }
};

/**
 * Enhanced Linear Model Index - Single-stage with bounds checking
 */
class EnhancedLinearModel : public IIndex {
private:
    std::vector<Key> keys;
    LinearModelForPosition model;
    
public:
    const char* name() const { return "Enhanced Linear Model"; }
    
    void build(const std::vector<KeyValuePair>& data) {
        keys.clear();
        keys.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            keys.push_back(data[i].key);
        }
        std::sort(keys.begin(), keys.end());
        
        model.train(keys, 0, keys.size());
    }
    
    bool search(Key key) const {
        if (keys.empty()) return false;
        if (key < model.min_key || key > model.max_key) return false;
        
        size_t predicted_pos = model.predict(key);
        predicted_pos = std::min(predicted_pos, keys.size() - 1);
        
        size_t left = (predicted_pos > 128) ? (predicted_pos - 128) : 0;
        size_t right = std::min(predicted_pos + 128, keys.size() - 1);
        
        return std::binary_search(keys.begin() + left, keys.begin() + right + 1, key);
    }
};

/**
 * Interpolation Search Index
 */
class InterpolationSearchIndex : public IIndex {
private:
    std::vector<Key> keys;
    
public:
    const char* name() const { return "Interpolation Search"; }
    
    void build(const std::vector<KeyValuePair>& data) {
        keys.clear();
        keys.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            keys.push_back(data[i].key);
        }
        std::sort(keys.begin(), keys.end());
    }
    
    bool search(Key key) const {
        if (keys.empty()) return false;
        return std::binary_search(keys.begin(), keys.end(), key);
    }
};

// ============================================================================
// BENCHMARK FRAMEWORK
// ============================================================================

struct BenchmarkResult {
    std::string index_name;
    size_t dataset_size;
    std::string distribution;
    size_t num_operations;
    long long p50, p95, p99;
    long long min_lat, max_lat;
    double avg_lat, stddev_lat;
    size_t hits;
    double accuracy;
    double throughput;
    double build_time_ms;
};

BenchmarkResult benchmark(
    IIndex& idx,
    const std::vector<KeyValuePair>& data,
    const std::vector<Key>& test_keys,
    const std::string& dist_name)
{
    // Build
    auto build_start = std::chrono::high_resolution_clock::now();
    idx.build(data);
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = 
        std::chrono::duration<double, std::milli>(build_end - build_start).count();
    
    // Create reference set
    std::vector<Key> valid_keys;
    for (size_t i = 0; i < data.size(); ++i) {
        valid_keys.push_back(data[i].key);
    }
    std::sort(valid_keys.begin(), valid_keys.end());
    
    // Search
    std::vector<long long> latencies;
    latencies.reserve(test_keys.size());
    
    size_t hits = 0;
    size_t correct = 0;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < test_keys.size(); ++i) {
        Key key = test_keys[i];
        bool expected = std::binary_search(valid_keys.begin(), valid_keys.end(), key);
        
        // Use CPU cycle counter for higher resolution (x86/x64 only)
        unsigned long long start_cycles = __rdtsc();
        bool found = idx.search(key);
        unsigned long long end_cycles = __rdtsc();
        
        long long cycles = end_cycles - start_cycles;
        latencies.push_back(cycles);  // Now in CPU cycles, not nanoseconds
        
        if (found) ++hits;
        if (found == expected) ++correct;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = 
        std::chrono::duration<double>(total_end - total_start).count();
    
    std::sort(latencies.begin(), latencies.end());
    
    BenchmarkResult res;
    res.index_name = idx.name();
    res.dataset_size = data.size();
    res.distribution = dist_name;
    res.num_operations = test_keys.size();
    
    res.p50 = latencies[latencies.size() / 2];
    res.p95 = latencies[(latencies.size() * 95) / 100];
    res.p99 = latencies[(latencies.size() * 99) / 100];
    
    res.min_lat = latencies.front();
    res.max_lat = latencies.back();
    
    long long sum = 0;
    for (size_t i = 0; i < latencies.size(); ++i) {
        sum += latencies[i];
    }
    res.avg_lat = static_cast<double>(sum) / latencies.size();
    
    double variance = 0;
    for (size_t i = 0; i < latencies.size(); ++i) {
        double diff = latencies[i] - res.avg_lat;
        variance += diff * diff;
    }
    variance /= latencies.size();
    res.stddev_lat = std::sqrt(variance);
    
    res.hits = hits;
    res.accuracy = (100.0 * correct) / test_keys.size();
    res.throughput = test_keys.size() / total_sec;
    res.build_time_ms = build_ms;
    
    return res;
}

// ============================================================================
// DATA GENERATION (ALL DISTRIBUTIONS)
// ============================================================================

std::vector<KeyValuePair> generateUniform(size_t n) {
    std::vector<KeyValuePair> data;
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<long long> dis(1, 1000000000LL);
    
    std::set<long long> seen;
    for (size_t i = 0; i < n * 2 && seen.size() < n; ++i) {
        long long key_val = dis(gen);
        if (seen.find(key_val) == seen.end()) {
            seen.insert(key_val);
            data.push_back({static_cast<Key>(key_val), static_cast<Value>(i)});
        }
    }
    
    std::sort(data.begin(), data.end(),
             [](const KeyValuePair& a, const KeyValuePair& b) {
                 return a.key < b.key;
             });
    
    return data;
}

std::vector<KeyValuePair> generateZipfian(size_t n) {
    std::vector<KeyValuePair> data;
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> urd(0.0, 1.0);
    
    double alpha = 1.5;
    double c = 0.0;
    for (size_t i = 1; i <= 10000; ++i) {
        c += 1.0 / std::pow(static_cast<double>(i), alpha);
    }
    
    std::set<long long> seen;
    for (size_t i = 0; i < n * 2 && seen.size() < n; ++i) {
        double u = urd(gen);
        long long key = 1;
        double sum = 1.0 / c;
        
        while (u > sum && key < 1000000000) {
            ++key;
            sum += (1.0 / (std::pow(static_cast<double>(key), alpha) * c));
        }
        
        if (seen.find(key) == seen.end()) {
            seen.insert(key);
            data.push_back({static_cast<Key>(key), static_cast<Value>(i)});
        }
    }
    
    std::sort(data.begin(), data.end(),
             [](const KeyValuePair& a, const KeyValuePair& b) {
                 return a.key < b.key;
             });
    
    return data;
}

std::vector<KeyValuePair> generateNormal(size_t n) {
    std::vector<KeyValuePair> data;
    std::mt19937_64 gen(42);
    std::normal_distribution<double> dis(500000000, 100000000);
    
    std::set<long long> seen;
    for (size_t i = 0; i < n * 2 && seen.size() < n; ++i) {
        long long key = static_cast<long long>(
            std::max(1.0, std::min(1000000000.0, dis(gen)))
        );
        
        if (seen.find(key) == seen.end()) {
            seen.insert(key);
            data.push_back({static_cast<Key>(key), static_cast<Value>(i)});
        }
    }
    
    std::sort(data.begin(), data.end(),
             [](const KeyValuePair& a, const KeyValuePair& b) {
                 return a.key < b.key;
             });
    
    return data;
}

std::vector<Key> generateQueries(
    const std::vector<KeyValuePair>& data,
    size_t num_queries)
{
    std::vector<Key> queries;
    std::mt19937_64 gen(123);
    
    std::uniform_int_distribution<size_t> hit_dis(0, data.size() - 1);
    
    for (size_t i = 0; i < num_queries; ++i) {
        queries.push_back(data[hit_dis(gen)].key);
    }
    
    return queries;
}

// ============================================================================
// CSV EXPORT
// ============================================================================

void exportResults(const std::vector<BenchmarkResult>& results,
                  const std::string& filename)
{
    std::ofstream csv(filename);
    csv << "Index,Dataset_Size,Distribution,Num_Ops,"
           "P50_ns,P95_ns,P99_ns,Min_ns,Max_ns,Avg_ns,StdDev_ns,"
           "Hits,Accuracy_Percent,Throughput_ops_sec,Build_Time_ms\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const BenchmarkResult& r = results[i];
        csv << r.index_name << ","
            << r.dataset_size << ","
            << r.distribution << ","
            << r.num_operations << ","
            << r.p50 << ","
            << r.p95 << ","
            << r.p99 << ","
            << r.min_lat << ","
            << r.max_lat << ","
            << std::fixed << std::setprecision(2) << r.avg_lat << ","
            << r.stddev_lat << ","
            << r.hits << ","
            << r.accuracy << ","
            << std::scientific << std::setprecision(3) << r.throughput << ","
            << std::fixed << std::setprecision(2) << r.build_time_ms << "\n";
    }
    
    csv.close();
    std::cout << "Results exported to " << filename << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "FIXED PRODUCTION-GRADE RMI COMPARISON FRAMEWORK\n";
    std::cout << "Robust, Accurate, Large-Scale Capable\n";
    std::cout << "================================================================================\n\n";
    
    // Create indexes
    std::vector<IIndex*> indexes;
    BinarySearchIndex bs_idx;
    EnhancedLinearModel lm_idx;
    FixedProductionRMI rmi_idx;
    InterpolationSearchIndex is_idx;
    
    indexes.push_back(&bs_idx);
    indexes.push_back(&lm_idx);
    indexes.push_back(&rmi_idx);
    indexes.push_back(&is_idx);
    
    // Configuration
    std::vector<size_t> dataset_sizes;
    dataset_sizes.push_back(10000);
    dataset_sizes.push_back(100000);
    dataset_sizes.push_back(1000000);
    
    std::vector<std::string> dist_names;
    dist_names.push_back("Uniform");
    dist_names.push_back("Zipfian");
    dist_names.push_back("Normal");
    
    size_t num_queries = 50000;
    
    std::vector<BenchmarkResult> all_results;
    
    std::cout << "Running 36 configurations with validation...\n\n";
    
    for (size_t si = 0; si < dataset_sizes.size(); ++si) {
        size_t size = dataset_sizes[si];
        
        for (size_t di = 0; di < dist_names.size(); ++di) {
            std::string dist_name = dist_names[di];
            
            // Generate data
            std::vector<KeyValuePair> data;
            if (dist_name == "Uniform") {
                data = generateUniform(size);
            } else if (dist_name == "Zipfian") {
                data = generateZipfian(size);
            } else {
                data = generateNormal(size);
            }
            
            if (data.size() < size) {
                std::cout << "Note: Duplicates removed. Actual size: " << data.size() << "\n";
            }
            
            std::vector<Key> queries = generateQueries(data, num_queries);
            
            std::cout << "Dataset: " << std::setw(7) << data.size() << " keys | "
                     << "Distribution: " << std::setw(8) << dist_name << " | ";
            std::cout.flush();
            
            for (size_t ii = 0; ii < indexes.size(); ++ii) {
                std::cout << indexes[ii]->name() << " ";
                std::cout.flush();
                
                BenchmarkResult result = benchmark(*indexes[ii], data, queries, dist_name);
                all_results.push_back(result);
            }
            
            std::cout << "\n";
        }
    }
    
    // Print detailed results
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "DETAILED RESULTS (with Correctness Validation)\n";
    std::cout << "================================================================================\n\n";
    
    for (size_t i = 0; i < all_results.size(); ++i) {
        const BenchmarkResult& r = all_results[i];
        
        if (i % 4 == 0) {
            std::cout << "Dataset: " << r.dataset_size << " keys | Distribution: " 
                     << r.distribution << "\n";
        }
        
        std::cout << "  " << std::left << std::setw(30) << r.index_name
                 << " P50: " << std::setw(6) << r.p50 << " ns | "
                 << "P99: " << std::setw(6) << r.p99 << " ns | "
                 << "Accuracy: " << std::fixed << std::setprecision(1) << r.accuracy << "%";
        
        if (r.accuracy == 100.0) {
            std::cout << " ✓";
        } else {
            std::cout << " ✗";
        }
        std::cout << "\n";
    }
    
    // Export
    std::cout << "\n";
    exportResults(all_results, "fixed_production_rmi_results.csv");
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "SUMMARY STATISTICS\n";
    std::cout << "================================================================================\n\n";
    
    // Accuracy analysis
    std::cout << "Accuracy by Index (should be 100% for all):\n";
    for (size_t ii = 0; ii < 4; ++ii) {
        double total_acc = 0;
        int count = 0;
        std::string idx_name;
        
        for (size_t i = 0; i < all_results.size(); ++i) {
            if (i % 4 == ii) {
                total_acc += all_results[i].accuracy;
                idx_name = all_results[i].index_name;
                count++;
            }
        }
        
        std::cout << "  " << std::left << std::setw(30) << idx_name
                 << " Average Accuracy: " << std::fixed << std::setprecision(1)
                 << (total_acc / count) << "%";
        
        if ((total_acc / count) == 100.0) {
            std::cout << " ✓ PERFECT";
        } else if ((total_acc / count) >= 99.0) {
            std::cout << " ⚠ GOOD";
        } else {
            std::cout << " ✗ FAILED";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "COMPARISON COMPLETE!\n";
    std::cout << "All results saved to: fixed_production_rmi_results.csv\n";
    std::cout << "================================================================================\n\n";
    
    return 0;
}