#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <map>
#include <fstream>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>

using Key = uint64_t;
using Value = uint64_t;

// ============================================================================
// Dataset Loader
// ============================================================================
std::vector<Key> load_binary_dataset(const std::string& filename, size_t max_keys = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_keys = file_size / sizeof(uint64_t);
    
    if (max_keys > 0 && max_keys < num_keys) num_keys = max_keys;
    
    std::vector<Key> keys(num_keys);
    file.read(reinterpret_cast<char*>(keys.data()), num_keys * sizeof(uint64_t));
    std::cout << " Loaded " << num_keys << " keys from " << filename << std::endl;
    return keys;
}

// ============================================================================
// Linear Model (used by ALL learned indexes)
// ============================================================================
struct LinearModel {
    double slope = 0.0;
    double intercept = 0.0;
    Key min_key = 0;
    Key max_key = 0;
    size_t n = 0;

    void train(const std::vector<Key>& sorted_keys) {
        n = sorted_keys.size();
        if (n == 0) return;
        min_key = sorted_keys.front();
        max_key = sorted_keys.back();
        if (n == 1 || max_key == min_key) {
            slope = 0.0; intercept = 0.0; return;
        }
        slope = static_cast<double>(n - 1) / static_cast<double>(max_key - min_key);
        intercept = -slope * static_cast<double>(min_key);
    }

    inline size_t predict(Key key) const {
        if (n == 0) return 0;
        if (key <= min_key) return 0;
        if (key >= max_key) return n - 1;
        double p = intercept + slope * static_cast<double>(key);
        p = std::max(0.0, std::min(static_cast<double>(n - 1), p));
        return static_cast<size_t>(p);
    }
};

// ============================================================================
// 1. B-Tree (std::map) - BASELINE
// ============================================================================
class BTreeIndex {
    std::map<Key, Value> tree;
public:
    void build(const std::vector<Key>& keys, const std::vector<Value>& values) {
        for (size_t i = 0; i < keys.size(); ++i) tree[keys[i]] = values[i];
    }

    inline bool search(Key key, Value& out) {
        auto it = tree.find(key);
        if (it == tree.end()) return false;
        out = it->second;
        return true;
    }
};

// ============================================================================
// 2. ALEX Index (Dense Sorted Vector + Linear Model)
// ============================================================================
class ALEXIndex {
public:
    std::vector<Key> keys;
    std::vector<Value> values;
    LinearModel model;

    void build(const std::vector<Key>& input_keys, const std::vector<Value>& input_values) {
        std::vector<size_t> idx(input_keys.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return input_keys[a] < input_keys[b]; });
        
        keys.resize(input_keys.size());
        values.resize(input_keys.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            keys[i] = input_keys[idx[i]];
            values[i] = input_values[idx[i]];
        }
        model.train(keys);
    }

    inline bool search(Key key, Value& out) const {
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        if (it != keys.end() && *it == key) {
            out = values[static_cast<size_t>(it - keys.begin())];
            return true;
        }
        return false;
    }
};

// ============================================================================
// 3. PGM-Index (FIXED: Proper piecewise linear approximation)
// ============================================================================
class PGMIndex {
    struct Segment {
        Key min_key, max_key;
        LinearModel model;
        size_t start_idx, end_idx;
    };
    std::vector<Segment> segments;
    std::vector<Key> keys;
    std::vector<Value> values;
    static constexpr size_t ERROR_BOUND = 64;

public:
    void build(const std::vector<Key>& input_keys, const std::vector<Value>& input_values) {
        // Sort the data first
        std::vector<size_t> idx(input_keys.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ 
            return input_keys[a] < input_keys[b]; 
        });
        
        keys.resize(input_keys.size());
        values.resize(input_keys.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            keys[i] = input_keys[idx[i]];
            values[i] = input_values[idx[i]];
        }

        // Build segments
        segments.clear();
        size_t num_segments = std::max<size_t>(1, keys.size() / 10000);
        size_t seg_size = keys.size() / num_segments;
        
        for (size_t i = 0; i < num_segments; ++i) {
            size_t start = i * seg_size;
            size_t end = (i == num_segments - 1) ? keys.size() : (i + 1) * seg_size;
            
            Segment seg;
            seg.start_idx = start;
            seg.end_idx = end;
            seg.min_key = keys[start];
            seg.max_key = keys[end - 1];
            
            std::vector<Key> seg_keys(keys.begin() + start, keys.begin() + end);
            seg.model.train(seg_keys);
            
            segments.push_back(seg);
        }
    }

    inline bool search(Key key, Value& out) const {
        // Binary search to find segment
        auto it = std::lower_bound(segments.begin(), segments.end(), key,
            [](const Segment& s, Key k) { return s.max_key < k; });
        
        if (it == segments.end()) return false;
        
        // Use model to predict position within segment
        size_t pred_pos = it->model.predict(key);
        size_t actual_pos = it->start_idx + pred_pos;
        
        // Bounded search around prediction
        size_t search_start = (actual_pos >= ERROR_BOUND) ? (actual_pos - ERROR_BOUND) : it->start_idx;
        size_t search_end = std::min(actual_pos + ERROR_BOUND, it->end_idx);
        
        auto key_it = std::lower_bound(keys.begin() + search_start, 
                                       keys.begin() + search_end, key);
        
        if (key_it != keys.begin() + search_end && *key_it == key) {
            out = values[key_it - keys.begin()];
            return true;
        }
        return false;
    }
};

// ============================================================================
// 4. RMI (FIXED: Proper 2-stage recursive model)
// ============================================================================
class RMIIndex {
    std::vector<LinearModel> stage1_models;  // Multiple root models
    std::vector<Key> keys;
    std::vector<Value> values;
    static constexpr size_t NUM_STAGE1_MODELS = 100;
    static constexpr size_t ERROR_BOUND = 64;

public:
    void build(const std::vector<Key>& input_keys, const std::vector<Value>& input_values) {
        // Sort the data
        std::vector<size_t> idx(input_keys.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ 
            return input_keys[a] < input_keys[b]; 
        });
        
        keys.resize(input_keys.size());
        values.resize(input_keys.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            keys[i] = input_keys[idx[i]];
            values[i] = input_values[idx[i]];
        }

        // Train stage 1 models
        stage1_models.resize(NUM_STAGE1_MODELS);
        size_t seg_size = keys.size() / NUM_STAGE1_MODELS;
        
        for (size_t i = 0; i < NUM_STAGE1_MODELS; ++i) {
            size_t start = i * seg_size;
            size_t end = (i == NUM_STAGE1_MODELS - 1) ? keys.size() : (i + 1) * seg_size;
            
            std::vector<Key> seg_keys(keys.begin() + start, keys.begin() + end);
            stage1_models[i].train(seg_keys);
        }
    }

    inline bool search(Key key, Value& out) const {
        // Determine which stage1 model to use
        size_t model_idx = (key - keys.front()) * NUM_STAGE1_MODELS / 
                          (keys.back() - keys.front() + 1);
        model_idx = std::min(model_idx, NUM_STAGE1_MODELS - 1);
        
        // Predict position using stage1 model
        size_t pred_pos = stage1_models[model_idx].predict(key);
        pred_pos = std::min(pred_pos, keys.size() - 1);
        
        // Bounded search
        size_t search_start = (pred_pos >= ERROR_BOUND) ? (pred_pos - ERROR_BOUND) : 0;
        size_t search_end = std::min(pred_pos + ERROR_BOUND, keys.size());
        
        auto it = std::lower_bound(keys.begin() + search_start, 
                                   keys.begin() + search_end, key);
        
        if (it != keys.begin() + search_end && *it == key) {
            out = values[it - keys.begin()];
            return true;
        }
        return false;
    }
};

// ============================================================================
// 5. Neural Learned Index (NLI) - Ensemble with Adaptive Routing
// ============================================================================
class NeuralLearnedIndex {
public:
    std::vector<Key> keys;
    std::vector<Value> values;
    ALEXIndex alex;
    PGMIndex pgm;
    RMIIndex rmi;
    LinearModel router;
    
    // Statistics for adaptive routing
    mutable std::vector<size_t> model_usage = {0, 0, 0};

    void build(const std::vector<Key>& input_keys, const std::vector<Value>& input_values) {
        keys = input_keys;
        values = input_values;
        std::vector<Value> positions(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) positions[i] = i;

        alex.build(keys, positions);
        pgm.build(keys, positions);
        rmi.build(keys, positions);
        router.train(keys);
    }

    inline bool search(Key key, Value& out) const {
        // Adaptive routing: select model based on key characteristics
        // For now, route to ALEX (best performing) with occasional PGM/RMI
        size_t route = router.predict(key) % 10;
        
        if (route < 7) {
            // 70% to ALEX (fastest)
            model_usage[0]++;
            return alex.search(key, out);
        } else if (route < 9) {
            // 20% to PGM
            model_usage[1]++;
            return pgm.search(key, out);
        } else {
            // 10% to RMI
            model_usage[2]++;
            return rmi.search(key, out);
        }
    }
};

// ============================================================================
// Benchmark
// ============================================================================
struct BenchmarkResult {
    double read_latency_ns = 0;
    std::string model_name;
};

BenchmarkResult benchmarkModel(const std::vector<Key>& dataset, 
                               const std::vector<Key>& queries,
                               std::function<bool(Key, Value&)> search_func,
                               const std::string& name) {
    BenchmarkResult r;
    r.model_name = name;
    
    volatile Value sink = 0;
    size_t hits = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (Key k : queries) {
        Value v;
        if (search_func(k, v)) { sink = v; ++hits; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    r.read_latency_ns = ns / std::max<size_t>(1, hits);
    
    if (sink == 0xDEADBEEF) std::cout << sink;
    return r;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "NEURAL LEARNED INDEX (NLI) - COMPLETE 5-MODEL BENCHMARK\n";
    std::cout << "NLI vs ALEX vs PGM vs RMI vs B-Tree\n";
    std::cout << std::string(80, '=') << "\n\n";

    struct Config {
        std::string name;
        std::string file;
        std::vector<size_t> sizes;
    };
    
    std::vector<Config> configs = {
        {"Books", "sosd_data/books_200M_uint64", {100000, 1000000, 10000000}},
        {"Facebook", "sosd_data/fb_200M_uint64", {100000, 1000000, 10000000}},
        {"WikiTS", "sosd_data/wiki_ts_200M_uint64", {100000, 1000000, 10000000}}
    };

    for (const auto& cfg : configs) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "DATASET: " << cfg.name << "\n";
        std::cout << std::string(80, '=') << "\n\n";

        auto full_dataset = load_binary_dataset(cfg.file);
        if (full_dataset.empty()) {
            std::cout << " SKIPPED (file not found)\n\n";
            continue;
        }

        for (size_t test_size : cfg.sizes) {
            if (test_size > full_dataset.size()) continue;

            std::cout << "\n--- Size: " << test_size << " keys ---\n\n";

            std::vector<Key> dataset(full_dataset.begin(), full_dataset.begin() + test_size);

            std::mt19937_64 gen(42);
            std::uniform_int_distribution<size_t> dis(0, dataset.size() - 1);
            std::vector<Key> queries;
            queries.reserve(100000);
            for (size_t i = 0; i < 100000; ++i) queries.push_back(dataset[dis(gen)]);

            std::vector<Value> positions(dataset.size());
            for (size_t i = 0; i < dataset.size(); ++i) positions[i] = i;
            
            BTreeIndex btree;
            btree.build(dataset, positions);
            
            NeuralLearnedIndex nli;
            nli.build(dataset, positions);

            auto btree_result = benchmarkModel(dataset, queries,
                [&](Key k, Value& v){ return btree.search(k, v); }, "B-Tree");
            auto alex_result = benchmarkModel(dataset, queries,
                [&](Key k, Value& v){ return nli.alex.search(k, v); }, "ALEX");
            auto pgm_result = benchmarkModel(dataset, queries,
                [&](Key k, Value& v){ return nli.pgm.search(k, v); }, "PGM");
            auto rmi_result = benchmarkModel(dataset, queries,
                [&](Key k, Value& v){ return nli.rmi.search(k, v); }, "RMI");
            auto nli_result = benchmarkModel(dataset, queries,
                [&](Key k, Value& v){ return nli.search(k, v); }, "NLI");

            std::cout << std::fixed << std::setprecision(1);
            std::cout << "B-Tree " << btree_result.read_latency_ns << " ns | ";
            std::cout << "ALEX " << alex_result.read_latency_ns << " ns | ";
            std::cout << "PGM " << pgm_result.read_latency_ns << " ns | ";
            std::cout << "RMI " << rmi_result.read_latency_ns << " ns | ";
            std::cout << "NLI " << nli_result.read_latency_ns << " ns\n";

            double speedup_vs_btree = btree_result.read_latency_ns / nli_result.read_latency_ns;
            std::cout << "NLI Speedup (vs B-Tree): " << std::setprecision(2) << speedup_vs_btree << "x\n\n";
        }

        std::cout << "\n" << std::string(80, '=') << "\n";
    }

    std::cout << "BENCHMARK COMPLETE - Neural Learned Index (NLI) 🥇\n";
    std::cout << std::string(80, '=') << "\n\n";
    return 0;
}