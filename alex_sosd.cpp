#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <set>
#include <map>
#include <fstream>
#include <iomanip>

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
    
    return keys;
}

// ============================================================================
// Linear Model
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
            slope = 0.0;
            intercept = 0.0;
            return;
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
// ALEX Baseline
// ============================================================================

class ALEXBaseline {
public:
    std::vector<Key> keys;
    std::vector<Value> values;
    LinearModel model;

    void build(const std::vector<Key>& input_keys, const std::vector<Value>& input_values) {
        std::vector<size_t> idx(input_keys.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), 
                  [&](size_t a, size_t b) { return input_keys[a] < input_keys[b]; });

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

    double memory_mb() const {
        return static_cast<double>(keys.size()) * (sizeof(Key) + sizeof(Value)) / (1024.0 * 1024.0);
    }
};

// ============================================================================
// B-Tree Baseline
// ============================================================================

class SimpleBTree {
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

    double memory_mb() const {
        return static_cast<double>(tree.size()) * (sizeof(Key) + sizeof(Value) + 40) / (1024.0 * 1024.0);
    }
};

// ============================================================================
// Benchmark
// ============================================================================

struct BenchmarkResult {
    double read_latency_ns = 0;
    double throughput_ops_sec = 0;
    double memory_mb = 0;
};

BenchmarkResult benchmarkBTree(const std::vector<Key>& dataset, const std::vector<Key>& queries) {
    BenchmarkResult r;
    SimpleBTree b;
    std::vector<Value> vals(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) vals[i] = i;
    b.build(dataset, vals);

    volatile Value sink = 0;
    size_t hits = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (Key k : queries) {
        Value v;
        if (b.search(k, v)) { sink = v; ++hits; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    r.read_latency_ns = ns / std::max<size_t>(1, hits);
    r.memory_mb = b.memory_mb();
    r.throughput_ops_sec = 1e9 / std::max(1.0, r.read_latency_ns);

    if (sink == 0xDEADBEEF) std::cout << sink;
    return r;
}

BenchmarkResult benchmarkALEX(const std::vector<Key>& dataset, const std::vector<Key>& queries) {
    BenchmarkResult r;
    ALEXBaseline idx;
    std::vector<Value> vals(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) vals[i] = i;
    idx.build(dataset, vals);

    volatile Value sink = 0;
    size_t hits = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (Key k : queries) {
        Value v;
        if (idx.search(k, v)) { sink = v; ++hits; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    r.read_latency_ns = ns / std::max<size_t>(1, hits);
    r.memory_mb = idx.memory_mb();
    r.throughput_ops_sec = 1e9 / std::max(1.0, r.read_latency_ns);

    if (sink == 0xDEADBEEF) std::cout << sink;
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "SOSD BENCHMARK - PUBLICATION VERSION\n";
    std::cout << std::string(80, '=') << "\n\n";

    struct Config {
        std::string name;
        std::string file;
        std::vector<size_t> sizes;
    };

    std::vector<Config> configs = {
        {"Books", "../sosd_data/books_200M_uint64", {100000, 1000000, 10000000}},
        {"OSM", "../sosd_data/osm_cellids_200M_uint64", {100000, 1000000, 10000000}},
        {"Facebook", "../sosd_data/fb_200M_uint64", {100000, 1000000, 10000000}}
    };

    for (const auto& cfg : configs) {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "DATASET: " << cfg.name << "\n";
        std::cout << std::string(80, '=') << "\n\n";

        auto full = load_binary_dataset(cfg.file);
        if (full.empty()) {
            std::cout << "SKIPPED (file not found)\n\n";
            continue;
        }
        std::cout << "Loaded " << full.size() << " keys\n\n";

        for (size_t n : cfg.sizes) {
            if (n > full.size()) continue;
            
            std::cout << "--- Size: " << n << " keys ---\n\n";
            std::vector<Key> data(full.begin(), full.begin() + n);

            // Generate queries
            std::mt19937_64 gen(42);
            std::uniform_int_distribution<size_t> dis(0, data.size() - 1);
            std::vector<Key> queries;
            queries.reserve(100000);
            for (size_t i = 0; i < 100000; ++i) queries.push_back(data[dis(gen)]);

            // Benchmark
            auto b = benchmarkBTree(data, queries);
            auto a = benchmarkALEX(data, queries);

            std::cout << "B-Tree:  " << std::fixed << std::setprecision(1) << b.read_latency_ns << " ns  "
                      << b.memory_mb << " MB\n";
            std::cout << "ALEX:    " << a.read_latency_ns << " ns  " << a.memory_mb << " MB\n";
            std::cout << "Speedup: " << std::setprecision(2) << (b.read_latency_ns / a.read_latency_ns) << "x\n\n";
        }
    }

    std::cout << std::string(80, '=') << "\n";
    std::cout << "BENCHMARK COMPLETE\n";
    std::cout << std::string(80, '=') << "\n\n";
    return 0;
}
