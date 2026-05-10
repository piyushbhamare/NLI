#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <set>
#include <map>
#include <iomanip>

using Key = long long;
using Value = long long;

// ============================================================================
// Phase-3 learned-index baseline (dense sorted vector)
// This is a *measurement baseline harness*: fast + correct reads.
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
        if (p < 0.0) p = 0.0;
        double hi = static_cast<double>(n - 1);
        if (p > hi) p = hi;
        return static_cast<size_t>(p);
    }
};

class ALEXBaselineDense {
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
        // Correct + fast for random reads: one lower_bound on dense vector
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        if (it != keys.end() && *it == key) {
            out = values[static_cast<size_t>(it - keys.begin())];
            return true;
        }
        return false;
    }

    inline void insert(Key key, Value value) {
        // Correct insert for benchmark (not real ALEX insertion behavior).
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        size_t pos = static_cast<size_t>(it - keys.begin());
        keys.insert(it, key);
        values.insert(values.begin() + pos, value);

        // Periodic retrain (kept cheap)
        if ((keys.size() & 1023u) == 0) model.train(keys);
    }

    double memory_mb() const {
        double bytes = static_cast<double>(keys.size()) * (sizeof(Key) + sizeof(Value));
        return bytes / (1024.0 * 1024.0);
    }
};

// ============================================================================
// B-tree baseline: std::map
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

    inline void insert(Key key, Value value) { tree[key] = value; }

    double memory_mb() const {
        // Rough RB-tree overhead estimate
        double bytes = static_cast<double>(tree.size()) * (sizeof(Key) + sizeof(Value) + 40);
        return bytes / (1024.0 * 1024.0);
    }
};

struct BenchmarkResult {
    double read_latency_ns = 0;
    double insert_latency_ns = 0;
    double throughput_ops_sec = 0;
    double memory_mb = 0;
};

static BenchmarkResult benchmarkBTree(const std::vector<Key>& dataset,
                                      const std::vector<Key>& read_queries,
                                      const std::vector<Key>& write_queries) {
    BenchmarkResult r;
    SimpleBTree b;

    std::vector<Value> vals(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) vals[i] = static_cast<Value>(i);
    b.build(dataset, vals);

    volatile Value sink = 0;
    size_t hits = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (Key k : read_queries) {
        Value v;
        if (b.search(k, v)) { sink = v; ++hits; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    r.read_latency_ns = ns / std::max<size_t>(1, hits);

    size_t max_ins = std::min<size_t>(write_queries.size(), 10000);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < max_ins; ++i) b.insert(write_queries[i], write_queries[i]);
    auto t3 = std::chrono::high_resolution_clock::now();

    double ins_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    r.insert_latency_ns = ins_ns / std::max<size_t>(1, max_ins);

    r.memory_mb = b.memory_mb();
    r.throughput_ops_sec = 1e9 / std::max(1.0, r.read_latency_ns);

    if (sink == -12345) std::cout << sink;
    return r;
}

static BenchmarkResult benchmarkALEX(const std::vector<Key>& dataset,
                                     const std::vector<Key>& read_queries,
                                     const std::vector<Key>& write_queries) {
    BenchmarkResult r;
    ALEXBaselineDense idx;

    std::vector<Value> vals(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) vals[i] = static_cast<Value>(i);
    idx.build(dataset, vals);

    volatile Value sink = 0;
    size_t hits = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (Key k : read_queries) {
        Value v;
        if (idx.search(k, v)) { sink = v; ++hits; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    r.read_latency_ns = ns / std::max<size_t>(1, hits);

    size_t max_ins = std::min<size_t>(write_queries.size(), 10000);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < max_ins; ++i) idx.insert(write_queries[i], write_queries[i]);
    auto t3 = std::chrono::high_resolution_clock::now();

    double ins_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    r.insert_latency_ns = ins_ns / std::max<size_t>(1, max_ins);

    r.memory_mb = idx.memory_mb();
    r.throughput_ops_sec = 1e9 / std::max(1.0, r.read_latency_ns);

    if (sink == -12345) std::cout << sink;
    return r;
}

int main() {
    std::cout << "\n================================================================================\n";
    std::cout << "PHASE 3: ALEX-baseline (Dense) vs B-TREE (std::map)\n";
    std::cout << "================================================================================\n\n";

    std::vector<size_t> sizes = {10000, 100000, 1000000};

    for (size_t n : sizes) {
        std::cout << "================================================================================\n";
        std::cout << "Dataset Size: " << n << " keys\n";
        std::cout << "================================================================================\n\n";

        std::mt19937_64 gen(42);
        std::uniform_int_distribution<Key> dis(1, 10000000);

        std::set<Key> uniq;
        while (uniq.size() < n) uniq.insert(dis(gen));
        std::vector<Key> dataset(uniq.begin(), uniq.end());

        std::uniform_int_distribution<size_t> idx_dis(0, dataset.size() - 1);
        std::vector<Key> read_q;
        std::vector<Key> write_q;
        read_q.reserve(100000);
        write_q.reserve(100000);

        for (size_t i = 0; i < 100000; ++i) {
            read_q.push_back(dataset[idx_dis(gen)]);
            write_q.push_back(dis(gen));
        }

        std::cout << "--- B-TREE BASELINE (std::map) ---\n";
        auto b = benchmarkBTree(dataset, read_q, write_q);
        std::cout << "  Read Latency:     " << std::fixed << std::setprecision(1) << b.read_latency_ns << " ns\n";
        std::cout << "  Insert Latency:   " << std::fixed << std::setprecision(1) << b.insert_latency_ns << " ns\n";
        std::cout << "  Throughput:       " << std::scientific << std::setprecision(2) << b.throughput_ops_sec << " ops/sec\n";
        std::cout << "  Memory:           " << std::fixed << std::setprecision(1) << b.memory_mb << " MB\n\n";

        std::cout << "--- ALEX BASELINE (dense sorted vector) ---\n";
        auto a = benchmarkALEX(dataset, read_q, write_q);
        std::cout << "  Read Latency:     " << std::fixed << std::setprecision(1) << a.read_latency_ns << " ns\n";
        std::cout << "  Insert Latency:   " << std::fixed << std::setprecision(1) << a.insert_latency_ns << " ns\n";
        std::cout << "  Throughput:       " << std::scientific << std::setprecision(2) << a.throughput_ops_sec << " ops/sec\n";
        std::cout << "  Memory:           " << std::fixed << std::setprecision(1) << a.memory_mb << " MB\n\n";

        std::cout << "--- COMPARISON ---\n";
        double rs = b.read_latency_ns / std::max(1.0, a.read_latency_ns);
        double is = b.insert_latency_ns / std::max(1.0, a.insert_latency_ns);
        double mr = b.memory_mb / std::max(1e-9, a.memory_mb);

        std::cout << "  ALEX Read Speedup:    " << std::fixed << std::setprecision(2);
        if (rs >= 1.0) std::cout << rs << "x FASTER\n";
        else std::cout << (1.0 / rs) << "x SLOWER\n";

        std::cout << "  ALEX Insert Speed:    " << std::fixed << std::setprecision(2);
        if (is >= 1.0) std::cout << is << "x FASTER\n";
        else std::cout << (1.0 / is) << "x SLOWER\n";

        std::cout << "  Memory Ratio:         " << std::fixed << std::setprecision(2) << mr << "x\n\n";
    }

    std::cout << "================================================================================\n";
    std::cout << "PHASE 3 COMPLETE - Baseline Measurement Harness\n";
    std::cout << "================================================================================\n\n";
    return 0;
}
