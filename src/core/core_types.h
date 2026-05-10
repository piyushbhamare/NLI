// src/core/types.h
#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace learned_index {

using Key = uint64_t;
using Value = uint64_t;

struct KeyValuePair {
    Key key;
    Value value;
    
    KeyValuePair() : key(0), value(0) {}
    KeyValuePair(Key k, Value v) : key(k), value(v) {}
    
    bool operator<(const KeyValuePair& other) const {
        return key < other.key;
    }
    
    bool operator==(const KeyValuePair& other) const {
        return key == other.key && value == other.value;
    }
};

struct SearchResult {
    bool found;
    size_t position;
    Value value;
    size_t probe_count;
    
    SearchResult() : found(false), position(0), value(0), probe_count(0) {}
    SearchResult(bool f, size_t p, Value v, size_t pc)
        : found(f), position(p), value(v), probe_count(pc) {}
};

struct IndexConfig {
    size_t max_model_error = 32;
    double gap_density_lower = 0.5;
    double gap_density_upper = 0.9;
    double drift_ewma_alpha = 0.9;
    double drift_psi_threshold = 0.2;
    size_t rebuild_threshold = 100000;
    size_t sample_size = 10000;
};

struct BenchmarkStats {
    double p50_latency_ns = 0.0;
    double p95_latency_ns = 0.0;
    double p99_latency_ns = 0.0;
    double avg_latency_ns = 0.0;
    double throughput_ops_sec = 0.0;
    size_t memory_bytes = 0;
    double avg_probe_count = 0.0;
};

struct ModelStats {
    double mean_absolute_error = 0.0;
    double max_error = 0.0;
    double rmse = 0.0;
    double r_squared = 0.0;
    size_t num_samples = 0;
};

} // namespace learned_index
