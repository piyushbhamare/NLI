#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
namespace learned_index {
using Key = uint64_t;
using Value = uint64_t;
struct KeyValuePair {
    Key key; Value value;
    KeyValuePair() : key(0), value(0) {}
    KeyValuePair(Key k, Value v) : key(k), value(v) {}
    bool operator<(const KeyValuePair& o) const { return key < o.key; }
};
struct SearchResult {
    bool found; size_t position; Value value; size_t probe_count;
    SearchResult() : found(false), position(0), value(0), probe_count(0) {}
    SearchResult(bool f, size_t p, Value v, size_t pc) : found(f), position(p), value(v), probe_count(pc) {}
};
struct ModelStats {
    double mean_absolute_error = 0.0;
    double max_error = 0.0;
    double rmse = 0.0;
    double r_squared = 0.0;
    size_t num_samples = 0;
};
}
