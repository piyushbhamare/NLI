// src/baseline/simple_learned_index.cpp
#include "simple_learned_index.h"
#include <algorithm>
#include <cmath>

namespace learned_index {

void SimpleLearnedIndex::bulkLoad(std::vector<KeyValuePair> data) {
    // Load and sort data
    data_.bulkLoad(std::move(data));
    
    if (data_.empty()) {
        return;
    }
    
    // Prepare training data: extract keys and positions
    std::vector<Key> keys;
    std::vector<size_t> positions;
    
    keys.reserve(data_.size());
    positions.reserve(data_.size());
    
    for (size_t i = 0; i < data_.size(); i++) {
        keys.push_back(data_.data()[i].key);
        positions.push_back(i);
    }
    
    // Train linear model
    model_.train(keys, positions);
    
    // Update statistics
    updateModelStats();
}

SearchResult SimpleLearnedIndex::lookup(Key key) const {
    if (data_.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    // Predict position using trained model
    size_t predicted_pos = model_.predictPosition(key, data_.size());
    
    // Exponential search around predicted position
    return exponentialSearch(key, predicted_pos);
}

bool SimpleLearnedIndex::insert(const KeyValuePair& kv) {
    return data_.insert(kv);
}

bool SimpleLearnedIndex::remove(Key key) {
    return data_.remove(key);
}

std::vector<KeyValuePair> SimpleLearnedIndex::rangeQuery(Key start, Key end) const {
    return data_.rangeQuery(start, end);
}

SearchResult SimpleLearnedIndex::exponentialSearch(
    Key key, size_t predicted_pos) const {
    
    const auto& arr = data_.data();
    if (arr.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    size_t n = arr.size();
    size_t probe_count = 1;
    
    // Check predicted position
    if (arr[predicted_pos].key == key) {
        return SearchResult(true, predicted_pos, arr[predicted_pos].value, 
                          probe_count);
    }
    
    // Determine search direction
    bool search_left = (predicted_pos > 0) && (key < arr[predicted_pos].key);
    bool search_right = (predicted_pos < n - 1) && (key > arr[predicted_pos].key);
    
    if (!search_left && !search_right) {
        return SearchResult(false, predicted_pos, 0, probe_count);
    }
    
    // Exponential search
    size_t left = predicted_pos;
    size_t right = predicted_pos;
    size_t step = 1;
    
    while (true) {
        // Expand left
        if (search_left && left > 0) {
            left = (left >= step) ? (left - step) : 0;
            probe_count++;
            
            if (arr[left].key == key) {
                return SearchResult(true, left, arr[left].value, probe_count);
            }
            
            if (arr[left].key < key) {
                break;  // Found lower bound
            }
        }
        
        // Expand right
        if (search_right && right < n - 1) {
            right = std::min(right + step, n - 1);
            probe_count++;
            
            if (arr[right].key == key) {
                return SearchResult(true, right, arr[right].value, probe_count);
            }
            
            if (arr[right].key > key) {
                break;  // Found upper bound
            }
        }
        
        if ((search_left && left == 0) || (search_right && right == n - 1)) {
            break;  // Exhausted search space
        }
        
        step *= 2;  // Exponential growth
    }
    
    // Binary search within [left, right]
    while (left <= right && right < n) {
        size_t mid = left + (right - left) / 2;
        probe_count++;
        
        if (arr[mid].key == key) {
            return SearchResult(true, mid, arr[mid].value, probe_count);
        } else if (arr[mid].key < key) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    
    return SearchResult(false, left, 0, probe_count);
}

void SimpleLearnedIndex::updateModelStats() {
    if (data_.empty()) {
        return;
    }
    
    const auto& arr = data_.data();
    std::vector<Key> keys;
    std::vector<size_t> positions;
    
    keys.reserve(arr.size());
    positions.reserve(arr.size());
    
    for (size_t i = 0; i < arr.size(); i++) {
        keys.push_back(arr[i].key);
        positions.push_back(i);
    }
    
    model_stats_ = model_.getStats(keys, positions);
    avg_error_ = model_stats_.mean_absolute_error;
    max_error_ = model_stats_.max_error;
    rmse_ = model_stats_.rmse;
    r_squared_ = model_stats_.r_squared;
}

} // namespace learned_index
