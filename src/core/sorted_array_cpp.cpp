// src/core/sorted_array.cpp
#include "sorted_array.h"
#include <algorithm>
#include <cmath>

namespace learned_index {

SortedArray::SortedArray(std::vector<KeyValuePair> data)
    : data_(std::move(data)) {
    std::sort(data_.begin(), data_.end());
}

void SortedArray::bulkLoad(std::vector<KeyValuePair> data) {
    data_ = std::move(data);
    std::sort(data_.begin(), data_.end());
}

SearchResult SortedArray::lookup(Key key) const {
    if (data_.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    size_t pos = binarySearch(key);
    size_t probe_count = static_cast<size_t>(std::log2(data_.size())) + 1;
    
    if (pos < data_.size() && data_[pos].key == key) {
        return SearchResult(true, pos, data_[pos].value, probe_count);
    }
    
    return SearchResult(false, pos, 0, probe_count);
}

bool SortedArray::insert(const KeyValuePair& kv) {
    auto it = std::lower_bound(data_.begin(), data_.end(), kv);
    
    if (it != data_.end() && it->key == kv.key) {
        it->value = kv.value;  // Update existing
        return false;
    }
    
    data_.insert(it, kv);
    return true;
}

bool SortedArray::remove(Key key) {
    auto it = std::lower_bound(data_.begin(), data_.end(), 
                               KeyValuePair(key, 0));
    
    if (it != data_.end() && it->key == key) {
        data_.erase(it);
        return true;
    }
    
    return false;
}

std::vector<KeyValuePair> SortedArray::rangeQuery(Key start, Key end) const {
    auto start_it = std::lower_bound(data_.begin(), data_.end(), 
                                     KeyValuePair(start, 0));
    auto end_it = std::upper_bound(data_.begin(), data_.end(), 
                                   KeyValuePair(end, 0));
    
    return std::vector<KeyValuePair>(start_it, end_it);
}

size_t SortedArray::binarySearch(Key key) const {
    size_t left = 0, right = data_.size();
    
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (data_[mid].key < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left;
}

} // namespace learned_index
