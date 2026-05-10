// src/core/search.cpp
/**
 * @file search.cpp
 * @brief Utility functions for search algorithms
 */

#include "types.h"
#include <cmath>
#include <algorithm>

namespace learned_index {

/**
 * @brief Exponential search around a predicted position
 * 
 * Used by learned indexes to expand search space when prediction error
 * is unknown. Combines exponential growth with binary search refinement.
 * 
 * Complexity: O(log n) average case, O(n) worst case
 * 
 * @param data Sorted array of key-value pairs
 * @param key Key to search for
 * @param predicted_pos Initial predicted position from model
 * @return SearchResult with position and probe count
 */
SearchResult exponentialSearch(const std::vector<KeyValuePair>& data,
                               Key key, size_t predicted_pos) {
    if (data.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    size_t n = data.size();
    size_t probe_count = 0;
    
    // Clamp predicted position
    predicted_pos = std::min(predicted_pos, n - 1);
    
    // Check predicted position
    probe_count++;
    if (data[predicted_pos].key == key) {
        return SearchResult(true, predicted_pos, data[predicted_pos].value, 
                          probe_count);
    }
    
    // Determine search direction
    bool search_left = (predicted_pos > 0) && (key < data[predicted_pos].key);
    bool search_right = (predicted_pos < n - 1) && (key > data[predicted_pos].key);
    
    if (!search_left && !search_right) {
        return SearchResult(false, predicted_pos, 0, probe_count);
    }
    
    // Exponential search
    size_t left = predicted_pos;
    size_t right = predicted_pos;
    size_t step = 1;
    
    while (true) {
        // Expand in search direction
        if (search_left && left > 0) {
            left = (left >= step) ? (left - step) : 0;
            probe_count++;
            
            if (data[left].key == key) {
                return SearchResult(true, left, data[left].value, probe_count);
            }
            
            if (data[left].key < key) {
                break;  // Found lower bound
            }
        }
        
        if (search_right && right < n - 1) {
            right = std::min(right + step, n - 1);
            probe_count++;
            
            if (data[right].key == key) {
                return SearchResult(true, right, data[right].value, probe_count);
            }
            
            if (data[right].key > key) {
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
        
        if (data[mid].key == key) {
            return SearchResult(true, mid, data[mid].value, probe_count);
        } else if (data[mid].key < key) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    
    return SearchResult(false, left, 0, probe_count);
}

/**
 * @brief Interpolation search (linear approximation)
 * 
 * Estimates position using linear interpolation between array bounds.
 * Better than binary search for uniformly distributed data.
 * 
 * Complexity: O(log log n) average, O(n) worst case
 * 
 * @param data Sorted array
 * @param key Search key
 * @return SearchResult
 */
SearchResult interpolationSearch(const std::vector<KeyValuePair>& data,
                                 Key key) {
    if (data.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    size_t left = 0, right = data.size() - 1;
    size_t probe_count = 0;
    
    while (left <= right && key >= data[left].key && key <= data[right].key) {
        probe_count++;
        
        // Linear interpolation
        double pos = left + ((double)(key - data[left].key) * 
                            (right - left)) / 
                            (data[right].key - data[left].key);
        
        size_t mid = static_cast<size_t>(pos);
        mid = std::min(mid, right);
        
        if (data[mid].key == key) {
            return SearchResult(true, mid, data[mid].value, probe_count);
        } else if (data[mid].key < key) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    
    // Final binary search
    while (left <= right) {
        size_t mid = left + (right - left) / 2;
        probe_count++;
        
        if (data[mid].key == key) {
            return SearchResult(true, mid, data[mid].value, probe_count);
        } else if (data[mid].key < key) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }
    
    return SearchResult(false, left, 0, probe_count);
}

} // namespace learned_index
