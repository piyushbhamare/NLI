#include "types.h"
#include <cmath>
#include <algorithm>
namespace learned_index {
SearchResult exponentialSearch(const std::vector<KeyValuePair>& data, Key key, size_t predicted_pos) {
    if (data.empty()) return SearchResult(false, 0, 0, 0);
    size_t n = data.size();
    predicted_pos = std::min(predicted_pos, n - 1);
    size_t probe_count = 1;
    if (data[predicted_pos].key == key) return SearchResult(true, predicted_pos, data[predicted_pos].value, probe_count);
    bool search_left = (predicted_pos > 0) && (key < data[predicted_pos].key);
    bool search_right = (predicted_pos < n - 1) && (key > data[predicted_pos].key);
    if (!search_left && !search_right) return SearchResult(false, predicted_pos, 0, probe_count);
    size_t left = predicted_pos, right = predicted_pos, step = 1;
    while (true) {
        if (search_left && left > 0) {
            left = (left >= step) ? (left - step) : 0; probe_count++;
            if (data[left].key == key) return SearchResult(true, left, data[left].value, probe_count);
            if (data[left].key < key) break;
        }
        if (search_right && right < n - 1) {
            right = std::min(right + step, n - 1); probe_count++;
            if (data[right].key == key) return SearchResult(true, right, data[right].value, probe_count);
            if (data[right].key > key) break;
        }
        if ((search_left && left == 0) || (search_right && right == n - 1)) break;
        step *= 2;
    }
    while (left <= right && right < n) {
        size_t mid = left + (right - left) / 2; probe_count++;
        if (data[mid].key == key) return SearchResult(true, mid, data[mid].value, probe_count);
        else if (data[mid].key < key) left = mid + 1;
        else { if (mid == 0) break; right = mid - 1; }
    }
    return SearchResult(false, left, 0, probe_count);
}
}
