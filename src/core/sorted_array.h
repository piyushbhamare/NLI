#pragma once
#include "types.h"
#include <vector>
#include <cstddef>
namespace learned_index {
class SortedArray {
public:
    SortedArray() = default;
    explicit SortedArray(std::vector<KeyValuePair> data);
    SearchResult lookup(Key key) const;
    bool insert(const KeyValuePair& kv);
    bool remove(Key key);
    void bulkLoad(std::vector<KeyValuePair> data);
    size_t size() const { return data_.size(); }
    const std::vector<KeyValuePair>& data() const { return data_; }
private:
    std::vector<KeyValuePair> data_;
    size_t binarySearch(Key key) const;
};
}
