// src/baseline/btree.h
#pragma once

#include "../core/types.h"
#include <vector>
#include <memory>
#include <shared_mutex>

namespace learned_index {

constexpr size_t BTREE_ORDER = 256;  ///< B+ tree node capacity

/**
 * @brief Node structure for B+ tree
 * 
 * B+ trees are classical database indexing structures that maintain
 * sorted keys with predictable O(log n) lookup, insert, delete.
 * Used as performance baseline for comparing learned indexes.
 */
struct BTreeNode {
    bool is_leaf;                                    ///< True if leaf node
    std::vector<Key> keys;                          ///< Stored keys
    std::vector<Value> values;                      ///< Values (leaf only)
    std::vector<std::shared_ptr<BTreeNode>> children;  ///< Child pointers
    std::shared_ptr<BTreeNode> next;               ///< Next leaf (leaf only)
    std::shared_ptr<BTreeNode> prev;               ///< Prev leaf (leaf only)
    
    explicit BTreeNode(bool leaf) : is_leaf(leaf), next(nullptr), prev(nullptr) {}
};

/**
 * @brief Classical B+ Tree index
 * 
 * Provides logarithmic time operations for lookup, insert, and delete.
 * Serves as the comparison baseline for learned indexes.
 * 
 * Time Complexity:
 *   - Lookup: O(log n)
 *   - Insert: O(log n) amortized
 *   - Delete: O(log n) amortized
 * 
 * Space: O(n)
 */
class BTree {
public:
    BTree();
    
    /**
     * @brief Lookup a key
     * @param key Search key
     * @return SearchResult with position and probe count
     */
    SearchResult lookup(Key key) const;
    
    /**
     * @brief Insert a key-value pair
     * @param key Key to insert
     * @param value Associated value
     * @return True if inserted, false if updated
     */
    bool insert(Key key, Value value);
    
    /**
     * @brief Remove a key
     * @param key Key to remove
     * @return True if removed, false if not found
     */
    bool remove(Key key);
    
    /**
     * @brief Bulk load from sorted data
     * @param data Vector of key-value pairs
     */
    void bulkLoad(const std::vector<KeyValuePair>& data);
    
    /**
     * @brief Range query [start, end]
     * @param start Start key
     * @param end End key
     * @return Matching key-value pairs
     */
    std::vector<KeyValuePair> rangeQuery(Key start, Key end) const;
    
    /// Get tree size
    size_t size() const { return size_; }
    
    /// Get tree height
    size_t height() const;
    
    /// Get node occupancy ratio
    double utilization() const;
    
private:
    std::shared_ptr<BTreeNode> root_;
    size_t size_;
    mutable std::shared_mutex mutex_;  ///< Basic concurrency support
    
    SearchResult lookupInternal(const std::shared_ptr<BTreeNode>& node,
                               Key key, size_t depth) const;
    bool insertInternal(std::shared_ptr<BTreeNode>& node, Key key, Value value);
    void split(std::shared_ptr<BTreeNode>& parent, size_t index);
    size_t heightInternal(const std::shared_ptr<BTreeNode>& node) const;
};

} // namespace learned_index
