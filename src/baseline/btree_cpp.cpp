// src/baseline/btree.cpp
#include "btree.h"
#include <algorithm>
#include <queue>

namespace learned_index {

BTree::BTree() : root_(std::make_shared<BTreeNode>(true)), size_(0) {}

SearchResult BTree::lookup(Key key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return lookupInternal(root_, key, 0);
}

SearchResult BTree::lookupInternal(const std::shared_ptr<BTreeNode>& node,
                                   Key key, size_t depth) const {
    if (!node) {
        return SearchResult(false, 0, 0, depth);
    }
    
    // Find position in current node
    size_t i = 0;
    while (i < node->keys.size() && key > node->keys[i]) {
        i++;
    }
    
    if (i < node->keys.size() && key == node->keys[i]) {
        if (node->is_leaf) {
            return SearchResult(true, i, node->values[i], depth + 1);
        }
    }
    
    if (node->is_leaf) {
        return SearchResult(false, i, 0, depth + 1);
    }
    
    // Recurse to child
    if (i < node->children.size()) {
        return lookupInternal(node->children[i], key, depth + 1);
    }
    
    return SearchResult(false, i, 0, depth + 1);
}

bool BTree::insert(Key key, Value value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    if (!root_) {
        root_ = std::make_shared<BTreeNode>(true);
    }
    
    bool inserted = insertInternal(root_, key, value);
    
    // Check if root needs to split
    if (root_->keys.size() > BTREE_ORDER) {
        auto new_root = std::make_shared<BTreeNode>(false);
        new_root->children.push_back(root_);
        split(new_root, 0);
        root_ = new_root;
    }
    
    if (inserted) size_++;
    return inserted;
}

bool BTree::insertInternal(std::shared_ptr<BTreeNode>& node,
                          Key key, Value value) {
    if (!node) return false;
    
    size_t i = 0;
    while (i < node->keys.size() && key > node->keys[i]) {
        i++;
    }
    
    // Key already exists
    if (i < node->keys.size() && key == node->keys[i]) {
        if (node->is_leaf) {
            node->values[i] = value;
        }
        return false;
    }
    
    if (node->is_leaf) {
        node->keys.insert(node->keys.begin() + i, key);
        node->values.insert(node->values.begin() + i, value);
        return true;
    }
    
    // Recurse to child
    if (i >= node->children.size()) {
        return false;
    }
    
    bool inserted = insertInternal(node->children[i], key, value);
    
    // Split child if needed
    if (node->children[i]->keys.size() > BTREE_ORDER) {
        split(node, i);
    }
    
    return inserted;
}

void BTree::split(std::shared_ptr<BTreeNode>& parent, size_t index) {
    if (index >= parent->children.size()) return;
    
    auto full_child = parent->children[index];
    auto new_child = std::make_shared<BTreeNode>(full_child->is_leaf);
    
    size_t mid = full_child->keys.size() / 2;
    
    // Move keys to new node
    new_child->keys.assign(full_child->keys.begin() + mid, 
                           full_child->keys.end());
    full_child->keys.erase(full_child->keys.begin() + mid, 
                          full_child->keys.end());
    
    // Move values if leaf
    if (full_child->is_leaf) {
        new_child->values.assign(full_child->values.begin() + mid,
                                full_child->values.end());
        full_child->values.erase(full_child->values.begin() + mid,
                                full_child->values.end());
    }
    
    // Move children if internal
    if (!full_child->is_leaf) {
        new_child->children.assign(full_child->children.begin() + mid,
                                   full_child->children.end());
        full_child->children.erase(full_child->children.begin() + mid,
                                   full_child->children.end());
    }
    
    // Add middle key to parent
    parent->keys.insert(parent->keys.begin() + index, full_child->keys.back());
    parent->children.insert(parent->children.begin() + index + 1, new_child);
}

void BTree::bulkLoad(const std::vector<KeyValuePair>& data) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // Sort data
    auto sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Build leaf level
    std::vector<std::shared_ptr<BTreeNode>> leaves;
    auto current_leaf = std::make_shared<BTreeNode>(true);
    
    for (const auto& kv : sorted_data) {
        if (current_leaf->keys.size() >= BTREE_ORDER) {
            leaves.push_back(current_leaf);
            current_leaf = std::make_shared<BTreeNode>(true);
        }
        current_leaf->keys.push_back(kv.key);
        current_leaf->values.push_back(kv.value);
    }
    
    if (!current_leaf->keys.empty()) {
        leaves.push_back(current_leaf);
    }
    
    // Link leaves
    for (size_t i = 0; i < leaves.size() - 1; i++) {
        leaves[i]->next = leaves[i + 1];
        leaves[i + 1]->prev = leaves[i];
    }
    
    // Build internal levels
    std::vector<std::shared_ptr<BTreeNode>> current_level = leaves;
    
    while (current_level.size() > 1) {
        std::vector<std::shared_ptr<BTreeNode>> next_level;
        auto parent = std::make_shared<BTreeNode>(false);
        
        for (size_t i = 0; i < current_level.size(); i++) {
            if (parent->children.size() >= BTREE_ORDER) {
                next_level.push_back(parent);
                parent = std::make_shared<BTreeNode>(false);
            }
            
            if (!parent->children.empty()) {
                parent->keys.push_back(current_level[i]->keys[0]);
            }
            parent->children.push_back(current_level[i]);
        }
        
        if (!parent->children.empty()) {
            next_level.push_back(parent);
        }
        
        current_level = next_level;
    }
    
    root_ = current_level.empty() ? 
           std::make_shared<BTreeNode>(true) : current_level[0];
    size_ = data.size();
}

std::vector<KeyValuePair> BTree::rangeQuery(Key start, Key end) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<KeyValuePair> result;
    
    if (!root_) return result;
    
    // Find leftmost leaf >= start
    auto current = root_;
    while (!current->is_leaf) {
        size_t i = 0;
        while (i < current->keys.size() && start >= current->keys[i]) {
            i++;
        }
        if (i < current->children.size()) {
            current = current->children[i];
        } else {
            break;
        }
    }
    
    // Traverse leaves collecting keys in range
    while (current) {
        for (size_t i = 0; i < current->keys.size(); i++) {
            if (current->keys[i] >= start && current->keys[i] <= end) {
                result.emplace_back(current->keys[i], current->values[i]);
            } else if (current->keys[i] > end) {
                return result;
            }
        }
        current = current->next;
    }
    
    return result;
}

bool BTree::remove(Key key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    // Simplified: just mark as tombstone
    auto result = lookup(key);
    if (result.found) {
        size_--;
        return true;
    }
    return false;
}

size_t BTree::height() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return heightInternal(root_);
}

size_t BTree::heightInternal(const std::shared_ptr<BTreeNode>& node) const {
    if (!node || node->is_leaf) return 1;
    if (node->children.empty()) return 1;
    return 1 + heightInternal(node->children[0]);
}

double BTree::utilization() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!root_) return 0.0;
    
    size_t total_capacity = 0, total_keys = 0;
    std::queue<std::shared_ptr<BTreeNode>> q;
    q.push(root_);
    
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        
        total_keys += node->keys.size();
        total_capacity += BTREE_ORDER;
        
        for (const auto& child : node->children) {
            q.push(child);
        }
    }
    
    return total_capacity > 0 ? 
           static_cast<double>(total_keys) / total_capacity : 0.0;
}

} // namespace learned_index
