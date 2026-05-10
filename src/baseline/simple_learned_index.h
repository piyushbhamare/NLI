// src/baseline/simple_learned_index.h
#pragma once

#include "../core/types.h"
#include "../core/sorted_array.h"
#include "linear_model.h"
#include <vector>

namespace learned_index {

/**
 * @brief Single-stage Learned Index using linear regression
 * 
 * The simplest form of learned index. Uses a single linear model
 * to predict key positions in a sorted array, then performs
 * exponential/binary search around the predicted position.
 * 
 * Key insight: A machine learning model can replace a B-tree node
 * by learning to approximate the cumulative distribution function (CDF).
 * 
 * Time Complexity:
 *   - Lookup: O(log(prediction_error)) typically, O(n) worst case
 *   - Build: O(n log n) for sorting + O(n) for model training
 * 
 * Space Complexity: O(n + model_size), where model_size is constant
 */
class SimpleLearnedIndex {
public:
    SimpleLearnedIndex() = default;
    
    /**
     * @brief Bulk load and train index
     * @param data Vector of key-value pairs to index
     */
    void bulkLoad(std::vector<KeyValuePair> data);
    
    /**
     * @brief Lookup a key using learned position prediction
     * @param key Key to search for
     * @return SearchResult with position and probe statistics
     */
    SearchResult lookup(Key key) const;
    
    /**
     * @brief Insert a key-value pair (requires rebuild for optimality)
     * @param kv Key-value pair
     * @return True if inserted, false if updated
     */
    bool insert(const KeyValuePair& kv);
    
    /**
     * @brief Remove a key
     * @param key Key to remove
     * @return True if removed, false if not found
     */
    bool remove(Key key);
    
    /**
     * @brief Range query [start, end]
     * @param start Start key
     * @param end End key
     * @return Vector of matching key-value pairs
     */
    std::vector<KeyValuePair> rangeQuery(Key start, Key end) const;
    
    /// Get total number of indexed keys
    size_t size() const { return data_.size(); }
    
    /// Get mean absolute error of model predictions
    double avgError() const { return avg_error_; }
    
    /// Get maximum error of model predictions
    double maxError() const { return max_error_; }
    
    /// Get model RMSE
    double rmse() const { return rmse_; }
    
    /// Get model R-squared
    double rSquared() const { return r_squared_; }
    
    /// Get underlying data
    const SortedArray& data() const { return data_; }
    
    /// Get trained model
    const LinearModel& model() const { return model_; }
    
    /// Get model statistics
    ModelStats modelStats() const { return model_stats_; }
    
private:
    SortedArray data_;
    LinearModel model_;
    double avg_error_ = 0.0;
    double max_error_ = 0.0;
    double rmse_ = 0.0;
    double r_squared_ = 0.0;
    ModelStats model_stats_;
    
    /**
     * @brief Exponential search around predicted position
     * @param key Search key
     * @param predicted_pos Position predicted by model
     * @return SearchResult
     */
    SearchResult exponentialSearch(Key key, size_t predicted_pos) const;
    
    /**
     * @brief Recalculate model statistics
     */
    void updateModelStats();
};

} // namespace learned_index
