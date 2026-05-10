#include "rmi_model.h"
#include <numeric>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace learned_index {

// ============================================================================
// LinearRegressionModel Implementation
// ============================================================================

void LinearRegressionModel::train(const std::vector<Key>& keys, 
                                 const std::vector<size_t>& positions) {
    if (keys.size() != positions.size() || keys.empty()) {
        slope_ = 0.0;
        intercept_ = 0.0;
        return;
    }
    
    // Store key range
    min_key_ = keys.front();
    max_key_ = keys.back();
    
    // Least squares regression: y = ax + b
    // a = (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
    // b = (Σy - a*Σx) / n
    
    size_t n = keys.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(keys[i]);
        double y = static_cast<double>(positions[i]);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-9) {
        slope_ = 0.0;
        intercept_ = sum_y / n;
    } else {
        slope_ = (n * sum_xy - sum_x * sum_y) / denominator;
        intercept_ = (sum_y - slope_ * sum_x) / n;
    }
}

size_t LinearRegressionModel::predict(Key key, size_t data_size) const {
    double predicted = slope_ * static_cast<double>(key) + intercept_;
    predicted = std::max(0.0, std::min(predicted, static_cast<double>(data_size) - 1));
    return static_cast<size_t>(predicted);
}

double LinearRegressionModel::meanAbsoluteError(
    const std::vector<Key>& keys, 
    const std::vector<size_t>& positions) const {
    
    if (keys.empty()) return 0.0;
    
    double total_error = 0.0;
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t predicted = predict(keys[i], positions.size());
        int actual = static_cast<int>(positions[i]);
        int pred = static_cast<int>(predicted);
        total_error += std::abs(actual - pred);
    }
    return total_error / keys.size();
}

double LinearRegressionModel::maxError(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    if (keys.empty()) return 0.0;
    
    double max_err = 0.0;
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t predicted = predict(keys[i], positions.size());
        int actual = static_cast<int>(positions[i]);
        int pred = static_cast<int>(predicted);
        double err = std::abs(actual - pred);
        max_err = std::max(max_err, err);
    }
    return max_err;
}

// ============================================================================
// RMIStage Implementation
// ============================================================================

void RMIStage::build(const std::vector<Key>& keys,
                    const std::vector<size_t>& targets,
                    bool is_leaf) {
    if (keys.empty()) return;
    
    // Partition keys into fanout buckets
    std::vector<std::vector<Key>> key_buckets(fanout_);
    std::vector<std::vector<size_t>> target_buckets(fanout_);
    
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t bucket_id = std::min(targets[i], fanout_ - 1);
        key_buckets[bucket_id].push_back(keys[i]);
        target_buckets[bucket_id].push_back(targets[i]);
    }
    
    // Train a model for each bucket
    for (size_t b = 0; b < fanout_; ++b) {
        if (key_buckets[b].empty()) {
            // Create empty model
            LinearRegressionModel model;
            models_.push_back(model);
        } else {
            // Train model on this bucket
            LinearRegressionModel model;
            std::vector<size_t> positions(key_buckets[b].size());
            
            if (is_leaf) {
                // Leaf: target_buckets contain actual positions
                for (size_t i = 0; i < key_buckets[b].size(); ++i) {
                    positions[i] = target_buckets[b][i];
                }
            } else {
                // Internal: renormalize to 0..fanout-1 within bucket
                for (size_t i = 0; i < key_buckets[b].size(); ++i) {
                    positions[i] = i;  // Relative position within bucket
                }
            }
            
            model.train(key_buckets[b], positions);
            models_.push_back(model);
            
            // If leaf stage, store segment info
            if (is_leaf && !key_buckets[b].empty()) {
                RMISegment seg;
                seg.start_pos = target_buckets[b].front();
                seg.end_pos = target_buckets[b].back() + 1;
                seg.min_key = key_buckets[b].front();
                seg.max_key = key_buckets[b].back();
                seg.model = model;
                seg.mae = model.meanAbsoluteError(key_buckets[b], target_buckets[b]);
                seg.max_error = model.maxError(key_buckets[b], target_buckets[b]);
                segments_.push_back(seg);
            }
        }
    }
}

size_t RMIStage::predict(Key key, size_t data_size) const {
    if (models_.empty()) return 0;
    
    // Predict using first model as router
    double val = models_[0].getSlope() * static_cast<double>(key) + 
                models_[0].getIntercept();
    
    size_t model_id = static_cast<size_t>(
        std::max(0.0, std::min(val, static_cast<double>(fanout_ - 1)))
    );
    
    if (model_id >= models_.size()) {
        model_id = models_.size() - 1;
    }
    
    return models_[model_id].predict(key, data_size);
}

// ============================================================================
// RecursiveModelIndex Implementation
// ============================================================================

RecursiveModelIndex::RecursiveModelIndex(size_t root_fanout,
                                        size_t internal_fanout,
                                        size_t leaf_fanout)
    : root_fanout_(root_fanout),
      internal_fanout_(internal_fanout),
      leaf_fanout_(leaf_fanout) {}

void RecursiveModelIndex::build(const std::vector<Key>& keys) {
    if (keys.size() < 2) {
        // Single key case
        stages_.clear();
        RMIStage leaf_stage(leaf_fanout_);
        std::vector<size_t> positions(1, 0);
        leaf_stage.build(keys, positions, true);
        stages_.push_back(leaf_stage);
        return;
    }
    
    stages_.clear();
    
    // Create initial target: uniform distribution to leaf_fanout buckets
    std::vector<size_t> targets(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        targets[i] = (i * leaf_fanout_) / keys.size();
        targets[i] = std::min(targets[i], leaf_fanout_ - 1);
    }
    
    // Build recursively
    buildStage(keys, targets, 0);
}

void RecursiveModelIndex::buildStage(const std::vector<Key>& keys,
                                     std::vector<size_t>& targets,
                                     size_t stage_level) {
    size_t fanout = (stage_level == 0) ? root_fanout_ : internal_fanout_;
    bool is_leaf = (stage_level == 1);  // Next stage is leaf
    
    RMIStage stage(fanout);
    stage.build(keys, targets, is_leaf);
    stages_.push_back(stage);
    
    if (!is_leaf) {
        // Continue building: create new targets from current stage
        std::vector<size_t> new_targets(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            new_targets[i] = stage.predict(keys[i], fanout);
        }
        buildStage(keys, new_targets, stage_level + 1);
    }
}

SearchResult RecursiveModelIndex::search(Key key,
                                        const std::vector<KeyValuePair>& data) const {
    if (data.empty()) {
        return SearchResult(false, 0, 0, 0);
    }
    
    // Navigate through stages
    size_t predicted_pos = 0;
    double error_bound = 1.0;
    
    for (size_t stage_idx = 0; stage_idx < stages_.size(); ++stage_idx) {
        const RMIStage& stage = stages_[stage_idx];
        predicted_pos = stage.predict(key, data.size());
        error_bound *= 2.0;  // Propagate error bounds
    }
    
    // Binary search around predicted position
    return exponentialSearch(key, data, predicted_pos, error_bound);
}

SearchResult RecursiveModelIndex::exponentialSearch(
    Key key,
    const std::vector<KeyValuePair>& data,
    size_t pred_pos,
    double error_bound) const {
    
    size_t probes = 0;
    
    // Clamp predicted position
    pred_pos = std::min(pred_pos, data.size() - 1);
    
    // Exponential search outward
    size_t left = pred_pos, right = pred_pos;
    size_t search_range = static_cast<size_t>(error_bound) + 10;
    
    while (left > 0 && data[left].key > key) {
        left = (left > search_range) ? left - search_range : 0;
        probes++;
    }
    
    while (right < data.size() - 1 && data[right].key < key) {
        right = std::min(right + search_range, data.size() - 1);
        probes++;
    }
    
    // Binary search in range
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        probes++;
        
        if (data[mid].key < key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    if (left < data.size() && data[left].key == key) {
        return SearchResult(true, left, data[left].value, probes);
    }
    
    return SearchResult(false, left, 0, probes);
}

RecursiveModelIndex::RMIStats RecursiveModelIndex::getStatistics() const {
    RMIStats stats;
    stats.num_stages = stages_.size();
    stats.num_models = 0;
    stats.num_leaf_segments = 0;
    stats.avg_mae = 0.0;
    stats.max_mae = 0.0;
    stats.root_mae = 0.0;
    
    for (size_t i = 0; i < stages_.size(); ++i) {
        stats.num_models += stages_[i].getNumModels();
        
        if (i == stages_.size() - 1) {
            stats.num_leaf_segments = stages_[i].getSegments().size();
            double sum_mae = 0.0;
            for (const auto& seg : stages_[i].getSegments()) {
                sum_mae += seg.mae;
                stats.max_mae = std::max(stats.max_mae, seg.mae);
            }
            if (stats.num_leaf_segments > 0) {
                stats.avg_mae = sum_mae / stats.num_leaf_segments;
            }
        }
        
        if (i == 0 && stages_[i].getNumModels() > 0) {
            const auto& root_model = stages_[i].getModel(0);
            stats.root_mae = root_model.getSlope();  // Simplified
        }
    }
    
    stats.total_model_size_bytes = stats.num_models * sizeof(LinearRegressionModel);
    return stats;
}

} // namespace learned_index