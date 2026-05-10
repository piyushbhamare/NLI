#ifndef LEARNED_INDEX_RMI_MODEL_H
#define LEARNED_INDEX_RMI_MODEL_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <cstddef>
#include "types.h"

namespace learned_index {

/**
 * @brief Linear regression model for position prediction
 * Implements: position = slope * key + intercept
 */
class LinearRegressionModel {
public:
    LinearRegressionModel() : slope_(0.0), intercept_(0.0), min_key_(0), max_key_(0) {}
    
    /**
     * Train model using least-squares regression
     * @param keys Sorted array of keys
     * @param positions Target positions (indices in sorted array)
     */
    void train(const std::vector<Key>& keys, const std::vector<size_t>& positions);
    
    /**
     * Predict position for a given key
     * @param key Input key
     * @param data_size Total data size (for clamping)
     * @return Predicted position
     */
    size_t predict(Key key, size_t data_size) const;
    
    /**
     * Get model parameters
     */
    double getSlope() const { return slope_; }
    double getIntercept() const { return intercept_; }
    Key getMinKey() const { return min_key_; }
    Key getMaxKey() const { return max_key_; }
    
    /**
     * Calculate model statistics
     */
    double meanAbsoluteError(const std::vector<Key>& keys, 
                            const std::vector<size_t>& positions) const;
    double maxError(const std::vector<Key>& keys, 
                   const std::vector<size_t>& positions) const;
    
private:
    double slope_;
    double intercept_;
    Key min_key_;
    Key max_key_;
};

/**
 * @brief Segment information in RMI leaf stage
 */
struct RMISegment {
    size_t start_pos;           // Start position in sorted array
    size_t end_pos;             // End position in sorted array
    Key min_key;                // Min key in segment
    Key max_key;                // Max key in segment
    LinearRegressionModel model; // Leaf model for this segment
    double mae;                 // Mean absolute error of model
    double max_error;           // Max error in segment
};

/**
 * @brief Single stage in RMI hierarchy
 * Organizes models as a forest (multiple trees if internal stage)
 */
class RMIStage {
public:
    explicit RMIStage(size_t fanout = 256) : fanout_(fanout) {}
    
    /**
     * Build stage from keys and target stage output
     * @param keys Sorted array of keys
     * @param targets Target outputs (child indices for internal, positions for leaf)
     * @param is_leaf Whether this is leaf stage
     */
    void build(const std::vector<Key>& keys, 
              const std::vector<size_t>& targets,
              bool is_leaf = false);
    
    /**
     * Predict at this stage
     * @param key Input key
     * @param data_size For leaf stage position clamping
     * @return Child index (internal) or predicted position (leaf)
     */
    size_t predict(Key key, size_t data_size = 0) const;
    
    /**
     * Get number of models in this stage
     */
    size_t getNumModels() const { return models_.size(); }
    
    /**
     * Get model at index
     */
    const LinearRegressionModel& getModel(size_t idx) const { 
        return models_.at(idx); 
    }
    
    /**
     * Get all segments (leaf stage only)
     */
    const std::vector<RMISegment>& getSegments() const { return segments_; }
    
    size_t getFanout() const { return fanout_; }
    
private:
    size_t fanout_;
    std::vector<LinearRegressionModel> models_;
    std::vector<RMISegment> segments_;  // Only for leaf stage
};

/**
 * @brief Complete Recursive Model Index structure
 * Multi-level hierarchy: Root → Intermediate stages → Leaf segments
 */
class RecursiveModelIndex {
public:
    /**
     * Construct RMI with specified parameters
     * @param root_fanout Fanout of root model
     * @param internal_fanout Fanout of internal stages
     * @param leaf_fanout Number of leaf segments
     */
    explicit RecursiveModelIndex(size_t root_fanout = 256, 
                                size_t internal_fanout = 256,
                                size_t leaf_fanout = 1024);
    
    /**
     * Build RMI from sorted keys
     * @param keys Sorted array of keys
     */
    void build(const std::vector<Key>& keys);
    
    /**
     * Search for key using RMI prediction + exponential search
     * @param key Key to search
     * @param data Vector of key-value pairs (sorted)
     * @return SearchResult with position and statistics
     */
    SearchResult search(Key key, const std::vector<KeyValuePair>& data) const;
    
    /**
     * Get number of stages in RMI
     */
    size_t getNumStages() const { return stages_.size(); }
    
    /**
     * Get stage at level
     */
    const RMIStage& getStage(size_t level) const { 
        return stages_.at(level); 
    }
    
    /**
     * Calculate RMI statistics
     */
    struct RMIStats {
        size_t num_stages;
        size_t num_models;
        size_t num_leaf_segments;
        double avg_mae;
        double max_mae;
        double root_mae;
        size_t total_model_size_bytes;
    };
    
    RMIStats getStatistics() const;
    
private:
    size_t root_fanout_;
    size_t internal_fanout_;
    size_t leaf_fanout_;
    std::vector<RMIStage> stages_;
    
    /**
     * Recursively build stages
     */
    void buildStage(const std::vector<Key>& keys,
                   std::vector<size_t>& targets,
                   size_t stage_level);
    
    /**
     * Search within bounded region after prediction
     */
    SearchResult exponentialSearch(Key key, 
                                  const std::vector<KeyValuePair>& data,
                                  size_t pred_pos,
                                  double error_bound) const;
};

} // namespace learned_index

#endif // LEARNED_INDEX_RMI_MODEL_H