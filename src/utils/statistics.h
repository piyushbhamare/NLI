// src/utils/statistics.h
#pragma once

#include "../core/types.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace learned_index {

/**
 * @brief Statistical analysis utilities
 */
class Statistics {
public:
    /**
     * @brief Compute percentile
     * @param data Sorted data
     * @param percentile Value 0-100
     * @return Percentile value
     */
    static double percentile(std::vector<double>& data, double p);
    
    /**
     * @brief Compute mean
     */
    static double mean(const std::vector<double>& data);
    
    /**
     * @brief Compute standard deviation
     */
    static double stddev(const std::vector<double>& data);
    
    /**
     * @brief Compute median
     */
    static double median(std::vector<double> data);
    
    /**
     * @brief Compute Population Stability Index (PSI)
     * Used for drift detection
     */
    static double psi(const std::vector<double>& expected,
                     const std::vector<double>& actual);
};

} // namespace learned_index
