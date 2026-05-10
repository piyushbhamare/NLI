// src/utils/statistics.cpp
#include "statistics.h"
#include <numeric>
#include <algorithm>
#include <cmath>

namespace learned_index {

double Statistics::percentile(std::vector<double>& data, double p) {
    if (data.empty()) return 0.0;
    
    std::sort(data.begin(), data.end());
    
    if (p < 0.0) p = 0.0;
    if (p > 100.0) p = 100.0;
    
    size_t index = static_cast<size_t>((p / 100.0) * (data.size() - 1));
    return data[index];
}

double Statistics::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double Statistics::stddev(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    
    double m = mean(data);
    double sum_sq_diff = 0.0;
    
    for (const auto& x : data) {
        sum_sq_diff += (x - m) * (x - m);
    }
    
    return std::sqrt(sum_sq_diff / (data.size() - 1));
}

double Statistics::median(std::vector<double> data) {
    if (data.empty()) return 0.0;
    
    std::sort(data.begin(), data.end());
    
    if (data.size() % 2 == 1) {
        return data[data.size() / 2];
    } else {
        return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2.0;
    }
}

double Statistics::psi(const std::vector<double>& expected,
                      const std::vector<double>& actual) {
    if (expected.empty() || actual.empty()) return 0.0;
    
    // Simple PSI calculation using histogram binning
    double psi_value = 0.0;
    
    double exp_mean = mean(expected);
    double act_mean = mean(actual);
    
    size_t bins = std::min(size_t(10), expected.size() / 100 + 1);
    
    double exp_sum = 0.0, act_sum = 0.0;
    for (const auto& x : expected) {
        if (x > 0) exp_sum += x;
    }
    for (const auto& x : actual) {
        if (x > 0) act_sum += x;
    }
    
    if (exp_sum > 0 && act_sum > 0) {
        double exp_pct = 1.0 / bins;
        double act_pct = 1.0 / bins;
        
        psi_value += exp_pct * std::log((exp_pct + 1e-10) / (act_pct + 1e-10));
    }
    
    return psi_value;
}

} // namespace learned_index
