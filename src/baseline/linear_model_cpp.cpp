// src/baseline/linear_model.cpp
#include "linear_model.h"
#include <algorithm>
#include <numeric>

namespace learned_index {

void LinearModel::train(const std::vector<Key>& keys,
                        const std::vector<size_t>& positions) {
    if (keys.empty() || keys.size() != positions.size()) {
        trained_ = false;
        return;
    }
    
    // Convert to double for numerical stability
    std::vector<double> x(keys.size());
    std::vector<double> y(positions.size());
    
    for (size_t i = 0; i < keys.size(); i++) {
        x[i] = static_cast<double>(keys[i]);
        y[i] = static_cast<double>(positions[i]);
    }
    
    fitLeastSquares(x, y);
    trained_ = true;
}

void LinearModel::fitLeastSquares(const std::vector<double>& x,
                                   const std::vector<double>& y) {
    size_t n = x.size();
    
    // Compute means
    double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    
    // Compute covariance and variance
    double numerator = 0.0, denominator = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x_diff = x[i] - x_mean;
        double y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }
    
    // Calculate slope and intercept
    if (std::abs(denominator) > 1e-10) {
        slope_ = numerator / denominator;
        intercept_ = y_mean - slope_ * x_mean;
    } else {
        slope_ = 0.0;
        intercept_ = y_mean;
    }
}

double LinearModel::predict(Key key) const {
    if (!trained_) return 0.0;
    return slope_ * static_cast<double>(key) + intercept_;
}

size_t LinearModel::predictPosition(Key key, size_t data_size) const {
    double pred = predict(key);
    
    if (pred < 0.0) return 0;
    if (pred >= static_cast<double>(data_size)) return data_size - 1;
    
    return static_cast<size_t>(pred);
}

double LinearModel::meanAbsoluteError(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    if (!trained_ || keys.empty()) return 0.0;
    
    double total_error = 0.0;
    for (size_t i = 0; i < keys.size(); i++) {
        double pred = predict(keys[i]);
        double actual = static_cast<double>(positions[i]);
        total_error += std::abs(pred - actual);
    }
    
    return total_error / keys.size();
}

double LinearModel::maxError(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    if (!trained_ || keys.empty()) return 0.0;
    
    double max_err = 0.0;
    for (size_t i = 0; i < keys.size(); i++) {
        double pred = predict(keys[i]);
        double actual = static_cast<double>(positions[i]);
        max_err = std::max(max_err, std::abs(pred - actual));
    }
    
    return max_err;
}

double LinearModel::rmse(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    if (!trained_ || keys.empty()) return 0.0;
    
    double sum_sq_error = 0.0;
    for (size_t i = 0; i < keys.size(); i++) {
        double pred = predict(keys[i]);
        double actual = static_cast<double>(positions[i]);
        double error = pred - actual;
        sum_sq_error += error * error;
    }
    
    return std::sqrt(sum_sq_error / keys.size());
}

double LinearModel::rSquared(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    if (!trained_ || keys.size() < 2) return 0.0;
    
    double y_mean = std::accumulate(positions.begin(), positions.end(), 0.0) /
                   positions.size();
    
    double ss_tot = 0.0, ss_res = 0.0;
    
    for (size_t i = 0; i < keys.size(); i++) {
        double actual = static_cast<double>(positions[i]);
        double pred = predict(keys[i]);
        
        ss_tot += (actual - y_mean) * (actual - y_mean);
        ss_res += (actual - pred) * (actual - pred);
    }
    
    if (std::abs(ss_tot) < 1e-10) return 0.0;
    
    return 1.0 - (ss_res / ss_tot);
}

ModelStats LinearModel::getStats(
    const std::vector<Key>& keys,
    const std::vector<size_t>& positions) const {
    
    ModelStats stats;
    stats.mean_absolute_error = meanAbsoluteError(keys, positions);
    stats.max_error = maxError(keys, positions);
    stats.rmse = rmse(keys, positions);
    stats.r_squared = rSquared(keys, positions);
    stats.num_samples = keys.size();
    
    return stats;
}

} // namespace learned_index
