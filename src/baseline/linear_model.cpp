#include "linear_model.h"
#include <algorithm>
#include <numeric>
#include <cmath>
namespace learned_index {
void LinearModel::train(const std::vector<Key>& keys, const std::vector<size_t>& positions) {
    if (keys.empty() || keys.size() != positions.size()) { trained_ = false; return; }
    std::vector<double> x(keys.size()), y(positions.size());
    for (size_t i = 0; i < keys.size(); i++) { x[i] = keys[i]; y[i] = positions[i]; }
    fitLeastSquares(x, y);
    trained_ = true;
}
void LinearModel::fitLeastSquares(const std::vector<double>& x, const std::vector<double>& y) {
    size_t n = x.size();
    double x_mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double num = 0.0, denom = 0.0;
    for (size_t i = 0; i < n; i++) { double xd = x[i] - x_mean, yd = y[i] - y_mean; num += xd * yd; denom += xd * xd; }
    slope_ = (denom > 1e-10) ? (num / denom) : 0.0;
    intercept_ = y_mean - slope_ * x_mean;
}
double LinearModel::predict(Key key) const { if (!trained_) return 0.0; return slope_ * key + intercept_; }
size_t LinearModel::predictPosition(Key key, size_t data_size) const {
    double pred = predict(key);
    if (pred < 0.0) return 0;
    if (pred >= (double)data_size) return data_size - 1;
    return (size_t)pred;
}
double LinearModel::meanAbsoluteError(const std::vector<Key>& keys, const std::vector<size_t>& positions) const {
    if (!trained_ || keys.empty()) return 0.0;
    double total = 0.0;
    for (size_t i = 0; i < keys.size(); i++) total += std::abs(predict(keys[i]) - (double)positions[i]);
    return total / keys.size();
}
double LinearModel::maxError(const std::vector<Key>& keys, const std::vector<size_t>& positions) const {
    if (!trained_ || keys.empty()) return 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < keys.size(); i++) max_err = std::max(max_err, std::abs(predict(keys[i]) - (double)positions[i]));
    return max_err;
}
ModelStats LinearModel::getStats(const std::vector<Key>& keys, const std::vector<size_t>& positions) const {
    ModelStats stats;
    stats.mean_absolute_error = meanAbsoluteError(keys, positions);
    stats.max_error = maxError(keys, positions);
    stats.num_samples = keys.size();
    return stats;
}
}
