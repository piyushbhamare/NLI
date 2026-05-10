#pragma once
#include "../core/types.h"
#include <vector>
#include <cstddef>
namespace learned_index {
class LinearModel {
public:
    LinearModel() : slope_(0.0), intercept_(0.0), trained_(false) {}
    void train(const std::vector<Key>& keys, const std::vector<size_t>& positions);
    double predict(Key key) const;
    size_t predictPosition(Key key, size_t data_size) const;
    double slope() const { return slope_; }
    double intercept() const { return intercept_; }
    bool isTrained() const { return trained_; }
    double meanAbsoluteError(const std::vector<Key>& keys, const std::vector<size_t>& pos) const;
    double maxError(const std::vector<Key>& keys, const std::vector<size_t>& pos) const;
    ModelStats getStats(const std::vector<Key>& keys, const std::vector<size_t>& pos) const;
private:
    double slope_, intercept_;
    bool trained_;
    void fitLeastSquares(const std::vector<double>& x, const std::vector<double>& y);
};
}
