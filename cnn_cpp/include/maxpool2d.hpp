#pragma once

#include <vector>

#include "layer.hpp"

class MaxPool2D : public Layer {
public:
    MaxPool2D(int kernel_size, int stride);

    Tensor forward(const Tensor& input) override;
    Tensor infer(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    int kernel_size_;
    int stride_;
    Tensor cached_input_;
    std::vector<int> max_indices_;
    Tensor run_forward(const Tensor& input, bool store_indices);
};
