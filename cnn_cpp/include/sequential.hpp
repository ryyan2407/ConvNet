#pragma once

#include <memory>
#include <vector>

#include "layer.hpp"

class Sequential {
public:
    void add(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
    Tensor predict(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void zero_grad();
    void update(float learning_rate);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
