#pragma once

#include "layer.hpp"

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor infer(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    Tensor cached_input_;
};
