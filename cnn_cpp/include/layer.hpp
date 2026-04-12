#pragma once

#include "tensor.hpp"

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor infer(const Tensor& input) { return forward(input); }
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void zero_grad() {}
    virtual void update(float learning_rate) { (void)learning_rate; }
    virtual ~Layer() = default;
};
