#pragma once

#include "layer.hpp"

class Softmax : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor infer(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    Tensor cached_output_;
    Tensor run_forward(const Tensor& input) const;
};
