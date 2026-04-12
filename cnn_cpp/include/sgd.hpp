#pragma once

#include "optimizer.hpp"

class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate);

    void zero_grad(Sequential& model) override;
    void step(Sequential& model) override;

private:
    float learning_rate_;
};
