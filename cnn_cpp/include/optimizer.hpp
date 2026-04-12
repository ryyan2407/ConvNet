#pragma once

#include "sequential.hpp"

class Optimizer {
public:
    virtual void zero_grad(Sequential& model) = 0;
    virtual void step(Sequential& model) = 0;
    virtual ~Optimizer() = default;
};
