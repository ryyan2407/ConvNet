#pragma once

#include <vector>

#include "tensor.hpp"

struct LabelView {
    const int* data;
    int size;

    int operator[](int index) const { return data[index]; }
};

class Loss {
public:
    virtual float forward(const Tensor& prediction, LabelView targets) = 0;
    virtual float forward(const Tensor& prediction, const std::vector<int>& targets) = 0;
    virtual Tensor backward() = 0;
    virtual ~Loss() = default;
};
