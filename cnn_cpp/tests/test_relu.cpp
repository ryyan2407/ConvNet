#include <cassert>

#include "relu.hpp"

int main() {
    Tensor input(1, 1, 2, 3);
    input(0, 0, 0, 0) = -2.0f;
    input(0, 0, 0, 1) = -0.1f;
    input(0, 0, 0, 2) = 0.0f;
    input(0, 0, 1, 0) = 1.5f;
    input(0, 0, 1, 1) = 3.0f;
    input(0, 0, 1, 2) = -7.0f;

    ReLU relu;
    Tensor output = relu.forward(input);

    assert(output(0, 0, 0, 0) == 0.0f);
    assert(output(0, 0, 0, 1) == 0.0f);
    assert(output(0, 0, 0, 2) == 0.0f);
    assert(output(0, 0, 1, 0) == 1.5f);
    assert(output(0, 0, 1, 1) == 3.0f);
    assert(output(0, 0, 1, 2) == 0.0f);

    return 0;
}
