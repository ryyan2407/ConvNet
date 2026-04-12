#include <cassert>
#include <vector>

#include "linear.hpp"

int main() {
    Tensor input(1, 1, 1, 3);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 0, 2) = 3.0f;

    Linear linear(3, 2);
    linear.set_weights({
        1.0f, 0.0f, -1.0f,
        0.5f, 0.5f, 0.5f
    });
    linear.set_bias({0.0f, 1.0f});

    Tensor output = linear.forward(input);
    assert(output.W() == 2);
    assert(output(0, 0, 0, 0) == -2.0f);
    assert(output(0, 0, 0, 1) == 4.0f);

    return 0;
}
