#include <cassert>

#include "softmax.hpp"
#include "utils.hpp"

int main() {
    Tensor input(1, 1, 1, 3);
    input(0, 0, 0, 0) = 1.0f;
    input(0, 0, 0, 1) = 2.0f;
    input(0, 0, 0, 2) = 3.0f;

    Softmax softmax;
    Tensor output = softmax.forward(input);

    const float sum = output(0, 0, 0, 0) + output(0, 0, 0, 1) + output(0, 0, 0, 2);
    assert(nearly_equal(sum, 1.0f, 1e-5f));
    assert(output(0, 0, 0, 2) > output(0, 0, 0, 1));
    assert(output(0, 0, 0, 1) > output(0, 0, 0, 0));

    return 0;
}
