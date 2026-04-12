#include <cassert>

#include "flatten.hpp"

int main() {
    Tensor input(1, 2, 2, 2);
    float value = 1.0f;
    for (int c = 0; c < 2; ++c) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                input(0, c, h, w) = value++;
            }
        }
    }

    Flatten flatten;
    Tensor output = flatten.forward(input);

    assert(output.N() == 1);
    assert(output.C() == 1);
    assert(output.H() == 1);
    assert(output.W() == 8);
    for (int i = 0; i < 8; ++i) {
        assert(output(0, 0, 0, i) == static_cast<float>(i + 1));
    }

    return 0;
}
