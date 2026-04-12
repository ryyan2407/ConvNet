#include <cassert>
#include <stdexcept>

#include "maxpool2d.hpp"

int main() {
    Tensor input(1, 1, 4, 4);
    const float values[16] = {
        1, 3, 2, 4,
        5, 6, 7, 8,
        0, 9, 3, 2,
        1, 4, 8, 6
    };

    int index = 0;
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = values[index++];
        }
    }

    MaxPool2D pool(2, 2);
    Tensor output = pool.forward(input);

    assert(output.H() == 2);
    assert(output.W() == 2);
    assert(output(0, 0, 0, 0) == 6.0f);
    assert(output(0, 0, 0, 1) == 8.0f);
    assert(output(0, 0, 1, 0) == 9.0f);
    assert(output(0, 0, 1, 1) == 8.0f);

    bool threw = false;
    try {
        MaxPool2D invalid_pool(3, 2);
        (void)invalid_pool.forward(input);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    return 0;
}
