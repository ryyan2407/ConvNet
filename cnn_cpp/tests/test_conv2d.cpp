#include <cassert>
#include <stdexcept>
#include <vector>

#include "conv2d.hpp"

int main() {
    Tensor input(1, 1, 4, 4);
    float value = 1.0f;
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = value++;
        }
    }

    Conv2D conv(1, 1, 3, 1, 0);
    conv.set_weights(std::vector<float>(9, 1.0f));
    conv.set_bias({0.0f});

    Tensor output = conv.forward(input);
    assert(output.N() == 1);
    assert(output.C() == 1);
    assert(output.H() == 2);
    assert(output.W() == 2);

    assert(output(0, 0, 0, 0) == 54.0f);
    assert(output(0, 0, 0, 1) == 63.0f);
    assert(output(0, 0, 1, 0) == 90.0f);
    assert(output(0, 0, 1, 1) == 99.0f);

    Conv2D padded(1, 1, 3, 1, 1);
    padded.set_weights(std::vector<float>(9, 1.0f));
    padded.set_bias({0.0f});
    Tensor padded_output = padded.forward(input);
    assert(padded_output.H() == 4);
    assert(padded_output.W() == 4);
    assert(padded_output(0, 0, 0, 0) == 14.0f);

    bool threw = false;
    try {
        Conv2D invalid_stride(1, 1, 3, 2, 0);
        (void)invalid_stride.forward(input);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    return 0;
}
