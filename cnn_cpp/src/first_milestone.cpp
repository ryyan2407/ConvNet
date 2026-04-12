#include <iostream>
#include <vector>

#include "conv2d.hpp"
#include "relu.hpp"

int main() {
    Tensor input(1, 1, 4, 4);
    float value = 1.0f;
    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            input(0, 0, h, w) = value++;
        }
    }

    Conv2D conv(1, 1, 3, 1, 1);
    conv.set_weights({
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f
    });
    conv.set_bias({0.0f});

    ReLU relu;
    Tensor conv_output = conv.forward(input);
    Tensor relu_output = relu.forward(conv_output);

    std::cout << "Milestone input shape: ";
    input.print_shape();
    std::cout << "Conv output:\n";
    conv_output.print_data();
    std::cout << "ReLU output:\n";
    relu_output.print_data();
    return 0;
}
