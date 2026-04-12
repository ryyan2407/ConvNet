#include "softmax.hpp"

#include <cmath>
#include <stdexcept>

Tensor Softmax::run_forward(const Tensor& input) const {
    if (input.C() != 1 || input.H() != 1) {
        throw std::runtime_error("Softmax expects input shape [N, 1, 1, features]");
    }

    Tensor output(input.N(), 1, 1, input.W());
    for (int n = 0; n < input.N(); ++n) {
        float max_value = input(n, 0, 0, 0);
        for (int i = 1; i < input.W(); ++i) {
            if (input(n, 0, 0, i) > max_value) {
                max_value = input(n, 0, 0, i);
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < input.W(); ++i) {
            const float value = std::exp(input(n, 0, 0, i) - max_value);
            output(n, 0, 0, i) = value;
            sum += value;
        }

        for (int i = 0; i < input.W(); ++i) {
            output(n, 0, 0, i) /= sum;
        }
    }
    return output;
}

Tensor Softmax::forward(const Tensor& input) {
    Tensor output = run_forward(input);
    cached_output_ = output;
    return output;
}

Tensor Softmax::infer(const Tensor& input) { return run_forward(input); }

Tensor Softmax::backward(const Tensor& grad_output) {
    Tensor grad_input(cached_output_.N(), cached_output_.C(), cached_output_.H(), cached_output_.W());
    for (int n = 0; n < cached_output_.N(); ++n) {
        for (int i = 0; i < cached_output_.W(); ++i) {
            float sum = 0.0f;
            for (int j = 0; j < cached_output_.W(); ++j) {
                const float jacobian = (i == j ? cached_output_(n, 0, 0, i) : 0.0f) -
                                       cached_output_(n, 0, 0, i) * cached_output_(n, 0, 0, j);
                sum += jacobian * grad_output(n, 0, 0, j);
            }
            grad_input(n, 0, 0, i) = sum;
        }
    }
    return grad_input;
}
