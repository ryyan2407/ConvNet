#include "flatten.hpp"

namespace {

Tensor flatten_forward_impl(const Tensor& input) {
    Tensor output(input.N(), 1, 1, input.C() * input.H() * input.W());
    for (int n = 0; n < input.N(); ++n) {
        int index = 0;
        for (int c = 0; c < input.C(); ++c) {
            for (int h = 0; h < input.H(); ++h) {
                for (int w = 0; w < input.W(); ++w) {
                    output(n, 0, 0, index++) = input(n, c, h, w);
                }
            }
        }
    }
    return output;
}

}  // namespace

Tensor Flatten::forward(const Tensor& input) {
    cached_input_ = input;
    return flatten_forward_impl(input);
}

Tensor Flatten::infer(const Tensor& input) { return flatten_forward_impl(input); }

Tensor Flatten::backward(const Tensor& grad_output) {
    Tensor grad_input(cached_input_.N(), cached_input_.C(), cached_input_.H(), cached_input_.W());
    for (int n = 0; n < cached_input_.N(); ++n) {
        int index = 0;
        for (int c = 0; c < cached_input_.C(); ++c) {
            for (int h = 0; h < cached_input_.H(); ++h) {
                for (int w = 0; w < cached_input_.W(); ++w) {
                    grad_input(n, c, h, w) = grad_output(n, 0, 0, index++);
                }
            }
        }
    }
    return grad_input;
}
